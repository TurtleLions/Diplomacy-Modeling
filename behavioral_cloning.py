import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as smp
import torch.multiprocessing as mp
import psutil
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import gc
from diplomacy import Game

from diplomacy_helpers import (
    build_global_vocab, 
    get_global_action_mask, 
    parse_state_to_tensor, 
    DiplomacyTransformer
)

import signal
import time

def log_gpu_memory(rank):
    if rank != 0:
        return
    allocated = torch.cuda.memory_allocated(rank) / (1024**3)
    reserved = torch.cuda.memory_reserved(rank) / (1024**3)
    print(f"[LOG] GPU 0 VRAM Allocated {allocated:.2f} GB | Reserved {reserved:.2f} GB")
    print("-" * 40)

# Global variables for background workers
worker_prov_to_idx = None
worker_order_to_idx = None
worker_num_provs = None
worker_none_idx = None
worker_vocab_size = None

def init_worker(p_idx, o_idx, n_provs, n_idx, v_size):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global worker_prov_to_idx, worker_order_to_idx, worker_num_provs, worker_none_idx, worker_vocab_size
    worker_prov_to_idx = p_idx
    worker_order_to_idx = o_idx
    worker_num_provs = n_provs
    worker_none_idx = n_idx
    worker_vocab_size = v_size

def warm_os_cache(filepath):
    print(f"Forcing OS page cache for {os.path.basename(filepath)}...")
    mapped_file = np.memmap(filepath, mode='r')
    
    # Touch one byte every 4096 bytes to load every page into RAM
    _ = mapped_file[::4096].copy()
    
    del mapped_file

def compress_mask_file():
    cache_dir = "/data/restanislao/diplomacy/"
    mask_path = os.path.join(cache_dir, "mask.bin")
    targets_path = os.path.join(cache_dir, "targets.bin")
    packed_mask_path = os.path.join(cache_dir, "mask_packed.bin")
    
    num_provs = 82
    
    targets_bytes = os.path.getsize(targets_path)
    total_samples = targets_bytes // (8 * num_provs)
    
    mask_bytes = os.path.getsize(mask_path)
    vocab_size = mask_bytes // (total_samples * num_provs)
    
    print(f"Detected {total_samples} samples and a vocab size of {vocab_size}")
    print("Loading original mask for compression...")
    
    mask_mmap = np.memmap(mask_path, dtype=np.bool_, mode='r', shape=(total_samples, num_provs, vocab_size))
    
    chunk_size = 10000
    total_chunks = (total_samples + chunk_size - 1) // chunk_size
    
    print(f"Packing into {packed_mask_path}...")
    with open(packed_mask_path, 'wb') as f:
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_samples)
            
            chunk = np.array(mask_mmap[start_idx:end_idx], copy=True)
            chunk_flat = chunk.reshape(end_idx - start_idx, -1)
            
            packed_chunk = np.packbits(chunk_flat, axis=1)
            f.write(packed_chunk.tobytes())
            
            f.flush()
            os.fsync(f.fileno())
            
            del chunk
            del chunk_flat
            del packed_chunk
            gc.collect()
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{total_chunks} chunks")
                
    print("\nCompression complete.")
    old_size = os.path.getsize(mask_path) / (1024**3)
    new_size = os.path.getsize(packed_mask_path) / (1024**3)
    print(f"Reduced mask from {old_size:.2f} GB to {new_size:.2f} GB")

# --- RESOURCE LOGGING ---
def log_system_resources(stage_name, rank=0):
    if rank != 0:
        return
        
    print(f"\n[LOG] Resource Usage - {stage_name}")
    ram_info = psutil.virtual_memory()
    print(f"System RAM Used {ram_info.used / (1024**3):.1f} GB of {ram_info.total / (1024**3):.1f} GB ({ram_info.percent}%)")
    print("-" * 40)

# --- 1. DATASET PREPARATION ---

class DiplomacyMemmapDataset(Dataset):
    def __init__(self, history_path, mask_packed_path, targets_path, total_samples, num_provs, vocab_size):
        self.history_path = history_path
        self.mask_packed_path = mask_packed_path
        self.targets_path = targets_path
        
        self.total_samples = total_samples
        self.num_provs = num_provs
        self.vocab_size = vocab_size
        
        # Calculate the exact number of booleans and bytes per sample
        self.bools_per_sample = num_provs * vocab_size
        self.bytes_per_sample = int(np.ceil(self.bools_per_sample / 8.0))
        
        self.history = None
        self.mask_packed = None
        self.targets = None
        
    def __len__(self):
        return self.total_samples
        
    def __getitem__(self, idx):
        if self.history is None:
            self.history = np.memmap(self.history_path, dtype=np.int8, mode='r', shape=(self.total_samples, 3, self.num_provs, 19))
            self.mask_packed = np.memmap(self.mask_packed_path, dtype=np.uint8, mode='r', shape=(self.total_samples, self.bytes_per_sample))
            self.targets = np.memmap(self.targets_path, dtype=np.int64, mode='r', shape=(self.total_samples, self.num_provs))
            
        # Return the raw compressed bytes directly to PyTorch
        return {
            'history': torch.from_numpy(self.history[idx].copy()), 
            'mask_packed': torch.from_numpy(self.mask_packed[idx].copy()), 
            'targets': torch.from_numpy(self.targets[idx].copy())  
        }

def _process_single_line(line):
    global worker_prov_to_idx, worker_order_to_idx, worker_num_provs, worker_none_idx, worker_vocab_size
    game_data = json.loads(line)
    
    if game_data.get('map', 'standard') != 'standard':
        return None
        
    game_engine = Game()
    provinces = list(game_engine.map.locs)
    history_buffer = np.zeros((3, worker_num_provs, 19), dtype=np.int8)
    
    g_histories, g_masks, g_targets = [], [], []
    
    for phase in game_data.get('phases', []):
        game_engine.set_state(phase['state'])
        current_state = parse_state_to_tensor(phase)
        
        history_buffer = np.roll(history_buffer, shift=1, axis=0)
        history_buffer[0] = current_state
        
        orders_dict = phase.get('orders', {})
        for power, text_orders in orders_dict.items():
            if not text_orders:
                continue
            
            mask = get_global_action_mask(game_engine, power, provinces, worker_order_to_idx)
            targets = np.full(worker_num_provs, worker_none_idx, dtype=np.int64)
            
            for order_str in text_orders:
                clean_order = order_str.replace('*', '')
                parts = clean_order.split()
                if len(parts) >= 2:
                    u_loc = parts[1].split('/')[0]
                    if u_loc in worker_prov_to_idx and clean_order in worker_order_to_idx:
                        p_idx = worker_prov_to_idx[u_loc]
                        targets[p_idx] = worker_order_to_idx[clean_order]
            
            # Compress the mask immediately to save RAM
            packed_mask = np.packbits(mask.flatten())
            
            g_histories.append(history_buffer.copy())
            g_masks.append(packed_mask)
            g_targets.append(targets)
            
    if not g_histories:
        return None
        
    return (
        np.stack(g_histories), 
        np.stack(g_masks), 
        np.stack(g_targets)
    )

def process_and_save_to_disk(json_path, cache_dir="./dataset_cache", max_games=None):
    os.makedirs(cache_dir, exist_ok=True)
    history_path = os.path.join(cache_dir, "history.bin")
    mask_packed_path = os.path.join(cache_dir, "mask_packed.bin")
    targets_path = os.path.join(cache_dir, "targets.bin")
    
    if os.path.exists(history_path) and os.path.exists(mask_packed_path) and os.path.exists(targets_path):
        print("Found existing binary cache skipping JSON parsing.")
        order_to_idx, _ = build_global_vocab()
        none_idx = order_to_idx['NONE']
        
        # Dynamically measure the map so it always evaluates to 82
        temp_engine = Game()
        num_provs = len(temp_engine.map.locs) 
        vocab_size = len(order_to_idx)
        
        targets_bytes = os.path.getsize(targets_path)
        total_samples = targets_bytes // (8 * num_provs)
        return history_path, mask_packed_path, targets_path, total_samples, num_provs, vocab_size, none_idx

    log_system_resources("Starting JSON Loading")
    order_to_idx, _ = build_global_vocab()
    none_idx = order_to_idx['NONE']
    vocab_size = len(order_to_idx)
    
    temp_engine = Game()
    provinces = list(temp_engine.map.locs)
    prov_to_idx = {p: i for i, p in enumerate(provinces)}
    num_provs = len(provinces)
    
    def line_generator():
        with open(json_path, 'r') as f:
            games_yielded = 0
            for line in f:
                if not line.strip():
                    continue
                yield line
                
                games_yielded += 1
                if max_games and games_yielded >= max_games:
                    break

    num_cores = min(8, smp.cpu_count())
    print(f"Starting standard multiprocessing pool with {num_cores} workers")
    
    total_samples = 0
    # REMOVED: buffer_limit, hist_buffer, mask_buffer, targ_buffer entirely!

    with open(history_path, 'wb') as f_hist, \
         open(mask_packed_path, 'wb') as f_mask, \
         open(targets_path, 'wb') as f_targ:
             
        with smp.Pool(
            processes=num_cores, 
            initializer=init_worker, 
            initargs=(prov_to_idx, order_to_idx, num_provs, none_idx, vocab_size),
            maxtasksperchild=10 # LOWERED to 10: Aggressively restarts workers to prevent Game() memory leaks
        ) as pool:
            try:
                # LOWERED chunksize to 5: Stops the workers from flooding the IPC queue
                for i, result in enumerate(pool.imap_unordered(_process_single_line, line_generator(), chunksize=5)):
                    if (i + 1) % 1000 == 0:
                        print(f"Processed {i + 1} games")
                        # Log RAM every 1000 games to prove it's stable
                        ram_used = psutil.virtual_memory().used / (1024**3)
                        print(f"  -> System RAM in use: {ram_used:.1f} GB")
                        
                    if result is None:
                        continue
                    
                    h_arr, m_arr, t_arr = result
                    
                    # Write directly to the OS buffer! No more np.concatenate!
                    f_hist.write(h_arr.tobytes())
                    f_mask.write(m_arr.tobytes())
                    f_targ.write(t_arr.tobytes())
                    
                    total_samples += len(h_arr)
                    
                    # Delete the arrays immediately to free RAM
                    del h_arr, m_arr, t_arr, result

            except KeyboardInterrupt:
                print("\nRun canceled by user. Forcibly terminating workers to prevent freezing.")
                pool.terminate()
                pool.join()
                raise
                    
    log_system_resources("Finished Writing Binary Cache")
    print(f"Total training samples generated {total_samples}")
    return history_path, mask_packed_path, targets_path, total_samples, num_provs, vocab_size, none_idx

# --- 2. DISTRIBUTED WORKER LOOP ---

def train_worker(rank, world_size, paths_and_metadata):
    history_path, mask_packed_path, targets_path, total_samples, num_provs, vocab_size, none_idx = paths_and_metadata
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    try:
        torch.set_num_threads(4) 
        
        batch_size_per_gpu = 512
        learning_rate = 1e-4
        epochs = 3
        
        dataset = DiplomacyMemmapDataset(history_path, mask_packed_path, targets_path, total_samples, num_provs, vocab_size)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size_per_gpu, 
            shuffle=False, 
            sampler=sampler,
            num_workers=4,              # 2 background workers per GPU
            pin_memory=True,
            prefetch_factor=2,          # Each worker cues up 2 batches
            persistent_workers=True     # Keeps workers alive between epochs
        )
        
        net = DiplomacyTransformer(num_provinces=num_provs, vocab_size=vocab_size).to(rank)
        net = DDP(net, device_ids=[rank], find_unused_parameters=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        
        if rank == 0:
            log_system_resources("Model Loaded to GPUs", rank)
            log_gpu_memory(rank)
            print("\n--- STARTING DISTRIBUTED BEHAVIORAL CLONING ---")
        
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            net.train()
            total_loss = 0.0
            
            epoch_start_time = time.time()
            batch_start_time = time.time()
            
            for batch_idx, batch in enumerate(dataloader):
                history = batch['history'].to(rank, dtype=torch.float32, non_blocking=True)
                targets = batch['targets'].to(rank, non_blocking=True)
                
                # Load the tiny compressed array into VRAM
                packed_mask = batch['mask_packed'].to(rank, non_blocking=True)
                
                # --- GPU BIT UNPACKING ---
                # Shift bits 7 to 0 to extract MSB first (matching np.packbits)
                shifts = torch.arange(7, -1, -1, device=rank, dtype=torch.uint8)
                unpacked = (packed_mask.unsqueeze(-1) >> shifts) & 1
                
                # Flatten the bits and trim any trailing padding
                bools_per_sample = num_provs * vocab_size
                mask_flat_bits = unpacked.view(packed_mask.size(0), -1)[:, :bools_per_sample].bool()
                mask = mask_flat_bits.view(packed_mask.size(0), num_provs, vocab_size)
                # -------------------------
                
                optimizer.zero_grad()
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # The rest of your forward pass remains completely untouched!
                    logits, state_value = net(history, targets)
                    
                    logits_flat = logits.view(-1, vocab_size)
                    targets_flat = targets.view(-1)
                    mask_flat = mask.view(-1, vocab_size)
                    
                    is_target_legal = mask_flat.gather(1, targets_flat.unsqueeze(1)).squeeze()
                    valid_training_mask = (targets_flat != none_idx) & is_target_legal
                    
                    if valid_training_mask.any():
                        valid_logits = logits_flat[valid_training_mask]
                        valid_masks = mask_flat[valid_training_mask]
                        valid_targets = targets_flat[valid_training_mask]
                        
                        valid_logits_masked = valid_logits.masked_fill(~valid_masks, -1e4)
                        loss = criterion(valid_logits_masked, valid_targets)
                    else:
                        loss = (logits_flat.mean() * 0.0)
                        
                    # TRICK DDP: Add the value_head to the autograd graph with a weight of 0
                    loss = loss + (state_value * 0).sum()
                
                # Backward pass remains the same
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()
                    
                if rank == 0 and batch_idx % 20 == 0:
                    elapsed = time.time() - batch_start_time
                    samples_processed = 20 * batch_size_per_gpu * world_size
                    throughput = samples_processed / elapsed if batch_idx > 0 else 0
                    
                    print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss {loss.item():.4f} | Time {elapsed:.2f}s | Speed {throughput:.0f} samples/s")
                    batch_start_time = time.time()
                    
            if rank == 0:
                epoch_time = time.time() - epoch_start_time
                avg_loss = total_loss / len(dataloader)
                print(f"\n==> Epoch {epoch+1} Complete")
                print(f"==> Average Loss {avg_loss:.4f} | Total Epoch Time {epoch_time:.2f}s")
                log_system_resources(f"End of Epoch {epoch+1}", rank)
                log_gpu_memory(rank)
                
        if rank == 0:
            save_path = "diplomacy_transformer_bc.pth"
            torch.save(net.module.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            
    finally:
        dist.destroy_process_group()

# --- 3. MAIN EXECUTION ---

def main():
    dataset_path = "./datasets/standard_no_press.jsonl"
    ssd_cache_directory = "/data/restanislao/diplomacy/"
    
    paths_and_metadata = process_and_save_to_disk(
        dataset_path, 
        cache_dir=ssd_cache_directory
    )
    
    history_path, mask_packed_path, targets_path, total_samples, num_provs, vocab_size, none_idx = paths_and_metadata
    
    print("\nPre-loading dataset into system RAM...")
    # warm_os_cache(history_path)
    # warm_os_cache(mask_packed_path)
    # warm_os_cache(targets_path)
    print("Caching complete. Starting workers.\n")
    
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("Error no GPUs detected")
        return
        
    print(f"Spawning {world_size} distributed workers")
    
    mp.spawn(
        train_worker,
        args=(world_size, paths_and_metadata),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()