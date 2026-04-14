"""
Behavioral Cloning pipeline for the Diplomacy Transformer.
Processes JSON game history into memory-mapped arrays and trains the unit-centric model using DDP.
"""
import os
import time
import json
import signal
import argparse
import multiprocessing as smp

import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from diplomacy import Game

from diplomacy_helpers import (
    build_global_vocab, 
    get_global_action_mask, 
    parse_state_to_tensor, 
    DiplomacyTransformer
)

# Constants
GLOBAL_POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
FEATURE_DIM = 46
MAX_SPARSE_MASK_LEN = 1200

# Global variables for multiprocessing workers.
worker_prov_to_idx = None
worker_order_to_idx = None
worker_num_provs = None
worker_none_idx = None
worker_vocab_size = None

def init_worker(p_idx, o_idx, n_provs, n_idx, v_size):
    """Initializes global variables for multiprocessing pool workers."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global worker_prov_to_idx, worker_order_to_idx, worker_num_provs, worker_none_idx, worker_vocab_size
    worker_prov_to_idx = p_idx
    worker_order_to_idx = o_idx
    worker_num_provs = n_provs
    worker_none_idx = n_idx
    worker_vocab_size = v_size

def log_system_resources(stage_name, rank=0):
    if rank != 0:
        return
    print(f"\n[LOG] Resource Usage - {stage_name}")
    ram_info = psutil.virtual_memory()
    print(f"System RAM Used: {ram_info.used / (1024**3):.1f} GB of {ram_info.total / (1024**3):.1f} GB ({ram_info.percent}%)")
    print("-" * 40)

def log_gpu_memory(rank):
    if rank != 0:
        return
    allocated = torch.cuda.memory_allocated(rank) / (1024**3)
    reserved = torch.cuda.memory_reserved(rank) / (1024**3)
    print(f"[LOG] GPU 0 VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
    print("-" * 40)


class DiplomacyMemmapDataset(Dataset):
    """Dataset class utilizing memory-mapped files for out-of-core training."""
    def __init__(self, history_path, mask_sparse_path, targets_path, total_samples, num_provs, vocab_size):
        self.history_path = history_path
        self.mask_sparse_path = mask_sparse_path
        self.targets_path = targets_path
        
        self.total_samples = total_samples
        self.num_provs = num_provs
        self.vocab_size = vocab_size
        
        self.history = None
        self.sparse_mask = None
        self.targets = None
        
    def __len__(self):
        return self.total_samples
        
    def __getitem__(self, idx):
        # Lazy initialization of memmap arrays for multiprocessing compatibility
        if self.history is None:
            self.history = np.memmap(self.history_path, dtype=np.int8, mode='r', shape=(self.total_samples, 3, self.num_provs, FEATURE_DIM))
            self.sparse_mask = np.memmap(self.mask_sparse_path, dtype=np.int32, mode='r', shape=(self.total_samples, MAX_SPARSE_MASK_LEN))
            self.targets = np.memmap(self.targets_path, dtype=np.int64, mode='r', shape=(self.total_samples, self.num_provs))
            
        return {
            'history': torch.from_numpy(self.history[idx].copy()),
            'sparse_mask': torch.from_numpy(self.sparse_mask[idx].copy()), 
            'targets': torch.from_numpy(self.targets[idx].copy())
        }

def _process_single_line(line):
    """Parses a single JSON game record and extracts state, masks, and target orders."""
    global worker_prov_to_idx, worker_order_to_idx, worker_num_provs, worker_none_idx, worker_vocab_size
    game_data = json.loads(line)
    
    if game_data.get('map', 'standard') != 'standard':
        return None
        
    game_engine = Game()
    provinces = list(game_engine.map.locs)
    
    agent_histories = {p: np.zeros((3, worker_num_provs, FEATURE_DIM), dtype=np.int8) for p in GLOBAL_POWERS}
    
    g_histories, g_masks, g_targets = [], [], []
    
    for phase in game_data.get('phases', []):
        game_engine.set_state(phase['state'])
        
        for power in GLOBAL_POWERS:
            current_state = parse_state_to_tensor(phase, observing_agent=power)
            agent_histories[power] = np.roll(agent_histories[power], shift=1, axis=0)
            agent_histories[power][0] = current_state
        
        orders_dict = phase.get('orders', {})
        for power, text_orders in orders_dict.items():
            if not text_orders:
                continue
            
            mask = get_global_action_mask(game_engine, power, provinces, worker_order_to_idx)
            targets = np.full(worker_num_provs, worker_none_idx, dtype=np.int64)
            
            for order_str in text_orders:
                clean_order = order_str.replace('*', '').upper()
                parts = clean_order.split()
                if len(parts) >= 2:
                    loc_full = parts[1]
                    if loc_full in worker_prov_to_idx and clean_order in worker_order_to_idx:
                        p_idx = worker_prov_to_idx[loc_full]
                        targets[p_idx] = worker_order_to_idx[clean_order]
            
            # Convert dense boolean mask to flat sparse indices
            flat_mask = mask.flatten()
            valid_indices = np.where(flat_mask)[0].astype(np.int32)
            num_valid = len(valid_indices)
            
            if num_valid > MAX_SPARSE_MASK_LEN:
                raise ValueError(f"Action space constraint violated: Found {num_valid} valid moves, max is {MAX_SPARSE_MASK_LEN}.")
                
            sparse_mask = np.full(MAX_SPARSE_MASK_LEN, -1, dtype=np.int32)
            sparse_mask[:num_valid] = valid_indices
            
            g_histories.append(agent_histories[power].copy())
            g_masks.append(sparse_mask)
            g_targets.append(targets)
            
    if not g_histories:
        return None
        
    return (
        np.stack(g_histories), 
        np.stack(g_masks), 
        np.stack(g_targets)
    )

def process_and_save_to_disk(json_path, cache_dir, max_games=None):
    """Processes raw JSON datasets and writes them to binary memmap caches."""
    os.makedirs(cache_dir, exist_ok=True)
    history_path = os.path.join(cache_dir, "history.bin")
    mask_sparse_path = os.path.join(cache_dir, "mask_sparse.bin")
    targets_path = os.path.join(cache_dir, "targets.bin")
    
    if os.path.exists(history_path) and os.path.exists(mask_sparse_path) and os.path.exists(targets_path):
        print("Found existing binary cache. Skipping JSON parsing.")
        order_to_idx, _ = build_global_vocab()
        none_idx = order_to_idx['NONE']
        
        temp_engine = Game()
        num_provs = len(temp_engine.map.locs) 
        vocab_size = len(order_to_idx)
        
        targets_bytes = os.path.getsize(targets_path)
        total_samples = targets_bytes // (8 * num_provs)
        return history_path, mask_sparse_path, targets_path, total_samples, num_provs, vocab_size, none_idx

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

    num_cores = min(21, smp.cpu_count())
    print(f"Starting multiprocessing pool with {num_cores} workers.")
    
    total_samples = 0
    
    with open(history_path, 'wb') as f_hist, \
         open(mask_sparse_path, 'wb') as f_mask, \
         open(targets_path, 'wb') as f_targ:
             
        with smp.Pool(
            processes=num_cores, 
            initializer=init_worker, 
            initargs=(prov_to_idx, order_to_idx, num_provs, none_idx, vocab_size),
            maxtasksperchild=10 
        ) as pool:
            try:
                for i, result in enumerate(pool.imap_unordered(_process_single_line, line_generator(), chunksize=5)):
                    if (i + 1) % 1000 == 0:
                        ram_used = psutil.virtual_memory().used / (1024**3)
                        print(f"Processed {i + 1} games | System RAM in use: {ram_used:.1f} GB")
                        
                    if result is None:
                        continue
                    
                    h_arr, m_arr, t_arr = result
                    
                    f_hist.write(h_arr.tobytes())
                    f_mask.write(m_arr.tobytes())
                    f_targ.write(t_arr.tobytes())
                    
                    total_samples += len(h_arr)
                    del h_arr, m_arr, t_arr, result

            except KeyboardInterrupt:
                print("\nProcess interrupted. Terminating workers.")
                pool.terminate()
                pool.join()
                raise
                    
    log_system_resources("Finished Writing Binary Cache")
    print(f"Total training samples generated: {total_samples}")
    return history_path, mask_sparse_path, targets_path, total_samples, num_provs, vocab_size, none_idx


def train_worker(rank, world_size, paths_and_metadata, args):
    """DDP worker function for training the Behavioral Cloning model."""
    history_path, mask_sparse_path, targets_path, total_samples, num_provs, vocab_size, none_idx = paths_and_metadata
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    try:
        # Core pinning to prevent thread contention
        core_start = rank * 6
        assigned_cores = list(range(core_start, core_start + 6))
        os.sched_setaffinity(0, assigned_cores)
        torch.set_num_threads(6)
        
        dataset = DiplomacyMemmapDataset(history_path, mask_sparse_path, targets_path, total_samples, num_provs, vocab_size)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        net = DiplomacyTransformer(num_provinces=num_provs, vocab_size=vocab_size, none_idx=none_idx).to(rank)
        net = DDP(net, device_ids=[rank], find_unused_parameters=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
        
        if rank == 0:
            log_system_resources("Model Loaded to GPUs", rank)
            log_gpu_memory(rank)
            print("\n--- Starting Distributed Behavioral Cloning ---")
        
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            net.train()
            total_loss = 0.0
            
            epoch_start_time = time.time()
            batch_start_time = time.time()
            
            for batch_idx, batch in enumerate(dataloader):
                history = batch['history'].to(rank, dtype=torch.float32, non_blocking=True)
                sparse_mask = batch['sparse_mask'].to(rank, dtype=torch.long, non_blocking=True)
                targets = batch['targets'].to(rank, non_blocking=True)
                
                # Reconstruct dense action mask from sparse indices
                batch_size = history.size(0)
                valid_mask = sparse_mask != -1
                
                row_offsets = torch.arange(batch_size, device=rank).unsqueeze(1) * (num_provs * vocab_size)
                global_indices = sparse_mask + row_offsets
                valid_global_indices = global_indices[valid_mask]
                
                batch_mask_flat = torch.zeros(batch_size * num_provs * vocab_size, dtype=torch.bool, device=rank)
                batch_mask_flat[valid_global_indices] = True
                dense_mask = batch_mask_flat.view(batch_size, num_provs, vocab_size)
                
                is_active_mask = (targets != none_idx)
                num_active_per_batch = is_active_mask.sum(dim=1)
                max_units = num_active_per_batch.max().item()
                
                optimizer.zero_grad()
                
                # Mixed-precision forward pass
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Encode State
                    state_repr, _ = net.module.encode_state(history)
                    
                    if max_units > 0:
                        # Pack Targets and Dense Masks to align with the dynamic (B, max_units) shape
                        padded_indices = torch.zeros((batch_size, max_units), dtype=torch.long, device=rank)
                        for b in range(batch_size):
                            valid_idx = torch.where(is_active_mask[b])[0]
                            if len(valid_idx) > 0:
                                valid_idx = valid_idx[torch.randperm(len(valid_idx), device=rank)]
                                padded_indices[b, :len(valid_idx)] = valid_idx

                        # Decode Unit-Centric Sequence        
                        logits, padding_mask = net.module.decode_full(state_repr, targets, is_active_mask, padded_indices)
                        
                        b_idx_expand = torch.arange(batch_size, device=rank).unsqueeze(1)
                        packed_targets = targets[b_idx_expand, padded_indices]
                        packed_masks = dense_mask[b_idx_expand, padded_indices, :]
                        
                        # Flatten for Loss calculation
                        logits_flat = logits.view(-1, vocab_size)
                        targets_flat = packed_targets.view(-1)
                        masks_flat = packed_masks.view(-1, vocab_size)
                        
                        # Verify target action is legal per engine rules (safety check)
                        is_target_legal = packed_masks.gather(2, packed_targets.unsqueeze(2)).squeeze(2)
                        
                        # Ignore padded slots and illegal historical moves
                        valid_training_mask = (~padding_mask).view(-1) & is_target_legal.view(-1)
                        
                        if valid_training_mask.any():
                            # Apply dense engine mask to prevent network from choosing illegal moves
                            logits_masked = logits_flat.masked_fill(~masks_flat, -1e4).float() 
                            loss = criterion(logits_masked[valid_training_mask], targets_flat[valid_training_mask])
                            if torch.isnan(loss):
                                print(f"NaN detected at Epoch {epoch+1}, Batch {batch_idx}!")
                                raise ValueError("Training halted due to NaN loss.")
                        else:
                            loss = (logits * 0).sum()
                    else:
                        # Edge Case: Phase has no valid orders (e.g., empty Spring Adjust)
                        loss = (state_repr * 0).sum()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()
                    
                if rank == 0 and batch_idx % 20 == 0:
                    elapsed = time.time() - batch_start_time
                    samples_processed = 20 * args.batch_size * world_size
                    throughput = samples_processed / elapsed if batch_idx > 0 else 0
                    
                    print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s | Speed: {throughput:.0f} samples/s")
                    batch_start_time = time.time()
                    
            if rank == 0:
                epoch_time = time.time() - epoch_start_time
                avg_loss = total_loss / len(dataloader)
                print(f"\n==> Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} | Epoch Time: {epoch_time:.2f}s")
                log_system_resources(f"End of Epoch {epoch+1}", rank)
                log_gpu_memory(rank)
                
        if rank == 0:
            torch.save(net.module.state_dict(), args.save_path)
            print(f"Model state dict saved to {args.save_path}")
            
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Diplomacy Behavioral Cloning Training")
    parser.add_argument("--dataset_path", type=str, default="/data/restanislao/datasets/standard_no_press.jsonl", help="Path to raw JSONL dataset")
    parser.add_argument("--cache_dir", type=str, default="/data/restanislao/diplomacy", help="Directory for binary memmap cache")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--save_path", type=str, default="diplomacy_transformer_bc.pth", help="Path to save the model weights")
    args = parser.parse_args()

    paths_and_metadata = process_and_save_to_disk(
        args.dataset_path, 
        cache_dir=args.cache_dir
    )
    
    print("\nCaching complete. Starting worker processes.\n")
    
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs detected. Distributed training requires at least 1 GPU.")
        
    print(f"Spawning {world_size} distributed workers...")
    
    mp.spawn(
        train_worker,
        args=(world_size, paths_and_metadata, args),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()