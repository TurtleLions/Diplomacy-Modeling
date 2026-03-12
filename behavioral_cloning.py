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

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

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
    def __init__(self, history_path, mask_path, targets_path, total_samples, num_provs, vocab_size):
        self.total_samples = total_samples
        
        self.history = np.memmap(history_path, dtype=np.float32, mode='r', shape=(total_samples, 3, num_provs, 16))
        self.mask = np.memmap(mask_path, dtype=np.bool_, mode='r', shape=(total_samples, num_provs, vocab_size))
        self.targets = np.memmap(targets_path, dtype=np.int64, mode='r', shape=(total_samples, num_provs))
        
    def __len__(self):
        return self.total_samples
        
    def __getitem__(self, idx):
        return {
            'history': torch.tensor(np.array(self.history[idx]), dtype=torch.float32),
            'mask': torch.tensor(np.array(self.mask[idx]), dtype=torch.bool),
            'targets': torch.tensor(np.array(self.targets[idx]), dtype=torch.long)
        }

def _process_single_line(args):
    line, prov_to_idx, order_to_idx, num_provs, none_idx, vocab_size = args
    game_data = json.loads(line)
    
    if game_data.get('map', 'standard') != 'standard':
        return None
        
    game_engine = Game()
    provinces = list(game_engine.map.locs)
    history_buffer = np.zeros((3, num_provs, 16), dtype=np.float32)
    
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
            
            mask = get_global_action_mask(game_engine, power, provinces, order_to_idx)
            targets = np.full(num_provs, none_idx, dtype=np.int64)
            
            for order_str in text_orders:
                clean_order = order_str.replace('*', '')
                parts = clean_order.split()
                if len(parts) >= 2:
                    u_loc = parts[1].split('/')[0]
                    if u_loc in prov_to_idx and clean_order in order_to_idx:
                        p_idx = prov_to_idx[u_loc]
                        targets[p_idx] = order_to_idx[clean_order]
            
            g_histories.append(history_buffer.copy())
            g_masks.append(mask)
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
    mask_path = os.path.join(cache_dir, "mask.bin")
    targets_path = os.path.join(cache_dir, "targets.bin")
    
    if os.path.exists(history_path) and os.path.exists(mask_path) and os.path.exists(targets_path):
        print("Found existing binary cache skipping JSON parsing.")
        order_to_idx, _ = build_global_vocab()
        none_idx = order_to_idx['NONE']
        num_provs = 81 
        vocab_size = len(order_to_idx)
        
        targets_bytes = os.path.getsize(targets_path)
        total_samples = targets_bytes // (8 * num_provs)
        return history_path, mask_path, targets_path, total_samples, num_provs, vocab_size, none_idx

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
                yield (line, prov_to_idx, order_to_idx, num_provs, none_idx, vocab_size)
                
                games_yielded += 1
                if max_games and games_yielded >= max_games:
                    break

    num_cores = min(24, smp.cpu_count())
    print(f"Starting standard multiprocessing pool with {num_cores} workers")
    
    total_samples = 0
    buffer_limit = 500
    
    hist_buffer = []
    mask_buffer = []
    targ_buffer = []
    
    with open(history_path, 'wb') as f_hist, \
         open(mask_path, 'wb') as f_mask, \
         open(targets_path, 'wb') as f_targ:
             
        with smp.Pool(processes=num_cores, initializer=init_worker) as pool:
            try:
                for i, result in enumerate(pool.imap_unordered(_process_single_line, line_generator(), chunksize=50)):
                    if (i + 1) % 1000 == 0:
                        print(f"Processed {i + 1} games")
                        
                    if result is None:
                        continue
                    
                    h_arr, m_arr, t_arr = result
                    
                    hist_buffer.append(h_arr)
                    mask_buffer.append(m_arr)
                    targ_buffer.append(t_arr)
                    total_samples += len(h_arr)
                    
                    del h_arr, m_arr, t_arr, result
                    
                    if len(hist_buffer) >= buffer_limit:
                        f_hist.write(np.concatenate(hist_buffer).tobytes())
                        f_mask.write(np.concatenate(mask_buffer).tobytes())
                        f_targ.write(np.concatenate(targ_buffer).tobytes())
                        
                        hist_buffer = []
                        mask_buffer = []
                        targ_buffer = []
                        gc.collect()
                
                # Flush remaining items
                if len(hist_buffer) > 0:
                    f_hist.write(np.concatenate(hist_buffer).tobytes())
                    f_mask.write(np.concatenate(mask_buffer).tobytes())
                    f_targ.write(np.concatenate(targ_buffer).tobytes())
                    
                    hist_buffer = []
                    mask_buffer = []
                    targ_buffer = []
                    gc.collect()

            except KeyboardInterrupt:
                print("\nRun canceled by user. Forcibly terminating workers to prevent freezing.")
                pool.terminate()
                pool.join()
                raise
                    
    log_system_resources("Finished Writing Binary Cache")
    print(f"Total training samples generated {total_samples}")
    return history_path, mask_path, targets_path, total_samples, num_provs, vocab_size, none_idx

# --- 2. DISTRIBUTED WORKER LOOP ---

def train_worker(rank, world_size, paths_and_metadata):
    history_path, mask_path, targets_path, total_samples, num_provs, vocab_size, none_idx = paths_and_metadata
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    core_start = rank * 6
    assigned_cores = list(range(core_start, core_start + 6))
    os.sched_setaffinity(0, assigned_cores)
    torch.set_num_threads(6)
    
    batch_size_per_gpu = 64
    learning_rate = 1e-4
    epochs = 10
    
    dataset = DiplomacyMemmapDataset(history_path, mask_path, targets_path, total_samples, num_provs, vocab_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size_per_gpu, 
        shuffle=False, 
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        multiprocessing_context="spawn" if hasattr(mp, 'get_context') else None
    )
    
    net = DiplomacyTransformer(num_provinces=num_provs, vocab_size=vocab_size).to(rank)
    net = DDP(net, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    if rank == 0:
        log_system_resources("Model Loaded to GPUs", rank)
        print("\n--- STARTING DISTRIBUTED BEHAVIORAL CLONING ---")
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        net.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            history = batch['history'].to(rank, non_blocking=True)
            mask = batch['mask'].to(rank, non_blocking=True)
            targets = batch['targets'].to(rank, non_blocking=True)
            
            optimizer.zero_grad()
            logits, _ = net(history)
            
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)
            mask_flat = mask.view(-1, vocab_size)
            
            is_target_legal = mask_flat.gather(1, targets_flat.unsqueeze(1)).squeeze()
            valid_training_mask = (targets_flat != none_idx) & is_target_legal
            
            if valid_training_mask.any():
                logits_masked = logits_flat.masked_fill(~mask_flat, -1e4)
                loss = criterion(logits_masked[valid_training_mask], targets_flat[valid_training_mask])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()
                
            if rank == 0 and batch_idx % 20 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss {loss.item():.4f}")
                
        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"==> Epoch {epoch+1} Complete | Avg Loss {avg_loss:.4f}")
            log_system_resources(f"End of Epoch {epoch+1}", rank)
            
    if rank == 0:
        save_path = "diplomacy_transformer_bc.pth"
        torch.save(net.module.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
    dist.destroy_process_group()

# --- 3. MAIN EXECUTION ---

def main():
    dataset_path = "./datasets/standard_no_press.jsonl"
    ssd_cache_directory = "/data/restanislao/diplomacy/"
    
    paths_and_metadata = process_and_save_to_disk(
        dataset_path, 
        cache_dir=ssd_cache_directory
    )
    
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