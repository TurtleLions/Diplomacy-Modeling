import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import psutil
import torch.distributed as dist
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

# --- RESOURCE LOGGING ---

def log_system_resources(stage_name, rank=0):
    # Only let the main process print to avoid console spam
    if rank != 0:
        return
        
    print(f"\n[LOG] Resource Usage - {stage_name}")
    
    ram_info = psutil.virtual_memory()
    ram_used_gb = ram_info.used / (1024 ** 3)
    ram_total_gb = ram_info.total / (1024 ** 3)
    ram_percent = ram_info.percent
    
    print(f"System RAM Used {ram_used_gb:.1f} GB of {ram_total_gb:.1f} GB ({ram_percent}%)")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            vram_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            vram_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            print(f"GPU {i} VRAM Allocated {vram_allocated:.2f} GB | Reserved {vram_reserved:.2f} GB")
    print("-" * 40)

# --- 1. DATASET PREPARATION ---

class DiplomacyTransformerDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'history': torch.tensor(sample['history'], dtype=torch.float32),
            'mask': torch.tensor(sample['mask'], dtype=torch.bool),
            'targets': torch.tensor(sample['targets'], dtype=torch.long)
        }

def _process_single_line(args):
    line, prov_to_idx, order_to_idx, num_provs, none_idx = args
    game_data = json.loads(line)
    
    if game_data.get('map', 'standard') != 'standard':
        return []
        
    game_engine = Game()
    provinces = list(game_engine.map.locs)
    history_buffer = np.zeros((3, num_provs, 16), dtype=np.float32)
    game_samples = []
    
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
            
            game_samples.append({
                'history': history_buffer.copy(),
                'mask': mask,
                'targets': targets
            })
            
    return game_samples

def process_jsonl_dataset(json_path, max_games=None):
    log_system_resources("Starting JSON Loading")
    print(f"Loading and processing data from {json_path}")
    
    order_to_idx, _ = build_global_vocab()
    none_idx = order_to_idx['NONE']
    
    temp_engine = Game()
    provinces = list(temp_engine.map.locs)
    prov_to_idx = {p: i for i, p in enumerate(provinces)}
    num_provs = len(provinces)
    
    with open(json_path, 'r') as f:
        lines = [line for line in f if line.strip()]
        
    if max_games:
        lines = lines[:max_games]

    log_system_resources("File Loaded into Memory")

    args_list = [(line, prov_to_idx, order_to_idx, num_provs, none_idx) for line in lines]
    samples = []
    
    num_cores = min(24, mp.cpu_count())
    print(f"Starting multiprocessing pool with {num_cores} workers")
    
    with mp.Pool(processes=num_cores) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_single_line, args_list, chunksize=50)):
            samples.extend(result)
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} games out of {len(lines)}")
                
    log_system_resources("Multiprocessing Pool Finished")
            
    print(f"Total games processed {len(lines)}")
    print(f"Total training samples generated {len(samples)}")
    return samples, num_provs, len(order_to_idx), none_idx

# --- 2. DISTRIBUTED WORKER LOOP ---

def train_worker(rank, world_size, samples, num_provs, vocab_size, none_idx):
    # Initialize the distributed process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # HARDWARE ALLOCATION
    # Calculate the block of 6 CPUs for this specific GPU
    core_start = rank * 6
    assigned_cores = list(range(core_start, core_start + 6))
    
    # Lock the operating system scheduler and PyTorch threads to these specific cores
    os.sched_setaffinity(0, assigned_cores)
    torch.set_num_threads(6)
    
    print(f"Worker {rank} initialized and locked to GPU {rank} and CPU cores {assigned_cores}")
    
    # TRAINING HYPERPARAMETERS
    # We divide the global batch size of 256 by 4 GPUs
    batch_size_per_gpu = 64
    learning_rate = 1e-4
    epochs = 10
    
    dataset = DiplomacyTransformerDataset(samples)
    
    # The DistributedSampler ensures each GPU gets a completely unique chunk of the dataset
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    
    # We lower num_workers to 4 here so we do not exceed the 6 CPUs allocated to this worker
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size_per_gpu, 
        shuffle=False, 
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    
    net = DiplomacyTransformer(
        num_provinces=num_provs, 
        vocab_size=vocab_size
    ).to(rank)
    
    # Wrap the model for distributed training
    net = DDP(net, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    if rank == 0:
        log_system_resources("Model Loaded to GPUs", rank)
        print("\n--- STARTING DISTRIBUTED BEHAVIORAL CLONING ---")
    
    for epoch in range(epochs):
        # We must set the epoch on the sampler so the data shuffles correctly each round
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
                
            # Only print from the main GPU to keep the terminal clean
            if rank == 0 and batch_idx % 20 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss {loss.item():.4f}")
                
        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"==> Epoch {epoch+1} Complete | Avg Loss {avg_loss:.4f}")
            log_system_resources(f"End of Epoch {epoch+1}", rank)
            
    # Save the model only once from the main worker
    if rank == 0:
        save_path = "diplomacy_transformer_bc.pth"
        torch.save(net.module.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
    dist.destroy_process_group()

# --- 3. MAIN EXECUTION ---

def main():
    dataset_path = "./datasets/standard_no_press.jsonl"
    max_games_to_load = None 
    
    # We parse the dataset once in the main loop to save massive amounts of system RAM
    samples, num_provs, vocab_size, none_idx = process_jsonl_dataset(dataset_path, max_games_to_load)
    
    world_size = torch.cuda.device_count()
    
    if world_size < 1:
        print("Error no GPUs detected")
        return
        
    print(f"Spawning {world_size} distributed workers")
    
    # Launch one distinct process per GPU
    mp.spawn(
        train_worker,
        args=(world_size, samples, num_provs, vocab_size, none_idx),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()