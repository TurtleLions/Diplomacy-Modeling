"""
Phase 2: Behavioral Cloning for the Feudal Diplomacy Agent.
Processes JSON game history into memory-mapped arrays and trains the Tactical Worker.
"""
import os
import time
import json
import signal
import argparse
import multiprocessing as smp

import psutil
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader

from diplomacy import Game
from HRLhelpers import FeudalDiplomacyAgent, build_global_vocab, parse_state_to_tensor, InteractionMatrixTracker

# --- GLOBAL CONSTANTS ---
FEATURE_DIM = 61
MAX_SPARSE_MASK_LEN = 4000

_DUMMY_GAME = Game()
GLOBAL_PROVINCES = sorted([prov.upper() for prov in list(_DUMMY_GAME.map.locs)])
GLOBAL_PROV_TO_IDX = {prov: i for i, prov in enumerate(GLOBAL_PROVINCES)}

GLOBAL_POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
GLOBAL_POWER_TO_IDX = {power: i for i, power in enumerate(GLOBAL_POWERS)}

GLOBAL_PROV_TYPES = {}
for prov in GLOBAL_PROVINCES:
    GLOBAL_PROV_TYPES[prov] = _DUMMY_GAME.map.area_type(prov).upper()

GLOBAL_HSCS = {
    'AUSTRIA': ['VIE', 'BUD', 'TRI'],
    'ENGLAND': ['LON', 'EDI', 'LVP'],
    'FRANCE': ['PAR', 'MAR', 'BRE'],
    'GERMANY': ['BER', 'MUN', 'KIE'],
    'ITALY': ['ROM', 'VEN', 'NAP'],
    'RUSSIA': ['MOS', 'SEV', 'WAR', 'STP'],
    'TURKEY': ['ANK', 'CON', 'SMY']
}

# --- OFFLINE GRAPH BUILDER ---

def build_distance_matrix(provinces):
    game = Game()
    G = nx.Graph()
    
    is_callable = callable(game.map.abut_list)
    if not is_callable:
        abut_dict = game.map.abut_list
    
    for loc in game.map.locs:
        if is_callable:
            borders = game.map.abut_list(loc)
        else:
            borders = abut_dict.get(loc, [])
            
        loc_base = loc.split('/')[0].upper() 
        
        for border in borders:
            border_base = border.split('/')[0].upper()
            if loc_base in provinces and border_base in provinces:
                G.add_edge(loc_base, border_base)
                
    num_provs = len(provinces)
    D = torch.zeros((num_provs, num_provs), dtype=torch.long)
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    
    for i, p1 in enumerate(provinces):
        for j, p2 in enumerate(provinces):
            if p1 in lengths and p2 in lengths[p1]:
                D[i, j] = min(lengths[p1][p2], 19) 
            else:
                D[i, j] = 19 
                
    return D

# --- DATA GENERATION & MEMMAP BUILDER ---

worker_prov_to_idx = None
worker_order_to_idx = None
worker_num_provs = None
worker_none_idx = None
worker_vocab_size = None
worker_max_mask_len = 4000

def init_worker(p_idx, o_idx, n_provs, n_idx, v_size):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global worker_prov_to_idx, worker_order_to_idx, worker_num_provs, worker_none_idx, worker_vocab_size
    worker_prov_to_idx = p_idx
    worker_order_to_idx = o_idx
    worker_num_provs = n_provs
    worker_none_idx = n_idx
    worker_vocab_size = v_size

def get_global_action_mask(game, power, provinces, order_to_idx):
    mask = np.zeros((len(provinces), len(order_to_idx)), dtype=np.bool_)
    orderable_locs = game.get_orderable_locations(power)
    all_possible = game.get_all_possible_orders()
    
    for i, prov in enumerate(provinces):
        prov_upper = prov.upper()
        if prov_upper in orderable_locs:
            for order in all_possible.get(prov_upper, []):
                clean_order = order.replace('*', '').upper()
                if clean_order in order_to_idx:
                    mask[i, order_to_idx[clean_order]] = True
        
        if not mask[i].any():
            mask[i, order_to_idx['NONE']] = True
            
    return mask

def _process_single_line(line):
    global worker_prov_to_idx, worker_order_to_idx, worker_num_provs, worker_none_idx, worker_vocab_size, worker_max_mask_len
    game_data = json.loads(line)
    
    if game_data.get('map', 'standard') != 'standard':
        return None
        
    game_engine = Game()
    tracker = InteractionMatrixTracker(num_agents=7, gamma=0.9)
    provinces = sorted([p.upper() for p in list(game_engine.map.locs)])
    
    agent_histories = {p: np.zeros((3, worker_num_provs, FEATURE_DIM), dtype=np.int8) for p in GLOBAL_POWERS}
    g_histories, g_masks, g_targets = [], [], []
    
    prev_state_for_obs = None
    bounces_for_obs = None
    
    for phase_data in game_data.get('phases', []):
        phase_name = phase_data.get('name')
        historical_orders = phase_data.get('orders', {})
        
        current_state_dict = game_engine.get_state()
        
        for power in GLOBAL_POWERS:
            power_upper = power.upper()
            
            obs_tensor = parse_state_to_tensor(
                turn_data={'name': phase_name, 'state': current_state_dict}, 
                observing_agent=power_upper,
                prev_state=prev_state_for_obs,
                bounces=bounces_for_obs,
                H_matrix=tracker.H 
            )
            
            agent_histories[power_upper] = np.roll(agent_histories[power_upper], shift=1, axis=0)
            agent_histories[power_upper][0] = obs_tensor
            
        for power, text_orders in historical_orders.items():
            if not text_orders: continue
            power_upper = power.upper()
            
            mask = get_global_action_mask(game_engine, power_upper, provinces, worker_order_to_idx)
            targets = np.full(worker_num_provs, worker_none_idx, dtype=np.int64)
            
            ordered_locs = set()
            
            for order_str in text_orders:
                clean_order = order_str.replace('*', '').upper()
                parts = clean_order.split()
                if len(parts) >= 2:
                    loc_full = parts[1]
                    ordered_locs.add(loc_full)
                    if loc_full in worker_prov_to_idx and clean_order in worker_order_to_idx:
                        p_idx = worker_prov_to_idx[loc_full]
                        targets[p_idx] = worker_order_to_idx[clean_order]
                        
            active_units = current_state_dict['units'].get(power_upper, [])
            for unit_str in active_units:
                clean_unit = unit_str.replace('*', '').upper()
                parts = clean_unit.split()
                if len(parts) >= 2:
                    u_type = parts[0]
                    loc_full = parts[1]
                    if loc_full not in ordered_locs and loc_full in worker_prov_to_idx:
                        hold_order = f"{u_type} {loc_full} H"
                        if hold_order in worker_order_to_idx:
                            p_idx = worker_prov_to_idx[loc_full]
                            targets[p_idx] = worker_order_to_idx[hold_order]
            
            flat_mask = mask.flatten()
            valid_indices = np.where(flat_mask)[0].astype(np.int32)
            num_valid = len(valid_indices)
            
            if num_valid > worker_max_mask_len:
                raise ValueError(f"Action space constraint violated: Found {num_valid} valid moves.")
                
            sparse_mask = np.full(worker_max_mask_len, -1, dtype=np.int32)
            sparse_mask[:num_valid] = valid_indices
            
            g_histories.append(agent_histories[power_upper].copy())
            g_masks.append(sparse_mask)
            g_targets.append(targets)

        prev_units = {a: current_state_dict['units'].get(a, []) for a in GLOBAL_POWERS}
        
        for power, orders in historical_orders.items():
            if orders:
                clean_orders = [o.replace('*', '').upper() for o in orders]
                game_engine.set_orders(power, clean_orders)
                
        delta_H = torch.zeros((7, 7), dtype=torch.float32)
        prov_to_power_idx = {}
        
        for p, units in prev_units.items():
            if p not in GLOBAL_POWER_TO_IDX: continue
            p_idx = GLOBAL_POWER_TO_IDX[p]
            for u in units:
                loc_base = u.split()[1].split('/')[0]
                prov_to_power_idx[loc_base] = p_idx
                
        prev_centers = current_state_dict.get('centers', {})
        for p, scs in prev_centers.items():
            if p not in GLOBAL_POWER_TO_IDX: continue
            p_idx = GLOBAL_POWER_TO_IDX[p]
            for sc in scs:
                if sc not in prov_to_power_idx:
                    prov_to_power_idx[sc] = p_idx
                    
        for agent, orders in historical_orders.items():
            if not orders: continue
            agent_upper = agent.upper()
            if agent_upper not in GLOBAL_POWER_TO_IDX: continue
            agent_idx = GLOBAL_POWER_TO_IDX[agent_upper]
            
            for order in orders:
                clean_order = order.replace('*', '').upper()
                try:
                    if ' S ' in clean_order or ' C ' in clean_order:
                        target_str = clean_order.replace(' S ', '|').replace(' C ', '|').split('|')[1]
                        parts = target_str.strip().split()
                        
                        if len(parts) >= 2:
                            target_loc = parts[1].split('/')[0]
                        elif len(parts) == 1:
                            target_loc = parts[0].split('/')[0]
                        else:
                            continue
                            
                        target_owner_idx = prov_to_power_idx.get(target_loc)
                        if target_owner_idx is not None and target_owner_idx != agent_idx:
                            delta_H[agent_idx, target_owner_idx] += 0.5
                            
                    elif ' - ' in clean_order:
                        target_loc = clean_order.split(' - ')[1].strip().split()[0].split('/')[0]
                        target_owner_idx = prov_to_power_idx.get(target_loc)
                        if target_owner_idx is not None and target_owner_idx != agent_idx:
                            delta_H[agent_idx, target_owner_idx] -= 1.0
                            delta_H[target_owner_idx, agent_idx] -= 1.0
                except Exception as e:
                    print(f"\n[PARSER WARNING] Ignored malformed order: '{order}' | Error: {e}")
                    continue
                        
        delta_H = torch.clamp(delta_H, -1.0, 1.0)
        tracker.update(delta_H)
        
        game_engine.process()
        
        prev_state_for_obs = current_state_dict
        bounces_for_obs = [loc for loc, msg in game_engine.get_order_status().items() if 'bounced' in msg or 'fails' in msg]
            
    if not g_histories:
        return None
        
    return (np.stack(g_histories), np.stack(g_masks), np.stack(g_targets))

def process_and_save_to_disk(json_path, cache_dir, max_games=None):
    os.makedirs(cache_dir, exist_ok=True)
    history_path = os.path.join(cache_dir, "history.bin")
    mask_sparse_path = os.path.join(cache_dir, "mask_sparse.bin")
    targets_path = os.path.join(cache_dir, "targets.bin")
    
    order_to_idx, _ = build_global_vocab()
    none_idx = order_to_idx['NONE']
    temp_engine = Game()
    num_provs = len(temp_engine.map.locs) 
    vocab_size = len(order_to_idx)
    
    if os.path.exists(history_path) and os.path.exists(mask_sparse_path) and os.path.exists(targets_path):
        targets_bytes = os.path.getsize(targets_path)
        total_samples = targets_bytes // (8 * num_provs)
        
        mask_bytes = os.path.getsize(mask_sparse_path)
        derived_mask_len = mask_bytes // (total_samples * 4)
        
        if mask_bytes % (total_samples * 4) != 0:
            print("\n[CRITICAL ERROR] The mask_sparse.bin file is corrupted (bytes do not align with sample count).")
            raise RuntimeError("Corrupted binary cache detected.")
            
        print(f"Found existing binary cache. Total Samples: {total_samples} | Detected Mask Length: {derived_mask_len}")
        return history_path, mask_sparse_path, targets_path, total_samples, num_provs, vocab_size, none_idx, derived_mask_len

    print("Building binary cache from JSON... (This may take a while)")
    provinces = sorted([p.upper() for p in list(temp_engine.map.locs)])
    prov_to_idx = {p: i for i, p in enumerate(provinces)}
    
    def line_generator():
        with open(json_path, 'r') as f:
            games_yielded = 0
            for line in f:
                if not line.strip(): continue
                yield line
                games_yielded += 1
                if max_games and games_yielded >= max_games: break

    num_cores = min(28, smp.cpu_count())
    print(f"Starting multiprocessing pool with {num_cores} workers.")
    total_samples = 0
    
    with open(history_path, 'wb') as f_hist, open(mask_sparse_path, 'wb') as f_mask, open(targets_path, 'wb') as f_targ:
        with smp.Pool(processes=num_cores, initializer=init_worker, initargs=(prov_to_idx, order_to_idx, num_provs, none_idx, vocab_size), maxtasksperchild=10) as pool:
            try:
                for i, result in enumerate(pool.imap_unordered(_process_single_line, line_generator(), chunksize=5)):
                    if (i + 1) % 1000 == 0:
                        ram_used = psutil.virtual_memory().used / (1024**3)
                        print(f"Processed {i + 1} games | System RAM in use: {ram_used:.1f} GB")
                    if result is None: continue
                    h_arr, m_arr, t_arr = result
                    f_hist.write(h_arr.tobytes())
                    f_mask.write(m_arr.tobytes())
                    f_targ.write(t_arr.tobytes())
                    total_samples += len(h_arr)
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                raise
                    
    print(f"Total training samples generated: {total_samples}")
    return history_path, mask_sparse_path, targets_path, total_samples, num_provs, vocab_size, none_idx, worker_max_mask_len

# --- DATASET & TRAINING ---

class DiplomacyMemmapDataset(Dataset):
    def __init__(self, history_path, mask_sparse_path, targets_path, total_samples, num_provs, vocab_size, derived_mask_len):
        self.history_path = history_path
        self.mask_sparse_path = mask_sparse_path
        self.targets_path = targets_path
        self.total_samples = total_samples
        self.num_provs = num_provs
        self.vocab_size = vocab_size
        self.derived_mask_len = derived_mask_len
        
        self.history = None
        self.sparse_mask = None
        self.targets = None
        
    def __len__(self): return self.total_samples
        
    def __getitem__(self, idx):
        if self.history is None:
            self.history = np.memmap(self.history_path, dtype=np.int8, mode='r', shape=(self.total_samples, 3, self.num_provs, FEATURE_DIM))
            self.sparse_mask = np.memmap(self.mask_sparse_path, dtype=np.int32, mode='r', shape=(self.total_samples, self.derived_mask_len))
            self.targets = np.memmap(self.targets_path, dtype=np.int64, mode='r', shape=(self.total_samples, self.num_provs))
            
        return {
            'history': torch.from_numpy(self.history[idx].copy()),
            'sparse_mask': torch.from_numpy(self.sparse_mask[idx].copy()), 
            'targets': torch.from_numpy(self.targets[idx].copy())
        }

def train_worker(rank, world_size, paths_and_metadata, args):
    history_path, mask_sparse_path, targets_path, total_samples, num_provs, vocab_size, none_idx, derived_mask_len = paths_and_metadata
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    core_start = rank * 6
    os.sched_setaffinity(os.getpid(), list(range(core_start, core_start + 6)))
    torch.set_num_threads(6)
    torch.cuda.set_device(rank)
    
    dataset = DiplomacyMemmapDataset(history_path, mask_sparse_path, targets_path, total_samples, num_provs, vocab_size, derived_mask_len)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler,
        num_workers=6, pin_memory=True, prefetch_factor=2, persistent_workers=True
    )
    
    agent = FeudalDiplomacyAgent(d_model=256, vocab_size=vocab_size).to(rank)
    
    temp_game = Game()
    provinces = [p.upper() for p in list(temp_game.map.locs)]
    distance_matrix = build_distance_matrix(provinces).to(rank)
    agent.D.copy_(distance_matrix) 
    
    agent = DDP(agent, device_ids=[rank], find_unused_parameters=True)
    criterion = nn.CrossEntropyLoss(ignore_index=none_idx) 
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
    
    if rank == 0:
        print(f"\n--- Starting Phase 2: Tactical Worker Behavioral Cloning ---")
        
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        agent.train()
        total_loss = 0.0
        batch_start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            S_mu = batch['history'][:, 0, :, :].to(rank, dtype=torch.float32, non_blocking=True) 
            sparse_mask = batch['sparse_mask'].to(rank, dtype=torch.long, non_blocking=True)
            targets = batch['targets'].to(rank, non_blocking=True)
            batch_size = S_mu.size(0)
            
            valid_mask = sparse_mask != -1
            row_offsets = torch.arange(batch_size, device=rank).unsqueeze(1) * (num_provs * vocab_size)
            global_indices = sparse_mask + row_offsets
            valid_global_indices = global_indices[valid_mask]
            
            batch_mask_flat = torch.zeros(batch_size * num_provs * vocab_size, dtype=torch.bool, device=rank)
            batch_mask_flat[valid_global_indices] = True
            dense_mask = batch_mask_flat.view(batch_size, num_provs, vocab_size)
            
            b_idx = torch.arange(batch_size, device=rank).view(-1, 1).expand(-1, num_provs)
            p_idx = torch.arange(num_provs, device=rank).view(1, -1).expand(batch_size, -1)

            valid_targets = dense_mask[b_idx, p_idx, targets]
            targets[~valid_targets] = none_idx

            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = agent(S_mu, bc_mode=True)
                logits_masked = logits.masked_fill(~dense_mask, -1e20).float()
                
                logits_flat = logits_masked.view(-1, vocab_size)
                targets_flat = targets.view(-1)
                
                loss = criterion(logits_flat, targets_flat)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
                
            if rank == 0 and batch_idx % 20 == 0:
                elapsed = time.time() - batch_start_time
                throughput = (20 * args.batch_size * world_size) / elapsed if batch_idx > 0 else 0
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f} | Speed: {throughput:.0f} samples/s")
                batch_start_time = time.time()
                
        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"\n==> Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f}")
            torch.save(agent.module.state_dict(), args.save_path)
            print(f"Agent state dict saved to {args.save_path}\n")

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="HRL Behavioral Cloning")
    parser.add_argument("--dataset_path", type=str, default="/data/restanislao/datasets/standard_no_press.jsonl", help="Path to raw JSONL dataset")
    parser.add_argument("--cache_dir", type=str, default="/data/restanislao/diplomacy", help="Directory for binary memmap cache")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--save_path", type=str, default="feudal_agent_bc.pth", help="Path to save the model weights")
    args = parser.parse_args()

    paths_and_metadata = process_and_save_to_disk(
        args.dataset_path, 
        cache_dir=args.cache_dir
    )
    
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs detected.")
        
    print(f"Spawning {world_size} distributed workers...")
    mp.spawn(train_worker, args=(world_size, paths_and_metadata, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()