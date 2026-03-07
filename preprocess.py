import json
import os
import torch
import numpy as np
from diplomacy import Game
from tqdm import tqdm
from multiprocessing import Pool, Manager

# --- CONSTANTS ---
ACTION_TYPES = ['NONE', 'H', '-', 'S', 'C', 'B', 'D', 'R']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}
POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
POWER_TO_IDX = {p: i for i, p in enumerate(POWERS)}

def get_vocab():
    game = Game()
    provinces = ['NONE'] + list(game.map.locs)
    return {p: i for i, p in enumerate(provinces)}

def robust_parse_order(parts, log_list=None):
    """
    Defensive parser to prevent IndexErrors. 
    Logs short strings that don't meet the [UNIT, LOC, ACTION] format.
    """
    if len(parts) < 3:
        if log_list is not None:
            log_list.append(" ".join(parts))
        return 'NONE', 'NONE', 'NONE'
    
    act_type = parts[2]
    t1, t2 = 'NONE', 'NONE'
    
    # Move/Retreat
    if act_type in ['-', 'R'] and len(parts) >= 4:
        t1 = parts[3]
    # Support/Convoy
    elif act_type in ['S', 'C'] and len(parts) >= 4:
        # Index 3 is usually 'A' or 'F', loc is index 4
        t1_idx = 4 if parts[3] in ['A', 'F'] and len(parts) > 4 else 3
        t1 = parts[t1_idx]
        
        if '-' in parts:
            try:
                idx = parts.index('-')
                if len(parts) > idx + 1:
                    t2 = parts[idx + 1]
            except ValueError:
                pass
    
    return act_type, t1, t2

worker_game = None

def worker_init():
    global worker_game
    worker_game = Game()

def process_batch(args):
    phases, prov_to_idx, skipped_log = args
    global worker_game
    if worker_game is None: worker_game = Game()
        
    provinces = worker_game.map.locs
    batch_samples = []

    for phase_data in phases:
        try:
            worker_game.set_state(phase_data['state'])
            state_tensor = np.zeros((len(provinces), 16), dtype=np.float32)
            
            # 1. State Tensor Generation
            for pow_name, u_list in phase_data['state'].get('units', {}).items():
                if pow_name not in POWER_TO_IDX: continue
                p_idx = POWER_TO_IDX[pow_name]
                for u_str in u_list:
                    parts = u_str.replace('*', '').split()
                    if len(parts) >= 2 and parts[1] in prov_to_idx:
                        loc_idx = prov_to_idx[parts[1]] - 1
                        state_tensor[loc_idx, p_idx] = 1.0
                        state_tensor[loc_idx, 7 if parts[0] == 'A' else 8] = 1.0

            for pow_name, c_list in centers.items():
                if pow_name not in POWER_TO_IDX: continue
                p_idx = POWER_TO_IDX[pow_name]
                for c in c_list:
                    if c in prov_to_idx:
                        state_tensor[prov_to_idx[c]-1, 9 + p_idx] = 1.0

            # 2. Sample Generation
            all_possible = worker_game.get_all_possible_orders()
            for power, orders in phase_data.get('orders', {}).items():
                if not orders: continue
                
                type_m, t1_m, t2_m = np.zeros((len(provinces), 8), dtype=bool), np.zeros((len(provinces), 82), dtype=bool), np.zeros((len(provinces), 82), dtype=bool)
                target_mat = np.zeros((len(provinces), 3), dtype=np.int64)
                
                orderable = worker_game.get_orderable_locations(power)
                for i, prov in enumerate(provinces):
                    if prov in orderable:
                        for o in all_possible.get(prov, []):
                            a, ta1, ta2 = robust_parse_order(o.split())
                            type_m[i, ACTION_TO_IDX.get(a, 0)] = True
                            t1_m[i, prov_to_idx.get(ta1, 0)] = True
                            t2_m[i, prov_to_idx.get(ta2, 0)] = True
                    else:
                        type_m[i, 0], t1_m[i, 0], t2_m[i, 0] = True, True, True

                for o_str in orders:
                    pts = o_str.replace('*', '').split()
                    if len(pts) >= 2 and pts[1] in prov_to_idx:
                        idx = prov_to_idx[pts[1]] - 1
                        a, ta1, ta2 = robust_parse_order(pts, skipped_log) # Log skips here
                        target_mat[idx] = [ACTION_TO_IDX.get(a, 0), prov_to_idx.get(ta1, 0), prov_to_idx.get(ta2, 0)]

                batch_samples.append({
                    'state': torch.from_numpy(state_tensor).half(),
                    'mask_type': torch.from_numpy(type_m),
                    'mask_t1': torch.from_numpy(t1_m),
                    'mask_t2': torch.from_numpy(t2_m),
                    'targets': torch.from_numpy(target_mat)
                })
        except Exception:
            continue
            
    return batch_samples

if __name__ == "__main__":
    manager = Manager()
    skipped_log = manager.list() # Thread-safe list for multi-process logging
    prov_to_idx = get_vocab()
    raw_phases = []
    
    print("Loading raw JSONL data...")
    with open("./datasets/standard_no_press.jsonl", 'r') as f:
        for line in f:
            if not line.strip(): continue
            game_obj = json.loads(line)
            if game_obj.get('map') == 'standard':
                raw_phases.extend(game_obj.get('phases', []))

    chunk_size = 50
    chunks = [(raw_phases[i:i + chunk_size], prov_to_idx, skipped_log) for i in range(0, len(raw_phases), chunk_size)]

    print(f"Pre-processing {len(raw_phases)} phases...")
    final_data = []
    with Pool(processes=os.cpu_count(), initializer=worker_init) as p:
        for result_batch in tqdm(p.imap_unordered(process_batch, chunks), total=len(chunks)):
            final_data.extend(result_batch)
    
    # Summary of skipped/malformed orders
    print(f"\nCompleted! Skipped/Malformed orders encountered: {len(skipped_log)}")
    if len(skipped_log) > 0:
        print("Sample of skipped orders:", list(set(skipped_log))[:10])
    
    print(f"Saving {len(final_data)} total samples...")
    torch.save(final_data, "processed_diplomacy.pt")