import json
import numpy as np
from diplomacy import Game
import functools
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --- DISTRIBUTED SETUP ---

def setup():
    """Initializes the distributed environment for ROCm/MI210."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    torch.cuda.set_device(local_rank)
    # Using 'nccl' as the alias for the RCCL backend on AMD
    dist.init_process_group(backend="nccl")
    return rank, world_size

def cleanup():
    dist.destroy_process_group()

# --- DATA PARSING & UTILITIES ---

ACTION_TYPES = ['NONE', 'H', '-', 'S', 'C', 'B', 'D', 'R']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}
IDX_TO_ACTION = {i: a for a, i in ACTION_TO_IDX.items()}

def get_province_vocab(game):
    provinces = ['NONE'] + list(game.map.locs)
    prov_to_idx = {p: i for i, p in enumerate(provinces)}
    idx_to_prov = {i: p for p, i in prov_to_idx.items()}
    return prov_to_idx, idx_to_prov

def get_adjacency_matrix(game):
    provinces = game.map.locs 
    prov_to_idx = {prov: i for i, prov in enumerate(provinces)}
    num_provs = len(provinces)
    adj = np.zeros((num_provs, num_provs), dtype=np.float32)

    for loc, neighbors in game.map.loc_abut.items():
        u_name = loc.split('/')[0].upper()
        if u_name in prov_to_idx:
            u_idx = prov_to_idx[u_name]
            for neighbor in neighbors:
                v_name = neighbor.split('/')[0].upper()
                if v_name in prov_to_idx:
                    v_idx = prov_to_idx[v_name]
                    adj[u_idx, v_idx] = 1.0

    adj += np.eye(num_provs)
    row_sum = adj.sum(1)
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    normalized_adj = np.diag(d_inv_sqrt) @ adj @ np.diag(d_inv_sqrt)
    return torch.tensor(normalized_adj, dtype=torch.float32)

def parse_state_to_tensor(turn_data):
    game = Game()
    provinces = game.map.locs 
    prov_to_idx = {prov: i for i, prov in enumerate(provinces)}
    powers = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
    power_to_idx = {power: i for i, power in enumerate(powers)}
    
    state_tensor = np.zeros((len(provinces), 16), dtype=np.float32)
    state_info = turn_data['state']
    
    for power, unit_list in state_info.get('units', {}).items():
        if power not in power_to_idx: continue
        p_idx_owner = power_to_idx[power]
        for unit_str in unit_list:
            parts = unit_str.replace('*', '').split()
            if len(parts) >= 2:
                u_type, u_loc = parts[0], parts[1]
                if u_loc in prov_to_idx:
                    idx = prov_to_idx[u_loc]
                    state_tensor[idx, p_idx_owner] = 1.0 
                    state_tensor[idx, 7 if u_type == 'A' else 8] = 1.0 

    for power, sc_list in state_info.get('centers', {}).items():
        if power not in power_to_idx: continue
        p_idx_owner = power_to_idx[power]
        for sc in sc_list:
            if sc in prov_to_idx:
                state_tensor[prov_to_idx[sc], 9 + p_idx_owner] = 1.0 
    return state_tensor

def robust_parse_order(parts):
    """Safely extracts target1 and target2 regardless of unit type inclusion."""
    act_type = parts[2]
    t1, t2 = 'NONE', 'NONE'

    if act_type in ['-', 'R'] and len(parts) >= 4:
        t1 = parts[3]
    elif act_type in ['S', 'C'] and len(parts) >= 4:
        # If parts[3] is 'A' or 'F', location is index 4. Otherwise, it's index 3.
        t1 = parts[4] if parts[3] in ['A', 'F'] and len(parts) > 4 else parts[3]
        if '-' in parts:
            dash_idx = parts.index('-')
            if len(parts) > dash_idx + 1:
                t2 = parts[dash_idx + 1]
    return act_type, t1, t2

def encode_human_order(order_str, prov_to_idx):
    target_array = np.array([ACTION_TO_IDX['NONE'], prov_to_idx['NONE'], prov_to_idx['NONE']], dtype=np.int64)
    parts = order_str.replace('*', '').split()
    if len(parts) < 3: return target_array
    
    act_type, t1, t2 = robust_parse_order(parts)
    target_array[0] = ACTION_TO_IDX.get(act_type, 0)
    target_array[1] = prov_to_idx.get(t1, 0)
    target_array[2] = prov_to_idx.get(t2, 0)
    return target_array

def get_compositional_action_mask(game, power, provinces, prov_to_idx):
    num_provs = len(provinces)
    type_mask = np.zeros((num_provs, len(ACTION_TYPES)), dtype=np.int8)
    t1_mask = np.zeros((num_provs, len(prov_to_idx)), dtype=np.int8)
    t2_mask = np.zeros((num_provs, len(prov_to_idx)), dtype=np.int8)
    
    all_possible_orders = game.get_all_possible_orders()
    orderable_locs = game.get_orderable_locations(power)
    
    for i, prov in enumerate(provinces):
        if prov in orderable_locs:
            for order in all_possible_orders.get(prov, []):
                parts = order.split()
                if len(parts) >= 3:
                    act_type, t1, t2 = robust_parse_order(parts)
                    type_mask[i, ACTION_TO_IDX.get(act_type, 0)] = 1
                    t1_mask[i, prov_to_idx.get(t1, 0)] = 1
                    t2_mask[i, prov_to_idx.get(t2, 0)] = 1
        else:
            type_mask[i, 0] = 1; t1_mask[i, 0] = 1; t2_mask[i, 0] = 1
    return {'type': type_mask, 'target1': t1_mask, 'target2': t2_mask}

# --- MODEL & DATASET ---

class DiplomacyDataset(Dataset):
    def __init__(self, json_data, game_engine, prov_to_idx):
        self.game = game_engine
        self.provinces = game_engine.map.locs
        self.prov_to_idx = prov_to_idx
        self.samples = [(p, pow, ords) for p in json_data for pow, ords in p.get('orders', {}).items() if ords]
        
    def __len__(self): return len(self.samples)
        
    def __getitem__(self, idx):
        phase_data, power, text_orders = self.samples[idx]
        state_tensor = parse_state_to_tensor(phase_data)
        self.game.set_state(phase_data['state'])
        mask_dict = get_compositional_action_mask(self.game, power, self.provinces, self.prov_to_idx)
        
        target_matrix = np.zeros((len(self.provinces), 3), dtype=np.int64)
        for order_str in text_orders:
            parts = order_str.replace('*', '').split()
            if len(parts) >= 2 and parts[1] in self.prov_to_idx:
                target_matrix[self.prov_to_idx[parts[1]]] = encode_human_order(order_str, self.prov_to_idx)
        
        return {
            'state': torch.tensor(state_tensor, dtype=torch.float32),
            'mask_type': torch.tensor(mask_dict['type'], dtype=torch.bool),
            'mask_t1': torch.tensor(mask_dict['target1'], dtype=torch.bool),
            'mask_t2': torch.tensor(mask_dict['target2'], dtype=torch.bool),
            'targets': torch.tensor(target_matrix, dtype=torch.long)
        }

class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.proj = nn.Linear(in_f, out_f)
    def forward(self, x, adj):
        return torch.relu(torch.matmul(adj, self.proj(x)))

class DiplomacyGCN(nn.Module):
    def __init__(self, adj, in_dim=16, hid_dim=256, vocab_size=83):
        super().__init__()
        self.register_buffer('adj', adj)
        self.gcn1 = GCNLayer(in_dim, hid_dim)
        self.gcn2 = GCNLayer(hid_dim, hid_dim)
        self.type_head = nn.Linear(hid_dim, 8)
        self.t1_head = nn.Linear(hid_dim, vocab_size)
        self.t2_head = nn.Linear(hid_dim, vocab_size)
    def forward(self, x):
        h = self.gcn2(self.gcn1(x, self.adj), self.adj)
        return self.type_head(h), self.t1_head(h), self.t2_head(h)

# --- TRAINING LOOP ---

def train_behavioral_cloning(rank, world_size):
    game_engine = Game()
    prov_to_idx, _ = get_province_vocab(game_engine)
    adj_matrix = get_adjacency_matrix(game_engine).to(rank)
    
    all_phases = []
    if os.path.exists("./datasets/standard_no_press.jsonl"):
        with open("./datasets/standard_no_press.jsonl", 'r') as f:
            for line in f:
                data = json.loads(line)
                if data.get('map') == 'standard': all_phases.extend(data.get('phases', []))
    
    dataset = DiplomacyDataset(all_phases, game_engine, prov_to_idx)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)

    net = DiplomacyGCN(adj=adj_matrix, in_dim=16, hid_dim=256, vocab_size=len(prov_to_idx)).to(rank)
    net = DDP(net, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(10):
        sampler.set_epoch(epoch)
        net.train()
        total_epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            states, targets = batch['state'].to(rank), batch['targets'].to(rank)
            m_type, m_t1, m_t2 = batch['mask_type'].to(rank), batch['mask_t1'].to(rank), batch['mask_t2'].to(rank)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                t_logits, t1_logits, t2_logits = net(states)
                
                t_logits_f, t1_logits_f, t2_logits_f = t_logits.view(-1, 8), t1_logits.view(-1, t1_logits.size(-1)), t2_logits.view(-1, t2_logits.size(-1))
                m_type_f, m_t1_f, m_t2_f = m_type.view(-1, 8), m_t1.view(-1, m_t1.size(-1)), m_t2.view(-1, m_t2.size(-1))
                targs_f = targets.view(-1, 3)

                is_leg_t = m_type_f.gather(1, targs_f[:, 0:1]).squeeze()
                is_leg_t1 = m_t1_f.gather(1, targs_f[:, 1:2]).squeeze()
                is_leg_t2 = m_t2_f.gather(1, targs_f[:, 2:3]).squeeze()
                valid_mask = (targs_f[:, 0] != 0) & is_leg_t & is_leg_t1 & is_leg_t2

                if valid_mask.any():
                    l_type = criterion(t_logits_f.masked_fill(~m_type_f, -1e4)[valid_mask], targs_f[valid_mask, 0])
                    l_t1 = criterion(t1_logits_f.masked_fill(~m_t1_f, -1e4)[valid_mask], targs_f[valid_mask, 1])
                    l_t2 = criterion(t2_logits_f.masked_fill(~m_t2_f, -1e4)[valid_mask], targs_f[valid_mask, 2])
                    loss = l_type + l_t1 + l_t2

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    total_epoch_loss += loss.item()

            if batch_idx % 50 == 0 and rank == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    if rank == 0: torch.save(net.module.state_dict(), "diplomacy_gcn_bc.pth")

if __name__ == "__main__":
    os.environ["RCCL_P2P_LEVEL"] = "PCIE" 
    rank, world_size = setup()
    train_behavioral_cloning(rank, world_size)
    cleanup()