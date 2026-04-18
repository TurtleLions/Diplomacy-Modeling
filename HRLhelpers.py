import os
import time
import argparse
import numpy as np
import networkx as nx
import json

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
from pettingzoo import ParallelEnv

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

# --- HELPER FUNCTIONS ---

def build_distance_matrix(provinces):
    game = Game()
    G = nx.Graph()
    is_callable = callable(game.map.abut_list)
    if not is_callable:
        abut_dict = game.map.abut_list
    
    for loc in game.map.locs:
        borders = game.map.abut_list(loc) if is_callable else abut_dict.get(loc, [])
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

def build_global_vocab(json_path="/data/restanislao/datasets/standard_no_press.jsonl", cache_path="vocab.txt"):
    """Scans the dataset to build or load the global vocabulary of all legal historical orders."""
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            sorted_orders = [line.strip() for line in f.readlines() if line.strip()]
        order_to_idx = {order: i for i, order in enumerate(sorted_orders)}
        return order_to_idx, {i: order for order, i in order_to_idx.items()}

    print("Building global vocabulary from dataset...")
    unique_orders = set(['NONE'])
    
    with open(json_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            game_data = json.loads(line)
            
            if game_data.get('map', 'standard') != 'standard':
                continue
                
            for phase in game_data.get('phases', []):
                for power, orders in phase.get('orders', {}).items():
                    if not orders:
                        continue
                        
                    for order_str in orders:
                        clean_order = order_str.replace('*', '').upper()
                        unique_orders.add(clean_order)
                        
    sorted_orders = sorted(list(unique_orders))
    
    with open(cache_path, 'w') as f:
        for order in sorted_orders:
            f.write(f"{order}\n")
            
    order_to_idx = {order: i for i, order in enumerate(sorted_orders)}
    return order_to_idx, {i: order for order, i in order_to_idx.items()}

def get_sparse_action_mask(game, power, provinces, order_to_idx, max_len=MAX_SPARSE_MASK_LEN):
    """Generates a memory-efficient sparse mask of legal actions for distributed training."""
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
            
    flat_mask = mask.flatten()
    valid_indices = np.where(flat_mask)[0].astype(np.int32)
    num_valid = len(valid_indices)
    
    if num_valid > max_len:
        raise ValueError(f"CRITICAL: Action space exceeds sparse limit ({num_valid} > {max_len}).")
        
    sparse_mask = np.full(max_len, -1, dtype=np.int32)
    sparse_mask[:num_valid] = valid_indices
    return sparse_mask

def parse_state_to_tensor(turn_data, observing_agent=None, prev_state=None, bounces=None, H_matrix=None):
    """
    Parses a single game phase into a standardized geometric feature tensor.
    
    Feature Vector Layout (Dim: 61):
      [0-6]   : Active unit presence (One-hot mapped to powers)
      [7]     : Unit is an Army
      [8]     : Unit is a Fleet
      [9-15]  : Supply Center ownership (One-hot mapped to powers)
      [16-18] : Coast specification (NC, SC, EC)
      [19-23] : Phase specific flags (Move, Retreat, Adjust, Spring, Fall/Winter)
      [24]    : Dislodged unit flag
      [25-31] : Build deficits (Broadcasted globally per power)
      [32-38] : Static Home Supply Center mapping
      [39-45] : Observing Agent identity mapping
      [46-48] : Province Type (Inland, Coast, Water)
      [49]    : Buildable Home SC (Empty, owned by observer, is HSC)
      [50]    : Standoff / Bounce occurred here last phase
      [51-57] : Previous SC ownership (One-hot mapped to powers)
      [58-60] : Relative Diplomatic Stance (Occupied by: Self, Ally, Enemy)
    """
    state_tensor = np.zeros((len(GLOBAL_PROV_TO_IDX), FEATURE_DIM), dtype=np.float32)
    state_info = turn_data.get('state', {})
    phase_name = turn_data.get('name', 'S1901M')
    
    is_m, is_r, is_a, is_spring, is_fall_winter = 1.0, 0.0, 0.0, 1.0, 0.0
    
    if len(phase_name) >= 6:
        season = phase_name[0].upper()
        p_type = phase_name[-1].upper()
        
        is_m = 1.0 if p_type == 'M' else 0.0
        is_r = 1.0 if p_type == 'R' else 0.0
        is_a = 1.0 if p_type == 'A' else 0.0
        is_spring = 1.0 if season == 'S' else 0.0
        is_fall_winter = 1.0 if season in ['F', 'W'] else 0.0
            
    state_tensor[:, 19] = is_m
    state_tensor[:, 20] = is_r
    state_tensor[:, 21] = is_a
    state_tensor[:, 22] = is_spring
    state_tensor[:, 23] = is_fall_winter

    for prov, p_idx in GLOBAL_PROV_TO_IDX.items():
        ptype = GLOBAL_PROV_TYPES.get(prov, 'LAND')
        if ptype in ['LAND', 'SHUT']: 
            state_tensor[p_idx, 46] = 1.0
        elif ptype in ['COAST', 'PORT']: 
            state_tensor[p_idx, 47] = 1.0
        elif ptype == 'WATER': 
            state_tensor[p_idx, 48] = 1.0

    for power, hsc_list in GLOBAL_HSCS.items():
        power_idx = GLOBAL_POWER_TO_IDX[power]
        for hsc in hsc_list:
            if hsc in GLOBAL_PROV_TO_IDX:
                p_idx = GLOBAL_PROV_TO_IDX[hsc]
                state_tensor[p_idx, 32 + power_idx] = 1.0

    units = state_info.get('units', {})
    centers = state_info.get('centers', {})
    dislodged = state_info.get('dislodged', {})

    occupied_bases = set()

    for power, unit_list in units.items():
        power_upper = power.upper()
        if power_upper not in GLOBAL_POWER_TO_IDX: continue
        power_idx = GLOBAL_POWER_TO_IDX[power_upper] 
        
        for unit_str in unit_list:
            clean_str = unit_str.replace('*', '').upper()
            parts = clean_str.split()
            if len(parts) >= 2:
                u_type = parts[0]
                loc_full = parts[1]
                loc_base = loc_full.split('/')[0]
                coast = loc_full.split('/')[1] if len(loc_full.split('/')) > 1 else None
                
                occupied_bases.add(loc_base)

                if loc_full in GLOBAL_PROV_TO_IDX:
                    p_idx = GLOBAL_PROV_TO_IDX[loc_full]
                    state_tensor[p_idx, power_idx] = 1.0
                    if u_type == 'A': state_tensor[p_idx, 7] = 1.0
                    elif u_type == 'F': state_tensor[p_idx, 8] = 1.0
                    
                    if coast == 'NC': state_tensor[p_idx, 16] = 1.0
                    elif coast == 'SC': state_tensor[p_idx, 17] = 1.0
                    elif coast == 'EC': state_tensor[p_idx, 18] = 1.0
                    
                    if observing_agent and H_matrix is not None:
                        obs_idx = GLOBAL_POWER_TO_IDX[observing_agent.upper()]
                        if power_idx == obs_idx:
                            state_tensor[p_idx, 58] = 1.0 # Self
                        elif H_matrix[obs_idx, power_idx].item() > 0.5:
                            state_tensor[p_idx, 59] = 1.0 # Ally
                        elif H_matrix[obs_idx, power_idx].item() < -0.5:
                            state_tensor[p_idx, 60] = 1.0 # Enemy

                if loc_base != loc_full and loc_base in GLOBAL_PROV_TO_IDX:
                    base_idx = GLOBAL_PROV_TO_IDX[loc_base]
                    state_tensor[base_idx, power_idx] = 1.0
                    if u_type == 'A': state_tensor[base_idx, 7] = 1.0
                    elif u_type == 'F': state_tensor[base_idx, 8] = 1.0
                    
                    if observing_agent and H_matrix is not None:
                        obs_idx = GLOBAL_POWER_TO_IDX[observing_agent.upper()]
                        if power_idx == obs_idx:
                            state_tensor[base_idx, 58] = 1.0
                        elif H_matrix[obs_idx, power_idx].item() > 0.5:
                            state_tensor[base_idx, 59] = 1.0
                        elif H_matrix[obs_idx, power_idx].item() < -0.5:
                            state_tensor[base_idx, 60] = 1.0

    for power, hsc_list in GLOBAL_HSCS.items():
        power_idx = GLOBAL_POWER_TO_IDX[power]
        for hsc in hsc_list:
            if hsc in GLOBAL_PROV_TO_IDX:
                p_idx = GLOBAL_PROV_TO_IDX[hsc]
                state_tensor[p_idx, 32 + power_idx] = 1.0

    for power_upper, power_idx in GLOBAL_POWER_TO_IDX.items():
        power_scs = len(centers.get(power_upper, []))
        power_units = len(units.get(power_upper, []))
        deficit = float(power_scs - power_units)
        state_tensor[:, 25 + power_idx] = deficit

    for power, unit_list in units.items():
        power_upper = power.upper()
        if power_upper not in GLOBAL_POWER_TO_IDX: continue
        power_idx = GLOBAL_POWER_TO_IDX[power_upper] 
        for unit_str in unit_list:
            clean_str = unit_str.replace('*', '').upper()
            parts = clean_str.split()
            if len(parts) >= 2:
                u_type = parts[0]
                loc_full = parts[1]
                
                loc_parts = loc_full.split('/')
                loc_base = loc_parts[0] 
                coast = loc_parts[1] if len(loc_parts) > 1 else None

                if loc_full in GLOBAL_PROV_TO_IDX:
                    p_idx = GLOBAL_PROV_TO_IDX[loc_full]
                    state_tensor[p_idx, power_idx] = 1.0
                    if u_type == 'A': state_tensor[p_idx, 7] = 1.0
                    elif u_type == 'F': state_tensor[p_idx, 8] = 1.0
                    
                    if coast == 'NC': state_tensor[p_idx, 16] = 1.0
                    elif coast == 'SC': state_tensor[p_idx, 17] = 1.0
                    elif coast == 'EC': state_tensor[p_idx, 18] = 1.0

                if loc_base != loc_full and loc_base in GLOBAL_PROV_TO_IDX:
                    base_idx = GLOBAL_PROV_TO_IDX[loc_base]
                    state_tensor[base_idx, power_idx] = 1.0
                    if u_type == 'A': state_tensor[base_idx, 7] = 1.0
                    elif u_type == 'F': state_tensor[base_idx, 8] = 1.0

    for power, unit_list in dislodged.items():
        power_upper = power.upper()
        if power_upper not in GLOBAL_POWER_TO_IDX: continue
        power_idx = GLOBAL_POWER_TO_IDX[power_upper]
        for unit_str in unit_list:
            clean_str = unit_str.replace('*', '').upper()
            parts = clean_str.split()
            if len(parts) >= 2:
                loc_full = parts[1]
                if loc_full in GLOBAL_PROV_TO_IDX:
                    p_idx = GLOBAL_PROV_TO_IDX[loc_full]
                    u_type = parts[0]
                    state_tensor[p_idx, power_idx] = 1.0
                    if u_type == 'A': state_tensor[p_idx, 7] = 1.0
                    elif u_type == 'F': state_tensor[p_idx, 8] = 1.0
                    state_tensor[p_idx, 24] = 1.0

    for power, sc_list in centers.items():
        power_upper = power.upper()
        if power_upper not in GLOBAL_POWER_TO_IDX: continue
        power_idx = GLOBAL_POWER_TO_IDX[power_upper]
        for sc in sc_list:
            sc_upper = sc.upper()
            if sc_upper in GLOBAL_PROV_TO_IDX:
                p_idx = GLOBAL_PROV_TO_IDX[sc_upper]
                state_tensor[p_idx, 9 + power_idx] = 1.0
    
    if observing_agent is not None:
        agent_upper = observing_agent.upper()
        if agent_upper in GLOBAL_POWER_TO_IDX:
            power_idx = GLOBAL_POWER_TO_IDX[agent_upper]
            state_tensor[:, 39 + power_idx] = 1.0
            my_scs = centers.get(agent_upper, [])
            for hsc in GLOBAL_HSCS.get(agent_upper, []):
                if hsc in my_scs and hsc not in occupied_bases and hsc in GLOBAL_PROV_TO_IDX:
                    state_tensor[GLOBAL_PROV_TO_IDX[hsc], 49] = 1.0

    if bounces is not None:
        for b_loc in bounces:
            b_loc_upper = b_loc.upper()
            if b_loc_upper in GLOBAL_PROV_TO_IDX:
                state_tensor[GLOBAL_PROV_TO_IDX[b_loc_upper], 50] = 1.0

    if prev_state is not None:
        prev_centers = prev_state.get('centers', {})
        for prev_power, prev_sc_list in prev_centers.items():
            prev_power_upper = prev_power.upper()
            if prev_power_upper in GLOBAL_POWER_TO_IDX:
                prev_p_idx = GLOBAL_POWER_TO_IDX[prev_power_upper]
                for sc in prev_sc_list:
                    sc_upper = sc.upper()
                    if sc_upper in GLOBAL_PROV_TO_IDX:
                        state_tensor[GLOBAL_PROV_TO_IDX[sc_upper], 51 + prev_p_idx] = 1.0

    return state_tensor

# --- STATE PROCESSORS ---

class InteractionMatrixTracker:
    """Tracks the 7x7 Diplomatic Interaction Matrix (H) using EMA."""
    def __init__(self, num_agents=7, gamma=0.9):
        self.num_agents = num_agents
        self.gamma = gamma
        self.H = torch.zeros((num_agents, num_agents), dtype=torch.float32)

    def update(self, delta_H):
        """delta_H should be a [-1, 1] tensor of shape (7, 7) derived from phase actions"""
        self.H = self.gamma * self.H + (1 - self.gamma) * delta_H
        return self.H

class MacroStatePooler(nn.Module):
    """Generates S_M via Cross-Attention Pooling over the Micro-State."""
    def __init__(self, d_model=256, num_queries=8, d_val=256):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        
        self.Q_strat = nn.Parameter(torch.randn(1, num_queries, d_model))
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_val)

    def forward(self, S_mu):
        batch_size = S_mu.size(0)
        
        Q = self.Q_strat.expand(batch_size, -1, -1) 
        K = self.W_K(S_mu)                          
        V = self.W_V(S_mu)                          
        
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_model ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        S_M = torch.bmm(attn_weights, V) 
        return S_M

# --- PHASE 1 ARCHITECTURES ---

class GraphBiasedAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, num_nodes=81):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        self.distance_bias = nn.Embedding(20, nhead) 

    def forward(self, x, distance_matrix_D, mask=None):
        B, L, _ = x.size()
        x_norm = self.norm(x)
        
        q = self.q_proj(x_norm).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        spatial_bias = self.distance_bias(distance_matrix_D).permute(2, 0, 1).unsqueeze(0)
        scores = scores + spatial_bias
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, self.d_model)
        return self.out_proj(out)

class TacticalWorker(nn.Module):
    """T_phi: Executes tactics guided by the latent strategy z_t."""
    def __init__(self, d_model=256, vocab_size=22231):
        super().__init__()
        self.d_model = d_model
        
        self.feature_projection = nn.Linear(FEATURE_DIM, d_model)
        self.encoder_transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        
        self.strategy_cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        self.strategy_norm = nn.LayerNorm(d_model)
        
        self.decoder_layer = GraphBiasedAttentionBlock(d_model, nhead=8)
        self.action_head = nn.Linear(d_model, vocab_size)

    def forward(self, S_mu_raw, z_t, distance_matrix_D):
        B, L, _ = S_mu_raw.size()
        x_emb = self.feature_projection(S_mu_raw)
        
        S_mu_encoded = self.encoder_transformer(x_emb) # Shape: (B, 82, 256)
        
        # z_t is shape (B, 8, 256)
        strat_context, _ = self.strategy_cross_attn(query=S_mu_encoded, key=z_t, value=z_t)
        
        S_mu_strat = self.strategy_norm(S_mu_encoded + strat_context)
        
        decoder_out = self.decoder_layer(S_mu_strat, distance_matrix_D)
        
        logits = self.action_head(decoder_out)
        return logits, S_mu_encoded

class InverseModel(nn.Module):
    """I_psi: Calculates the achieved z based on state transitions per theater."""
    def __init__(self, d_val=256, d_model=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_val * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, d_model),
            nn.Tanh() 
        )

    def forward(self, S_M_t, S_M_t_next):
        x = torch.cat([S_M_t, S_M_t_next], dim=-1)
        
        return self.mlp(x) # Output shape: (B, 8, 256)

class MacroManager(nn.Module):
    """
    M_theta: High-Performance Cross-Attention + GRU Strategist.
    Uses z_prev to query the 8 theaters of war, then formally gates the strategy update.
    """
    def __init__(self, d_model=256, num_theaters=8):
        super().__init__()
        
        self.H_proj = nn.Sequential(
            nn.Linear(7 * 7, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, d_model)
        )
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        
        self.gru_cell = nn.GRUCell(input_size=d_model, hidden_size=d_model)
        
        self.z_norm = nn.LayerNorm(d_model)

        self.z_out = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.z_out.weight)
        nn.init.zeros_(self.z_out.bias)

    def forward(self, S_M_t, H_t, z_prev):
        B = S_M_t.size(0)
        H_emb = self.H_proj(H_t.view(B, -1)).unsqueeze(1) # Shape: (B, 1, d_model)
        
        # Broadcast H_emb across the 8 theaters
        query = z_prev + H_emb.expand(-1, 8, -1) 
        
        attn_out, _ = self.cross_attn(query=query, key=S_M_t, value=S_M_t)
        attn_out = self.attn_norm(attn_out + query)
        
        # Flatten Batch and Theaters to process all 8 regions independently and simultaneously
        attn_out_flat = attn_out.view(B * 8, -1)
        z_prev_flat = z_prev.view(B * 8, -1)
        
        z_t_flat = self.gru_cell(attn_out_flat, z_prev_flat)
        
        # Unflatten back to sequence
        z_t = z_t_flat.view(B, 8, -1)
        
        z_t = torch.tanh(self.z_out(self.z_norm(z_t)))
        return z_t # Shape: (B, 8, 256)

# --- THE FEUDAL ENVELOPE ---

class FeudalDiplomacyAgent(nn.Module):
    def __init__(self, d_model=256, vocab_size=22231):
        super().__init__()
        self.pooler = MacroStatePooler(d_model=d_model)
        self.worker = TacticalWorker(d_model=d_model, vocab_size=vocab_size)
        self.manager = MacroManager(d_model=d_model)
        self.inverse_model = InverseModel(d_model=d_model)
        
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        num_provs = len(GLOBAL_PROVINCES)
        self.register_buffer("D", torch.zeros((num_provs, num_provs), dtype=torch.long)) 
        
    def forward_phase_2_bc(self, S_mu_raw):
        """Pre-training: z_t is forced to 0. (One-Shot Prediction)"""
        B = S_mu_raw.size(0)
        z_zero = torch.zeros((B, 8, self.worker.d_model), device=S_mu_raw.device, dtype=S_mu_raw.dtype)
        logits, _ = self.worker(S_mu_raw, z_zero, self.D)
        return logits

    def forward(self, mb_obs, mb_H, mb_prev_z, worker_z_target):
        """Unified forward pass to trigger DDP gradient hooks."""
        x_emb = self.worker.feature_projection(mb_obs)
        S_mu_encoded = self.worker.encoder_transformer(x_emb)
        S_M = self.pooler(S_mu_encoded)
        
        predicted_z = self.manager(S_M, mb_H, mb_prev_z)
        
        logits, _ = self.worker(mb_obs, worker_z_target, self.D)
        
        values_pred = self.value_head(S_M.mean(dim=1)).squeeze(-1).float()
        
        return S_M, predicted_z, logits, values_pred

    def step(self, S_mu_raw, H_t, z_prev):
        """Inference Step (Phase 4)."""
        x_emb = self.worker.feature_projection(S_mu_raw)
        S_mu_encoded = self.worker.encoder_transformer(x_emb)
        S_M_t = self.pooler(S_mu_encoded)
        z_t = self.manager(S_M_t, H_t, z_prev)
        return z_t, S_M_t

class DiplomacyTransformerEnv(ParallelEnv):
    """
    PettingZoo Parallel Environment wrapper for the game of Diplomacy.
    Calculates step rewards, draw conditions, and manages historical state buffers.
    """
    metadata = {'render_modes': ['human'], "name": "diplomacy_transformer_v0"}

    def __init__(self, history_length=3):
        self.possible_agents = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
        self.game = Game()
        
        self.provinces = GLOBAL_PROVINCES[:] 
        self.num_provinces = len(self.provinces)
        
        self.history_length = history_length
        self.step_count = 0
        
        self.order_to_idx, self.idx_to_order = build_global_vocab()
        self.vocab_size = len(self.order_to_idx)
        
        self.tracker = InteractionMatrixTracker(num_agents=7, gamma=0.9)
        
        self.state_history = {
            a: np.zeros((self.history_length, self.num_provinces, FEATURE_DIM), dtype=np.float32) 
            for a in self.possible_agents
        }

        self.stalemate_counter = 0
        self.stalemate_threshold = 3 
        self.last_year_sc_owners = {}

    def _update_history(self, agent, new_state_tensor):
        self.state_history[agent] = np.roll(self.state_history[agent], shift=1, axis=0)
        self.state_history[agent][0] = new_state_tensor

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.game = Game()
        self.step_count = 0
        self.tracker = InteractionMatrixTracker(num_agents=7, gamma=0.9)
        
        self.state_history = {
            a: np.zeros((self.history_length, self.num_provinces, FEATURE_DIM), dtype=np.float32) 
            for a in self.possible_agents
        }
        
        self.stalemate_counter = 0
        self.last_year_sc_owners = {sc: a for a in self.agents for sc in self.game.get_centers(a)}

        current_state = self.game.get_state()
        current_phase = self.game.get_current_phase()

        bounces = [loc for loc, msg in self.game.get_order_status().items() if 'bounced' in msg or 'fails' in msg]

        observations = {}
        for a in self.agents:
            agent_obs = parse_state_to_tensor(
                turn_data={'name': current_phase, 'state': current_state}, 
                observing_agent=a,
                prev_state=None,
                bounces=None,
                H_matrix=self.tracker.H
            )
            self._update_history(a, agent_obs)
            observations[a] = self.state_history[a].copy()
            
        infos = {a: {
            'action_mask': get_sparse_action_mask(self.game, a, self.provinces, self.order_to_idx),
            'H_matrix': self.tracker.H.cpu().numpy()
        } for a in self.agents}
        return observations, infos

    def step(self, actions, progress=0.0):
        self.step_count += 1
        anneal_factor = max(0.0, 1.0 - progress)
        prev_sc_counts = {a: len(self.game.get_centers(a)) for a in self.possible_agents}
        self.game.clear_orders()

        prev_state_dict = self.game.get_state()
        prev_units = {a: prev_state_dict['units'].get(a, []) for a in self.agents}
        prev_phase_type = self.game.get_current_phase()[-1]
                        
        rewards = {a: 0.0 for a in self.agents}
        
        legality_metrics = {a: {'proposed': 0, 'illegal_dropped': 0, 'illegal_prov_indices': []} for a in self.agents}
        
        for agent, action_indices in actions.items():
            text_orders = []
            
            orderable_locs = self.game.get_orderable_locations(agent)
            all_possible = self.game.get_all_possible_orders()
            
            for i, order_idx in enumerate(action_indices):
                order_str = self.idx_to_order[int(order_idx)]
                if order_str != 'NONE':
                    legality_metrics[agent]['proposed'] += 1
                    
                    prov_upper = self.provinces[i].upper()
                    is_legal = False
                    
                    if prov_upper in orderable_locs:
                        legal_orders_for_prov = [
                            o.replace('*', '').upper() for o in all_possible.get(prov_upper, [])
                        ]
                        
                        if order_str in legal_orders_for_prov:
                            is_legal = True
                            
                    if is_legal:
                        text_orders.append(order_str)
                    else:
                        legality_metrics[agent]['illegal_dropped'] += 1
                        legality_metrics[agent]['illegal_prov_indices'].append(i)
                        
            self.game.set_orders(agent, text_orders)

        delta_H = torch.zeros((7, 7), dtype=torch.float32)
        
        prov_to_power_idx = {}
        
        for p, units in prev_units.items():
            if p not in GLOBAL_POWER_TO_IDX: continue
            p_idx = GLOBAL_POWER_TO_IDX[p]
            for u in units:
                loc_base = u.split()[1].split('/')[0]
                prov_to_power_idx[loc_base] = p_idx
                
        prev_centers = prev_state_dict.get('centers', {})
        for p, scs in prev_centers.items():
            if p not in GLOBAL_POWER_TO_IDX: continue
            p_idx = GLOBAL_POWER_TO_IDX[p]
            for sc in scs:
                if sc not in prov_to_power_idx:
                    prov_to_power_idx[sc] = p_idx
                
        for agent in self.agents:
            agent_idx = GLOBAL_POWER_TO_IDX[agent]
            text_orders = self.game.get_orders(agent) 
            
            for order in text_orders:
                try:
                    if ' S ' in order or ' C ' in order:
                        target_str = order.replace(' S ', '|').replace(' C ', '|').split('|')[1]
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
                            
                    elif ' - ' in order:
                        target_loc = order.split(' - ')[1].strip().split()[0].split('/')[0]
                        
                        target_owner_idx = prov_to_power_idx.get(target_loc)
                        if target_owner_idx is not None and target_owner_idx != agent_idx:
                            delta_H[agent_idx, target_owner_idx] -= 1.0
                            delta_H[target_owner_idx, agent_idx] -= 1.0
                except Exception as e:
                    print(f"\n[PARSER WARNING] Ignored malformed order: '{order}' | Error: {e}")
                    continue
                        
        delta_H = torch.clamp(delta_H, -1.0, 1.0)
        self.tracker.update(delta_H)

        self.game.process()
        
        current_state_dict = self.game.get_state()
        current_phase = self.game.get_current_phase()
        bounces = [loc for loc, msg in self.game.get_order_status().items() if 'bounced' in msg or 'fails' in msg]

        observations = {}
        for a in self.agents:
            agent_obs = parse_state_to_tensor(
                turn_data={'name': current_phase, 'state': current_state_dict}, 
                observing_agent=a,
                prev_state=prev_state_dict,
                bounces=bounces,
                H_matrix=self.tracker.H 
            )
            self._update_history(a, agent_obs)
            observations[a] = self.state_history[a].copy()
        
        is_done = False
        if self.step_count >= 100: 
            is_done = True
            
        current_state_dict = self.game.get_state()
        current_phase = self.game.get_current_phase()
        all_map_scs = self.game.map.scs
        
        if current_phase.startswith('S') and current_phase.endswith('M') and self.step_count > 1:
            current_sc_owners_map = {sc: a for a in self.possible_agents for sc in self.game.get_centers(a)}
            if current_sc_owners_map == self.last_year_sc_owners:
                self.stalemate_counter += 1
            else:
                self.stalemate_counter = 0
                self.last_year_sc_owners = current_sc_owners_map.copy()
            
            if self.stalemate_counter >= self.stalemate_threshold:
                is_done = True

        if not is_done:
            for a in self.agents:
                if len(self.game.get_centers(a)) >= 18:
                    is_done = True
                    break

        for agent in self.agents:
            current_scs = self.game.get_centers(agent)
            current_sc_count = len(current_scs)
            agent_units = current_state_dict['units'].get(agent, [])
            prev_agent_units = prev_units.get(agent, [])
            
            # rewards[agent] -= 0.05
            
            # Supply Center Deltas
            sc_delta = current_sc_count - prev_sc_counts.get(agent, 0)
            if sc_delta != 0:
                rewards[agent] += (sc_delta * 5.0)
                        
            occupied_unowned_scs = 0
            for unit_str in agent_units:
                prov_base = unit_str.split()[1].split('/')[0] 
                if prov_base in all_map_scs and prov_base not in current_scs:
                    occupied_unowned_scs += 1
            rewards[agent] += (occupied_unowned_scs * 0.2) * anneal_factor
                
            # Dislodgement Penalty
            current_dislodged = current_state_dict.get('dislodged', {}).get(agent, [])
            if len(current_dislodged) > 0:
                rewards[agent] -= (len(current_dislodged) * 0.5) * anneal_factor
                
        # Terminal States & Truncation Multipliers
        if is_done:
            solo_winner = None
            for agent in self.agents:
                if len(self.game.get_centers(agent)) >= 18:
                    solo_winner = agent
                    break
            
            if solo_winner:
                rewards[solo_winner] += 150.0
            else:
                # Sum of Squares Draw Scoring
                total_sq_scs = sum(len(self.game.get_centers(a)) ** 2 for a in self.agents)
                if total_sq_scs > 0:
                    for agent in self.agents:
                        agent_scs = len(self.game.get_centers(agent))
                        if agent_scs > 0:
                            sos_share = ((agent_scs ** 2) / total_sq_scs) * 100.0
                            rewards[agent] += sos_share

        for agent in self.agents:
            current_scs = self.game.get_centers(agent)
            agent_units = current_state_dict.get('units', {}).get(agent, [])
            if len(current_scs) == 0 and len(agent_units) == 0:
                rewards[agent] -= 50.0

        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}

        for agent in self.agents:
            current_scs = self.game.get_centers(agent)
            agent_units = current_state_dict.get('units', {}).get(agent, [])
            
            if len(current_scs) >= 18 or (len(current_scs) == 0 and len(agent_units) == 0):
                terminations[agent] = True
            elif is_done: 
                truncations[agent] = True

        infos = {a: {
            'action_mask': get_sparse_action_mask(self.game, a, self.provinces, self.order_to_idx), 
            'legality_metrics': legality_metrics[a],
            'H_matrix': self.tracker.H.cpu().numpy()
        } for a in self.agents}
        
        self.agents = [a for a in self.agents if not terminations[a] and not truncations[a]]
        return observations, rewards, terminations, truncations, infos