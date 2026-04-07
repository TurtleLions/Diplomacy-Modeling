"""
Core helper module for the Diplomacy Reinforcement Learning pipeline.
Contains the custom Transformer architecture, feature extraction logic, 
and the Parallel PettingZoo environment wrapper.
"""
import os
import json
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.checkpoint import checkpoint
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, MultiDiscrete

from diplomacy import Game

# --- GLOBAL CONSTANTS ---
FEATURE_DIM = 46
MAX_SPARSE_MASK_LEN = 2000

_DUMMY_GAME = Game()
GLOBAL_PROVINCES = [prov.upper() for prov in list(_DUMMY_GAME.map.locs)]
GLOBAL_PROV_TO_IDX = {prov: i for i, prov in enumerate(GLOBAL_PROVINCES)}

GLOBAL_POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
GLOBAL_POWER_TO_IDX = {power: i for i, power in enumerate(GLOBAL_POWERS)}

GLOBAL_HSCS = {
    'AUSTRIA': ['VIE', 'BUD', 'TRI'],
    'ENGLAND': ['LON', 'EDI', 'LVP'],
    'FRANCE': ['PAR', 'MAR', 'BRE'],
    'GERMANY': ['BER', 'MUN', 'KIE'],
    'ITALY': ['ROM', 'VEN', 'NAP'],
    'RUSSIA': ['MOS', 'SEV', 'WAR', 'STP'],
    'TURKEY': ['ANK', 'CON', 'SMY']
}

class KVCacheAttentionBlock(nn.Module):
    """
    A custom Transformer block that supports causal decoding with a Key-Value cache.
    Optimizes autoregressive action sampling by caching previous attention states.
    """
    def __init__(self, d_model, nhead, dim_feedforward=1024):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x, kv_cache=None, padding_mask=None):
        batch_size, seq_len, _ = x.size()
        
        x_norm = self.norm1(x)
        
        q = self.q_proj(x_norm).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        new_kv_cache = (k, v)

        # Build custom attention mask to handle causal + unit padding constraints
        if padding_mask is not None or seq_len > 1:
            causal_mask = torch.tril(torch.ones((seq_len, k.size(2)), dtype=torch.bool, device=x.device))
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1) # (1, 1, seq_len, total_seq_len)
            
            if padding_mask is not None:
                valid_keys = (~padding_mask).unsqueeze(1).unsqueeze(2) # (batch, 1, 1, total_seq_len)
                
                is_query_padded = padding_mask.unsqueeze(1).unsqueeze(-1) # (batch, 1, seq_len, 1)
                
                attn_mask = causal_mask & (valid_keys | is_query_padded)
            else:
                attn_mask = causal_mask
                
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            # Single step decoding without padding
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        x = x + self.out_proj(attn_out) 
        out = x + self.ffn(self.norm2(x))
        
        return out, new_kv_cache

class DiplomacyTransformer(nn.Module):
    """
    End-to-end policy and value network for the game of Diplomacy.
    Utilizes an encoder to process historical board states and a unit-centric 
    causal decoder to autoregressively generate simultaneous orders.
    """
    def __init__(self, input_dim=FEATURE_DIM, d_model=256, nhead=8, num_layers=8, num_provinces=82, history_length=3, vocab_size=22231):
        super().__init__()
        
        self.num_provinces = num_provinces
        self.history_length = history_length
        self.d_model = d_model
        
        # State Encoder (Full Board Context)
        self.feature_projection = nn.Linear(input_dim, d_model)
        self.province_embedding = nn.Embedding(num_provinces, d_model)
        self.time_embedding = nn.Embedding(history_length, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.value_head = nn.Linear(d_model * num_provinces, 1)

        # Autoregressive Action Decoder (Unit-Centric)
        self.action_embedding = nn.Embedding(vocab_size, d_model)
        self.start_token_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        self.causal_decoder_block = KVCacheAttentionBlock(d_model=d_model, nhead=nhead)
        self.action_head = nn.Linear(d_model, vocab_size)

        p_idx = torch.arange(num_provinces).unsqueeze(0).unsqueeze(0).expand(1, history_length, -1).clone()
        t_idx = torch.arange(history_length).unsqueeze(0).unsqueeze(-1).expand(1, -1, num_provinces).clone()
        self.register_buffer('prov_indices', p_idx)
        self.register_buffer('time_indices', t_idx)

    def encode_state(self, x):
        """Processes the historical board state and extracts the current temporal representation."""
        batch_size = x.size(0)
        p_idx = self.prov_indices.expand(batch_size, -1, -1)
        t_idx = self.time_indices.expand(batch_size, -1, -1)
        
        x_proj = self.feature_projection(x)
        x_emb = x_proj + self.province_embedding(p_idx) + self.time_embedding(t_idx)
        
        seq_input = x_emb.reshape(batch_size, self.history_length * self.num_provinces, self.d_model)
        transformer_out = checkpoint(
            self.transformer, 
            seq_input, 
            use_reentrant=False
        )
        
        out_reshaped = transformer_out.reshape(batch_size, self.history_length, self.num_provinces, self.d_model)
        current_state_repr = out_reshaped[:, 0, :, :] 
        
        flat_current_state = current_state_repr.reshape(batch_size, -1)
        state_value = self.value_head(flat_current_state)
        
        return current_state_repr, state_value

    def decode_step(self, prev_action, active_state_repr, step_idx, kv_cache=None):
        """
        Executes a single step of autoregressive decoding for a specific active unit.
        active_state_repr: (batch_size, max_active_units, d_model) pre-gathered tensor.
        """
        batch_size = active_state_repr.size(0)
        
        if kv_cache is None:
            action_emb = self.start_token_embedding.expand(batch_size, -1, -1).squeeze(1)
        else:
            action_emb = self.action_embedding(prev_action)
            
        # Extract the state for the specific unit in the dynamic sequence
        unit_state = active_state_repr[:, step_idx, :]
        
        decoder_input = (action_emb + unit_state).unsqueeze(1)
        
        decoder_out, new_kv_cache = self.causal_decoder_block(decoder_input, kv_cache)
        logits = self.action_head(decoder_out.squeeze(1))
        
        return logits, new_kv_cache

    def decode_full(self, state_repr, actions, active_mask, return_hidden=False):
        """
        Unit-centric Teacher-forced forward pass utilized during Behavioral Cloning.
        Compresses the 82-province tensor down to only active units before decoding.
        """
        batch_size = state_repr.size(0)
        max_units = active_mask.sum(dim=1).max().item()
        
        if max_units == 0:
            dummy_out = torch.zeros((batch_size, 0, self.action_head.out_features), device=state_repr.device)
            dummy_mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=state_repr.device)
            return dummy_out, dummy_mask

        active_states = []
        active_acts = []
        
        for b in range(batch_size):
            valid_indices = torch.where(active_mask[b])[0]
            count = len(valid_indices)
            
            b_states = state_repr[b, valid_indices, :]
            b_acts = actions[b, valid_indices]
            
            if count < max_units:
                pad_states = torch.zeros((max_units - count, self.d_model), device=state_repr.device)
                pad_acts = torch.zeros((max_units - count,), dtype=torch.long, device=actions.device)
                b_states = torch.cat([b_states, pad_states], dim=0)
                b_acts = torch.cat([b_acts, pad_acts], dim=0)
                
            active_states.append(b_states)
            active_acts.append(b_acts)
            
        active_states = torch.stack(active_states) # (B, max_units, D)
        active_acts = torch.stack(active_acts)     # (B, max_units)

        if active_states.size(0) > 0 and not hasattr(self, '_debug_printed'):
            print("\n--- DEBUG: DECODE_FULL SEQUENCE ---")
            print(f"Max Units in Sequence: {max_units}")
            print(f"Active Mask Sums:      {active_mask.sum(dim=1)[:5]}")
            print(f"Packed Actions:        {active_acts[0][:5]}")
            self._debug_printed = True # Ensure it only prints once
        
        seq_lengths = active_mask.sum(dim=1, keepdim=True)
        idx = torch.arange(max_units, device=state_repr.device).unsqueeze(0)
        padding_mask = idx >= seq_lengths 
        
        action_emb = self.action_embedding(active_acts[:, :-1])
        start_emb = self.start_token_embedding.expand(batch_size, 1, -1)
        shifted_emb = torch.cat([start_emb, action_emb], dim=1)
        
        decoder_input = shifted_emb + active_states
        
        def decoder_wrapper(inp, p_mask):
            out, _ = self.causal_decoder_block(inp, kv_cache=None, padding_mask=p_mask)
            return out

        decoder_out = checkpoint(decoder_wrapper, decoder_input, padding_mask, use_reentrant=False)
        
        if return_hidden:
            return decoder_out
            
        action_logits = self.action_head(decoder_out)
        return action_logits, padding_mask

    def forward(self, x, actions, active_mask, return_hidden=False):
        """Standard full forward pass combining state encoding and unit-centric decoding."""
        state_repr, state_value = self.encode_state(x)
        action_logits, padding_mask = self.decode_full(state_repr, actions, active_mask, return_hidden=return_hidden)
        return action_logits, state_value, padding_mask


def build_global_vocab(json_path="./datasets/standard_no_press.jsonl", cache_path="vocab.txt"):
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
                        clean_order = order_str.replace('*', '')
                        unique_orders.add(clean_order)
                        
    sorted_orders = sorted(list(unique_orders))
    
    with open(cache_path, 'w') as f:
        for order in sorted_orders:
            f.write(f"{order}\n")
            
    order_to_idx = {order: i for i, order in enumerate(sorted_orders)}
    return order_to_idx, {i: order for order, i in order_to_idx.items()}

def get_global_action_mask(game, power, provinces, order_to_idx):
    """Generates a dense boolean mask for all legal actions available to a given power."""
    mask = np.zeros((len(provinces), len(order_to_idx)), dtype=np.bool_)
    orderable_locs = game.get_orderable_locations(power)
    
    for i, prov in enumerate(provinces):
        if prov in orderable_locs:
            for order in game.get_all_possible_orders().get(prov, []):
                if order in order_to_idx:
                    mask[i, order_to_idx[order]] = True
        
        if not mask[i].any():
            mask[i, order_to_idx['NONE']] = True
            
    return mask

def get_sparse_action_mask(game, power, provinces, order_to_idx, max_len=MAX_SPARSE_MASK_LEN):
    """Generates a memory-efficient sparse mask of legal actions for distributed training."""
    mask = np.zeros((len(provinces), len(order_to_idx)), dtype=np.bool_)
    orderable_locs = game.get_orderable_locations(power)
    
    base_loc_to_orders = {}
    for loc in orderable_locs:
        base_prov = loc.split('/')[0].upper()
        base_loc_to_orders[base_prov] = game.get_all_possible_orders().get(loc, [])
    
    for i, prov in enumerate(provinces):
        prov_upper = prov.upper()
        if prov_upper in base_loc_to_orders:
            for order in base_loc_to_orders[prov_upper]:
                clean_order = order.replace('*', '').upper()
                if clean_order in order_to_idx:
                    mask[i, order_to_idx[clean_order]] = True
        
        if not mask[i].any():
            mask[i, order_to_idx['NONE']] = True
            
    flat_mask = mask.flatten()
    valid_indices = np.where(flat_mask)[0].astype(np.int32)
    num_valid = len(valid_indices)
    
    if num_valid > max_len:
        print(f"Warning: Action space exceeds sparse limit ({num_valid}). Randomly truncating valid actions.")
        np.random.shuffle(valid_indices)
        valid_indices = valid_indices[:max_len]
        num_valid = max_len
        
    sparse_mask = np.full(max_len, -1, dtype=np.int32)
    sparse_mask[:num_valid] = valid_indices
    return sparse_mask

def parse_state_to_tensor(turn_data, observing_agent=None):
    """
    Parses a single game phase into a standardized geometric feature tensor.
    
    Feature Vector Layout (Dim: 46):
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

    units = state_info.get('units', {})
    centers = state_info.get('centers', {})
    dislodged = state_info.get('dislodged', {})

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
                u_loc = loc_parts[0]
                coast = loc_parts[1] if len(loc_parts) > 1 else None

                if u_loc in GLOBAL_PROV_TO_IDX:
                    p_idx = GLOBAL_PROV_TO_IDX[u_loc]
                    state_tensor[p_idx, power_idx] = 1.0
                    if u_type == 'A': state_tensor[p_idx, 7] = 1.0
                    elif u_type == 'F': state_tensor[p_idx, 8] = 1.0
                    
                    if coast == 'NC': state_tensor[p_idx, 16] = 1.0
                    elif coast == 'SC': state_tensor[p_idx, 17] = 1.0
                    elif coast == 'EC': state_tensor[p_idx, 18] = 1.0

    for power, unit_list in dislodged.items():
        power_upper = power.upper()
        if power_upper not in GLOBAL_POWER_TO_IDX: continue
        power_idx = GLOBAL_POWER_TO_IDX[power_upper]
        for unit_str in unit_list:
            clean_str = unit_str.replace('*', '').upper()
            parts = clean_str.split()
            if len(parts) >= 2:
                u_loc = parts[1].split('/')[0]
                if u_loc in GLOBAL_PROV_TO_IDX:
                    p_idx = GLOBAL_PROV_TO_IDX[u_loc]
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

    return state_tensor

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
        
        self.state_history = {
            a: np.zeros((self.history_length, self.num_provinces, FEATURE_DIM), dtype=np.float32) 
            for a in self.possible_agents
        }
        
        self.stalemate_counter = 0
        self.last_year_sc_owners = {sc: a for a in self.agents for sc in self.game.get_centers(a)}


        observations = {}
        for a in self.agents:
            agent_obs = parse_state_to_tensor({'name': self.game.get_current_phase(), 'state': self.game.get_state()}, observing_agent=a)
            self._update_history(a, agent_obs)
            observations[a] = self.state_history[a].copy()
            
        infos = {a: {'action_mask': get_sparse_action_mask(self.game, a, self.provinces, self.order_to_idx)} for a in self.agents}
        return observations, infos

    def step(self, actions):
        self.step_count += 1
        
        prev_sc_counts = {a: len(self.game.get_centers(a)) for a in self.possible_agents}

        self.game.clear_orders()

        prev_state_dict = self.game.get_state()
        prev_units = {a: prev_state_dict['units'].get(a, []) for a in self.agents}
        prev_phase_type = self.game.get_current_phase()[-1]
        
        global_orders = {}
        if prev_phase_type == 'M':
            for a, action_indices in actions.items():
                for order_idx in action_indices:
                    o_str = self.idx_to_order[int(order_idx)]
                    if o_str != 'NONE':
                        u_loc = o_str.split()[1].split('/')[0]
                        global_orders[u_loc] = o_str
                        
        rewards = {a: 0.0 for a in self.agents}
        
        for agent, action_indices in actions.items():
            text_orders = []
            for i, order_idx in enumerate(action_indices):
                order_str = self.idx_to_order[int(order_idx)]
                if order_str != 'NONE':
                    text_orders.append(order_str)
            self.game.set_orders(agent, text_orders)

        self.game.process()
        
        observations = {}
        for a in self.agents:
            agent_obs = parse_state_to_tensor({'name': self.game.get_current_phase(), 'state': self.game.get_state()}, observing_agent=a)
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
            
            rewards[agent] -= 0.05
            
            # Supply Center Deltas
            sc_delta = current_sc_count - prev_sc_counts.get(agent, 0)
            if sc_delta != 0:
                rewards[agent] += sc_delta * 10.0
                    
            # Tactical Advancement
            if prev_phase_type == 'M':
                curr_locs = {u.replace('*', '').upper().split()[1].split('/')[0] for u in agent_units if len(u.split()) >= 2}
                prev_locs = {u.replace('*', '').upper().split()[1].split('/')[0] for u in prev_agent_units if len(u.split()) >= 2}
                
                advanced_locs = curr_locs - prev_locs
                for loc in advanced_locs:
                    rewards[agent] += 0.1

                    if loc in all_map_scs and loc not in current_scs:
                        rewards[agent] += 1.0 
                        
            # Dislodgement Penalty
            current_dislodged = current_state_dict.get('dislodged', {}).get(agent, [])
            if len(current_dislodged) > 0:
                rewards[agent] -= (len(current_dislodged) * 0.5)
                
            # Terminal States & Truncation Multipliers
            if current_sc_count >= 18:
                rewards[agent] += 100.0
            elif current_sc_count == 0 and len(agent_units) == 0:
                rewards[agent] -= 50.0 
            elif is_done:
                base_truncation = current_sc_count * 3.0
                if self.stalemate_counter >= self.stalemate_threshold:
                    base_truncation -= 20.0 
                rewards[agent] += base_truncation

        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}

        for agent in self.agents:
            current_scs = self.game.get_centers(agent)
            agent_units = current_state_dict.get('units', {}).get(agent, [])
            
            if len(current_scs) >= 18 or (len(current_scs) == 0 and len(agent_units) == 0):
                terminations[agent] = True
            elif is_done: 
                truncations[agent] = True

        infos = {a: {'action_mask': get_sparse_action_mask(self.game, a, self.provinces, self.order_to_idx)} for a in self.agents}
        
        self.agents = [a for a in self.agents if not terminations[a] and not truncations[a]]

        return observations, rewards, terminations, truncations, infos