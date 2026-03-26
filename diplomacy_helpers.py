import torch
import torch.nn as nn
from diplomacy import Game
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, MultiDiscrete
import numpy as np
import functools
from diplomacy import Game
import torch.optim as optim
from torch.distributions import Categorical
import os
import json

_DUMMY_GAME = Game()
GLOBAL_PROVINCES = list(_DUMMY_GAME.map.locs)
GLOBAL_PROV_TO_IDX = {prov: i for i, prov in enumerate(GLOBAL_PROVINCES)}

GLOBAL_POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
GLOBAL_POWER_TO_IDX = {power: i for i, power in enumerate(GLOBAL_POWERS)}

class KVCacheAttentionBlock(nn.Module):
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
        
        # Standard Transformer FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x, kv_cache=None):
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

        is_causal = (seq_len > 1)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        # First residual add
        x = x + self.out_proj(attn_out) 
        
        # Second residual add over the FFN
        out = x + self.ffn(self.norm2(x))
        
        return out, new_kv_cache

class DiplomacyTransformer(nn.Module):
    def __init__(self, input_dim=16, d_model=256, nhead=8, num_layers=8, num_provinces=81, history_length=3, vocab_size=14000):
        super().__init__()
        
        self.num_provinces = num_provinces
        self.history_length = history_length
        self.d_model = d_model
        
        # --- THE ENCODER (Unchanged) ---
        self.feature_projection = nn.Linear(input_dim, d_model)
        self.province_embedding = nn.Embedding(num_provinces, d_model)
        self.time_embedding = nn.Embedding(history_length, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.value_head = nn.Linear(d_model * num_provinces, 1)

        # --- NEW: THE AUTOREGRESSIVE DECODER ---
        # 1. Embed the previously chosen action so we can feed it into the next step
        self.action_embedding = nn.Embedding(vocab_size, d_model)
        # Dedicated learned start token
        self.start_token_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 2. Our custom KV Cache block (you can stack multiple of these if needed)
        self.causal_decoder_block = KVCacheAttentionBlock(d_model=d_model, nhead=nhead)
        
        # 3. The final projection to logits
        self.action_head = nn.Linear(d_model, vocab_size)

        # PRE-COMPUTE INDICES
        p_idx = torch.arange(num_provinces).unsqueeze(0).unsqueeze(0).expand(1, history_length, -1)
        t_idx = torch.arange(history_length).unsqueeze(0).unsqueeze(-1).expand(1, -1, num_provinces)
        self.register_buffer('prov_indices', p_idx)
        self.register_buffer('time_indices', t_idx)

    def encode_state(self, x):
        """
        STEP 1: Process the board state once. 
        Call this outside your province-loop.
        """
        batch_size = x.size(0)
        p_idx = self.prov_indices.expand(batch_size, -1, -1)
        t_idx = self.time_indices.expand(batch_size, -1, -1)
        
        x_proj = self.feature_projection(x)
        x_emb = x_proj + self.province_embedding(p_idx) + self.time_embedding(t_idx)
        
        seq_input = x_emb.reshape(batch_size, self.history_length * self.num_provinces, self.d_model)
        transformer_out = self.transformer(seq_input)
        
        out_reshaped = transformer_out.reshape(batch_size, self.history_length, self.num_provinces, self.d_model)
        current_state_repr = out_reshaped[:, 0, :, :] 
        
        flat_current_state = current_state_repr.reshape(batch_size, -1)
        state_value = self.value_head(flat_current_state)
        
        return current_state_repr, state_value

    def decode_step(self, prev_action, prov_idx, state_repr, kv_cache=None):
        batch_size = state_repr.size(0)
        
        # Use the start embedding for the very first province, otherwise use the action embedding
        if prov_idx == 0:
            action_emb = self.start_token_embedding.expand(batch_size, -1, -1).squeeze(1)
        else:
            action_emb = self.action_embedding(prev_action)
            
        prov_state = state_repr[:, prov_idx, :]
        decoder_input = (action_emb + prov_state).unsqueeze(1)
        
        decoder_out, new_kv_cache = self.causal_decoder_block(decoder_input, kv_cache)
        logits = self.action_head(decoder_out.squeeze(1))
        
        return logits, new_kv_cache

    def forward(self, x, actions):
        batch_size = x.size(0)
        state_repr, state_value = self.encode_state(x)
        
        # Embed all actions EXCEPT the last one (shifting right)
        action_emb = self.action_embedding(actions[:, :-1])
        
        # Expand the start token for the batch and prepend it
        start_emb = self.start_token_embedding.expand(batch_size, 1, -1)
        shifted_emb = torch.cat([start_emb, action_emb], dim=1)
        
        decoder_input = shifted_emb + state_repr
        decoder_out, _ = self.causal_decoder_block(decoder_input, kv_cache=None)
        
        action_logits = self.action_head(decoder_out)
        return action_logits, state_value

def build_global_vocab(json_path="./datasets/standard_no_press.jsonl", cache_path="vocab.txt"):
    # 1. Load from cache if it exists for instant startup
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            sorted_orders = [line.strip() for line in f.readlines() if line.strip()]
        order_to_idx = {order: i for i, order in enumerate(sorted_orders)}
        return order_to_idx, {i: order for order, i in order_to_idx.items()}

    print("Building global vocabulary from dataset (this may take a minute)...")
    unique_orders = set(['NONE'])
    
    # 2. Scan the dataset to find every historically played order
    with open(json_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            game_data = json.loads(line)
            
            # Skip non-standard maps just in case
            if game_data.get('map', 'standard') != 'standard':
                continue
                
            for phase in game_data.get('phases', []):
                for power, orders in phase.get('orders', {}).items():
                    # NEW Add a safety check to skip None or empty lists
                    if not orders:
                        continue
                        
                    for order_str in orders:
                        # Clean the order string to match the engine's formatting
                        clean_order = order_str.replace('*', '')
                        unique_orders.add(clean_order)
                        
    sorted_orders = sorted(list(unique_orders))
    
    # 3. Save to a text file so we never have to parse the JSON again
    with open(cache_path, 'w') as f:
        for order in sorted_orders:
            f.write(f"{order}\n")
            
    order_to_idx = {order: i for i, order in enumerate(sorted_orders)}
    return order_to_idx, {i: order for order, i in order_to_idx.items()}

def get_global_action_mask(game, power, provinces, order_to_idx):
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

def get_sparse_action_mask(game, power, provinces, order_to_idx, max_len=1200):
    # 1. Generate the dense mask normally
    mask = np.zeros((len(provinces), len(order_to_idx)), dtype=np.bool_)
    orderable_locs = game.get_orderable_locations(power)
    
    # --- COASTAL MAPPING FIX ---
    # Map specific coastal locations (STP/SC) to their base province (STP)
    base_loc_to_orders = {}
    for loc in orderable_locs:
        base_prov = loc.split('/')[0]
        base_loc_to_orders[base_prov] = game.get_all_possible_orders().get(loc, [])
    # ---------------------------
    
    for i, prov in enumerate(provinces):
        # Check against the base province mapping instead of orderable_locs
        if prov in base_loc_to_orders:
            for order in base_loc_to_orders[prov]:
                if order in order_to_idx:
                    mask[i, order_to_idx[order]] = True
        
        if not mask[i].any():
            mask[i, order_to_idx['NONE']] = True
            
    # 2. Convert to Sparse Indices
    flat_mask = mask.flatten()
    valid_indices = np.where(flat_mask)[0].astype(np.int32)
    num_valid = len(valid_indices)
    
    if num_valid > max_len:
        raise ValueError(f"FATAL: Exceeded max sparse mask length: {num_valid}")
        
    sparse_mask = np.full(max_len, -1, dtype=np.int32)
    sparse_mask[:num_valid] = valid_indices
    return sparse_mask

class DiplomacyTransformerEnv(ParallelEnv):
    metadata = {'render_modes': ['human'], "name": "diplomacy_transformer_v0"}

    def __init__(self, history_length=3):
        self.possible_agents = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
        self.game = Game()
        
        self.provinces = GLOBAL_PROVINCES[:] 
        self.num_provinces = len(self.provinces)
        
        self.history_length = history_length
        self.step_count = 0
        
        # 1. Initialize the Global Vocabulary once
        self.order_to_idx, self.idx_to_order = build_global_vocab()
        self.vocab_size = len(self.order_to_idx)
        
        # 2. State history buffer to feed the Transformer
        # Shape: (History_Length, 81 Provinces, 16 Features)
        self.state_history = np.zeros((self.history_length, self.num_provinces, 16), dtype=np.float32)

    def _update_history(self, new_state_tensor):
        # Roll the history buffer backward (oldest state drops out)
        self.state_history = np.roll(self.state_history, shift=1, axis=0)
        # Insert the newest state at index 0
        self.state_history[0] = new_state_tensor

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.game = Game()
        self.step_count = 0
        
        # Clear history buffer with zeros
        self.state_history = np.zeros((self.history_length, self.num_provinces, 16), dtype=np.float32)
        
        # Get first state and update history
        obs_tensor = parse_state_to_tensor({'state': self.game.get_state()})
        self._update_history(obs_tensor)
        
        # All agents receive the exact same global history tensor
        observations = {a: self.state_history.copy() for a in self.agents} 
        
        # Generate exact global action masks for the new single head
        infos = {a: {'action_mask': get_sparse_action_mask(self.game, a, self.provinces, self.order_to_idx)} for a in self.agents}
        return observations, infos

    def step(self, actions):
        self.step_count += 1
        self.game.clear_orders()
        
        # --- CAPTURE "BEFORE" STATE FOR REWARDS ---
        prev_sc_owners = {sc: a for a in self.possible_agents for sc in self.game.get_centers(a)}
        prev_state_dict = self.game.get_state()
        prev_units = {a: prev_state_dict['units'].get(a, []) for a in self.agents}
        prev_phase_type = self.game.get_current_phase()[-1] # Gets 'M', 'R', or 'A'
        
        rewards = {a: 0.0 for a in self.agents}
        
        # --- DECODE ACTIONS ---
        for agent, action_indices in actions.items():
            text_orders = []
            for i, order_idx in enumerate(action_indices):
                order_str = self.idx_to_order[order_idx]
                if order_str != 'NONE':
                    text_orders.append(order_str)
            self.game.set_orders(agent, text_orders)

        # Adjudicate the turn
        self.game.process()
        
        # --- UPDATE STATE & HISTORY ---
        new_obs_tensor = parse_state_to_tensor({'state': self.game.get_state()})
        self._update_history(new_obs_tensor)
        observations = {a: self.state_history.copy() for a in self.agents}
        
        # --- REWARDS & TERMINATION ---
        is_done = False
        if self.step_count >= 150: 
            is_done = True
        
        current_state_dict = self.game.get_state()
        
        for agent in self.agents:
            current_scs = self.game.get_centers(agent)
            current_units = current_state_dict['units'].get(agent, [])
            agent_prev_units = prev_units.get(agent, [])
            
            # 1. Base Existence Reward
            rewards[agent] += len(current_scs) * 0.01  
            
            # 2. Strategic Milestones (Massively increased to encourage risk)
            for sc in current_scs:
                if sc not in prev_sc_owners: 
                    rewards[agent] += 5.0     # Took a neutral SC
                elif prev_sc_owners[sc] != agent: 
                    rewards[agent] += 10.0    # Stole an enemy SC!
                    
            for sc, owner in prev_sc_owners.items():
                if owner == agent and sc not in current_scs: 
                    rewards[agent] -= 10.0    # Lost an SC
            
            # 3. Tactical Micro-Rewards (The Loop Breaker)
            if prev_phase_type == 'M': # Only apply to Movement phases
                for unit in current_units:
                    if unit not in agent_prev_units:
                        # Unit successfully moved to a new province
                        rewards[agent] += 0.5
                    else:
                        # Unit stayed in place (Held or Bounced)
                        rewards[agent] -= 0.1 
            
            # 4. Endgame Conditions
            if len(current_scs) >= 18:
                rewards[agent] += 100.0
                is_done = True
            if len(current_scs) == 0 and len(current_units) == 0:
                rewards[agent] -= 50.0

        terminations = {a: is_done for a in self.agents}
        infos = {a: {'action_mask': get_sparse_action_mask(self.game, a, self.provinces, self.order_to_idx)} for a in self.agents}
        
        self.agents = [a for a in self.agents if not terminations[a] and (len(self.game.get_centers(a)) > 0 or len(self.game.get_state()['units'].get(a, [])) > 0)]
        
        return observations, rewards, terminations, {a: False for a in self.agents}, infos

def parse_state_to_tensor(turn_data):
    # REMOVED: game = Game()
    state_tensor = np.zeros((len(GLOBAL_PROV_TO_IDX), 16), dtype=np.float32)
    state_info = turn_data['state']
    units = state_info.get('units', {})
    centers = state_info.get('centers', {})

    for power, unit_list in units.items():
        if power not in GLOBAL_POWER_TO_IDX: continue
        power_idx = GLOBAL_POWER_TO_IDX[power] # Fixed to ensure alignment
        for unit_str in unit_list:
            clean_str = unit_str.replace('*', '')
            parts = clean_str.split()
            if len(parts) >= 2:
                u_type, u_loc = parts[0], parts[1].split('/')[0] 
                if u_loc in GLOBAL_PROV_TO_IDX:
                    p_idx = GLOBAL_PROV_TO_IDX[u_loc]
                    state_tensor[p_idx, power_idx] = 1.0
                    if u_type == 'A': state_tensor[p_idx, 7] = 1.0
                    elif u_type == 'F': state_tensor[p_idx, 8] = 1.0

    for power, sc_list in centers.items():
        if power not in GLOBAL_POWER_TO_IDX: continue
        power_idx = GLOBAL_POWER_TO_IDX[power]
        for sc in sc_list:
            if sc in GLOBAL_PROV_TO_IDX:
                p_idx = GLOBAL_PROV_TO_IDX[sc]
                state_tensor[p_idx, 9 + power_idx] = 1.0
                
    return state_tensor

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Environment and get Vocab Size
    env = DiplomacyTransformerEnv(history_length=3)
    vocab_size = env.vocab_size
    
    # Initialize Model and Optimizer (Pass env.num_provinces here)
    net = DiplomacyTransformer(num_provinces=env.num_provinces, vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    
    num_updates = 1000
    num_steps = 150
    
    for update in range(num_updates):
        obs, infos = env.reset()
        active_agents = env.agents
        
        # Rollout buffers
        batch_obs, batch_actions, batch_logprobs, batch_rewards = [], [], [], []
        
        net.eval()
        for step in range(num_steps):
            if not active_agents:
                break
                
            actions_to_send = {}
            for agent in active_agents:
                # Shape: (1, History, 81, 16)
                agent_obs = torch.tensor(obs[agent], device=device).unsqueeze(0)
                batch_size = agent_obs.size(0) # FIXED: Define batch_size
                
                # FIXED: Convert Sparse Mask back to Dense Boolean Mask
                sparse_mask = torch.tensor(infos[agent]['action_mask'], device=device)
                dense_mask = torch.zeros((env.num_provinces, vocab_size), dtype=torch.bool, device=device)
                valid_indices = sparse_mask[sparse_mask != -1]
                
                if len(valid_indices) > 0:
                    # Map flat sparse indices back to (province, action) 2D coordinates
                    prov_indices = valid_indices // vocab_size
                    act_indices = valid_indices % vocab_size
                    dense_mask[prov_indices, act_indices] = True
                
                with torch.no_grad():
                    state_repr, value = net.encode_state(agent_obs)
                    
                    actions = []
                    logprobs = []
                    
                    current_action = torch.zeros(batch_size, dtype=torch.long, device=device)
                    kv_cache = None
                    
                    for prov_idx in range(net.num_provinces):
                        logits, kv_cache = net.decode_step(current_action, prov_idx, state_repr, kv_cache)
                        
                        # Apply the dense mask for THIS province
                        prov_mask = dense_mask[prov_idx, :].unsqueeze(0) 
                        logits = logits.masked_fill(~prov_mask, -1e9)
                        
                        dist = Categorical(logits=logits)
                        current_action = dist.sample()
                        
                        actions.append(current_action)
                        logprobs.append(dist.log_prob(current_action))

                    final_actions = torch.stack(actions, dim=1)
                    final_logprobs = torch.stack(logprobs, dim=1).sum(dim=1) 
                    
                # FIXED: Send final_actions instead of current_action
                actions_to_send[agent] = final_actions.squeeze(0).cpu().numpy()
                
                batch_obs.append(agent_obs)
                batch_actions.append(final_actions) # FIXED
                batch_logprobs.append(final_logprobs) # FIXED
                
            obs, rewards, terms, truncs, infos = env.step(actions_to_send)
            active_agents = env.agents
            
            for agent in actions_to_send.keys():
                batch_rewards.append(rewards.get(agent, 0.0))
                
        print(f"Update {update} | Steps: {len(batch_rewards)} | Avg Reward: {np.mean(batch_rewards):.4f}")