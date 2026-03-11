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

class DiplomacyTransformer(nn.Module):
    def __init__(self, input_dim=16, d_model=128, nhead=8, num_layers=4, num_provinces=81, history_length=3, vocab_size=14000):
        super().__init__()
        
        self.num_provinces = num_provinces
        self.history_length = history_length
        self.d_model = d_model
        
        # 1. Project the raw 16-feature input into the hidden dimension
        self.feature_projection = nn.Linear(input_dim, d_model)
        
        # 2. Spatial and Temporal Embeddings
        self.province_embedding = nn.Embedding(num_provinces, d_model)
        self.time_embedding = nn.Embedding(history_length, d_model)
        
        # 3. The Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Single Output Head (Predicting exactly 1 legal order per province)
        # We only predict actions for the current time-step, so we need a dedicated action head
        self.action_head = nn.Linear(d_model, vocab_size)
        
        # Optional: A critic head for Actor-Critic PPO
        self.value_head = nn.Linear(d_model * num_provinces, 1)

    def forward(self, x):
        # Expected input x shape: (Batch, History_Length, Provinces, Input_Dim)
        # Example: (32, 3, 81, 16)
        batch_size = x.size(0)
        
        # Create positional indices for embeddings
        prov_indices = torch.arange(self.num_provinces, device=x.device).unsqueeze(0).unsqueeze(0).expand(batch_size, self.history_length, -1)
        time_indices = torch.arange(self.history_length, device=x.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.num_provinces)
        
        # Project features and add embeddings
        x_proj = self.feature_projection(x)
        x_emb = x_proj + self.province_embedding(prov_indices) + self.time_embedding(time_indices)
        
        # Flatten time and provinces into a single sequence for the Transformer
        # New shape: (Batch, History_Length * Provinces, d_model)
        seq_input = x_emb.view(batch_size, self.history_length * self.num_provinces, self.d_model)
        
        # Pass through self-attention
        transformer_out = self.transformer(seq_input)
        
        # Reshape back to isolate the current time-step (assuming index 0 is the current turn)
        # Shape: (Batch, History_Length, Provinces, d_model)
        out_reshaped = transformer_out.view(batch_size, self.history_length, self.num_provinces, self.d_model)
        
        # Extract only the current time-step's representations to predict moves
        current_state_repr = out_reshaped[:, 0, :, :] # Shape: (Batch, 81, d_model)
        
        # Generate logits for the global action vocabulary
        action_logits = self.action_head(current_state_repr) # Shape: (Batch, 81, Vocab_Size)
        
        # Calculate state value (Critic) using the flattened current state
        flat_current_state = current_state_repr.view(batch_size, -1)
        state_value = self.value_head(flat_current_state)
        
        return action_logits, state_value

def build_global_vocab():
    dummy_game = Game()
    unique_orders = set(['NONE'])
    for prov, orders in dummy_game.get_all_possible_orders().items():
        for order in orders:
            unique_orders.add(order)
            
    sorted_orders = sorted(list(unique_orders))
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

class DiplomacyTransformerEnv(ParallelEnv):
    metadata = {'render_modes': ['human'], "name": "diplomacy_transformer_v0"}

    def __init__(self, history_length=3):
        self.possible_agents = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
        self.game = Game()
        self.provinces = list(self.game.map.locs)
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
        infos = {a: {'action_mask': get_global_action_mask(self.game, a, self.provinces, self.order_to_idx)} for a in self.agents}
        
        return observations, infos

    def step(self, actions):
        self.step_count += 1
        self.game.clear_orders()
        prev_sc_owners = {sc: a for a in self.possible_agents for sc in self.game.get_centers(a)}
        
        rewards = {a: 0.0 for a in self.agents}
        
        # --- DECODE ACTIONS ---
        for agent, action_indices in actions.items():
            text_orders = []
            
            # action_indices is now a simple 1D array of length 81 (one integer per province)
            for i, order_idx in enumerate(action_indices):
                order_str = self.idx_to_order[order_idx]
                
                # If the network chose 'NONE' or the string is empty, we skip it
                if order_str != 'NONE':
                    text_orders.append(order_str)
            
            # Submit the structurally perfect orders directly to the engine
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
        
        for agent in self.agents:
            current_scs = self.game.get_centers(agent)
            agent_units = self.game.get_state()['units'].get(agent, [])
            
            # Basic SC reward scaling
            rewards[agent] += len(current_scs) * 0.005
            
            for sc in current_scs:
                if sc not in prev_sc_owners: rewards[agent] += 1.0
                elif prev_sc_owners[sc] != agent: rewards[agent] += 2.0
            for sc, owner in prev_sc_owners.items():
                if owner == agent and sc not in current_scs: rewards[agent] -= 2.0
            
            if len(current_scs) >= 18:
                rewards[agent] += 100.0
                is_done = True
            if len(current_scs) == 0 and len(agent_units) == 0:
                rewards[agent] -= 50.0

        terminations = {a: is_done for a in self.agents}
        infos = {a: {'action_mask': get_global_action_mask(self.game, a, self.provinces, self.order_to_idx)} for a in self.agents}
        
        self.agents = [a for a in self.agents if not terminations[a] and (len(self.game.get_centers(a)) > 0 or len(self.game.get_state()['units'].get(a, [])) > 0)]
        
        return observations, rewards, terminations, {a: False for a in self.agents}, infos

def parse_state_to_tensor(turn_data):
    game = Game()
    provinces = game.map.locs
    prov_to_idx = {prov: i for i, prov in enumerate(provinces)}
    powers = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
    power_to_idx = {power: i for i, power in enumerate(powers)}

    state_tensor = np.zeros((len(provinces), 16), dtype=np.float32)
    state_info = turn_data['state']
    units = state_info.get('units', {})
    centers = state_info.get('centers', {})

    for power, unit_list in units.items():
        if power not in power_to_idx: continue
        power_idx = power_to_idx[power]
        for unit_str in unit_list:
            clean_str = unit_str.replace('*', '')
            parts = clean_str.split()
            if len(parts) >= 2:
                u_type, u_loc = parts[0], parts[1].split('/')[0] 
                if u_loc in prov_to_idx:
                    p_idx = prov_to_idx[u_loc]
                    state_tensor[p_idx, power_idx] = 1.0
                    if u_type == 'A': state_tensor[p_idx, 7] = 1.0
                    elif u_type == 'F': state_tensor[p_idx, 8] = 1.0

    for power, sc_list in centers.items():
        if power not in power_to_idx: continue
        power_idx = power_to_idx[power]
        for sc in sc_list:
            if sc in prov_to_idx:
                p_idx = prov_to_idx[sc]
                state_tensor[p_idx, 9 + power_idx] = 1.0
    return state_tensor

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Environment and get Vocab Size
    env = DiplomacyTransformerEnv(history_length=3)
    vocab_size = env.vocab_size
    
    # Initialize Model and Optimizer (Pass env.num_provinces here)
    net = DiplomacyTransformer(num_provinces=env.num_provinces, vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    
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
                mask = torch.tensor(infos[agent]['action_mask'], device=device)
                
                with torch.no_grad():
                    logits, value = net(agent_obs)
                    # Apply the global mask
                    logits = logits.squeeze(0).masked_fill(~mask, -1e9)
                    
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                    
                actions_to_send[agent] = action.cpu().numpy()
                
                batch_obs.append(agent_obs)
                batch_actions.append(action)
                batch_logprobs.append(logprob)
                
            obs, rewards, terms, truncs, infos = env.step(actions_to_send)
            active_agents = env.agents
            
            for agent in actions_to_send.keys():
                batch_rewards.append(rewards.get(agent, 0.0))
                
        print(f"Update {update} | Steps: {len(batch_rewards)} | Avg Reward: {np.mean(batch_rewards):.4f}")