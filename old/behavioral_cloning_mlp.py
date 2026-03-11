import json
import numpy as np
from diplomacy import Game
import functools
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, MultiDiscrete
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os

def parse_state_to_tensor(turn_data):
    """
    Parses a Diplomacy JSON state into a (81, 16) NumPy tensor.
    Features per province (16 total):
    - [0:7]: One-hot vector of which power has a unit here.
    - [7]: 1 if unit is an Army (A).
    - [8]: 1 if unit is a Fleet (F).
    - [9:16]: One-hot vector of which power owns the Supply Center here.
    """
    
    # Initialize a dummy game just to pull the official map data
    game = Game()
    provinces = game.map.locs 
    prov_to_idx = {prov: i for i, prov in enumerate(provinces)}
    
    powers = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
    power_to_idx = {power: i for i, power in enumerate(powers)}
    
    # Initialize a zero tensor: 81 provinces, 16 features each
    state_tensor = np.zeros((len(provinces), 16), dtype=np.float32)
    
    # Extract the relevant dictionaries from your JSON
    state_info = turn_data['state']
    units = state_info.get('units', {})
    centers = state_info.get('centers', {})
    
    # 1. Encode Unit Positions and Types
    for power, unit_list in units.items():
        if power not in power_to_idx: continue
        power_idx = power_to_idx[power]
        
        for unit_str in unit_list:
            # Clean up the string (e.g., "*A BUR" means a dislodged unit)
            clean_str = unit_str.replace('*', '') 
            parts = clean_str.split()
            
            if len(parts) >= 2:
                u_type = parts[0]
                # Grab the standard 3-letter code, slicing off coast identifiers (like /NC) for the base node
                u_loc = parts[1] 
                
                if u_loc in prov_to_idx:
                    p_idx = prov_to_idx[u_loc]
                    # Flag the owner
                    state_tensor[p_idx, power_idx] = 1.0 
                    # Flag the unit type
                    if u_type == 'A':
                        state_tensor[p_idx, 7] = 1.0 
                    elif u_type == 'F':
                        state_tensor[p_idx, 8] = 1.0 
                        
    # 2. Encode Supply Center Ownership
    for power, sc_list in centers.items():
        if power not in power_to_idx: continue
        power_idx = power_to_idx[power]
        
        for sc in sc_list:
            if sc in prov_to_idx:
                p_idx = prov_to_idx[sc]
                # Flag the SC owner
                state_tensor[p_idx, 9 + power_idx] = 1.0 
                
    return state_tensor

# --- Example Usage ---
# Assuming 'sample_json' is loaded from your dataset containing the "S1901M" phase dict
# tensor = parse_state_to_tensor(sample_json['phases'][0])
# print(f"Tensor shape: {tensor.shape}")

ACTION_TYPES = ['NONE', 'H', '-', 'S', 'C', 'B', 'D', 'R']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}
IDX_TO_ACTION = {i: a for a, i in ACTION_TO_IDX.items()}

def get_province_vocab(game):
    """
    Returns the vocabulary for targets (Size: 82).
    Index 0 is always 'NONE' for actions that don't need a target.
    """
    provinces = ['NONE'] + list(game.map.locs)
    prov_to_idx = {p: i for i, p in enumerate(provinces)}
    idx_to_prov = {i: p for p, i in prov_to_idx.items()}
    return prov_to_idx, idx_to_prov

def get_compositional_action_mask(game, power, provinces, prov_to_idx):
    """
    Parses the legal text orders into grammatical tokens and builds 
    independent binary masks for Action Type, Target 1, and Target 2.
    """
    num_provs = len(provinces)
    
    # Initialize the 3 mask matrices with 0
    type_mask = np.zeros((num_provs, len(ACTION_TYPES)), dtype=np.int8)
    t1_mask = np.zeros((num_provs, len(prov_to_idx)), dtype=np.int8)
    t2_mask = np.zeros((num_provs, len(prov_to_idx)), dtype=np.int8)
    
    all_possible_orders = game.get_all_possible_orders()
    orderable_locs = game.get_orderable_locations(power)
    
    for i, prov in enumerate(provinces):
        if prov in orderable_locs:
            legal_orders = all_possible_orders.get(prov, [])
            for order in legal_orders:
                # String parsing: e.g., "A PAR S A MAR - BUR" -> ['A', 'PAR', 'S', 'A', 'MAR', '-', 'BUR']
                parts = order.split()
                
                if len(parts) >= 3:
                    act_type = parts[2]
                    t1 = 'NONE'
                    t2 = 'NONE'
                    
                    # Parse based on grammatical structure
                    if act_type in ['-', 'R']:
                        t1 = parts[3]
                    elif act_type == 'S':
                        t1 = parts[4] # Skip the supported unit's 'A' or 'F'
                        if len(parts) > 5 and parts[5] == '-':
                            t2 = parts[6]
                    elif act_type == 'C':
                        t1 = parts[4]
                        t2 = parts[6]
                        
                    # Lookup the indices
                    act_idx = ACTION_TO_IDX.get(act_type, 0)
                    t1_idx = prov_to_idx.get(t1, 0)
                    t2_idx = prov_to_idx.get(t2, 0)
                    
                    # Flip the mask values to 1
                    type_mask[i, act_idx] = 1
                    t1_mask[i, t1_idx] = 1
                    t2_mask[i, t2_idx] = 1
        else:
            # If the player has no unit here, force the 'NONE' dummy action
            type_mask[i, ACTION_TO_IDX['NONE']] = 1
            t1_mask[i, prov_to_idx['NONE']] = 1
            t2_mask[i, prov_to_idx['NONE']] = 1
            
    return {'type': type_mask, 'target1': t1_mask, 'target2': t2_mask}

def decode_compositional_order(province, action_array, game, idx_to_action, idx_to_prov):
    """
    Translates a [Type, Target1, Target2] integer array back into a DAIDE/diplomacy text string.
    Example input for Paris: [2, 14, 0] -> "A PAR - BUR"
    """
    act_idx, t1_idx, t2_idx = action_array
    
    act_str = idx_to_action[act_idx]
    t1_str = idx_to_prov[t1_idx]
    t2_str = idx_to_prov[t2_idx]
    
    if act_str == 'NONE':
        return None
        
    # We need to know if the unit in the province is an Army or Fleet to format the string properly
    unit_type = "A" # Default
    owner = game.get_state()['units']
    for power, units in owner.items():
        for u in units:
            if province in u:
                unit_type = u[0] # Grab the 'A' or 'F'
                break
                
    # Base unit string: e.g., "A PAR"
    base_unit = f"{unit_type} {province}"
    
    # Construct the string based on the grammar
    if act_str in ['H', 'B', 'D']:
        # e.g., A PAR H
        return f"{base_unit} {act_str}"
        
    elif act_str in ['-', 'R']:
        # e.g., A PAR - BUR
        return f"{base_unit} {act_str} {t1_str}"
        
    elif act_str == 'S':
        # Support can be a hold (A PAR S A MAR) or a move (A PAR S A MAR - BUR)
        if t2_str == 'NONE':
            return f"{base_unit} S {t1_str}"
        else:
            return f"{base_unit} S {t1_str} - {t2_str}"
            
    elif act_str == 'C':
        # e.g., F ENG C A LON - BRE
        return f"{base_unit} C {t1_str} - {t2_str}"
        
    return None

def encode_human_order(order_str, prov_to_idx):
    """
    Translates a human text order into a [Type, Target1, Target2] integer array.
    Example: "A PAR - BUR" -> [2, 14, 0]
    """
    # Default to 'NONE' tokens
    target_array = np.array([
        ACTION_TO_IDX['NONE'], 
        prov_to_idx['NONE'], 
        prov_to_idx['NONE']
    ], dtype=np.int64)
    
    parts = order_str.replace('*', '').split()
    if len(parts) < 3: 
        return target_array
        
    act_type = parts[2]
    t1 = 'NONE'
    t2 = 'NONE'
    
    if act_type in ['-', 'R']:
        t1 = parts[3]
    elif act_type == 'S':
        t1 = parts[4] # Skip the supported unit's 'A' or 'F'
        if len(parts) > 5 and parts[5] == '-':
            t2 = parts[6]
    elif act_type == 'C':
        t1 = parts[4]
        t2 = parts[6]
        
    target_array[0] = ACTION_TO_IDX.get(act_type, 0)
    target_array[1] = prov_to_idx.get(t1, 0)
    target_array[2] = prov_to_idx.get(t2, 0)
    
    return target_array

class DiplomacyEnv(ParallelEnv):
    metadata = {'render_modes': ['human'], "name": "diplomacy_v0"}

    def __init__(self):
        # 1. Define the agents
        self.possible_agents = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
        self.agents = self.possible_agents[:]
        
        self.game = Game()
        self.prov_to_idx, self.idx_to_prov = get_province_vocab(self.game)
        self.provinces = self.game.map.locs # The 81 base provinces
        self.num_provinces = len(self.provinces)
        
        # 2. Define the Compositional Action Space
        # Shape: (81 provinces, 3 tokens per province)
        # Token sizes: [8 (Action Types), 82 (Target 1), 82 (Target 2)]
        token_bounds = np.array([len(ACTION_TYPES), len(self.prov_to_idx), len(self.prov_to_idx)])
        action_shape = np.tile(token_bounds, (self.num_provinces, 1))
        
        self.action_spaces = {
            agent: MultiDiscrete(action_shape) 
            for agent in self.possible_agents
        }
        
        # 3. Define the Observation Space (Your 81x16 Tensor)
        self.observation_spaces = {
            agent: Box(low=0.0, high=1.0, shape=(self.num_provinces, 16), dtype=np.float32)
            for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        """
        Restarts the game to Spring 1901.
        Returns the initial observations and infos (containing action masks).
        """
        self.agents = self.possible_agents[:]
        self.game = Game()
        
        # 1. Generate the state tensor
        live_state_wrapper = {'state': self.game.get_state()}
        global_obs_tensor = parse_state_to_tensor(live_state_wrapper)
        observations = {agent: global_obs_tensor.copy() for agent in self.agents} 
        
        # 2. Generate the Action Masks
        infos = {agent: {} for agent in self.agents} 
        for agent in self.agents:
            infos[agent]['action_mask'] = get_compositional_action_mask(
                self.game, agent, self.provinces, self.prov_to_idx
            )
            
        return observations, infos

    def step(self, actions):
        """
        Receives a dictionary of actions from all agents.
        Format: {'FRANCE': [[Type, T1, T2], ... 81 times], 'ENGLAND': ...}
        """
        # 1. Clear out previous orders in the engine
        self.game.clear_orders()
        
        # 2. Map predicted integers back to strings and set them in the engine
        for agent, action_matrix in actions.items():
            text_orders = []
            
            # Iterate through the 81 rows (provinces)
            for prov_idx, action_array in enumerate(action_matrix):
                province_str = self.provinces[prov_idx]
                
                # Use the decoder function we built earlier
                order_str = decode_compositional_order(
                    province_str, action_array, self.game, 
                    IDX_TO_ACTION, self.idx_to_prov
                )
                
                if order_str is not None:
                    text_orders.append(order_str)
                    
            # Pass the list of text strings to the adjudicator
            self.game.set_orders(agent, text_orders)
            
        # 3. Resolve the phase
        self.game.process()
        
        # 4. Build the new observations
        live_state_wrapper = {'state': self.game.get_state()}
        global_obs_tensor = parse_state_to_tensor(live_state_wrapper)
        observations = {agent: global_obs_tensor.copy() for agent in self.agents}
        
        # 5. Calculate Rewards (e.g., Number of Supply Centers owned)
        rewards = {}
        for agent in self.agents:
            # Get the number of centers the agent currently owns
            sc_count = len(self.game.get_centers(agent))
            rewards[agent] = float(sc_count) 
            
        # 6. Check for Game Over (Someone reaches 18 SCs, or we hit a max phase count)
        is_done = False
        for agent in self.agents:
            if len(self.game.get_centers(agent)) >= 18:
                is_done = True
                
        terminations = {agent: is_done for agent in self.agents}
        truncations = {agent: False for agent in self.agents} # Used for turn limits
        
        # 7. Generate new action masks for the next turn
        infos = {agent: {} for agent in self.agents} 
        for agent in self.agents:
            infos[agent]['action_mask'] = get_compositional_action_mask(
                self.game, agent, self.provinces, self.prov_to_idx
            )
            
        # PettingZoo standard: remove eliminated agents
        # (In Diplomacy, an agent is usually considered eliminated if they have 0 SCs and 0 units)
        self.agents = [
            agent for agent in self.agents 
            if not terminations[agent] and 
            (len(self.game.get_centers(agent)) > 0 or len(self.game.get_state()['units'][agent]) > 0)
        ]
        
        return observations, rewards, terminations, truncations, infos
    

class DiplomacyDataset(Dataset):
    def __init__(self, json_data, game_engine, prov_to_idx):
        self.game = game_engine
        self.provinces = game_engine.map.locs
        self.prov_to_idx = prov_to_idx
        
        # Flatten the data: each sample is a (phase_data, power) pair
        self.samples = []
        for phase_data in json_data:
            for power, text_orders in phase_data.get('orders', {}).items():
                if text_orders: # Only include if they actually submitted orders
                    self.samples.append((phase_data, power, text_orders))
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        phase_data, power, text_orders = self.samples[idx]
        
        # 1. Generate State
        state_tensor = parse_state_to_tensor(phase_data)
        
        # 2. Reconstruct Game for Masks
        self.game.set_state(phase_data['state'])
        mask_dict = get_compositional_action_mask(self.game, power, self.provinces, self.prov_to_idx)
        
        # 3. Generate Targets
        target_matrix = np.zeros((len(self.provinces), 3), dtype=np.int64)
        for order_str in text_orders:
            parts = order_str.replace('*', '').split()
            if len(parts) >= 2:
                prov_str = parts[1]
                if prov_str in self.prov_to_idx:
                    p_idx = self.prov_to_idx[prov_str]
                    target_matrix[p_idx] = encode_human_order(order_str, self.prov_to_idx)
        
        return {
            'state': torch.tensor(state_tensor, dtype=torch.float32),
            'mask_type': torch.tensor(mask_dict['type'], dtype=torch.bool),
            'mask_t1': torch.tensor(mask_dict['target1'], dtype=torch.bool),
            'mask_t2': torch.tensor(mask_dict['target2'], dtype=torch.bool),
            'targets': torch.tensor(target_matrix, dtype=torch.long)
        }

class DiplomacyNet(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=256, target_vocab_size=83):
        super(DiplomacyNet, self).__init__()
        
        # This encoder processes the 16 features of EACH province 
        # into a rich 256-dimensional hidden state.
        self.province_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # The Three Output Heads
        # Action Type (8 possibilities)
        self.type_head = nn.Linear(hidden_dim, 8)
        
        # Target 1 (Dynamically sized, usually 83)
        self.t1_head = nn.Linear(hidden_dim, target_vocab_size)
        
        # Target 2 (Dynamically sized, usually 83)
        self.t2_head = nn.Linear(hidden_dim, target_vocab_size)

    def forward(self, x):
        # x shape: (Batch_Size, 82, 16)
        
        # Encode features
        hidden = self.province_encoder(x) 
        
        # Branch into the three token predictions
        type_logits = self.type_head(hidden) 
        t1_logits = self.t1_head(hidden)     
        t2_logits = self.t2_head(hidden)     
        
        return type_logits, t1_logits, t2_logits

if __name__ == "__main__":
    print("Initializing Engine and Vocab...")
    game_engine = Game()
    prov_to_idx, idx_to_prov = get_province_vocab(game_engine)
    
    # 1. Load the actual dataset
    dataset_filepath = "./datasets/standard_no_press.jsonl" 
    
    print(f"Loading data from {dataset_filepath}...")
    all_phases = []
    
    # Parse the JSONL file line-by-line
    with open(dataset_filepath, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                game_data = json.loads(line)
                
                # Check the top-level game map to ensure it's standard
                if game_data.get('map', 'standard') != 'standard':
                    continue
                
                # Append phases, double-checking the state map just to be safe
                for phase in game_data.get('phases', []):
                    state_map = phase.get('state', {}).get('map', 'standard')
                    if state_map == 'standard':
                        all_phases.append(phase)
                        
            except json.JSONDecodeError:
                continue # Skip corrupted lines
            
            # Limit the dataset size for testing
            if len(all_phases) > 2000:
                print("Limiting to 2000 phases for initial testing.")
                break
                
    print(f"Total standard phases loaded: {len(all_phases)}")

    print("\nBuilding PyTorch Dataset...")
    dataset = DiplomacyDataset(all_phases, game_engine, prov_to_idx)
    
    # Use batch_size > 1 to leverage vectorization and speed up training
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Initialize Network, Optimizer, and Loss Function
    target_vocab_length = len(prov_to_idx)
    net = DiplomacyNet(input_dim=16, hidden_dim=256, target_vocab_size=target_vocab_length)
    
    # Lowering the learning rate slightly for batched real-world data
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # ignore_index ensures that if a target doesn't map correctly, it doesn't break the loss
    criterion = nn.CrossEntropyLoss()
    
    print("\n--- STARTING TRAINING LOOP ON REAL DATA ---")
    epochs = 10 # Adjust as needed
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            states = batch['state']
            mask_type = batch['mask_type']
            mask_t1 = batch['mask_t1']
            mask_t2 = batch['mask_t2']
            targets = batch['targets'] 
            
            optimizer.zero_grad()
            
            type_logits, t1_logits, t2_logits = net(states)
            
            # Apply Masks 
            type_logits = type_logits.masked_fill(~mask_type, -1e9)
            t1_logits = t1_logits.masked_fill(~mask_t1, -1e9)
            t2_logits = t2_logits.masked_fill(~mask_t2, -1e9)
            
            # Flatten everything
            type_logits_flat = type_logits.view(-1, type_logits.size(-1))
            t1_logits_flat = t1_logits.view(-1, t1_logits.size(-1))
            t2_logits_flat = t2_logits.view(-1, t2_logits.size(-1))
            
            mask_type_flat = mask_type.view(-1, mask_type.size(-1))
            mask_t1_flat = mask_t1.view(-1, mask_t1.size(-1))
            mask_t2_flat = mask_t2.view(-1, mask_t2.size(-1))
            
            targets_flat = targets.view(-1, 3)
            
            # --- THE FIX: Identify which human targets are actually legal ---
            # We "gather" the boolean mask value at the exact index the human chose.
            is_legal_type = mask_type_flat.gather(1, targets_flat[:, 0:1]).squeeze()
            is_legal_t1 = mask_t1_flat.gather(1, targets_flat[:, 1:2]).squeeze()
            is_legal_t2 = mask_t2_flat.gather(1, targets_flat[:, 2:3]).squeeze()
            
            # A valid target is one that has a unit (not NONE) AND the human order is 100% legal
            valid_mask = (targets_flat[:, 0] != 0) & is_legal_type & is_legal_t1 & is_legal_t2
            
            # Filter the logits and targets to ONLY include valid, legal human moves
            valid_type_logits = type_logits_flat[valid_mask]
            valid_t1_logits = t1_logits_flat[valid_mask]
            valid_t2_logits = t2_logits_flat[valid_mask]
            valid_targets = targets_flat[valid_mask]
            
            # Calculate Loss ONLY on these valid units
            if valid_targets.size(0) > 0: 
                loss_type = criterion(valid_type_logits, valid_targets[:, 0])
                loss_t1 = criterion(valid_t1_logits, valid_targets[:, 1])
                loss_t2 = criterion(valid_t2_logits, valid_targets[:, 2])
                
                loss = loss_type + loss_t1 + loss_t2
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0 and batch_idx > 0:
                print(f"   Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"=== Epoch {epoch+1:02d}/{epochs} Completed | Avg Loss: {avg_loss:.4f} ===")
        
    # 3. Save the trained weights
    save_path = "diplomacy_bc_model.pth"
    torch.save(net.state_dict(), save_path)
    print(f"\nTraining complete! Model weights saved to '{save_path}'")