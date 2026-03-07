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
from torch.distributions import Categorical

def get_adjacency_matrix(game):
    """
    Builds a normalized adjacency matrix using 'loc_abut' from 
    Diplomacy 1.1.2 engine.
    """
    # Use the 81 base provinces from your state tensor
    provinces = game.map.locs 
    prov_to_idx = {prov: i for i, prov in enumerate(provinces)}
    num_provs = len(provinces)
    
    adj = np.zeros((num_provs, num_provs), dtype=np.float32)

    # loc_abut: {'LVP': ['CLY', 'edi', 'IRI', ...], ...}
    for loc, neighbors in game.map.loc_abut.items():
        # Standardize source: "STP/NC" -> "STP"
        u_name = loc.split('/')[0].upper()
        if u_name in prov_to_idx:
            u_idx = prov_to_idx[u_name]
            
            for neighbor in neighbors:
                # Standardize neighbor: "yor" or "Bal" -> "YOR" or "BAL"
                # Then strip coastal identifier
                v_name = neighbor.split('/')[0].upper()
                
                if v_name in prov_to_idx:
                    v_idx = prov_to_idx[v_name]
                    adj[u_idx, v_idx] = 1.0

    # 1. Add Self-Loops (Nodes need to see their own features)
    adj += np.eye(num_provs)
    
    # 2. Symmetric Normalization: D^-1/2 * A * D^-1/2
    # This prevents nodes with high connectivity from drowning out others
    row_sum = adj.sum(1)
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    
    normalized_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    return torch.tensor(normalized_adj, dtype=torch.float32)

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

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: (Batch, 81, In_Features)
        # adj: (81, 81)
        
        # Projection: (Batch, 81, Out_Features)
        support = self.projection(x) 
        
        # Neighborhood Aggregation: (Batch, 81, Out_Features)
        # adj is broadcasted across the Batch dimension
        output = torch.matmul(adj, support) 
        return torch.relu(output)

class DiplomacyGCN(nn.Module):
    def __init__(self, adj, input_dim=16, hidden_dim=256, target_vocab_size=83):
        super(DiplomacyGCN, self).__init__()
        
        # register_buffer ensures the adjacency matrix moves to GPU automatically with the model
        self.register_buffer('adj', adj)
        
        # Encoder: Information flows across the map
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        
        # Output Heads (Predicting actions for each of the 81 provinces)
        self.type_head = nn.Linear(hidden_dim, 8)
        self.t1_head = nn.Linear(hidden_dim, target_vocab_size)
        self.t2_head = nn.Linear(hidden_dim, target_vocab_size)

    def forward(self, x):
        # x shape: (Batch, 81, 16)
        
        # Message passing: Provinces 'talk' to their neighbors
        h = self.gcn1(x, self.adj)
        h = self.gcn2(h, self.adj)
        
        # Decoders
        type_logits = self.type_head(h)
        t1_logits = self.t1_head(h)
        t2_logits = self.t2_head(h)
        
        return type_logits, t1_logits, t2_logits

if __name__ == "__main__":
    env = DiplomacyEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    adj_matrix = get_adjacency_matrix(env.game).to(device)
    target_vocab_length = len(env.prov_to_idx)
    net = DiplomacyGCN(
        adj=adj_matrix, 
        input_dim=16, 
        hidden_dim=256, 
        target_vocab_size=target_vocab_length
    ).to(device)
    
    # Load BC weights
    model_path = "diplomacy_gcn_bc_2_5644.pth"
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded pre-trained BC weights from {model_path}")
    else:
        print("No BC weights found. Starting from scratch.")
        
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    
    num_episodes = 10000 
    entropy_coef = 0.05 
    gamma = 0.99 
    
    print("\n--- STARTING UPGRADED RL SELF-PLAY ---")
    
    for episode in range(num_episodes):
        obs, infos = env.reset()
        print(f"\n[Episode {episode+1}/{num_episodes}] Game Started!")

        episode_log_probs = {agent: [] for agent in env.possible_agents}
        episode_rewards = {agent: [] for agent in env.possible_agents}
        episode_entropies = {agent: [] for agent in env.possible_agents}
        
        prev_sc_counts = {agent: float(len(env.game.get_centers(agent))) for agent in env.possible_agents}
        
        step_count = 0
        while env.agents: 
            current_phase = env.game.phase
            
            actions = {}
            turn_log_probs = {}
            turn_entropies = {}
            
            # 1. Collect actions from all live agents
            for agent in env.agents:
                state_tensor = torch.tensor(obs[agent]).unsqueeze(0).to(device)
                
                mask_dict = infos[agent]['action_mask']
                mask_type = torch.tensor(mask_dict['type'], dtype=torch.bool).unsqueeze(0).to(device)
                mask_t1 = torch.tensor(mask_dict['target1'], dtype=torch.bool).unsqueeze(0).to(device)
                mask_t2 = torch.tensor(mask_dict['target2'], dtype=torch.bool).unsqueeze(0).to(device)
                
                type_logits, t1_logits, t2_logits = net(state_tensor)
                
                # Apply Masks
                type_logits = type_logits.masked_fill(~mask_type, -1e9)
                t1_logits = t1_logits.masked_fill(~mask_t1, -1e9)
                t2_logits = t2_logits.masked_fill(~mask_t2, -1e9)
                
                type_dist = Categorical(logits=type_logits)
                t1_dist = Categorical(logits=t1_logits)
                t2_dist = Categorical(logits=t2_logits)
                
                type_action = type_dist.sample()
                t1_action = t1_dist.sample()
                t2_action = t2_dist.sample()
                
                active_unit_mask = (type_action != 0)
                
                if active_unit_mask.any():
                    joint_log_prob = (
                        type_dist.log_prob(type_action)[active_unit_mask] + 
                        t1_dist.log_prob(t1_action)[active_unit_mask] + 
                        t2_dist.log_prob(t2_action)[active_unit_mask]
                    )
                    turn_log_probs[agent] = joint_log_prob.sum()
                    
                    joint_entropy = (
                        type_dist.entropy()[active_unit_mask] + 
                        t1_dist.entropy()[active_unit_mask] + 
                        t2_dist.entropy()[active_unit_mask]
                    )
                    turn_entropies[agent] = joint_entropy.sum()
                
                actions[agent] = torch.stack([
                    type_action.squeeze(0), 
                    t1_action.squeeze(0), 
                    t2_action.squeeze(0)
                ], dim=-1).cpu().numpy()

            # --- PRE-STEP DATA FOR REWARDS ---
            all_supply_centers = env.game.map.scs
            prev_sc_owners = {sc: p for p in env.possible_agents for sc in env.game.get_centers(p)}
            neutral_scs_before = [sc for sc in all_supply_centers if sc not in prev_sc_owners]

            # 2. Step the environment
            next_obs, rewards, terminations, truncations, next_infos = env.step(actions)
            
            # 3. Calculate Shaped Rewards
            for agent in env.possible_agents:
                if agent not in turn_log_probs: continue
                
                current_sc_list = env.game.get_centers(agent)
                current_sc_count = len(current_sc_list)
                
                # Maintenance reward
                step_reward = current_sc_count * 0.1 

                # Capture/Loss rewards
                for sc in current_sc_list:
                    if sc not in prev_sc_owners: step_reward += 1.0 # Neutral
                    elif prev_sc_owners[sc] != agent: step_reward += 1.5 # Steal

                for sc, owner in prev_sc_owners.items():
                    if owner == agent and sc not in current_sc_list:
                        step_reward -= 2.0 # Lost SC

                # Retreat penalty
                agent_units = env.game.get_state()['units'].get(agent, [])
                dislodged_count = sum(1 for u in agent_units if '*' in u)
                step_reward -= (dislodged_count * 0.5)

                episode_log_probs[agent].append(turn_log_probs[agent])
                episode_entropies[agent].append(turn_entropies[agent])
                episode_rewards[agent].append(step_reward)
                prev_sc_counts[agent] = current_sc_count

            obs, infos = next_obs, next_infos
            step_count += 1
            if step_count > 100: break
                
        # --- UPDATE PHASE ---
        print(f"\n  Episode Finished. Steps: {step_count} | Adjudicating...")
        policy_loss = []
        for agent in env.possible_agents:
            if not episode_log_probs[agent]: continue
            
            returns, R = [], 0
            for r in reversed(episode_rewards[agent]):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns).to(device)
            
            if returns.std() > 0:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            agent_log_probs = torch.stack(episode_log_probs[agent])
            agent_entropies = torch.stack(episode_entropies[agent])
            
            loss = -(agent_log_probs * returns).mean() - (entropy_coef * agent_entropies.mean())
            policy_loss.append(loss)
            
        if policy_loss:
            optimizer.zero_grad()
            total_loss = torch.stack(policy_loss).sum()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            print(f"  Total Policy Loss: {total_loss.item():.4f}")
        
        print(f"  Final SCs: { {k: int(v) for k, v in prev_sc_counts.items()} }")

    torch.save(net.state_dict(), "diplomacy_rl_model.pth")
    print("\nRL Training complete! Model saved.")