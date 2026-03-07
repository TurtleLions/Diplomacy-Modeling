import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from diplomacy import Game
import multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, MultiDiscrete

def get_adjacency_matrix(game):
    """
    Builds a normalized adjacency matrix using 'loc_abut' from 
    Diplomacy 1.1.2 engine.
    """
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
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    
    normalized_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    return torch.tensor(normalized_adj, dtype=torch.float32)

def parse_state_to_tensor(turn_data):
    """
    Parses a Diplomacy JSON state into a (81, 16) NumPy tensor.
    """
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
                u_type = parts[0]
                u_loc = parts[1] 
                
                if u_loc in prov_to_idx:
                    p_idx = prov_to_idx[u_loc]
                    state_tensor[p_idx, power_idx] = 1.0 
                    if u_type == 'A':
                        state_tensor[p_idx, 7] = 1.0 
                    elif u_type == 'F':
                        state_tensor[p_idx, 8] = 1.0 
                        
    for power, sc_list in centers.items():
        if power not in power_to_idx: continue
        power_idx = power_to_idx[power]
        
        for sc in sc_list:
            if sc in prov_to_idx:
                p_idx = prov_to_idx[sc]
                state_tensor[p_idx, 9 + power_idx] = 1.0 
                
    return state_tensor

ACTION_TYPES = ['NONE', 'H', '-', 'S', 'C', 'B', 'D', 'R']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}
IDX_TO_ACTION = {i: a for a, i in ACTION_TO_IDX.items()}

def get_province_vocab(game):
    """
    Returns the vocabulary for targets (Size: 82).
    """
    provinces = ['NONE'] + list(game.map.locs)
    prov_to_idx = {p: i for i, p in enumerate(provinces)}
    idx_to_prov = {i: p for p, i in prov_to_idx.items()}
    return prov_to_idx, idx_to_prov

def get_compositional_action_mask(game, power, provinces, prov_to_idx):
    """
    Parses the legal text orders into grammatical tokens and builds 
    independent binary masks.
    """
    num_provs = len(provinces)
    
    type_mask = np.zeros((num_provs, len(ACTION_TYPES)), dtype=np.int8)
    t1_mask = np.zeros((num_provs, len(prov_to_idx)), dtype=np.int8)
    t2_mask = np.zeros((num_provs, len(prov_to_idx)), dtype=np.int8)
    
    all_possible_orders = game.get_all_possible_orders()
    orderable_locs = game.get_orderable_locations(power)
    
    for i, prov in enumerate(provinces):
        if prov in orderable_locs:
            legal_orders = all_possible_orders.get(prov, [])
            for order in legal_orders:
                parts = order.split()
                
                if len(parts) >= 3:
                    act_type = parts[2]
                    t1 = 'NONE'
                    t2 = 'NONE'
                    
                    if act_type in ['-', 'R']:
                        t1 = parts[3]
                    elif act_type == 'S':
                        t1 = parts[4] 
                        if len(parts) > 5 and parts[5] == '-':
                            t2 = parts[6]
                    elif act_type == 'C':
                        t1 = parts[4]
                        t2 = parts[6]
                        
                    act_idx = ACTION_TO_IDX.get(act_type, 0)
                    t1_idx = prov_to_idx.get(t1, 0)
                    t2_idx = prov_to_idx.get(t2, 0)
                    
                    type_mask[i, act_idx] = 1
                    t1_mask[i, t1_idx] = 1
                    t2_mask[i, t2_idx] = 1
        else:
            type_mask[i, ACTION_TO_IDX['NONE']] = 1
            t1_mask[i, prov_to_idx['NONE']] = 1
            t2_mask[i, prov_to_idx['NONE']] = 1
            
    return {'type': type_mask, 'target1': t1_mask, 'target2': t2_mask}

def decode_compositional_order(province, action_array, game, idx_to_action, idx_to_prov):
    """
    Translates integer array back into text string.
    """
    act_idx, t1_idx, t2_idx = action_array
    
    act_str = idx_to_action[act_idx]
    t1_str = idx_to_prov[t1_idx]
    t2_str = idx_to_prov[t2_idx]
    
    if act_str == 'NONE':
        return None
        
    unit_type = "A" 
    owner = game.get_state()['units']
    for power, units in owner.items():
        for u in units:
            if province in u:
                unit_type = u[0] 
                break
                
    base_unit = f"{unit_type} {province}"
    
    if act_str in ['H', 'B', 'D']:
        return f"{base_unit} {act_str}"
    elif act_str in ['-', 'R']:
        return f"{base_unit} {act_str} {t1_str}"
    elif act_str == 'S':
        if t2_str == 'NONE':
            return f"{base_unit} S {t1_str}"
        else:
            return f"{base_unit} S {t1_str} - {t2_str}"
    elif act_str == 'C':
        return f"{base_unit} C {t1_str} - {t2_str}"
        
    return None

def encode_human_order(order_str, prov_to_idx):
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
        t1 = parts[4] 
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
        from gymnasium.spaces import Box, MultiDiscrete
        import functools
        
        self.possible_agents = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
        self.agents = self.possible_agents[:]
        
        self.game = Game()
        self.prov_to_idx, self.idx_to_prov = get_province_vocab(self.game)
        self.provinces = self.game.map.locs 
        self.num_provinces = len(self.provinces)
        
        token_bounds = np.array([len(ACTION_TYPES), len(self.prov_to_idx), len(self.prov_to_idx)])
        action_shape = np.tile(token_bounds, (self.num_provinces, 1))
        
        self.action_spaces = {
            agent: MultiDiscrete(action_shape) 
            for agent in self.possible_agents
        }
        
        self.observation_spaces = {
            agent: Box(low=0.0, high=1.0, shape=(self.num_provinces, 16), dtype=np.float32)
            for agent in self.possible_agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.game = Game()
        
        live_state_wrapper = {'state': self.game.get_state()}
        global_obs_tensor = parse_state_to_tensor(live_state_wrapper)
        observations = {agent: global_obs_tensor.copy() for agent in self.agents} 
        
        infos = {agent: {} for agent in self.agents} 
        for agent in self.agents:
            infos[agent]['action_mask'] = get_compositional_action_mask(
                self.game, agent, self.provinces, self.prov_to_idx
            )
            
        return observations, infos

    def step(self, actions):
        self.game.clear_orders()
        
        for agent, action_matrix in actions.items():
            text_orders = []
            for prov_idx, action_array in enumerate(action_matrix):
                province_str = self.provinces[prov_idx]
                order_str = decode_compositional_order(
                    province_str, action_array, self.game, 
                    IDX_TO_ACTION, self.idx_to_prov
                )
                if order_str is not None:
                    text_orders.append(order_str)
            self.game.set_orders(agent, text_orders)
            
        self.game.process()
        
        live_state_wrapper = {'state': self.game.get_state()}
        global_obs_tensor = parse_state_to_tensor(live_state_wrapper)
        observations = {agent: global_obs_tensor.copy() for agent in self.agents}
        
        rewards = {}
        for agent in self.agents:
            sc_count = len(self.game.get_centers(agent))
            rewards[agent] = float(sc_count) 
            
        is_done = False
        for agent in self.agents:
            if len(self.game.get_centers(agent)) >= 18:
                is_done = True
                
        terminations = {agent: is_done for agent in self.agents}
        truncations = {agent: False for agent in self.agents} 
        
        infos = {agent: {} for agent in self.agents} 
        for agent in self.agents:
            infos[agent]['action_mask'] = get_compositional_action_mask(
                self.game, agent, self.provinces, self.prov_to_idx
            )
            
        self.agents = [
            agent for agent in self.agents 
            if not terminations[agent] and 
            (len(self.game.get_centers(agent)) > 0 or len(self.game.get_state()['units'][agent]) > 0)
        ]
        
        return observations, rewards, terminations, truncations, infos
    
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        support = self.projection(x) 
        output = torch.matmul(adj, support) 
        return torch.relu(output)

class DiplomacyGCN(nn.Module):
    def __init__(self, adj, input_dim=16, hidden_dim=256, target_vocab_size=83):
        super(DiplomacyGCN, self).__init__()
        self.register_buffer('adj', adj)
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.type_head = nn.Linear(hidden_dim, 8)
        self.t1_head = nn.Linear(hidden_dim, target_vocab_size)
        self.t2_head = nn.Linear(hidden_dim, target_vocab_size)

    def forward(self, x):
        h = self.gcn1(x, self.adj)
        h = self.gcn2(h, self.adj)
        type_logits = self.type_head(h)
        t1_logits = self.t1_head(h)
        t2_logits = self.t2_head(h)
        return type_logits, t1_logits, t2_logits

# --- 1. THE ASYNC CPU WORKER ---
def worker(remote, parent_remote):
    parent_remote.close()
    env = DiplomacyEnv() 
    
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, rewards, terms, truncs, infos = env.step(data)
                remote.send((obs, rewards, terms, truncs, infos, env.agents))
            elif cmd == 'reset':
                obs, infos = env.reset()
                remote.send((obs, infos, env.agents))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_possible_agents':
                remote.send(env.possible_agents)
        except EOFError:
            break

# --- 2. THE VECTORIZER MANAGER ---
class SubprocVecDiplomacy:
    def __init__(self, num_envs=7):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.ps = [
            mp.Process(target=worker, args=(work_remote, remote))
            for (work_remote, remote) in zip(self.work_remotes, self.remotes)
        ]
        for p in self.ps:
            p.daemon = True 
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def step(self, actions_list):
        for remote, action_dict in zip(self.remotes, actions_list):
            remote.send(('step', action_dict))
        return [remote.recv() for remote in self.remotes]
        
    def get_possible_agents(self):
        self.remotes[0].send(('get_possible_agents', None))
        return self.remotes[0].recv()

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

# --- 3. THE DISTRIBUTED TRAINING LOOP ---
if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    device_id = local_rank % num_gpus
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    NUM_ENVS_PER_GPU = 7 

    if global_rank == 0:
        print(f"--- STARTING VECTORIZED DISTRIBUTED RL ---")
        print(f"GPUs active: {num_gpus}. Parallel envs per GPU: {NUM_ENVS_PER_GPU}.")

    vec_env = SubprocVecDiplomacy(num_envs=NUM_ENVS_PER_GPU)
    possible_agents = vec_env.get_possible_agents()
    
    dummy_env = DiplomacyEnv()
    adj_matrix = get_adjacency_matrix(dummy_env.game).to(device)
    target_vocab_length = len(dummy_env.prov_to_idx)
    del dummy_env 
    
    net = DiplomacyGCN(
        adj=adj_matrix, 
        input_dim=16, 
        hidden_dim=256, 
        target_vocab_size=target_vocab_length
    ).to(device)
    
    net = DDP(net, device_ids=[device_id])
    
    model_path = "diplomacy_gcn_bc_2_5644.pth"
    if os.path.exists(model_path):
        net.module.load_state_dict(torch.load(model_path, map_location=device))
        if global_rank == 0:
            print(f"Successfully loaded BC weights.")
    
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    
    # --- TRAINING HYPERPARAMETERS ---
    num_epochs = 100            # Total number of epochs
    accum_steps = 4             # Episodes per epoch (Gradient Accumulation Steps)
    entropy_coef = 0.05 
    gamma = 0.99 

    for epoch in range(num_epochs):
        if global_rank == 0:
            print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
            
        epoch_loss = 0.0
        optimizer.zero_grad() # Zero gradients at the start of the epoch
        
        for accum_step in range(accum_steps):
            env_results = vec_env.reset()
            
            ep_data = {
                e: {
                    a: {'obs': [], 'm_type': [], 'm_t1': [], 'm_t2': [], 
                        'act_type': [], 'act_t1': [], 'act_t2': [], 'rewards': []} 
                    for a in possible_agents
                } for e in range(NUM_ENVS_PER_GPU)
            }
            
            step_count = 0
            
            # --- GAME ROLLOUT PHASE (NO GRADIENTS) ---
            with torch.no_grad():
                while step_count < 100: 
                    actions_to_send = [{} for _ in range(NUM_ENVS_PER_GPU)]
                    step_took_action = {e: {a: False for a in possible_agents} for e in range(NUM_ENVS_PER_GPU)}
                    
                    for agent in possible_agents:
                        active_env_indices = []
                        obs_list = []
                        mask_type_list, mask_t1_list, mask_t2_list = [], [], []
                        
                        for i in range(NUM_ENVS_PER_GPU):
                            if len(env_results[i]) == 3:
                                obs_dict, infos_dict, active_agents = env_results[i]
                            else:
                                obs_dict, _, _, _, infos_dict, active_agents = env_results[i]
                                
                            if agent in active_agents:
                                active_env_indices.append(i)
                                obs_list.append(torch.tensor(obs_dict[agent]))
                                
                                mask_dict = infos_dict[agent]['action_mask']
                                mask_type_list.append(torch.tensor(mask_dict['type'], dtype=torch.bool))
                                mask_t1_list.append(torch.tensor(mask_dict['target1'], dtype=torch.bool))
                                mask_t2_list.append(torch.tensor(mask_dict['target2'], dtype=torch.bool))

                        if not active_env_indices:
                            continue 
                        
                        batch_obs = torch.stack(obs_list).to(device) 
                        batch_mask_type = torch.stack(mask_type_list).to(device)
                        batch_mask_t1 = torch.stack(mask_t1_list).to(device)
                        batch_mask_t2 = torch.stack(mask_t2_list).to(device)

                        type_logits, t1_logits, t2_logits = net(batch_obs)
                        
                        type_logits = type_logits.masked_fill(~batch_mask_type, -1e9)
                        t1_logits = t1_logits.masked_fill(~batch_mask_t1, -1e9)
                        t2_logits = t2_logits.masked_fill(~batch_mask_t2, -1e9)
                        
                        type_dist = Categorical(logits=type_logits)
                        t1_dist = Categorical(logits=t1_logits)
                        t2_dist = Categorical(logits=t2_logits)
                        
                        type_action = type_dist.sample()
                        t1_action = t1_dist.sample()
                        t2_action = t2_dist.sample()
                        
                        for idx, env_idx in enumerate(active_env_indices):
                            active_unit_mask = (type_action[idx] != 0)
                            
                            if active_unit_mask.any():
                                ep_data[env_idx][agent]['obs'].append(obs_list[idx].clone())
                                ep_data[env_idx][agent]['m_type'].append(mask_type_list[idx].clone())
                                ep_data[env_idx][agent]['m_t1'].append(mask_t1_list[idx].clone())
                                ep_data[env_idx][agent]['m_t2'].append(mask_t2_list[idx].clone())
                                
                                ep_data[env_idx][agent]['act_type'].append(type_action[idx].cpu().clone())
                                ep_data[env_idx][agent]['act_t1'].append(t1_action[idx].cpu().clone())
                                ep_data[env_idx][agent]['act_t2'].append(t2_action[idx].cpu().clone())
                                
                                step_took_action[env_idx][agent] = True
                            
                            actions_to_send[env_idx][agent] = torch.stack([
                                type_action[idx], 
                                t1_action[idx], 
                                t2_action[idx]
                            ], dim=-1).cpu().numpy()

                    next_env_results = vec_env.step(actions_to_send)
                    
                    for i in range(NUM_ENVS_PER_GPU):
                        obs, rewards, terms, truncs, infos, active_agents = next_env_results[i]
                        for agent in possible_agents:
                            if step_took_action[i][agent]:
                                ep_data[i][agent]['rewards'].append(float(rewards.get(agent, 0.0)))
                    
                    env_results = next_env_results
                    
                    all_done = all(len(result[5]) == 0 for result in env_results)
                    if all_done:
                        break
                        
                    step_count += 1
                    
            # --- SYNCHRONIZED UPDATE PHASE (GRADIENTS ENABLED) ---
            all_obs, all_m_type, all_m_t1, all_m_t2 = [], [], [], []
            all_act_type, all_act_t1, all_act_t2 = [], [], []
            all_returns = []
            
            for i in range(NUM_ENVS_PER_GPU):
                for agent in possible_agents:
                    data = ep_data[i][agent]
                    if len(data['rewards']) == 0: continue
                    
                    returns, R = [], 0
                    for r in reversed(data['rewards']):
                        R = r + gamma * R
                        returns.insert(0, R)
                    returns = torch.tensor(returns, dtype=torch.float32).to(device)
                    
                    if returns.std() > 0:
                        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                    
                    all_returns.append(returns)
                    all_obs.extend(data['obs'])
                    all_m_type.extend(data['m_type'])
                    all_m_t1.extend(data['m_t1'])
                    all_m_t2.extend(data['m_t2'])
                    all_act_type.extend(data['act_type'])
                    all_act_t1.extend(data['act_t1'])
                    all_act_t2.extend(data['act_t2'])
                    
            if all_obs:
                b_obs = torch.stack(all_obs).to(device)
                b_m_type = torch.stack(all_m_type).to(device)
                b_m_t1 = torch.stack(all_m_t1).to(device)
                b_m_t2 = torch.stack(all_m_t2).to(device)
                b_act_type = torch.stack(all_act_type).to(device)
                b_act_t1 = torch.stack(all_act_t1).to(device)
                b_act_t2 = torch.stack(all_act_t2).to(device)
                b_returns = torch.cat(all_returns).to(device)
                
                # Context manager for DDP to only network-sync on the final accumulation step
                is_last_accum_step = (accum_step == accum_steps - 1)
                sync_context = net.no_sync() if not is_last_accum_step else nullcontext()
                
                with sync_context:
                    type_logits, t1_logits, t2_logits = net(b_obs)
                    
                    type_logits = type_logits.masked_fill(~b_m_type, -1e9)
                    t1_logits = t1_logits.masked_fill(~b_m_t1, -1e9)
                    t2_logits = t2_logits.masked_fill(~b_m_t2, -1e9)
                    
                    type_dist = Categorical(logits=type_logits)
                    t1_dist = Categorical(logits=t1_logits)
                    t2_dist = Categorical(logits=t2_logits)
                    
                    active_unit_mask = (b_act_type != 0)
                    
                    log_p = type_dist.log_prob(b_act_type) + t1_dist.log_prob(b_act_t1) + t2_dist.log_prob(b_act_t2)
                    log_p = (log_p * active_unit_mask).sum(dim=1)
                    
                    entropy = type_dist.entropy() + t1_dist.entropy() + t2_dist.entropy()
                    entropy = (entropy * active_unit_mask).sum(dim=1)
                    
                    # Scale loss by accum_steps so the sum equals the true mean
                    loss = (-(log_p * b_returns).mean() - (entropy_coef * entropy.mean())) / accum_steps
                    
                    loss.backward()
                    epoch_loss += loss.item() # Keep track of total scaled loss

        # --- APPLY ACCUMULATED GRADIENTS AT END OF EPOCH ---
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()
        
        if global_rank == 0:
            print(f"  Master Node - Avg Epoch Loss: {epoch_loss:.4f}")
            checkpoint_path = f"diplomacy_rl_model_epoch_{epoch+1}.pth"
            torch.save(net.module.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    if global_rank == 0:
        print("\nTraining complete!")

    vec_env.close()
    dist.destroy_process_group()