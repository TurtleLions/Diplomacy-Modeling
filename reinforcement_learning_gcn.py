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

# --- 1. UTILITIES & ENVIRONMENT (Unchanged Logic, Optimized for Speed) ---
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
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    normalized_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    return torch.tensor(normalized_adj, dtype=torch.float32)

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
                u_type, u_loc = parts[0], parts[1] 
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

ACTION_TYPES = ['NONE', 'H', '-', 'S', 'C', 'B', 'D', 'R']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}
IDX_TO_ACTION = {i: a for a, i in ACTION_TO_IDX.items()}

def get_province_vocab(game):
    provinces = ['NONE'] + list(game.map.locs)
    prov_to_idx = {p: i for i, p in enumerate(provinces)}
    return prov_to_idx, {i: p for p, i in prov_to_idx.items()}

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
                    act_type = parts[2]
                    t1, t2 = 'NONE', 'NONE'
                    if act_type in ['-', 'R']: t1 = parts[3]
                    elif act_type == 'S':
                        t1 = parts[4] 
                        if len(parts) > 5 and parts[5] == '-': t2 = parts[6]
                    elif act_type == 'C':
                        t1, t2 = parts[4], parts[6]
                        
                    type_mask[i, ACTION_TO_IDX.get(act_type, 0)] = 1
                    t1_mask[i, prov_to_idx.get(t1, 0)] = 1
                    t2_mask[i, prov_to_idx.get(t2, 0)] = 1
        else:
            type_mask[i, ACTION_TO_IDX['NONE']] = 1
            t1_mask[i, prov_to_idx['NONE']] = 1
            t2_mask[i, prov_to_idx['NONE']] = 1
            
    return {'type': type_mask, 'target1': t1_mask, 'target2': t2_mask}

def decode_compositional_order(province, action_array, game, idx_to_action, idx_to_prov):
    act_str = idx_to_action[action_array[0]]
    t1_str, t2_str = idx_to_prov[action_array[1]], idx_to_prov[action_array[2]]
    
    if act_str == 'NONE': return None
    unit_type = "A" 
    for power, units in game.get_state()['units'].items():
        for u in units:
            if province in u:
                unit_type = u[0] 
                break
                
    base_unit = f"{unit_type} {province}"
    if act_str in ['H', 'B', 'D']: return f"{base_unit} {act_str}"
    elif act_str in ['-', 'R']: return f"{base_unit} {act_str} {t1_str}"
    elif act_str == 'S':
        return f"{base_unit} S {t1_str}" if t2_str == 'NONE' else f"{base_unit} S {t1_str} - {t2_str}"
    elif act_str == 'C': return f"{base_unit} C {t1_str} - {t2_str}"
    return None

class DiplomacyEnv(ParallelEnv):
    metadata = {'render_modes': ['human'], "name": "diplomacy_v0"}

    def __init__(self):
        self.possible_agents = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
        self.game = Game()
        self.prov_to_idx, self.idx_to_prov = get_province_vocab(self.game)
        self.provinces = list(self.game.map.locs)
        self.num_provinces = len(self.provinces)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.game = Game()
        obs_tensor = parse_state_to_tensor({'state': self.game.get_state()})
        observations = {a: obs_tensor.copy() for a in self.agents} 
        infos = {a: {'action_mask': get_compositional_action_mask(self.game, a, self.provinces, self.prov_to_idx)} for a in self.agents}
        return observations, infos

    def step(self, actions):
        self.game.clear_orders()
        prev_sc_owners = {sc: a for a in self.possible_agents for sc in self.game.get_centers(a)}
        
        for agent, action_matrix in actions.items():
            text_orders = [
                decode_compositional_order(self.provinces[i], arr, self.game, IDX_TO_ACTION, self.idx_to_prov)
                for i, arr in enumerate(action_matrix)
            ]
            self.game.set_orders(agent, [o for o in text_orders if o])
            
        self.game.process()
        obs_tensor = parse_state_to_tensor({'state': self.game.get_state()})
        observations = {a: obs_tensor.copy() for a in self.agents}
        
        rewards = {a: 0.0 for a in self.agents}
        is_done = False
        
        for agent in self.agents:
            current_scs = self.game.get_centers(agent)
            agent_units = self.game.get_state()['units'].get(agent, [])
            
            rewards[agent] += len(current_scs) * 0.05
            for sc in current_scs:
                if sc not in prev_sc_owners: rewards[agent] += 1.0
                elif prev_sc_owners[sc] != agent: rewards[agent] += 2.0
            for sc, owner in prev_sc_owners.items():
                if owner == agent and sc not in current_scs: rewards[agent] -= 2.0
                    
            rewards[agent] -= (sum(1 for u in agent_units if '*' in u) * 0.5) 
            
            if len(current_scs) >= 18:
                rewards[agent] += 100.0
                is_done = True
            if len(current_scs) == 0 and len(agent_units) == 0:
                rewards[agent] -= 50.0

        terminations = {a: is_done for a in self.agents}
        infos = {a: {'action_mask': get_compositional_action_mask(self.game, a, self.provinces, self.prov_to_idx)} for a in self.agents}
        self.agents = [a for a in self.agents if not terminations[a] and (len(self.game.get_centers(a)) > 0 or len(self.game.get_state()['units'].get(a, [])) > 0)]
        return observations, rewards, terminations, {a: False for a in self.agents}, infos

# --- 2. DEEPER GCN WITH RESIDUALS & VALUE HEAD ---
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        out = torch.relu(torch.matmul(adj, self.projection(x)))
        # Residual connection if dimensions match
        if x.shape[-1] == out.shape[-1]:
            return x + out
        return out

class DiplomacyActorCritic(nn.Module):
    def __init__(self, adj, input_dim=16, hidden_dim=256, target_vocab_size=83):
        super().__init__()
        self.register_buffer('adj', adj)
        
        # Increased depth for better global propagation
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn4 = GCNLayer(hidden_dim, hidden_dim)
        
        # Actor Heads
        self.type_head = nn.Linear(hidden_dim, 8)
        self.t1_head = nn.Linear(hidden_dim, target_vocab_size)
        self.t2_head = nn.Linear(hidden_dim, target_vocab_size)
        
        # Critic (Value) Head - Outputs 1 value per province, pooled for state value
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.gcn1(x, self.adj)
        h = self.gcn2(h, self.adj)
        h = self.gcn3(h, self.adj)
        h = self.gcn4(h, self.adj)
        
        type_logits = self.type_head(h)
        t1_logits = self.t1_head(h)
        t2_logits = self.t2_head(h)
        
        # Mean pool the province values to get a single state value
        state_value = self.value_head(h).mean(dim=-2) 
        return type_logits, t1_logits, t2_logits, state_value

# --- 3. VECTORIZER ---
def worker(remote, parent_remote):
    parent_remote.close()
    env = DiplomacyEnv() 
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step': remote.send((*env.step(data), env.agents))
            elif cmd == 'reset': remote.send((*env.reset(), env.agents))
            elif cmd == 'close':
                remote.close()
                break
        except EOFError: break

class SubprocVecDiplomacy:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote)) for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        for p in self.ps:
            p.daemon = True 
            p.start()
        for remote in self.work_remotes: remote.close()

    def reset(self):
        for remote in self.remotes: remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def step(self, actions_list):
        for remote, action_dict in zip(self.remotes, actions_list): remote.send(('step', action_dict))
        return [remote.recv() for remote in self.remotes]
        
    def close(self):
        for remote in self.remotes: remote.send(('close', None))
        for p in self.ps: p.join()

# --- 4. HIGH-PERFORMANCE PPO TRAINING LOOP ---
if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    NUM_ENVS = 14 # Scale this up to feed the GPU
    NUM_STEPS = 100
    NUM_AGENTS = 7

    if global_rank == 0:
        print(f"--- STARTING PPO DISTRIBUTED RL ---")
        print(f"Envs per GPU: {NUM_ENVS} | Steps per rollout: {NUM_STEPS}")

    vec_env = SubprocVecDiplomacy(num_envs=NUM_ENVS)
    dummy_env = DiplomacyEnv()
    possible_agents = dummy_env.possible_agents
    adj_matrix = get_adjacency_matrix(dummy_env.game).to(device)
    MAP_PROVINCES = len(dummy_env.provinces)
    VOCAB_SIZE = len(dummy_env.prov_to_idx)
    del dummy_env 

    net = DiplomacyActorCritic(adj=adj_matrix, target_vocab_size=PROVINCES).to(device)
    net = DDP(net, device_ids=[local_rank])
    optimizer = optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)

    # Pre-allocate Tensors (Zero memory fragmentation)
    b_obs = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS, MAP_PROVINCES, 16), dtype=torch.float32, device=device)
    b_actions = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS, MAP_PROVINCES, 3), dtype=torch.long, device=device)
    b_logprobs = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS), dtype=torch.float32, device=device)
    b_rewards = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS), dtype=torch.float32, device=device)
    b_dones = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS), dtype=torch.float32, device=device)
    b_values = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS), dtype=torch.float32, device=device)
    b_masks = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS), dtype=torch.bool, device=device) 
    
    # Mask buffers (Targets use VOCAB_SIZE, types use 8)
    b_m_type = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS, MAP_PROVINCES, 8), dtype=torch.bool, device=device)
    b_m_t1 = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS, MAP_PROVINCES, VOCAB_SIZE), dtype=torch.bool, device=device)
    b_m_t2 = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS, MAP_PROVINCES, VOCAB_SIZE), dtype=torch.bool, device=device)

    # PPO Hyperparams
    num_updates = 1000
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef_start = 0.05
    ent_coef_end = 0.001
    v_coef = 0.5
    update_epochs = 4
    
    agent_to_idx = {a: i for i, a in enumerate(possible_agents)}

    next_env_results = vec_env.reset()
    next_done = torch.zeros((NUM_ENVS, NUM_AGENTS), device=device)

    for update in range(1, num_updates + 1):
        # Entropy schedule
        frac = 1.0 - (update - 1.0) / num_updates
        ent_coef = frac * ent_coef_start + (1 - frac) * ent_coef_end

        # --- ROLLOUT PHASE ---
        net.eval()
        for step in range(NUM_STEPS):
            b_dones[step] = next_done
            actions_to_send = [{} for _ in range(NUM_ENVS)]
            
            with torch.no_grad():
                for i in range(NUM_ENVS):
                    obs_dict, infos_dict, active_agents = next_env_results[i] if len(next_env_results[i]) == 3 else (*next_env_results[i][0:2], next_env_results[i][-2], next_env_results[i][-1])[0:3] # handle reset vs step outputs
                    if len(next_env_results[i]) == 6: # Step output
                        obs_dict, step_rewards, terms, _, infos_dict, active_agents = next_env_results[i]
                        for a in possible_agents:
                            b_rewards[step-1, i, agent_to_idx[a]] = step_rewards.get(a, 0.0)
                            next_done[i, agent_to_idx[a]] = float(terms.get(a, False))
                    
                    for a in active_agents:
                        a_idx = agent_to_idx[a]
                        b_masks[step, i, a_idx] = True
                        b_obs[step, i, a_idx] = torch.tensor(obs_dict[a], device=device)
                        
                        m_dict = infos_dict[a]['action_mask']
                        b_m_type[step, i, a_idx] = torch.tensor(m_dict['type'], device=device)
                        b_m_t1[step, i, a_idx] = torch.tensor(m_dict['target1'], device=device)
                        b_m_t2[step, i, a_idx] = torch.tensor(m_dict['target2'], device=device)

                # Batched Inference over active agents
                flat_obs = b_obs[step][b_masks[step]]
                if flat_obs.shape[0] > 0:
                    type_l, t1_l, t2_l, values = net(flat_obs)
                    
                    type_l = type_l.masked_fill(~b_m_type[step][b_masks[step]], -1e9)
                    t1_l = t1_l.masked_fill(~b_m_t1[step][b_masks[step]], -1e9)
                    t2_l = t2_l.masked_fill(~b_m_t2[step][b_masks[step]], -1e9)
                    
                    type_dist, t1_dist, t2_dist = Categorical(logits=type_l), Categorical(logits=t1_l), Categorical(logits=t2_l)
                    a_type, a_t1, a_t2 = type_dist.sample(), t1_dist.sample(), t2_dist.sample()
                    
                    active_unit_mask = (a_type != 0)
                    log_p = type_dist.log_prob(a_type) + t1_dist.log_prob(a_t1) + t2_dist.log_prob(a_t2)
                    
                    # Distribute back to buffers
                    b_values[step][b_masks[step]] = values.squeeze()
                    b_logprobs[step][b_masks[step]] = (log_p * active_unit_mask).sum(dim=1)
                    
                    idx_counter = 0
                    for i in range(NUM_ENVS):
                        for a in possible_agents:
                            if b_masks[step, i, agent_to_idx[a]]:
                                act_matrix = torch.stack([a_type[idx_counter], a_t1[idx_counter], a_t2[idx_counter]], dim=-1)
                                b_actions[step, i, agent_to_idx[a]] = act_matrix
                                actions_to_send[i][a] = act_matrix.cpu().numpy()
                                idx_counter += 1

            next_env_results = vec_env.step(actions_to_send)
            
        # Get next value for GAE
        with torch.no_grad():
            next_value = torch.zeros((NUM_ENVS, NUM_AGENTS), device=device)
            # Simplification: estimate 0 for next value if done, else forward pass (omitted for brevity, assume 0 for terminal)
            
        # --- GAE CALCULATION ---
        advantages = torch.zeros_like(b_rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(NUM_STEPS)):
            if t == NUM_STEPS - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - b_dones[t + 1]
                nextvalues = b_values[t + 1]
            delta = b_rewards[t] + gamma * nextvalues * nextnonterminal - b_values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + b_values

        # Flatten buffers (Only keep valid agent steps)
        valid = b_masks.view(-1)
        flat_obs = b_obs.view(-1, MAP_PROVINCES, 16)[valid]
        flat_act = b_actions.view(-1, MAP_PROVINCES, 3)[valid]
        flat_logprobs = b_logprobs.view(-1)[valid]
        flat_adv = advantages.view(-1)[valid]
        flat_ret = returns.view(-1)[valid]
        flat_val = b_values.view(-1)[valid]
        
        flat_m_type = b_m_type.view(-1, MAP_PROVINCES, 8)[valid]
        flat_m_t1 = b_m_t1.view(-1, MAP_PROVINCES, VOCAB_SIZE)[valid]
        flat_m_t2 = b_m_t2.view(-1, MAP_PROVINCES, VOCAB_SIZE)[valid]

        # Global Batch Normalization (The Fix)
        if flat_adv.shape[0] > 1:
            flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        # --- UPDATE PHASE ---
        net.train()
        for epoch in range(update_epochs):
            # In a full setup, you'd mini-batch this. For simplicity, full-batch update here.
            type_l, t1_l, t2_l, new_val = net(flat_obs)
            
            type_l = type_l.masked_fill(~flat_m_type, -1e9)
            t1_l = t1_l.masked_fill(~flat_m_t1, -1e9)
            t2_l = t2_l.masked_fill(~flat_m_t2, -1e9)
            
            type_dist, t1_dist, t2_dist = Categorical(logits=type_l), Categorical(logits=t1_l), Categorical(logits=t2_l)
            
            active_unit_mask = (flat_act[..., 0] != 0)
            new_logp = type_dist.log_prob(flat_act[..., 0]) + t1_dist.log_prob(flat_act[..., 1]) + t2_dist.log_prob(flat_act[..., 2])
            new_logp = (new_logp * active_unit_mask).sum(dim=1)
            
            entropy = type_dist.entropy() + t1_dist.entropy() + t2_dist.entropy()
            entropy = (entropy * active_unit_mask).sum(dim=1).mean()

            logratio = new_logp - flat_logprobs
            ratio = logratio.exp()
            
            pg_loss1 = -flat_adv * ratio
            pg_loss2 = -flat_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            v_loss = 0.5 * ((new_val.squeeze() - flat_ret) ** 2).mean()
            
            loss = pg_loss - ent_coef * entropy + v_loss * v_coef
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

        # Cleanup buffers for next update
        b_masks.zero_()
        b_rewards.zero_()

        if global_rank == 0:
            avg_reward = b_rewards.sum() / (NUM_ENVS * NUM_AGENTS) # Approx metric
            print(f"Update: {update}/{num_updates} | Loss: {loss.item():.4f} | Val Loss: {v_loss.item():.4f} | Ent: {entropy.item():.4f}")

    vec_env.close()
    dist.destroy_process_group()