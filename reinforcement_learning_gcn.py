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
from diplomacy_utils import DiplomacyEnv, DiplomacyGCN, get_adjacency_matrix

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use RCCL for AMD GPUs
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, num_envs_per_gpu):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    # 1. Initialize Vector of Environments
    # Total cores used: 4 GPUs * 7 envs = 28 cores
    envs = [DiplomacyEnv() for _ in range(num_envs_per_gpu)]
    
    # 2. Setup Model with DDP
    # We only need to compute the adj matrix once
    sample_env = DiplomacyEnv()
    adj_matrix = get_adjacency_matrix(sample_env.game).to(device)
    
    model = DiplomacyGCN(
        adj=adj_matrix, 
        input_dim=16, 
        hidden_dim=512, # Increased capacity for 4 GPUs
        target_vocab_size=len(sample_env.prov_to_idx)
    ).to(device)
    
    model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Hyperparameters
    gamma = 0.99
    entropy_coef = 0.05
    
    for epoch in range(1000):
        # Reset all environments
        all_obs = []
        all_infos = []
        for e in envs:
            o, i = e.reset()
            all_obs.append(o)
            all_infos.append(i)
            
        # Storage for gradients
        batch_log_probs = []
        batch_rewards = []
        batch_entropies = []
        
        # Run games in parallel
        active_envs = list(range(num_envs_per_gpu))
        step_count = 0
        
        while active_envs and step_count < 120:
            # A. Batch all observations from all agents in all active envs
            # Shape: [Active_Envs * 7, 81, 16]
            flat_obs = []
            flat_masks = {'type': [], 't1': [], 't2': []}
            env_agent_map = [] # To map flat index back to (env_idx, agent_name)

            for e_idx in active_envs:
                for agent in envs[e_idx].agents:
                    flat_obs.append(all_obs[e_idx][agent])
                    m = all_infos[e_idx][agent]['action_mask']
                    flat_masks['type'].append(m['type'])
                    flat_masks['t1'].append(m['target1'])
                    flat_masks['t2'].append(m['target2'])
                    env_agent_map.append((e_idx, agent))

            if not flat_obs: break

            obs_t = torch.tensor(np.array(flat_obs), dtype=torch.float32).to(device)
            m_type = torch.tensor(np.array(flat_masks['type']), dtype=torch.bool).to(device)
            m_t1 = torch.tensor(np.array(flat_masks['t1']), dtype=torch.bool).to(device)
            m_t2 = torch.tensor(np.array(flat_masks['t2']), dtype=torch.bool).to(device)

            # B. Single Forward Pass on MI210 (Massive speedup)
            type_logits, t1_logits, t2_logits = model(obs_t)

            # C. Mask and Sample
            type_logits.masked_fill_(~m_type, -1e9)
            t1_logits.masked_fill_(~m_t1, -1e9)
            t2_logits.masked_fill_(~m_t2, -1e9)

            dist_type, dist_t1, dist_t2 = Categorical(logits=type_logits), Categorical(logits=t1_logits), Categorical(logits=t2_logits)
            
            act_type, act_t1, act_t2 = dist_type.sample(), dist_t1.sample(), dist_t2.sample()
            
            # Compute log_probs only for provinces with units
            # active_unit_mask: [Batch, 81]
            active_unit_mask = (act_type != 0) 
            
            # Step environments
            actions_per_env = {i: {} for i in active_envs}
            log_probs_per_env = {i: {} for i in active_envs}
            entropy_per_env = {i: {} for i in active_envs}

            for idx, (e_idx, agent) in enumerate(env_agent_map):
                # Construct action matrix for the engine
                actions_per_env[e_idx][agent] = torch.stack([act_type[idx], act_t1[idx], act_t2[idx]], dim=-1).cpu().numpy()
                
                # REINFORCE log_prob summation
                if active_unit_mask[idx].any():
                    lp = (dist_type.log_prob(act_type[idx])[active_unit_mask[idx]] + 
                          dist_t1.log_prob(act_t1[idx])[active_unit_mask[idx]] + 
                          dist_t2.log_prob(act_t2[idx])[active_unit_mask[idx]]).sum()
                    log_probs_per_env[e_idx][agent] = lp
                    
                    ent = (dist_type.entropy()[idx][active_unit_mask[idx]] + 
                           dist_t1.entropy()[idx][active_unit_mask[idx]] + 
                           dist_t2.entropy()[idx][active_unit_mask[idx]]).sum()
                    entropy_per_env[e_idx][agent] = ent

            # D. Physical Step (The CPU Bottleneck)
            # Parallelizing this part further is possible via threading, but DDP handles the sync
            new_active_envs = []
            for e_idx in active_envs:
                o, r, d, t, i = envs[e_idx].step(actions_per_env[e_idx])
                
                # Logic to store rewards and check if env is done
                # (Same as your previous logic, but indexed by e_idx)
                # ... [Insert Reward Shaping Logic Here] ...
                
                if envs[e_idx].agents:
                    all_obs[e_idx], all_infos[e_idx] = o, i
                    new_active_envs.append(e_idx)
            
            active_envs = new_active_envs
            step_count += 1

        # 3. Distributed Backprop
        # DDP automatically averages gradients across the 4 GPUs during .backward()
        if batch_log_probs:
            optimizer.zero_grad()
            # Calculate loss...
            # total_loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch} complete on 4 GPUs.")

    cleanup()

if __name__ == "__main__":
    world_size = 4  # 4 MI210s
    envs_per_gpu = 7 # 7 * 4 = 28 cores utilized
    torch.multiprocessing.spawn(train, args=(world_size, envs_per_gpu), nprocs=world_size, join=True)