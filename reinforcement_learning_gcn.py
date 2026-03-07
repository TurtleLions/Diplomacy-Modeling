import os
import json
import csv
import logging
import numpy as np
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from torch.distributions import Categorical

from diplomacy import Game
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, MultiDiscrete

# Assuming these are in your local diplomacy_utils.py
from diplomacy_utils import DiplomacyEnv, DiplomacyGCN, get_adjacency_matrix

# Setup basic terminal logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("DiplomacyTrainer")

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # NCCL/RCCL for AMD GPUs
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, num_envs_per_gpu):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    # 1. Initialize Vector of Environments (28 total across cluster)
    envs = [DiplomacyEnv() for _ in range(num_envs_per_gpu)]
    sample_env = envs[0]
    
    # 2. Setup Model with DDP
    adj_matrix = get_adjacency_matrix(sample_env.game).to(device)
    model = DiplomacyGCN(
        adj=adj_matrix, 
        input_dim=16, 
        hidden_dim=512, 
        target_vocab_size=len(sample_env.prov_to_idx)
    ).to(device)
    
    model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Hyperparameters
    gamma = 0.99
    entropy_coef = 0.05
    max_steps = 120
    
    # Local CSV Logging Setup (Rank 0 only)
    if rank == 0:
        csv_file = open('training_log.csv', mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'loss', 'avg_reward', 'steps'])

    for epoch in range(1000):
        # Buffer for log_probs, rewards, and entropies
        env_log_probs = {i: {agent: [] for agent in sample_env.possible_agents} for i in range(num_envs_per_gpu)}
        env_rewards = {i: {agent: [] for agent in sample_env.possible_agents} for i in range(num_envs_per_gpu)}
        env_entropies = {i: {agent: [] for agent in sample_env.possible_agents} for i in range(num_envs_per_gpu)}
        
        all_obs = []
        all_infos = []
        for e in envs:
            o, i = e.reset()
            all_obs.append(o)
            all_infos.append(i)
            
        active_envs = list(range(num_envs_per_gpu))
        step_count = 0
        total_epoch_reward = 0
        
        # --- VECTORIZED GAME LOOP ---
        while active_envs and step_count < max_steps:
            flat_obs = []
            flat_masks = {'type': [], 't1': [], 't2': []}
            env_agent_map = [] 

            # A. Prepare Batch from all active environments
            for e_idx in active_envs:
                for agent in envs[e_idx].agents:
                    flat_obs.append(all_obs[e_idx][agent])
                    m = all_infos[e_idx][agent]['action_mask']
                    flat_masks['type'].append(m['type'])
                    flat_masks['t1'].append(m['target1'])
                    flat_masks['t2'].append(m['target2'])
                    env_agent_map.append((e_idx, agent))

            if not flat_obs: break

            # Tensor conversion [Batch, 81, 16]
            obs_t = torch.tensor(np.array(flat_obs), dtype=torch.float32).to(device)
            m_type = torch.tensor(np.array(flat_masks['type']), dtype=torch.bool).to(device)
            m_t1 = torch.tensor(np.array(flat_masks['t1']), dtype=torch.bool).to(device)
            m_t2 = torch.tensor(np.array(flat_masks['t2']), dtype=torch.bool).to(device)

            # B. Forward Pass & Sampling
            type_logits, t1_logits, t2_logits = model(obs_t)
            type_logits.masked_fill_(~m_type, -1e9)
            t1_logits.masked_fill_(~m_t1, -1e9)
            t2_logits.masked_fill_(~m_t2, -1e9)

            dist_type = Categorical(logits=type_logits)
            dist_t1 = Categorical(logits=t1_logits)
            dist_t2 = Categorical(logits=t2_logits)
            
            act_type, act_t1, act_t2 = dist_type.sample(), dist_t1.sample(), dist_t2.sample()
            active_unit_mask = (act_type != 0) 

            # C. Corrected Vectorized Log Prob & Entropy (Fixes the IndexError)
            batch_lp = (dist_type.log_prob(act_type) + dist_t1.log_prob(act_t1) + dist_t2.log_prob(act_t2)) * active_unit_mask
            batch_ent = (dist_type.entropy() + dist_t1.entropy() + dist_t2.entropy()) * active_unit_mask

            actions_per_env = {i: {} for i in active_envs}
            pre_step_sc_owners = {e_idx: {sc: p for p in sample_env.possible_agents for sc in envs[e_idx].game.get_centers(p)} for e_idx in active_envs}

            # Map batch results back to individual envs
            for idx, (e_idx, agent) in enumerate(env_agent_map):
                actions_per_env[e_idx][agent] = torch.stack([act_type[idx], act_t1[idx], act_t2[idx]], dim=-1).cpu().numpy()
                
                if active_unit_mask[idx].any():
                    env_log_probs[e_idx][agent].append(batch_lp[idx].sum())
                    env_entropies[e_idx][agent].append(batch_ent[idx].sum())

            # D. Execute Step & Calculate Local Rewards
            new_active_envs = []
            for e_idx in active_envs:
                next_obs, rewards, terminations, truncations, next_infos = envs[e_idx].step(actions_per_env[e_idx])
                
                for agent in sample_env.possible_agents:
                    if agent in actions_per_env[e_idx]:
                        curr_sc_list = envs[e_idx].game.get_centers(agent)
                        step_reward = len(curr_sc_list) * 0.1 
                        
                        # Capture / Loss Rewards
                        for sc in curr_sc_list:
                            if sc not in pre_step_sc_owners[e_idx]: step_reward += 1.0 
                            elif pre_step_sc_owners[e_idx][sc] != agent: step_reward += 1.5 
                        for sc, owner in pre_step_sc_owners[e_idx].items():
                            if owner == agent and sc not in curr_sc_list: step_reward -= 2.0 

                        # Retreat penalty
                        dislodged = sum(1 for u in envs[e_idx].game.get_state()['units'].get(agent, []) if '*' in u)
                        step_reward -= (dislodged * 0.5)
                        
                        env_rewards[e_idx][agent].append(step_reward)
                        total_epoch_reward += step_reward
                
                if envs[e_idx].agents:
                    all_obs[e_idx], all_infos[e_idx] = next_obs, next_infos
                    new_active_envs.append(e_idx)
            
            active_envs = new_active_envs
            step_count += 1

        # --- POLICY GRADIENT UPDATE ---
        policy_losses = []
        for i in range(num_envs_per_gpu):
            for agent in sample_env.possible_agents:
                if not env_log_probs[i][agent]: continue
                
                # Discounted Rewards
                returns, R = [], 0
                for r in reversed(env_rewards[i][agent]):
                    R = r + gamma * R
                    returns.insert(0, R)
                
                returns = torch.tensor(returns).to(device)
                if returns.std() > 0:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                
                l_probs = torch.stack(env_log_probs[i][agent])
                entropies = torch.stack(env_entropies[i][agent])
                
                loss = -(l_probs * returns).mean() - (entropy_coef * entropies.mean())
                policy_losses.append(loss)

        if policy_losses:
            optimizer.zero_grad()
            total_loss = torch.stack(policy_losses).sum()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if rank == 0:
                avg_reward = total_epoch_reward / (num_envs_per_gpu * 7)
                logger.info(f"Epoch {epoch} | Loss: {total_loss.item():.4f} | Avg Reward: {avg_reward:.2f}")
                csv_writer.writerow([epoch, f"{total_loss.item():.4f}", f"{avg_reward:.2f}", step_count])
                csv_file.flush() # Ensure it writes to disk

    if rank == 0:
        torch.save(model.module.state_dict(), "diplomacy_rl_model_mi210.pth")
        csv_file.close()
    cleanup()

if __name__ == "__main__":
    WORLD_SIZE = 4
    ENVS_PER_GPU = 7 # Utilizes all 28 cores
    spawn(train, args=(WORLD_SIZE, ENVS_PER_GPU), nprocs=WORLD_SIZE, join=True)