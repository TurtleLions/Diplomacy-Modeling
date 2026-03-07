import os
import csv
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ExponentialLR

# Standard Diplomacy imports
from diplomacy import Game
from diplomacy_utils import DiplomacyEnv, DiplomacyGCN, get_adjacency_matrix

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, num_envs_per_gpu):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    # 1. Init Environments & Model
    envs = [DiplomacyEnv() for _ in range(num_envs_per_gpu)]
    sample_env = envs[0]
    adj_matrix = get_adjacency_matrix(sample_env.game).to(device)
    
    model = DiplomacyGCN(
        adj=adj_matrix, 
        input_dim=16, 
        hidden_dim=512, 
        target_vocab_size=len(sample_env.prov_to_idx)
    ).to(device)
    
    model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.999) # Decays LR over time
    
    gamma, entropy_coef, max_steps = 0.99, 0.05, 120
    
    if rank == 0:
        csv_file = open('training_log.csv', mode='w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(['epoch', 'loss', 'avg_reward', 'steps', 'lr'])
        print(f"--- Launching Training on 4 MI210s | 28 Environments Total ---")

    for epoch in range(1000):
        # Nested buffers to ensure perfect alignment per agent per env
        env_buffers = {i: {agent: {'lp': [], 're': [], 'en': []} 
                       for agent in sample_env.possible_agents} 
                       for i in range(num_envs_per_gpu)}
        
        all_obs, all_infos = [], []
        for e in envs:
            o, i = e.reset()
            all_obs.append(o); all_infos.append(i)
            
        active_envs = list(range(num_envs_per_gpu))
        step_count, total_epoch_reward = 0, 0
        
        # --- VECTORIZED GAME LOOP ---
        while active_envs and step_count < max_steps:
            flat_obs, flat_masks, env_agent_map = [], {'type': [], 't1': [], 't2': []}, []

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
            type_logits, t1_logits, t2_logits = model(obs_t)
            
            # Applying masks
            for logits, m_key in zip([type_logits, t1_logits, t2_logits], ['type', 't1', 't2']):
                mask = torch.tensor(np.array(flat_masks[m_key]), dtype=torch.bool).to(device)
                logits.masked_fill_(~mask, -1e9)

            d_type, d_t1, d_t2 = Categorical(logits=type_logits), Categorical(logits=t1_logits), Categorical(logits=t2_logits)
            a_type, a_t1, a_t2 = d_type.sample(), d_t1.sample(), d_t2.sample()
            unit_mask = (a_type != 0)

            # Parallel Log-Prob & Entropy Calculation
            batch_lp = (d_type.log_prob(a_type) + d_t1.log_prob(a_t1) + d_t2.log_prob(a_t2)) * unit_mask
            batch_en = (d_type.entropy() + d_t1.entropy() + d_t2.entropy()) * unit_mask

            actions_per_env = {i: {} for i in active_envs}
            pre_step_scs = {e_idx: {sc: p for p in sample_env.possible_agents 
                            for sc in envs[e_idx].game.get_centers(p)} for e_idx in active_envs}

            for idx, (e_idx, agent) in enumerate(env_agent_map):
                actions_per_env[e_idx][agent] = torch.stack([a_type[idx], a_t1[idx], a_t2[idx]], dim=-1).cpu().numpy()
                # Store log-prob/entropy only if they exist for this step
                env_buffers[e_idx][agent]['lp'].append(batch_lp[idx].sum())
                env_buffers[e_idx][agent]['en'].append(batch_en[idx].sum())

            new_active_envs = []
            for e_idx in active_envs:
                next_obs, rewards, terminations, truncations, next_infos = envs[e_idx].step(actions_per_env[e_idx])
                
                for agent in sample_env.possible_agents:
                    if agent in actions_per_env[e_idx]:
                        # Reward Shaping
                        sc_list = envs[e_idx].game.get_centers(agent)
                        r = len(sc_list) * 0.1
                        for sc in sc_list:
                            if sc not in pre_step_scs[e_idx]: r += 1.0
                            elif pre_step_scs[e_idx][sc] != agent: r += 1.5
                        for sc, owner in pre_step_scs[e_idx].items():
                            if owner == agent and sc not in sc_list: r -= 2.0
                        
                        env_buffers[e_idx][agent]['re'].append(r)
                        total_epoch_reward += r
                
                if envs[e_idx].agents:
                    all_obs[e_idx], all_infos[e_idx] = next_obs, next_infos
                    new_active_envs.append(e_idx)
            
            active_envs, step_count = new_active_envs, step_count + 1

        # --- UPDATE PHASE ---
        policy_losses = []
        for i in range(num_envs_per_gpu):
            for agent in sample_env.possible_agents:
                buf = env_buffers[i][agent]
                if not buf['lp'] or len(buf['lp']) != len(buf['re']): continue
                
                returns, R = [], 0
                for r in reversed(buf['re']):
                    R = r + gamma * R
                    returns.insert(0, R)
                
                ret_t = torch.tensor(returns).to(device)
                if ret_t.std() > 0: ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)
                
                loss = -(torch.stack(buf['lp']) * ret_t).mean() - (entropy_coef * torch.stack(buf['en']).mean())
                policy_losses.append(loss)

        if policy_losses:
            optimizer.zero_grad()
            total_loss = torch.stack(policy_losses).sum()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()

            if rank == 0:
                avg_r = total_epoch_reward / (num_envs_per_gpu * 7)
                curr_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch:03d} | Loss: {total_loss.item():.4f} | Avg R: {avg_r:.2f} | LR: {curr_lr:.2e}")
                writer.writerow([epoch, f"{total_loss.item():.4f}", f"{avg_r:.2f}", step_count, f"{curr_lr:.2e}"])
                csv_file.flush()

    if rank == 0:
        torch.save(model.module.state_dict(), "diplomacy_rl_mi210.pth")
        csv_file.close()
    cleanup()

if __name__ == "__main__":
    spawn(train, args=(4, 7), nprocs=4, join=True)