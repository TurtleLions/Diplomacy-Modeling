"""
Distributed Proximal Policy Optimization (PPO) pipeline for the Diplomacy Transformer.
Features asynchronous environment rollouts, double-buffered experience collection to mask
CPU latency, and multi-GPU training via Distributed Data Parallel (DDP).
"""
import os
import time
import datetime
import argparse
import threading
import queue
import contextlib
import torch.multiprocessing as mp  
import multiprocessing.connection
import random
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from diplomacy import Game

import wandb
from dotenv import load_dotenv
from gymnasium.vector import AsyncVectorEnv

from HRLhelpers import FeudalDiplomacyAgent, DiplomacyTransformerEnv, build_global_vocab, build_distance_matrix, InteractionMatrixTracker

C_INT = 0.5

def rebuild_packed_masks(sparse_masks, num_provs, vocab_size, none_idx, device):
    """
    Directly builds the dynamically sized `packed_masks` and `is_active_mask` 
    from sparse indices without materializing the massive (B, 82, V) dense tensor.
    Fully vectorized for maximum GPU throughput.
    """
    batch_size = sparse_masks.size(0)
    
    valid_mask = sparse_masks != -1
    valid_flat_indices = sparse_masks[valid_mask].long()
    
    b_indices = torch.arange(batch_size, device=device).view(batch_size, 1).expand(batch_size, sparse_masks.size(1))
    valid_b_indices = b_indices[valid_mask]
    
    valid_prov_indices = valid_flat_indices // vocab_size
    valid_action_indices = valid_flat_indices % vocab_size
    
    is_not_none = valid_action_indices != none_idx
    active_b = valid_b_indices[is_not_none]
    active_p = valid_prov_indices[is_not_none]
    
    is_active_mask = torch.zeros((batch_size, num_provs), dtype=torch.bool, device=device)
    is_active_mask[active_b, active_p] = True
    
    max_active = is_active_mask.sum(dim=1).max().item()
    
    if max_active == 0:
        packed_masks = torch.zeros((batch_size, 0, vocab_size), dtype=torch.bool, device=device)
        padded_indices = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        return is_active_mask, packed_masks, padded_indices, max_active
        
    noise = torch.rand((batch_size, num_provs), device=device)
    
    noise.masked_fill_(~is_active_mask, float('inf')) 
    
    ranks = noise.argsort(dim=1).argsort(dim=1)
    
    prov_to_active_idx = torch.full((batch_size, num_provs), -1, dtype=torch.long, device=device)
    prov_to_active_idx[is_active_mask] = ranks[is_active_mask]
    
    padded_indices = torch.zeros((batch_size, max_active), dtype=torch.long, device=device)
    active_b_all, active_p_all = torch.where(is_active_mask)
    active_seq_idx = prov_to_active_idx[active_b_all, active_p_all]
    padded_indices[active_b_all, active_seq_idx] = active_p_all
    
    a_idx = prov_to_active_idx[valid_b_indices, valid_prov_indices]
    keep = a_idx != -1
    kb = valid_b_indices[keep]
    ka = a_idx[keep]
    kact = valid_action_indices[keep]
    
    packed_masks = torch.zeros((batch_size, max_active, vocab_size), dtype=torch.bool, device=device)
    packed_masks[kb, ka, kact] = True
    
    return is_active_mask, packed_masks, padded_indices, max_active

def worker(remote, parent_remote):
    """Background worker process for executing environment steps."""
    torch.set_num_threads(1)
    parent_remote.close()
    env = DiplomacyTransformerEnv(history_length=1) 
    
    episode_rewards = {a: 0.0 for a in env.possible_agents}
    
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step': 
                action_dict, progress = data
                obs, rewards, terms, truncs, infos = env.step(action_dict, progress=progress)
                
                for a, r in rewards.items():
                    episode_rewards[a] += r
                
                env_is_done = len(terms) == 0 or all(terms.values()) or all(truncs.values()) or len(env.agents) == 0
                
                if env_is_done:
                    terminal_obs = obs
                    obs, infos = env.reset()

                    for agent, t_obs in terminal_obs.items():
                        if agent in infos:
                            infos[agent]['__terminal_observation'] = t_obs
                    
                    infos['episode_reward'] = episode_rewards.copy()
                    
                    episode_rewards = {a: 0.0 for a in env.possible_agents}

                remote.send((obs, rewards, terms, truncs, infos, env.agents))
            elif cmd == 'reset': 
                episode_rewards = {a: 0.0 for a in env.possible_agents}
                remote.send((*env.reset(), env.agents))
            elif cmd == 'close':
                remote.close()
                break
        except EOFError: 
            break
        except Exception as e:
            print(f"Worker process terminated unexpectedly: {e}")
            remote.close()
            break

class SubprocVecDiplomacy:
    """Asynchronous vector environment using multiprocessing pipes for parallel simulation."""
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote)) for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        for p in self.ps:
            p.daemon = True 
            p.start()
        for remote in self.work_remotes: 
            remote.close()

    def _gather_results(self):
        """Safely polls ready pipes to prevent OS buffer lockups."""
        results = [None] * self.num_envs
        
        # Keep a dictionary mapping each active remote to its original index
        remotes_left = {remote: i for i, remote in enumerate(self.remotes)}
        
        while remotes_left:
            ready_remotes = multiprocessing.connection.wait(remotes_left.keys())
            
            for remote in ready_remotes:
                idx = remotes_left[remote]
                results[idx] = remote.recv() # Clear the buffer instantly
                del remotes_left[remote]     # Remove from the polling pool
                
        return results

    def reset(self):
        for remote in self.remotes: 
            remote.send(('reset', None))
        return self._gather_results()

    def step(self, actions_list, progress_list):
        for remote, action_dict, prog in zip(self.remotes, actions_list, progress_list): 
            remote.send(('step', (action_dict, prog)))
        return self._gather_results()
        
    def close(self):
        for remote in self.remotes: 
            remote.send(('close', None))
        for p in self.ps: 
            p.join()

def evaluate_against_baseline(live_net, baseline_net, device, update_num, live_power, game_index, save_dir="./eval_games"):
    """Runs a single evaluation game for a specific assigned power."""
    os.makedirs(save_dir, exist_ok=True)
    env = DiplomacyTransformerEnv(history_length=1)
    obs, infos = env.reset()
    
    log_path = os.path.join(save_dir, f"eval_update_{update_num}_{live_power}_game_{game_index}.txt")
    
    live_h_memory = {a: torch.zeros((1, 8, 256), dtype=torch.bfloat16, device=device) for a in env.possible_agents}

    # Helper function to process inference for a specific subset of agents
    def get_actions(net, agents, net_device, is_baseline=False):
        if not agents:
            return {}
        
        obs_tensor = torch.stack([torch.tensor(obs[a][0]) for a in agents]).to(device=net_device, dtype=torch.bfloat16)
        sparse_masks_tensor = torch.stack([torch.tensor(infos[a]['action_mask'], dtype=torch.long) for a in agents]).to(net_device)
        
        B = obs_tensor.size(0)
        MAP_PROVINCES = 82
        VOCAB_SIZE = 22231
        
        valid_mask = sparse_masks_tensor != -1
        row_offsets = torch.arange(B, device=net_device).unsqueeze(1) * (MAP_PROVINCES * VOCAB_SIZE)
        global_indices = sparse_masks_tensor + row_offsets
        valid_global_indices = global_indices[valid_mask]
        
        dense_mask = torch.zeros(B * MAP_PROVINCES * VOCAB_SIZE, dtype=torch.bool, device=net_device)
        dense_mask[valid_global_indices.long()] = True
        dense_mask = dense_mask.view(B, MAP_PROVINCES, VOCAB_SIZE)
        
        with torch.no_grad():
            autocast_device = 'cuda' if net_device.type == 'cuda' else 'cpu'
            with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                
                if is_baseline:
                    z_eval = torch.zeros((B, 8, 256), dtype=torch.bfloat16, device=net_device)
                else:
                    x_emb = net.worker.feature_projection(obs_tensor)
                    S_mu_encoded = net.worker.encoder_transformer(x_emb)
                    S_M = net.pooler(S_mu_encoded)

                    real_H = torch.stack([torch.tensor(infos[a]['H_matrix'], dtype=torch.float32) for a in agents]).to(net_device)
                    
                    h_prev = torch.cat([live_h_memory[a] for a in agents], dim=0)
                    
                    z_eval, h_eval = net.manager(S_M, real_H, h_prev)
                    z_eval = z_eval.to(torch.bfloat16)
                    
                    for i, a in enumerate(agents):
                            live_h_memory[a] = h_eval[i].unsqueeze(0).detach()

                logits, _ = net.worker(obs_tensor, z_eval, net.D)
                
                logits = torch.nan_to_num(logits, nan=-1e8, posinf=1e8, neginf=-1e8)
                logits = logits.float().masked_fill(~dense_mask, -1e20)
                final_actions = torch.argmax(logits, dim=-1)
                
        return {a: final_actions[i].cpu().numpy() for i, a in enumerate(agents)}

    with open(log_path, "w") as f:
        f.write(f"Evaluation Game - Update {update_num} | Game {game_index}/10\n")
        f.write(f"Live Network Power: {live_power} | Baseline Power: ALL OTHERS\n")
        f.write("=" * 40 + "\n")
        
        step_count = 0
        while len(env.agents) > 0 and step_count < 150:
            phase_name = env.game.get_current_phase()
            f.write(f"\n--- Phase {phase_name} ---\n")
            active_agents = env.agents
            
            live_agents = [a for a in active_agents if a == live_power]
            baseline_agents = [a for a in active_agents if a != live_power]
            
            baseline_device = next(baseline_net.parameters()).device
            
            live_actions = get_actions(live_net, live_agents, device, is_baseline=False)
            baseline_actions = get_actions(baseline_net, baseline_agents, baseline_device, is_baseline=True)
            
            action_dict = {**live_actions, **baseline_actions}
            
            for agent in active_agents:
                f.write(f"{agent} ({'LIVE' if agent == live_power else 'BASELINE'}) Orders\n")
                orders_issued = False
                for prov_idx, order_idx in enumerate(action_dict[agent]):
                    order_str = env.idx_to_order[order_idx.item()]
                    if order_str != 'NONE':
                        f.write(f"  {order_str}\n")
                        orders_issued = True
                if not orders_issued:
                    f.write("  (No valid orders)\n")
                    
            obs, rewards, terms, truncs, infos = env.step(action_dict)
            step_count += 1
            
        f.write("\n" + "=" * 40 + "\n")
        f.write("FINAL SUPPLY CENTER COUNTS\n")
        
        live_scs = 0
        for agent in env.possible_agents:
            scs = env.game.get_centers(agent)
            count = len(scs)
            if agent == live_power:
                live_scs = count
            f.write(f"{agent} ({'LIVE' if agent == live_power else 'BASELINE'}): {count}\n")
            
    return live_scs

def parse_args():
    parser = argparse.ArgumentParser(description="PPO Training for Diplomacy")
    parser.add_argument("--num_envs", type=int, default=42, help="Number of parallel environments per GPU")
    parser.add_argument("--num_steps", type=int, default=512, help="Number of steps per rollout")
    parser.add_argument("--num_updates", type=int, default=1000, help="Total number of PPO updates")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO policy clipping coefficient")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--v_coef", type=float, default=0.1, help="Value function loss coefficient")
    parser.add_argument("--kl_coef", type=float, default=0.01, help="KL divergence penalty coefficient")
    parser.add_argument("--update_epochs", type=int, default=6, help="Number of epochs per PPO update")
    parser.add_argument("--bc_weights", type=str, default="feudal_agent_bc.pth", help="Path to pre-trained Behavioral Cloning weights")
    parser.add_argument("--resume_weights", type=str, default=None, help="Path to RL checkpoint to resume training from")
    parser.add_argument("--bc_kl_coef", type=float, default=0.1, help="KL divergence penalty coefficient for behavioral cloning")
    return parser.parse_args()

def rollout_worker(local_rank, device, args, buffers, actor_net, bc_baseline_net, 
                   free_buffers_queue, ready_buffers_queue, worker_stats, 
                   NUM_AGENTS, MAP_PROVINCES, VOCAB_SIZE, NONE_IDX, possible_agents, agent_to_idx,
                   actor_weights_lock, start_update=1):
    """Background process responsible for filling the experience buffer asynchronously."""
    torch.cuda.set_device(device) 
    inference_stream = torch.cuda.Stream(device=device)
    
    vec_env = SubprocVecDiplomacy(num_envs=args.num_envs)
    
    buffer_idx = 0
    next_env_results = vec_env.reset()
    next_done = torch.zeros((args.num_envs, NUM_AGENTS), device=device)
    
    batch_H = torch.zeros((args.num_envs, NUM_AGENTS, 7, 7), dtype=torch.float32, device=device)
    batch_h = torch.zeros((args.num_envs, NUM_AGENTS, 8, 256), dtype=torch.bfloat16, device=device)
    batch_prev_h = torch.zeros((args.num_envs, NUM_AGENTS, 8, 256), dtype=torch.bfloat16, device=device)
    batch_S_M = torch.zeros((args.num_envs, NUM_AGENTS, 8, 256), dtype=torch.bfloat16, device=device)

    for update in range(start_update, args.num_updates + 1):
        buffer_idx = free_buffers_queue.get()

        num_learning = 3
        learning_assignment = torch.zeros((args.num_envs, NUM_AGENTS), dtype=torch.bool, device=device)
        for i in range(args.num_envs):
            perm = torch.randperm(NUM_AGENTS, device=device)
            learning_assignment[i, perm[:num_learning]] = True

        current_progress = (update - 1) / max(1, args.num_updates - 1)

        buf = buffers[buffer_idx]
        with torch.cuda.stream(inference_stream):
            buf['masks'].zero_()
            buf['rewards'].zero_()
            
            env_step_time, gpu_forward_time = 0.0, 0.0
            local_proposed, local_dropped = 0, 0
            local_ep_reward_sum, local_ep_count = 0.0, 0
        
            for step in range(args.num_steps):
                actions_to_send = [{} for _ in range(args.num_envs)]
                
                active_obs_list, active_sparse_list, active_indices = [], [], []
                baseline_obs_list, baseline_sparse_list, baseline_metadata = [], [], []
                h_matrix_list, h_indices = [], []
                
                with torch.no_grad():
                    for i in range(args.num_envs):
                        if len(next_env_results[i]) == 3:
                            obs_dict, infos_dict, active_agents = next_env_results[i]
                        else:
                            obs_dict, step_rewards, terms, truncs, infos_dict, active_agents = next_env_results[i]

                        if 'episode_reward' in infos_dict:
                            for r in infos_dict['episode_reward'].values():
                                local_ep_reward_sum += r
                                local_ep_count += 1

                        for a in active_agents:
                            a_idx = agent_to_idx[a]

                            if 'H_matrix' in infos_dict[a]:
                                h_matrix_list.append(infos_dict[a]['H_matrix'])
                                h_indices.append((i, a_idx))
                            
                            if learning_assignment[i, a_idx]:
                                buf['masks'][step, i, a_idx] = True
                                active_obs_list.append(obs_dict[a][0])
                                active_sparse_list.append(infos_dict[a]['action_mask'])
                                active_indices.append((i, a_idx))

                                if 'legality_metrics' in infos_dict[a]:
                                    bad_indices = infos_dict[a]['legality_metrics'].get('illegal_prov_indices', [])
                                    for bad_idx in bad_indices:
                                        buf['unit_penalties'][step, i, a_idx, bad_idx] = -1.0
                                    local_proposed += infos_dict[a]['legality_metrics'].get('proposed', 0)
                                    local_dropped += infos_dict[a]['legality_metrics'].get('illegal_dropped', 0)
                            else:
                                baseline_obs_list.append(obs_dict[a][0])
                                baseline_sparse_list.append(infos_dict[a]['action_mask'])
                                baseline_metadata.append((i, a))

                    if h_matrix_list:
                        h_env_idx = [idx[0] for idx in h_indices]
                        h_agt_idx = [idx[1] for idx in h_indices]
                        batch_H[h_env_idx, h_agt_idx] = torch.tensor(np.stack(h_matrix_list), dtype=torch.float32, device=device)

                    if active_obs_list:
                        act_env_idx = [idx[0] for idx in active_indices]
                        act_agt_idx = [idx[1] for idx in active_indices]
                        
                        batched_active_obs = torch.tensor(np.stack(active_obs_list), dtype=torch.bfloat16, device=device)
                        batched_active_sparse = torch.tensor(np.stack(active_sparse_list), dtype=torch.int32, device=device)
                        
                        buf['obs'][step, act_env_idx, act_agt_idx] = batched_active_obs
                        buf['sparse_masks'][step, act_env_idx, act_agt_idx] = batched_active_sparse

                    t_gpu_start = time.time()

                    flat_obs = buf['obs'][step][buf['masks'][step]]
                    if flat_obs.shape[0] > 0:
                        
                        active_H = batch_H[buf['masks'][step]]
                        active_h = batch_h[buf['masks'][step]]
                        active_prev_h = batch_prev_h[buf['masks'][step]]
                        
                        with actor_weights_lock:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                x_emb = actor_net.worker.feature_projection(flat_obs)
                                S_mu_encoded = actor_net.worker.encoder_transformer(x_emb)
                                active_S_M = actor_net.pooler(S_mu_encoded)
                                
                                new_z, new_h = actor_net.manager(active_S_M, active_H, active_h)
                                new_z = new_z.to(torch.bfloat16)
                                
                                logits, _ = actor_net.worker(flat_obs, new_z, actor_net.D)
                                pooled_S_M = active_S_M.mean(dim=1)
                                values = actor_net.value_head(pooled_S_M).squeeze(-1).float()
                        
                        batch_S_M[buf['masks'][step]] = active_S_M
                        batch_prev_h[buf['masks'][step]] = active_h
                        batch_h[buf['masks'][step]] = new_h
                        
                        buf['S_M'][step][buf['masks'][step]] = active_S_M
                        buf['z'][step][buf['masks'][step]] = new_z
                        buf['h'][step][buf['masks'][step]] = new_h
                        buf['prev_h'][step][buf['masks'][step]] = active_h
                        buf['H'][step][buf['masks'][step]] = active_H

                        active_sparse = buf['sparse_masks'][step][buf['masks'][step]]
                        
                        is_active_mask, packed_masks, padded_idx, max_active = rebuild_packed_masks(
                            active_sparse, MAP_PROVINCES, VOCAB_SIZE, NONE_IDX, device
                        )
                        
                        B, max_act = padded_idx.shape
                        b_idx = torch.arange(B, device=device).view(B, 1).expand(B, max_act)
                        
                        active_logits = logits[b_idx, padded_idx, :].float()
                        active_logits = active_logits.masked_fill(~packed_masks, -1e20)
                        
                        seq_lens = is_active_mask.sum(dim=1)
                        valid_pack_mask = torch.arange(max_act, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
                        active_logits = active_logits.masked_fill(~valid_pack_mask.unsqueeze(-1), 0.0)
                        
                        dist_cat = Categorical(logits=active_logits)
                        packed_actions = dist_cat.sample()
                        packed_logprobs = dist_cat.log_prob(packed_actions)
                        
                        final_actions = torch.full((flat_obs.size(0), MAP_PROVINCES), NONE_IDX, dtype=torch.long, device=device)
                        final_logprobs = torch.zeros((flat_obs.size(0), MAP_PROVINCES), dtype=torch.float32, device=device)
                        
                        valid_b, valid_seq = torch.where(valid_pack_mask)
                        valid_p = padded_idx[valid_b, valid_seq]
                        
                        final_actions[valid_b, valid_p] = packed_actions[valid_b, valid_seq]
                        final_logprobs[valid_b, valid_p] = packed_logprobs[valid_b, valid_seq]
                        
                        buf['values'][step][buf['masks'][step]] = values
                        buf['logprobs'][step][buf['masks'][step]] = final_logprobs
                        
                        idx_counter = 0
                        for i in range(args.num_envs):
                            for a in possible_agents:
                                if buf['masks'][step, i, agent_to_idx[a]]:
                                    act_array = final_actions[idx_counter]
                                    buf['actions'][step, i, agent_to_idx[a]] = act_array
                                    actions_to_send[i][a] = act_array.cpu().numpy()
                                    idx_counter += 1
                                    
                    if baseline_obs_list:
                        bc_obs_tensor = torch.tensor(np.stack(baseline_obs_list), dtype=torch.bfloat16, device=device)
                        bc_sparse_tensor = torch.tensor(np.stack(baseline_sparse_list), dtype=torch.int32, device=device)
                        
                        with torch.no_grad():
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                # Baseline receives a dummy 0 strategy vector
                                bc_z = torch.zeros(bc_obs_tensor.size(0), 8, 256, device=device, dtype=torch.bfloat16)
                                bc_logits, _ = bc_baseline_net.worker(bc_obs_tensor, bc_z, bc_baseline_net.D)

                            # Fast One-Shot Masking for Baseline
                            valid_mask_bc = bc_sparse_tensor != -1
                            row_offsets_bc = torch.arange(bc_obs_tensor.size(0), device=device).unsqueeze(1) * (MAP_PROVINCES * VOCAB_SIZE)
                            global_indices_bc = bc_sparse_tensor + row_offsets_bc
                            valid_global_indices_bc = global_indices_bc[valid_mask_bc]

                            dense_mask_bc = torch.zeros(bc_obs_tensor.size(0) * MAP_PROVINCES * VOCAB_SIZE, dtype=torch.bool, device=device)
                            dense_mask_bc[valid_global_indices_bc.long()] = True
                            dense_mask_bc = dense_mask_bc.view(bc_obs_tensor.size(0), MAP_PROVINCES, VOCAB_SIZE)

                            bc_logits = bc_logits.float().masked_fill(~dense_mask_bc, -1e20)
                            bc_final_actions = torch.argmax(bc_logits, dim=-1) # Greedy sample for BC proxy

                        for idx, (env_idx, agent_name) in enumerate(baseline_metadata):
                            actions_to_send[env_idx][agent_name] = bc_final_actions[idx].cpu().numpy()

                    gpu_forward_time += (time.time() - t_gpu_start)

                t_env_start = time.time()
                progress_to_send = [(update - 1) / max(1, args.num_updates - 1) for _ in range(args.num_envs)]
                next_env_results = vec_env.step(actions_to_send, progress_to_send)
                env_step_time += (time.time() - t_env_start)

                terminal_obs_to_encode = []
                terminal_indices = []

                with torch.no_grad():
                    for i in range(args.num_envs):
                        if len(next_env_results[i]) > 3:
                            obs_dict, step_rewards, terms, truncs, infos_dict, active_agents = next_env_results[i]

                            for a in possible_agents:
                                if a in infos_dict and '__terminal_observation' in infos_dict[a]:
                                    if learning_assignment[i, agent_to_idx[a]]:
                                        terminal_obs_to_encode.append(infos_dict[a]['__terminal_observation'][0])
                                        terminal_indices.append((i, agent_to_idx[a]))
                                        
                                        batch_h[i, agent_to_idx[a]].zero_()
                                        batch_prev_h[i, agent_to_idx[a]].zero_()
                                        batch_S_M[i, agent_to_idx[a]].zero_()
                            
                            for a in possible_agents:
                                if learning_assignment[i, agent_to_idx[a]]:
                                    buf['rewards'][step, i, agent_to_idx[a]] = step_rewards.get(a, 0.0)
                                    buf['extrinsic_rewards'][step, i, agent_to_idx[a]] = step_rewards.get(a, 0.0)
                                    buf['dones'][step, i, agent_to_idx[a]] = float(terms.get(a, False))
                                    buf['truncations'][step, i, agent_to_idx[a]] = float(truncs.get(a, False))
                                    next_done[i, agent_to_idx[a]] = float(terms.get(a, False))

                    if terminal_obs_to_encode:
                        term_tensor = torch.tensor(np.stack(terminal_obs_to_encode), dtype=torch.bfloat16, device=device)
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            x_emb_t = actor_net.worker.feature_projection(term_tensor)
                            S_M_t = actor_net.pooler(actor_net.worker.encoder_transformer(x_emb_t))
                            term_v = actor_net.value_head(S_M_t.mean(dim=1)).float().squeeze(-1)
                        for list_idx, (env_idx, agent_idx) in enumerate(terminal_indices):
                            buf['terminal_values'][step, env_idx, agent_idx] = term_v[list_idx]
                            buf['terminal_S_M'][step, env_idx, agent_idx] = S_M_t[list_idx]

        # Calculate GAE Advantages
        with torch.no_grad():
            next_value = torch.zeros((args.num_envs, NUM_AGENTS), device=device)
            obs_to_encode = []
            indices_to_update = []
            
            for i in range(args.num_envs):
                infos_dict = next_env_results[i][4]
                obs_dict = next_env_results[i][0]
                active_agents_list = next_env_results[i][5]
                
                for a in possible_agents:
                    if a in infos_dict and '__terminal_observation' in infos_dict[a]:
                        obs_to_encode.append(infos_dict[a]['__terminal_observation'][0])
                        indices_to_update.append((i, agent_to_idx[a]))
                    elif a in active_agents_list and a in obs_dict:
                        obs_to_encode.append(obs_dict[a][0])
                        indices_to_update.append((i, agent_to_idx[a]))
                    
            if obs_to_encode:
                obs_tensor = torch.tensor(np.stack(obs_to_encode), dtype=torch.bfloat16, device=device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    x_emb = actor_net.worker.feature_projection(obs_tensor)
                    S_mu_enc = actor_net.worker.encoder_transformer(x_emb)
                    S_M_next = actor_net.pooler(S_mu_enc)
                    pooled_S_M_next = S_M_next.mean(dim=1)
                    next_v = actor_net.value_head(pooled_S_M_next).float()
            
            for list_idx, (env_idx, agent_idx) in enumerate(indices_to_update):
                val = next_v[list_idx].float().squeeze(-1)
                next_value[env_idx, agent_idx] = val
                buf['terminal_values'][step, env_idx, agent_idx] = val
            
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Flatten the states to pass through the inverse model
                flat_S_M = buf['S_M'].view(-1, 8, 256)
                flat_S_M_next = buf['S_M_next'].view(-1, 8, 256)
                flat_z = buf['z'].view(-1, 8, 256)
                
                # Predict what Z was actually achieved
                flat_z_achieved = actor_net.inverse_model(flat_S_M, flat_S_M_next)
                
                # Calculate cosine similarity and average across the 8 theaters
                intrinsic_rewards_flat = F.cosine_similarity(flat_z_achieved.float(), flat_z.float(), dim=-1).mean(dim=1)
                
                # Reshape back to buffer dimensions
                intrinsic_rewards = intrinsic_rewards_flat.view(args.num_steps, args.num_envs, NUM_AGENTS)
                
                # Add scaled intrinsic reward to the extrinsic reward
                current_c_int = min(0.02, 0.02 * (update / 100.0))
                buf['rewards'] += (current_c_int * intrinsic_rewards)
        
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - buf['dones'][t]
                is_trunc = buf['truncations'][t] == 1.0
                nextvalues = torch.where(is_trunc, buf['terminal_values'][t], buf['values'][t + 1])
                
            delta = buf['rewards'][t] + args.gamma * nextvalues * nextnonterminal - buf['values'][t]
            buf['advantages'][t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        buf['returns'] = buf['advantages'] + buf['values']
        
        buf['S_M_next'][:-1] = buf['S_M'][1:]
        
        if obs_to_encode:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                x_emb_next = actor_net.worker.feature_projection(obs_tensor)
                S_M_terminal = actor_net.pooler(actor_net.worker.encoder_transformer(x_emb_next))
            for list_idx, (env_idx, agent_idx) in enumerate(indices_to_update):
                buf['S_M_next'][-1, env_idx, agent_idx] = S_M_terminal[list_idx]
                buf['terminal_S_M'][-1, env_idx, agent_idx] = S_M_terminal[list_idx]

        is_done_or_trunc = (buf['dones'] == 1.0) | (buf['truncations'] == 1.0)
        is_done_expanded = is_done_or_trunc.unsqueeze(-1).unsqueeze(-1).expand_as(buf['S_M_next'])
        buf['S_M_next'] = torch.where(is_done_expanded, buf['terminal_S_M'], buf['S_M_next'])

        inference_stream.synchronize()

        worker_stats[0] = env_step_time
        worker_stats[1] = gpu_forward_time
        worker_stats[2] = local_proposed
        worker_stats[3] = local_dropped
        worker_stats[4] = local_ep_reward_sum
        worker_stats[5] = local_ep_count
    
        ready_buffers_queue.put(buffer_idx)

def main():
    mp.set_start_method('spawn', force=True)
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)

    args = parse_args()

    NUM_AGENTS = 7
    HISTORY_LENGTH = 1

    if global_rank == 0:
        print("--- Initiating Distributed PPO Pipeline ---")
        load_dotenv()
        wandb_key = os.environ.get("WANDB_KEY")
        if wandb_key:
            wandb.login(key=wandb_key)
        else:
            print("Warning: WANDB_KEY not found in .env file.")

    if global_rank == 0:
        # Rank 0 builds the vocabulary cache file for all workers
        build_global_vocab() 
        
    dist.barrier() 

    # Initialize environment metadata
    dummy_env = DiplomacyTransformerEnv()
    possible_agents = dummy_env.possible_agents
    MAP_PROVINCES = dummy_env.num_provinces
    VOCAB_SIZE = dummy_env.vocab_size
    NONE_IDX = dummy_env.order_to_idx['NONE']
    del dummy_env

    if global_rank == 0:
        print(f"Environment Initialized: {MAP_PROVINCES} Provinces | Action Space: {VOCAB_SIZE}")
    
    # Model Initialization
    net = FeudalDiplomacyAgent(d_model=256, vocab_size=VOCAB_SIZE).to(device)
    actor_net = FeudalDiplomacyAgent(d_model=256, vocab_size=VOCAB_SIZE).to(device)
    bc_baseline_net = FeudalDiplomacyAgent(d_model=256, vocab_size=VOCAB_SIZE).to(device)

    
    temp_game = Game()
    provinces = sorted([p.upper() for p in list(temp_game.map.locs)])
    
    D_matrix = build_distance_matrix(provinces).to(device)
    net.D.copy_(D_matrix)
    actor_net.D.copy_(D_matrix)
    bc_baseline_net.D.copy_(D_matrix)
    
    if os.path.exists(args.bc_weights):
        bc_state_dict = torch.load(args.bc_weights, map_location=device)
        net.load_state_dict(bc_state_dict, strict=False)
        
        actor_net.load_state_dict(net.state_dict())
        bc_baseline_net.load_state_dict(bc_state_dict, strict=False)

        # Freeze reference models
        actor_net.eval()
        for param in actor_net.parameters(): param.requires_grad = False
            
        if global_rank == 0: 
            print("Successfully loaded pre-trained BC weights for policy initialization.")

    loaded_opt_state = None
    start_update = 1

    if args.resume_weights and os.path.exists(args.resume_weights):
        checkpoint = torch.load(args.resume_weights, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'], strict=False)
            actor_net.load_state_dict(checkpoint['model_state_dict'], strict=False)
            loaded_opt_state = checkpoint['optimizer_state_dict']
            
            start_update = checkpoint.get('update', 0) + 1 
        else:
            net.load_state_dict(checkpoint, strict=False)
            actor_net.load_state_dict(checkpoint, strict=False)
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if global_rank == 0:
            print(f"Successfully resumed RL training from checkpoint: {args.resume_weights}")

    net = DDP(net, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    
    def warmup_schedule(update):
        warmup_updates = 10
        if update < warmup_updates:
            return float(update) / float(max(1, warmup_updates))
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)
    
    if loaded_opt_state is not None:
        optimizer.load_state_dict(loaded_opt_state)
        if global_rank == 0:
            print("Successfully restored Optimizer momentum and variance states.")

    def create_buffer():
        """Creates a memory-pinned tensor buffer for experience collection."""
        return {
            'obs': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 82, 61), dtype=torch.bfloat16, device=device),
            'actions': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 82), dtype=torch.long, device=device),
            'logprobs': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 82), dtype=torch.float32, device=device),
            'rewards': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'extrinsic_rewards': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'dones': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'values': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'masks': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.bool, device=device),
            'truncations': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'terminal_values': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'sparse_masks': torch.full((args.num_steps, args.num_envs, NUM_AGENTS, 4000), -1, dtype=torch.int32, device=device),
            'advantages': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'returns': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'S_M': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 8, 256), dtype=torch.bfloat16, device=device),
            'S_M_next': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 8, 256), dtype=torch.bfloat16, device=device),
            'z': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 8, 256), dtype=torch.bfloat16, device=device),
            'h': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 8, 256), dtype=torch.bfloat16, device=device),
            'prev_h': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 8, 256), dtype=torch.bfloat16, device=device),
            'H': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 7, 7), dtype=torch.float32, device=device),
            'unit_penalties': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 82), dtype=torch.float32, device=device),
            'terminal_S_M': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 8, 256), dtype=torch.bfloat16, device=device),
        }
        
    # Double-buffering architecture masks CPU environment latency behind GPU backpropagation
    buffers = {0: create_buffer(), 1: create_buffer()}
    
    free_buffers_queue = queue.Queue()
    ready_buffers_queue = queue.Queue()
    
    free_buffers_queue.put(0)
    free_buffers_queue.put(1)

    thread_stats = {"env_time": 0.0, "gpu_fwd_time": 0.0, "proposed": 0, "dropped": 0}

    actor_net.share_memory()
    bc_baseline_net.share_memory()

    worker_stats = torch.zeros(6, dtype=torch.float32, device=device)
    last_avg_ep_reward = 0.0
    agent_to_idx = {a: i for i, a in enumerate(possible_agents)}

    inference_stream = torch.cuda.Stream(device=device)
    actor_weights_lock = threading.Lock()
    rollout_process = threading.Thread(target=rollout_worker, args=(
        local_rank, device, args, buffers, actor_net, bc_baseline_net,
        free_buffers_queue, ready_buffers_queue, worker_stats,
        NUM_AGENTS, MAP_PROVINCES, VOCAB_SIZE, NONE_IDX, possible_agents, agent_to_idx,
        actor_weights_lock, start_update
    ))
    rollout_process.start()

    timestamp_list = [None]
    if global_rank == 0:
        timestamp_list[0] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    dist.broadcast_object_list(timestamp_list, src=0)
    timestamp = timestamp_list[0]

    run_name = f"ppo_run_{timestamp}"
    base_dir = f"/data/restanislao/model_runs/{run_name}"
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    eval_dir = os.path.join(base_dir, "eval_games")

    if global_rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        
        writer = SummaryWriter(log_dir=f"/data/restanislao/tb_runs/{run_name}")
        wandb.init(
            project="diplomacy-ppo",
            name=run_name,
            config=vars(args),
            dir="/data/restanislao/wandb"
        )
        wandb.watch(net.module, log="all", log_freq=10)

    dist.barrier()

    for param in net.module.worker.parameters():
            param.requires_grad = False

    if start_update >= 11:
        for param in net.module.worker.parameters():
            param.requires_grad = True
        if global_rank == 0:
            print("Resuming past warmup: TacticalWorker parameters unfrozen.")

    for update in range(start_update, args.num_updates + 1):
        start_time = time.time()

        if update == 11:
            for param in net.module.worker.parameters():
                param.requires_grad = True
            if global_rank == 0:
                print("Warmup complete: TacticalWorker parameters unfrozen.")
        
        buffer_idx = ready_buffers_queue.get()
        
        with actor_weights_lock:
            with torch.no_grad():
                for param, actor_param in zip(net.module.parameters(), actor_net.parameters()):
                    actor_param.data.copy_(param)
        torch.cuda.current_stream().synchronize()

        buf = buffers[buffer_idx]
        env_step_time = worker_stats[0].item()
        gpu_forward_time = worker_stats[1].item()
        proposed_actions = int(worker_stats[2].item())
        illegal_dropped = int(worker_stats[3].item())
        
        valid = buf['masks'].view(-1)
        flat_obs = buf['obs'].view(-1, MAP_PROVINCES, 61)[valid] 
        flat_act = buf['actions'].view(-1, MAP_PROVINCES)[valid]
        flat_logprobs = buf['logprobs'].view(-1, MAP_PROVINCES)[valid]
        flat_adv = buf['advantages'].view(-1)[valid]
        flat_ret = buf['returns'].view(-1)[valid]
        flat_sparse_masks = buf['sparse_masks'].view(-1, 4000)[valid]
        
        flat_S_M = buf['S_M'].view(-1, 8, 256)[valid]
        flat_S_M_next = buf['S_M_next'].view(-1, 8, 256)[valid]
        flat_z = buf['z'].view(-1, 8, 256)[valid]
        flat_h = buf['h'].view(-1, 8, 256)[valid]
        flat_prev_h = buf['prev_h'].view(-1, 8, 256)[valid]
        flat_H = buf['H'].view(-1, 7, 7)[valid]

        free_buffers_queue.put(buffer_idx)
        inference_stream.wait_stream(torch.cuda.current_stream())

        b_size = flat_obs.shape[0]
        
        # DDP min-batch synchronization to prevent NCCL hanging
        local_b_size = torch.tensor([b_size], dtype=torch.long, device=device)
        dist.all_reduce(local_b_size, op=dist.ReduceOp.MIN)
        min_b_size = local_b_size.item()

        if b_size > min_b_size:
            perm = torch.randperm(b_size, device=device)
            flat_obs = flat_obs[perm][:min_b_size]
            flat_act = flat_act[perm][:min_b_size]
            flat_logprobs = flat_logprobs[perm][:min_b_size]
            flat_adv = flat_adv[perm][:min_b_size]
            flat_ret = flat_ret[perm][:min_b_size]
            flat_sparse_masks = flat_sparse_masks[perm][:min_b_size]
            
            flat_S_M = flat_S_M[perm][:min_b_size]
            flat_S_M_next = flat_S_M_next[perm][:min_b_size]
            flat_z = flat_z[perm][:min_b_size]
            flat_h = flat_h[perm][:min_b_size]
            flat_prev_h = flat_prev_h[perm][:min_b_size]
            flat_H = flat_H[perm][:min_b_size]
            b_size = min_b_size

        # Advantage Normalization across GPUs
        if flat_adv.shape[0] > 1:
            local_sum, local_sq_sum = flat_adv.sum(), (flat_adv ** 2).sum()
            local_count = torch.tensor(flat_adv.shape[0], dtype=torch.float32, device=device)
        else:
            local_sum = local_sq_sum = local_count = torch.tensor(0.0, device=device)
            
        stats = torch.stack([local_sum, local_sq_sum, local_count])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        if stats[2] > 1:
            global_mean = stats[0] / stats[2]
            global_var = (stats[1] / stats[2]) - (global_mean ** 2)
            global_std = torch.sqrt(torch.clamp(global_var, min=1e-8))
            flat_adv = (flat_adv - global_mean) / (global_std + 1e-8)

        t_update_start = time.time()
        net.train()
        
        mb_size = 512
        accum_steps = 16
        optimizer.zero_grad() 

        epoch_pg_loss_sum = 0.0
        epoch_manager_loss_sum = 0.0
        epoch_feasibility_error_sum = 0.0
        epoch_bc_kl_sum = 0.0
        epoch_entropy_sum = 0.0
        epoch_total_loss_sum = 0.0
        epoch_intrinsic_reward_sum = 0.0
        epoch_inv_loss_sum = 0.0
        epoch_z_var_sum = 0.0
        epoch_inv_grad_norm_sum = 0.0
        current_c_int = min(0.5, 0.5 * (update / 200.0))
        track_steps = 0

        target_kl = 0.02
        global_avg_kl = 0.0

        for epoch in range(args.update_epochs):
            perm = torch.randperm(b_size, device=device)
            epoch_obs = flat_obs[perm]
            epoch_act = flat_act[perm]
            epoch_logprobs = flat_logprobs[perm]
            epoch_adv = flat_adv[perm]
            epoch_ret = flat_ret[perm]
            epoch_sparse = flat_sparse_masks[perm]
            
            epoch_S_M = flat_S_M[perm]
            epoch_S_M_next = flat_S_M_next[perm]
            epoch_z = flat_z[perm]
            epoch_h = flat_h[perm]
            epoch_prev_h = flat_prev_h[perm]
            epoch_H = flat_H[perm]

            start_indices = list(range(0, b_size, mb_size))
            epoch_kl_sum, epoch_kl_steps = 0.0, 0

            for step_idx, start in enumerate(start_indices):
                end = start + mb_size
                
                mb_obs = epoch_obs[start:end].to(dtype=torch.bfloat16)
                mb_act = epoch_act[start:end]
                mb_logprobs = epoch_logprobs[start:end]
                mb_adv = epoch_adv[start:end]
                mb_ret = epoch_ret[start:end]
                mb_sparse_gpu = epoch_sparse[start:end].to(device)
                
                mb_S_M = epoch_S_M[start:end].to(dtype=torch.bfloat16)
                mb_S_M_next = epoch_S_M_next[start:end].to(dtype=torch.bfloat16)
                mb_z = epoch_z[start:end].to(dtype=torch.bfloat16)
                mb_h = epoch_h[start:end].to(dtype=torch.bfloat16)
                mb_prev_h = epoch_prev_h[start:end].to(dtype=torch.bfloat16)
                mb_H = epoch_H[start:end]

                is_last_batch = (step_idx + 1) == len(start_indices)
                sync_this_step = (step_idx + 1) % accum_steps == 0 or is_last_batch
                my_context = net.no_sync() if not sync_this_step else contextlib.nullcontext()
                
                with my_context:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        live_S_M, predicted_z, predicted_h, logits, values_pred, z_achieved_raw = net(
                            mb_obs, mb_H, mb_prev_h, mb_z.to(torch.bfloat16), bc_mode=False, mb_S_M_next=mb_S_M_next
                        )
                        logits = torch.nan_to_num(logits, nan=-1e8, posinf=1e8, neginf=-1e8)
                        z_achieved = z_achieved_raw.float()

                        intrinsic_reward = F.cosine_similarity(z_achieved, mb_z.detach().float(), dim=-1)
                        epoch_intrinsic_reward_sum += intrinsic_reward.detach().mean().item()

                        flat_mb_z = mb_z.view(-1, 256).float()
                        flat_z_achieved = z_achieved_raw.view(-1, 256).float()
                        flat_predicted_z = predicted_z.view(-1, 256).float()

                        flat_mb_z_det = flat_mb_z.detach()
                        flat_z_achieved_det = flat_z_achieved.detach()
                        
                        inv_loss_mgr = F.mse_loss(flat_predicted_z, flat_z_achieved_det)
                        inv_loss_inv = F.mse_loss(flat_z_achieved, flat_mb_z_det)
                        
                        target_std = 1.0
                        std_z_achieved = torch.sqrt(z_achieved_raw.float().var(dim=0) + 1e-04)
                        std_predicted_z = torch.sqrt(mb_z.float().var(dim=0) + 1e-04)
                        
                        var_loss_achieved = torch.mean(F.relu(target_std - std_z_achieved))
                        var_loss_predicted = torch.mean(F.relu(target_std - std_predicted_z))
                        
                        manager_loss = inv_loss_mgr + (1.0 * var_loss_predicted)
                        inverse_model_loss = inv_loss_inv + (1.0 * var_loss_achieved)
                        
                        z_variance = z_achieved_raw.var(dim=0).mean().item() if z_achieved_raw.size(0) > 1 else 0.0

                        epoch_inv_loss_sum += inverse_model_loss.item()
                        epoch_z_var_sum += z_variance
                        
                        v_loss = F.huber_loss(values_pred, mb_ret.float(), delta=10.0)
                        distance_penalty = F.mse_loss(mb_z.detach(), z_achieved.detach())
                        
                        is_active, packed_masks, padded_idx, max_act_live = rebuild_packed_masks(
                            mb_sparse_gpu, MAP_PROVINCES, VOCAB_SIZE, NONE_IDX, device
                        )
                        
                        B_live, max_act_live = padded_idx.shape
                        
                        pg_loss = torch.tensor(0.0, device=device)
                        entropy = torch.tensor(0.0, device=device)
                        bc_kl_penalty = torch.tensor(0.0, device=device)
                        kl_divergence = torch.tensor(0.0, device=device)
                        
                        if max_act_live > 0:
                            b_idx_live = torch.arange(B_live, device=device).view(B_live, 1).expand(B_live, max_act_live)
                            
                            active_logits = logits[b_idx_live, padded_idx, :].float()
                            active_logits = active_logits.masked_fill(~packed_masks, -1e20)
                            
                            seq_lens = is_active.sum(dim=1)
                            valid_pack_mask = torch.arange(max_act_live, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
                            active_logits = active_logits.masked_fill(~valid_pack_mask.unsqueeze(-1), 0.0)
                            
                            dist_cat = Categorical(logits=active_logits)
                            packed_actions = torch.gather(mb_act, 1, padded_idx)
                            new_logprobs = dist_cat.log_prob(packed_actions)
                            
                            old_logprobs_packed = torch.gather(mb_logprobs, 1, padded_idx)
                            ratio = torch.exp(new_logprobs - old_logprobs_packed)
                            
                            valid_ratio_mask = (packed_actions != NONE_IDX) & valid_pack_mask
                            
                            if valid_ratio_mask.any():
                                worker_adv = mb_adv.unsqueeze(1)
                                
                                surr1 = ratio * worker_adv
                                surr2 = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * worker_adv
                                
                                pg_loss = -torch.min(surr1, surr2)[valid_ratio_mask].mean()
                                entropy = dist_cat.entropy()[valid_ratio_mask].mean()
                                
                                with torch.no_grad():
                                    bc_z = torch.zeros_like(mb_z)
                                    bc_logits, _ = bc_baseline_net.worker(mb_obs, bc_z, bc_baseline_net.D)
                                    bc_logits = torch.nan_to_num(bc_logits, nan=-1e8, posinf=1e8, neginf=-1e8)
                                    
                                    bc_active_logits = bc_logits[b_idx_live, padded_idx, :].float()
                                    bc_active_logits = bc_active_logits.masked_fill(~packed_masks, -1e20)
                                    bc_active_logits = bc_active_logits.masked_fill(~valid_pack_mask.unsqueeze(-1), 0.0)
                                    
                                    bc_dist = Categorical(logits=bc_active_logits)
                                    bc_logprobs = bc_dist.log_prob(packed_actions)
                                    bc_logprobs = torch.clamp(bc_logprobs, min=-20.0)
                                    
                                kl_divergence = ((ratio - 1.0) - torch.log(ratio))[valid_ratio_mask].mean()
                                
                                kl_div_vector = new_logprobs - bc_logprobs
                                raw_bc_kl = kl_div_vector[valid_ratio_mask].mean()
                                progress = min(1.0, update / 100.0) 
                                dynamic_target_kl = 0.02 + (0.48 * progress) 
                                
                                bc_kl_penalty = F.relu(raw_bc_kl - dynamic_target_kl)

                        # Final loss aggregation
                        unscaled_loss = (pg_loss 
                                         - (args.ent_coef * entropy) 
                                         + (0.05 * manager_loss) 
                                         + (0.05 * inverse_model_loss)
                                         + (args.bc_kl_coef * bc_kl_penalty) 
                                         + (args.v_coef * v_loss))

                        epoch_pg_loss_sum += pg_loss.item()
                        epoch_manager_loss_sum += manager_loss.item()
                        epoch_feasibility_error_sum += distance_penalty.mean().item()
                        epoch_bc_kl_sum += bc_kl_penalty.item() 
                        epoch_entropy_sum += entropy.item()
                        epoch_total_loss_sum += unscaled_loss.item()
                        track_steps += 1

                        current_block_start = (step_idx // accum_steps) * accum_steps
                        current_block_end = min(current_block_start + accum_steps, len(start_indices))
                        actual_accum_steps = current_block_end - current_block_start
                        
                        loss = unscaled_loss / actual_accum_steps
                        loss.backward()
                        inv_grad_norm = torch.nn.utils.clip_grad_norm_(net.module.inverse_model.parameters(), float('inf')).item()
                        epoch_inv_grad_norm_sum += inv_grad_norm

                if sync_this_step:
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_kl_sum += kl_divergence.item()
                epoch_kl_steps += 1

            local_epoch_kl = epoch_kl_sum / max(1, epoch_kl_steps)
            epoch_kl_tensor = torch.tensor([local_epoch_kl], device=device)
            dist.all_reduce(epoch_kl_tensor, op=dist.ReduceOp.SUM)
            global_epoch_kl = epoch_kl_tensor.item() / dist.get_world_size()
            
            global_avg_kl = global_epoch_kl
            
            if global_epoch_kl > target_kl * 1.5:
                if global_rank == 0:
                    print(f"Early stopping triggered at epoch {epoch+1} due to high KL: {global_epoch_kl:.4f}")
                break
            
        args.kl_coef = max(0.0001, min(5.0, args.kl_coef))

        local_metrics = torch.tensor([
            epoch_pg_loss_sum / max(1, track_steps),
            epoch_manager_loss_sum / max(1, track_steps),
            epoch_feasibility_error_sum / max(1, track_steps),
            epoch_bc_kl_sum / max(1, track_steps),
            epoch_entropy_sum / max(1, track_steps),
            epoch_total_loss_sum / max(1, track_steps),
            epoch_intrinsic_reward_sum / max(1, track_steps),
            epoch_inv_loss_sum / max(1, track_steps),
            epoch_z_var_sum / max(1, track_steps),
            epoch_inv_grad_norm_sum / max(1, track_steps),
            worker_stats[4].item(),
            worker_stats[5].item()
        ], device=device)

        dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)
        global_metrics = local_metrics[:10] / dist.get_world_size()

        avg_pg_loss = global_metrics[0].item()
        avg_manager_loss = global_metrics[1].item()
        avg_feasibility_error = global_metrics[2].item()
        avg_bc_kl = global_metrics[3].item()
        avg_entropy = global_metrics[4].item()
        avg_total_loss = global_metrics[5].item()
        avg_intrinsic_reward = global_metrics[6].item()
        avg_inv_loss = global_metrics[7].item()
        avg_z_var = global_metrics[8].item()
        avg_inv_grad_norm = global_metrics[9].item()
        global_ep_reward_sum = local_metrics[10].item()
        global_ep_count = local_metrics[11].item()

        if global_ep_count > 0:
            last_avg_ep_reward = global_ep_reward_sum / global_ep_count

        update_time = time.time() - t_update_start
        
        scheduler.step()

        if global_rank == 0:
            total_time = time.time() - start_time
            global_steps = args.num_envs * args.num_steps * NUM_AGENTS * dist.get_world_size()
            sps = int(global_steps / total_time)  
            total_active_steps = buf['masks'].sum().item()
            if total_active_steps > 0:
                avg_extrinsic_step_reward = buf['extrinsic_rewards'].sum().item() / total_active_steps
                avg_total_step_reward = buf['rewards'].sum().item() / total_active_steps
            else:
                avg_extrinsic_step_reward = 0.0
                avg_total_step_reward = 0.0

            illegal_rate = (illegal_dropped / max(1, proposed_actions)) * 100.0
            
            print(f"Update {update}/{args.num_updates} | SPS: {sps} | Extrinsic Step: {avg_extrinsic_step_reward:.2f} | Total Step (w/ Intrinsic): {avg_total_step_reward:.2f}")
            print(f"  CPU Time: {env_step_time:.2f}s | GPU Fwd: {gpu_forward_time:.2f}s | Bwd: {update_time:.2f}s")
            print(f"  Losses -> Total: {avg_total_loss:.4f} | PG: {avg_pg_loss:.4f} | Mgr(InfoNCE): {avg_manager_loss:.4f} | Inv(InfoNCE): {avg_inv_loss:.4f}")
            print(f"  Metrics -> Feasibility Err: {avg_feasibility_error:.4f} | BC_KL: {avg_bc_kl:.4f} | Z_Var: {avg_z_var:.4f}")
            print(f"  Legality Audit: {proposed_actions - illegal_dropped}/{proposed_actions} legal actions ({illegal_rate:.1f}% illegal)")
            print(f"  Avg Episode Return: {last_avg_ep_reward:.2f} (Completed {int(global_ep_count)} episodes this update)")

            writer.add_scalar("Perf/SPS", sps, update)
            writer.add_scalar("Reward/Extrinsic_Manager", avg_extrinsic_step_reward, update)
            writer.add_scalar("Reward/Avg_Episodic_Return", last_avg_ep_reward, update)
            writer.add_scalar("Reward/Intrinsic_Worker", avg_intrinsic_reward, update)
            writer.add_scalar("Loss/Total_Loss", avg_total_loss, update)
            writer.add_scalar("Loss/PG_Surrogate", avg_pg_loss, update)
            writer.add_scalar("Loss/Manager_InfoNCE", avg_manager_loss, update)
            writer.add_scalar("Metrics/Manager_Feasibility_Error", avg_feasibility_error, update)
            writer.add_scalar("Loss/BC_KL_Penalty", avg_bc_kl, update)
            writer.add_scalar("Loss/Entropy", avg_entropy, update)
            writer.add_scalar("Metrics/Illegal_Action_Rate", illegal_rate, update)
            writer.add_scalar("Loss/Inverse_Model_InfoNCE", avg_inv_loss, update)
            writer.add_scalar("Metrics/Z_Achieved_Variance", avg_z_var, update)
            writer.add_scalar("Metrics/Inverse_Grad_Norm", avg_inv_grad_norm, update)
            writer.add_scalar("Hyperparameters/C_INT", current_c_int, update)

            wandb.log({
                "Perf/SPS": sps,
                "Reward/Extrinsic_Manager": avg_extrinsic_step_reward,
                "Reward/Total_Step_Reward": avg_total_step_reward,
                "Reward/Avg_Episodic_Return": last_avg_ep_reward,
                "Reward/Intrinsic_Worker": avg_intrinsic_reward,
                "Loss/Total_Loss": avg_total_loss,
                "Loss/PG_Surrogate": avg_pg_loss,
                "Loss/Manager_InfoNCE": avg_manager_loss,
                "Metrics/Manager_Feasibility_Error": avg_feasibility_error,
                "Loss/BC_KL_Penalty": avg_bc_kl,
                "Loss/Entropy": avg_entropy,
                "Loss/KL_Div": global_avg_kl,
                "Metrics/Illegal_Action_Rate": illegal_rate,
                "Loss/Inverse_Model_InfoNCE": avg_inv_loss,
                "Metrics/Z_Achieved_Variance": avg_z_var,
                "Metrics/Inverse_Grad_Norm": avg_inv_grad_norm,
                "Hyperparameters/C_INT": current_c_int,
                "Metrics/Proposed_Actions": proposed_actions,
                "global_step": update * global_steps,
            }, step=update)

            if update % 1 == 0:
                ckpt_path = os.path.join(ckpt_dir, f"diplomacy_APPO_update_{update}.pth")
                if global_rank == 0:
                    checkpoint = {
                        'update': update,
                        'model_state_dict': net.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                    }
                    print(f"  -> Saved checkpoint to {ckpt_path}")
                    torch.save(checkpoint, ckpt_path)
            
        EVAL_FREQ = 10
        
        if update % EVAL_FREQ == 0:
            eval_start_time = time.time()
            torch.cuda.empty_cache()
            
            attention_cache = []
            worker_attn_cache = []
            hook_handle = None
            worker_hook_handle = None
            if global_rank == 0:
                def get_attn_hook(module, inp, out):
                    attention_cache.clear()
                    attention_cache.append(out[1].detach().cpu().numpy())
                
                def get_worker_attn_hook(module, inp, out):
                    worker_attn_cache.clear()
                    worker_attn_cache.append(out[1].detach().cpu().numpy())
                
                # Attach hook to the MacroManager's cross attention layer
                hook_handle = net.module.manager.cross_attn.register_forward_hook(get_attn_hook)

                worker_hook_handle = net.module.worker.strategy_cross_attn.register_forward_hook(get_worker_attn_hook)

            world_size = dist.get_world_size()
            my_powers = [p for i, p in enumerate(possible_agents) if i % world_size == global_rank]
            
            local_eval_scs = torch.zeros(NUM_AGENTS, dtype=torch.float32, device=device)
            
            net.eval()
            GAMES_PER_POWER = 3
            total_eval_games = 7 * GAMES_PER_POWER
            
            # Calculate Categorical Outcomes
            solos, survivals, eliminations = 0, 0, 0

            for power in my_powers:
                power_sc_sum = 0
                for game_idx in range(GAMES_PER_POWER):
                    sc = evaluate_against_baseline(
                        live_net=net.module,
                        baseline_net=bc_baseline_net,
                        device=device, 
                        update_num=update, 
                        live_power=power, 
                        game_index=game_idx + 1, 
                        save_dir=eval_dir
                    )
                    power_sc_sum += float(sc)
                    
                    if sc >= 18:
                        solos += 1
                    elif sc > 0:
                        survivals += 1
                    else:
                        eliminations += 1
                        
                local_eval_scs[agent_to_idx[power]] = power_sc_sum / GAMES_PER_POWER
            net.train()

            dist.all_reduce(local_eval_scs, op=dist.ReduceOp.SUM)
            
            local_outcomes = torch.tensor([solos, survivals, eliminations], dtype=torch.float32, device=device)
            dist.all_reduce(local_outcomes, op=dist.ReduceOp.SUM)
            
            if global_rank == 0:
                avg_eval_sc = local_eval_scs.mean().item()
                eval_duration = time.time() - eval_start_time

                total_solos, total_survivals, total_elims = local_outcomes.tolist()
                total_games = float(NUM_AGENTS * GAMES_PER_POWER)
                        
                solo_rate = (total_solos / total_games) * 100.0
                survival_rate = (total_survivals / total_games) * 100.0
                elimination_rate = (total_elims / total_games) * 100.0

                print(f"\n--- Evaluation Results (Update {update}) ---")
                for i, power in enumerate(possible_agents):
                    print(f"  {power}: {local_eval_scs[i].item():.1f} SCs")
                print(f"  Average SCs: {avg_eval_sc:.2f}")
                print(f"  Outcomes -> Solo: {solo_rate:.1f}% | Survive: {survival_rate:.1f}% | Eliminated: {elimination_rate:.1f}%")
                print(f"  Eval Duration: {eval_duration:.2f}s\n")
                
                wandb.log({
                    "Eval/Avg_SCs": avg_eval_sc,
                    "Eval_Metrics/Solo_Rate_Pct": solo_rate,
                    "Eval_Metrics/Survival_Rate_Pct": survival_rate,
                    "Eval_Metrics/Elimination_Rate_Pct": elimination_rate,
                    "Perf/Eval_Time_s": eval_duration,
                    "global_step": update * global_steps
                }, step=update)
                
                for i, power in enumerate(possible_agents):
                    wandb.log({f"Eval/{power}_SCs": local_eval_scs[i].item()}, step=update)
                
                if hook_handle is not None:
                    hook_handle.remove() 
                
                if len(attention_cache) > 0:
                    attn_matrix = attention_cache[0][0] 
                    
                    # Intercept the raw values in the console
                    print(f"\n[DEBUG] MacroManager Attn - Max: {attn_matrix.max():.4f}, Min: {attn_matrix.min():.4f}")
                    
                    plt.figure(figsize=(8, 6))
                    # REMOVED vmin/vmax to force auto-scaling. ADDED annot=True to see the math.
                    sns.heatmap(attn_matrix, cmap="viridis", annot=True, fmt=".3f")
                    plt.title(f"MacroManager Cross-Attention - Update {update}")
                    plt.xlabel("Key (S_M Theaters)")
                    plt.ylabel("Query (z_prev + H_t)")
                    
                    wandb.log({"Attention/MacroManager_Map": wandb.Image(plt)}, step=update)
                    plt.close()

                # --- UPDATE 2: TacticalWorker Plot ---
                if worker_hook_handle is not None:
                    worker_hook_handle.remove()
                    
                if len(worker_attn_cache) > 0:
                    w_attn_matrix = worker_attn_cache[0][0] 
                    
                    # Intercept the raw values in the console
                    print(f"[DEBUG] TacticalWorker Attn - Max: {w_attn_matrix.max():.4f}, Min: {w_attn_matrix.min():.4f}\n")
                    
                    plt.figure(figsize=(6, 12)) 
                    # REMOVED vmin/vmax. (annot=False left intact because 82 rows of numbers is unreadable)
                    sns.heatmap(w_attn_matrix, cmap="magma")
                    plt.title(f"Worker Strategy Execution - Update {update}")
                    plt.xlabel("Strategy Vectors (z_t)")
                    plt.ylabel("Provinces (S_mu)")
                    
                    wandb.log({"Attention/TacticalWorker_Map": wandb.Image(plt)}, step=update)
                    plt.close()
                
                if flat_H.shape[0] > 0:
                    sample_H = flat_H[-1].detach().cpu().numpy()
                    
                    plt.figure(figsize=(7, 6))
                    sns.heatmap(sample_H, cmap="RdBu", annot=True, fmt=".1f", vmin=-1.0, vmax=1.0,
                                xticklabels=possible_agents, yticklabels=possible_agents)
                    plt.title(f"Sample Diplomatic Belief Matrix (H) - Update {update}")
                    
                    wandb.log({"Attention/Diplomatic_Matrix_H": wandb.Image(plt)}, step=update)
                    plt.close()
                    
            torch.cuda.empty_cache()

    rollout_process.join()
    if global_rank == 0:
        writer.close()
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()