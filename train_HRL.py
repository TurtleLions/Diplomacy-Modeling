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
import concurrent.futures
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
import glob

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
    
    # Find the actual maximum number of units in this specific batch
    max_active = int(is_active_mask.sum(dim=1).max().item())
    
    BIN_SIZE = 8
    if max_active > 0:
        max_active = ((max_active + BIN_SIZE - 1) // BIN_SIZE) * BIN_SIZE
        
    max_active = min(max_active, num_provs)
    
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

def worker(remote, parent_remote, env_idx, shared_obs, shared_masks, shared_H):
    """Background worker process using PyTorch Shared Memory."""
    torch.set_num_threads(1)
    parent_remote.close()
    env = DiplomacyTransformerEnv(history_length=1) 
    
    episode_rewards = {a: 0.0 for a in env.possible_agents}
    agent_to_idx = {a: i for i, a in enumerate(env.possible_agents)}
    
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

                for a in env.agents:
                    a_idx = agent_to_idx[a]
                    shared_obs[env_idx, a_idx].copy_(torch.from_numpy(obs[a]))
                    shared_masks[env_idx, a_idx].copy_(torch.from_numpy(infos[a]['action_mask']))
                    shared_H[env_idx, a_idx].copy_(torch.from_numpy(infos[a]['H_matrix']))
                    
                    del obs[a]
                    del infos[a]['action_mask']
                    del infos[a]['H_matrix']

                remote.send((obs, rewards, terms, truncs, infos, env.agents))

            elif cmd == 'reset': 
                episode_rewards = {a: 0.0 for a in env.possible_agents}
                obs, infos = env.reset()
                
                for a in env.agents:
                    a_idx = agent_to_idx[a]
                    shared_obs[env_idx, a_idx].copy_(torch.from_numpy(obs[a]))
                    shared_masks[env_idx, a_idx].copy_(torch.from_numpy(infos[a]['action_mask']))
                    shared_H[env_idx, a_idx].copy_(torch.from_numpy(infos[a]['H_matrix']))
                    
                    del obs[a]
                    del infos[a]['action_mask']
                    del infos[a]['H_matrix']

                remote.send((obs, infos, env.agents))

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
    """Asynchronous vector environment using IPC pipes + Shared Memory for parallel simulation."""
    def __init__(self, num_envs):
        self.num_envs = num_envs
        
        # Pre-allocate massive shared memory blocks for all workers to write to
        NUM_AGENTS = 7
        self.shared_obs = torch.zeros((num_envs, NUM_AGENTS, 1, 82, 61), dtype=torch.int8).share_memory_()
        self.shared_masks = torch.full((num_envs, NUM_AGENTS, 4000), -1, dtype=torch.int32).share_memory_()
        self.shared_H = torch.zeros((num_envs, NUM_AGENTS, 7, 7), dtype=torch.float32).share_memory_()
        
        self.possible_agents = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
        self.agent_to_idx = {a: i for i, a in enumerate(self.possible_agents)}

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.ps = [
            mp.Process(
                target=worker, 
                args=(work_remote, remote, i, self.shared_obs, self.shared_masks, self.shared_H)
            ) 
            for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes))
        ]
        
        for p in self.ps:
            p.daemon = True 
            p.start()
        for remote in self.work_remotes: 
            remote.close()

    def _reconstruct_data(self, idx, msg):
        """Zero-Copy Reattachment: Restores arrays to the dictionaries instantly from RAM."""
        if len(msg) == 6:
            obs, rewards, terms, truncs, infos, active_agents = msg
            for a in active_agents:
                a_idx = self.agent_to_idx[a]
                obs[a] = self.shared_obs[idx, a_idx].numpy()
                infos[a]['action_mask'] = self.shared_masks[idx, a_idx].numpy()
                infos[a]['H_matrix'] = self.shared_H[idx, a_idx].numpy()
            return (obs, rewards, terms, truncs, infos, active_agents)
        else:
            obs, infos, active_agents = msg
            for a in active_agents:
                a_idx = self.agent_to_idx[a]
                obs[a] = self.shared_obs[idx, a_idx].numpy()
                infos[a]['action_mask'] = self.shared_masks[idx, a_idx].numpy()
                infos[a]['H_matrix'] = self.shared_H[idx, a_idx].numpy()
            return (obs, infos, active_agents)

    def _gather_results(self):
        """Safely polls ready pipes to prevent OS buffer lockups."""
        results = [None] * self.num_envs
        remotes_left = {remote: i for i, remote in enumerate(self.remotes)}
        
        while remotes_left:
            ready_remotes = multiprocessing.connection.wait(remotes_left.keys())
            
            for remote in ready_remotes:
                idx = remotes_left[remote]
                msg = remote.recv()
                
                results[idx] = self._reconstruct_data(idx, msg)
                del remotes_left[remote]
                
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
    
    live_h_memory = {a: torch.zeros((1, 8, 512), dtype=torch.bfloat16, device=device) for a in env.possible_agents}

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
                    z_eval = torch.zeros((B, 8, 512), dtype=torch.bfloat16, device=net_device)
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
                
                logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
                logits = torch.clamp(logits, min=-50.0, max=50.0)
                logits = logits.float().masked_fill(~dense_mask, -1e20)
                if is_baseline:
                    dist_eval = Categorical(logits=logits)
                    final_actions = dist_eval.sample()
                else:
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
    parser.add_argument("--num_envs", type=int, default=35, help="Number of parallel environments per GPU")
    parser.add_argument("--num_steps", type=int, default=512, help="Number of steps per rollout")
    parser.add_argument("--num_updates", type=int, default=1000, help="Total number of PPO updates")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO policy clipping coefficient")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--v_coef", type=float, default=0.1, help="Value function loss coefficient")
    parser.add_argument("--kl_coef", type=float, default=0.01, help="KL divergence penalty coefficient")
    parser.add_argument("--update_epochs", type=int, default=3, help="Number of epochs per PPO update")
    parser.add_argument("--bc_weights", type=str, default="feudal_agent_bc.pth", help="Path to pre-trained Behavioral Cloning weights")
    parser.add_argument("--resume_weights", type=str, default=None, help="Path to RL checkpoint to resume training from")
    parser.add_argument("--bc_kl_coef", type=float, default=0.01, help="KL divergence penalty coefficient for behavioral cloning")
    parser.add_argument("--run_name", type=str, default=None, help="Target run name (e.g., ppo_run_20260427_164600) to resume wandb logging and folder paths.")
    return parser.parse_args()

def rollout_worker(local_rank, device, args, buffers, actor_net, bc_baseline_net, history_nets, ckpt_dir,
                   free_buffers_queue, ready_buffers_queue, worker_stats, 
                   NUM_AGENTS, MAP_PROVINCES, VOCAB_SIZE, NONE_IDX, possible_agents, agent_to_idx,
                   actor_weights_lock, start_update=1):
    """Background process responsible for filling the experience buffer asynchronously."""
    torch.cuda.set_device(device) 
    inference_stream = torch.cuda.Stream(device=device, priority=1)
    
    vec_env = SubprocVecDiplomacy(num_envs=args.num_envs)
    
    buffer_idx = 0
    next_env_results = vec_env.reset()
    next_done = torch.zeros((args.num_envs, NUM_AGENTS), device=device)
    
    batch_H = torch.zeros((args.num_envs, NUM_AGENTS, 7, 7), dtype=torch.float32, device=device)
    batch_h = torch.zeros((args.num_envs, NUM_AGENTS, 8, 512), dtype=torch.bfloat16, device=device)
    batch_prev_h = torch.zeros((args.num_envs, NUM_AGENTS, 8, 512), dtype=torch.bfloat16, device=device)
    batch_S_M = torch.zeros((args.num_envs, NUM_AGENTS, 8, 512), dtype=torch.bfloat16, device=device)

    NUM_HISTORY_NETS = len(history_nets)

    def refresh_historical_pool():
        available_ckpts = glob.glob(os.path.join(ckpt_dir, "*.pth"))
        if not available_ckpts: 
            if local_rank == 0:
                print("  [League] No historical checkpoints found yet. History nets retaining base weights.")
            return
            
        loaded_ckpts = []
        for net in history_nets:
            chosen_ckpt = random.choice(available_ckpts)
            loaded_ckpts.append(os.path.basename(chosen_ckpt))
            try:
                state_dict = torch.load(chosen_ckpt, map_location='cpu', weights_only=True)
                if 'model_state_dict' in state_dict:
                    clean_dict = {k.replace('module.', ''): v for k, v in state_dict['model_state_dict'].items()}
                    net.load_state_dict(clean_dict, strict=False)
                else:
                    net.load_state_dict(state_dict, strict=False)
            except Exception as e:
                if local_rank == 0:
                    print(f"  [League Error] Failed to load {os.path.basename(chosen_ckpt)}: {e}")
                pass
                
        if local_rank == 0:
            print(f"  [League Refresh] Opponents loaded: {', '.join(loaded_ckpts)}")

    for update in range(start_update, args.num_updates + 1):
        buffer_idx = free_buffers_queue.get()

        if update % 5 == 0 or update == start_update:
            refresh_historical_pool()

        num_learning = 2

        opp_probs = [0.2] + [0.8 / max(1, NUM_HISTORY_NETS)] * NUM_HISTORY_NETS if NUM_HISTORY_NETS > 0 else [1.0]
        opp_probs = np.array(opp_probs) / sum(opp_probs)

        net_assignment = torch.zeros((args.num_envs, NUM_AGENTS), dtype=torch.long, device=device)
        learning_assignment = torch.zeros((args.num_envs, NUM_AGENTS), dtype=torch.bool, device=device)
        
        for i in range(args.num_envs):
            perm = torch.randperm(NUM_AGENTS, device=device)
            actor_agents = perm[:num_learning]
            net_assignment[i, actor_agents] = 1
            learning_assignment[i, actor_agents] = True
            
            for a_idx in perm[num_learning:]:
                opp_type = np.random.choice([0] + list(range(2, 2 + NUM_HISTORY_NETS)), p=opp_probs)
                net_assignment[i, a_idx] = opp_type

        current_progress = (update - 1) / max(1, args.num_updates - 1)
        buf = buffers[buffer_idx]
        
        with torch.cuda.stream(inference_stream):
            buf['masks'].zero_()
            buf['extrinsic_rewards'].zero_()
            buf['intrinsic_rewards'].zero_()
            
            env_step_time, gpu_forward_time = 0.0, 0.0
            local_proposed, local_dropped = 0, 0
            local_ep_reward_sum, local_ep_count = 0.0, 0
            terminal_cache = []
        
            for step in range(args.num_steps):
                actions_to_send = [{} for _ in range(args.num_envs)]
                
                # Dictionary to group states by assigned Network ID
                net_obs_lists = {k: [] for k in range(2 + NUM_HISTORY_NETS)}
                net_sparse_lists = {k: [] for k in range(2 + NUM_HISTORY_NETS)}
                net_indices = {k: [] for k in range(2 + NUM_HISTORY_NETS)}
                
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
                            
                            net_id = net_assignment[i, a_idx].item()
                            net_obs_lists[net_id].append(obs_dict[a][0])
                            net_sparse_lists[net_id].append(infos_dict[a]['action_mask'])
                            net_indices[net_id].append((i, a_idx))

                            if net_id == 1:
                                buf['masks'][step, i, a_idx] = True
                                if 'legality_metrics' in infos_dict[a]:
                                    bad_indices = infos_dict[a]['legality_metrics'].get('illegal_prov_indices', [])
                                    for bad_idx in bad_indices:
                                        buf['unit_penalties'][step, i, a_idx, bad_idx] = -1.0
                                    local_proposed += infos_dict[a]['legality_metrics'].get('proposed', 0)
                                    local_dropped += infos_dict[a]['legality_metrics'].get('illegal_dropped', 0)

                    if h_matrix_list:
                        h_env_idx = [idx[0] for idx in h_indices]
                        h_agt_idx = [idx[1] for idx in h_indices]
                        batch_H[h_env_idx, h_agt_idx] = torch.tensor(np.stack(h_matrix_list), dtype=torch.float32, device=device)

                    if net_obs_lists[1]:
                        act_env_idx = [idx[0] for idx in net_indices[1]]
                        act_agt_idx = [idx[1] for idx in net_indices[1]]
                        
                        batched_active_obs = torch.tensor(np.stack(net_obs_lists[1]), dtype=torch.int8, device=device)
                        batched_active_sparse = torch.tensor(np.stack(net_sparse_lists[1]), dtype=torch.int32, device=device)
                        
                        buf['obs'][step, act_env_idx, act_agt_idx] = batched_active_obs
                        buf['sparse_masks'][step, act_env_idx, act_agt_idx] = batched_active_sparse

                    t_gpu_start = time.time()

                    flat_obs = buf['obs'][step][buf['masks'][step]].to(torch.bfloat16)
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
                                
                                logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
                                logits = torch.clamp(logits, min=-50.0, max=50.0)
                                
                                pooled_S_M = active_S_M.mean(dim=1)
                                ext_values = actor_net.extrinsic_value_head(pooled_S_M).squeeze(-1).float()
                                int_values = actor_net.intrinsic_value_head(pooled_S_M).squeeze(-1).float()
                        
                        batch_S_M[buf['masks'][step]] = active_S_M
                        batch_prev_h[buf['masks'][step]] = active_h
                        batch_h[buf['masks'][step]] = new_h
                        
                        buf['S_M'][step][buf['masks'][step]] = active_S_M
                        buf['z'][step][buf['masks'][step]] = new_z
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
                        
                        buf['ext_values'][step][buf['masks'][step]] = ext_values
                        buf['int_values'][step][buf['masks'][step]] = int_values
                        buf['logprobs'][step][buf['masks'][step]] = final_logprobs
                        
                        idx_counter = 0
                        for i in range(args.num_envs):
                            for a in possible_agents:
                                if buf['masks'][step, i, agent_to_idx[a]]:
                                    act_array = final_actions[idx_counter]
                                    buf['actions'][step, i, agent_to_idx[a]] = act_array
                                    actions_to_send[i][a] = act_array.cpu().numpy()
                                    idx_counter += 1

                    with torch.no_grad():
                        for net_id in range(2 + NUM_HISTORY_NETS):
                            if net_id == 1 or not net_obs_lists[net_id]: continue 
                            
                            opp_obs = torch.tensor(np.stack(net_obs_lists[net_id]), dtype=torch.int8, device=device).to(torch.bfloat16)
                            opp_sparse = torch.tensor(np.stack(net_sparse_lists[net_id]), dtype=torch.int32, device=device)
                            
                            env_idxs = [idx[0] for idx in net_indices[net_id]]
                            agt_idxs = [idx[1] for idx in net_indices[net_id]]
                            
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                if net_id == 0:
                                    
                                    opp_z = torch.zeros(opp_obs.size(0), 8, 512, device=device, dtype=torch.bfloat16)
                                    opp_logits, _ = bc_baseline_net.worker(opp_obs, opp_z, bc_baseline_net.D)
                                else:
                                    net = history_nets[net_id - 2]
                                    opp_H = batch_H[env_idxs, agt_idxs]
                                    opp_prev_h = batch_h[env_idxs, agt_idxs]
                                    
                                    opp_x_emb = net.worker.feature_projection(opp_obs)
                                    opp_S_mu = net.worker.encoder_transformer(opp_x_emb)
                                    opp_S_M = net.pooler(opp_S_mu)
                                    
                                    opp_z, opp_new_h = net.manager(opp_S_M, opp_H, opp_prev_h)
                                    opp_z = opp_z.to(torch.bfloat16)
                                    
                                    batch_prev_h[env_idxs, agt_idxs] = opp_prev_h
                                    batch_h[env_idxs, agt_idxs] = opp_new_h
                                    
                                    opp_logits, _ = net.worker(opp_obs, opp_z, net.D)

                            # Masking and Sampling 
                            opp_logits = torch.nan_to_num(opp_logits, nan=0.0, posinf=50.0, neginf=-50.0)
                            opp_logits = torch.clamp(opp_logits, min=-50.0, max=50.0)
                            
                            valid_mask_opp = opp_sparse != -1
                            row_offsets_opp = torch.arange(opp_obs.size(0), device=device).unsqueeze(1) * (MAP_PROVINCES * VOCAB_SIZE)
                            global_indices_opp = opp_sparse + row_offsets_opp
                            valid_global_indices_opp = global_indices_opp[valid_mask_opp]

                            dense_mask_opp = torch.zeros(opp_obs.size(0) * MAP_PROVINCES * VOCAB_SIZE, dtype=torch.bool, device=device)
                            dense_mask_opp[valid_global_indices_opp.long()] = True
                            dense_mask_opp = dense_mask_opp.view(opp_obs.size(0), MAP_PROVINCES, VOCAB_SIZE)
                            
                            opp_logits = opp_logits.float().masked_fill(~dense_mask_opp, -1e20)
                            opp_dist = Categorical(logits=opp_logits)
                            opp_final_actions = opp_dist.sample()

                            for list_idx, (env_idx, agt_idx) in enumerate(net_indices[net_id]):
                                agt_name = possible_agents[agt_idx]
                                actions_to_send[env_idx][agt_name] = opp_final_actions[list_idx].cpu().numpy()

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
                                    batch_h[i, agent_to_idx[a]].zero_()
                                    batch_prev_h[i, agent_to_idx[a]].zero_()
                                    batch_S_M[i, agent_to_idx[a]].zero_()

                                    if learning_assignment[i, agent_to_idx[a]]:
                                        terminal_obs_to_encode.append(infos_dict[a]['__terminal_observation'][0])
                                        terminal_indices.append((i, agent_to_idx[a]))
                            
                            for a in possible_agents:
                                if learning_assignment[i, agent_to_idx[a]]:
                                    buf['extrinsic_rewards'][step, i, agent_to_idx[a]] = step_rewards.get(a, 0.0)
                                    buf['dones'][step, i, agent_to_idx[a]] = float(terms.get(a, False))
                                    buf['truncations'][step, i, agent_to_idx[a]] = float(truncs.get(a, False))
                                    next_done[i, agent_to_idx[a]] = float(terms.get(a, False))

                    if terminal_obs_to_encode:
                        term_tensor = torch.tensor(np.stack(terminal_obs_to_encode), dtype=torch.bfloat16, device=device)
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            x_emb_t = actor_net.worker.feature_projection(term_tensor)
                            S_M_t = actor_net.pooler(actor_net.worker.encoder_transformer(x_emb_t))
                            term_v_ext = actor_net.extrinsic_value_head(S_M_t.mean(dim=1)).float().squeeze(-1)
                            term_v_int = actor_net.intrinsic_value_head(S_M_t.mean(dim=1)).float().squeeze(-1)
                        for list_idx, (env_idx, agent_idx) in enumerate(terminal_indices):
                            buf['ext_terminal_values'][step, env_idx, agent_idx] = term_v_ext[list_idx]
                            buf['int_terminal_values'][step, env_idx, agent_idx] = term_v_int[list_idx]
                            terminal_cache.append((step, env_idx, agent_idx, S_M_t[list_idx].detach().cpu()))

        # Calculate GAE Advantages
        with torch.no_grad():
            next_ext_value = torch.zeros((args.num_envs, NUM_AGENTS), device=device)
            next_int_value = torch.zeros((args.num_envs, NUM_AGENTS), device=device)
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
                    next_ext_v = actor_net.extrinsic_value_head(pooled_S_M_next).float()
                    next_int_v = actor_net.intrinsic_value_head(pooled_S_M_next).float()
            
            for list_idx, (env_idx, agent_idx) in enumerate(indices_to_update):
                buf['ext_terminal_values'][step, env_idx, agent_idx] = next_ext_v[list_idx].float().squeeze(-1)
                buf['int_terminal_values'][step, env_idx, agent_idx] = next_int_v[list_idx].float().squeeze(-1)
                next_ext_value[env_idx, agent_idx] = next_ext_v[list_idx].float().squeeze(-1)
                next_int_value[env_idx, agent_idx] = next_int_v[list_idx].float().squeeze(-1)
            buf['S_M_next'][:-1] = buf['S_M'][1:]
            
            if obs_to_encode:
                for list_idx, (env_idx, agent_idx) in enumerate(indices_to_update):
                    buf['S_M_next'][-1, env_idx, agent_idx] = S_M_next[list_idx]

            for t_step, env_idx, agent_idx, term_sm in terminal_cache:
                buf['S_M_next'][t_step, env_idx, agent_idx] = term_sm.to(device)
                
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                flat_S_M = buf['S_M'].view(-1, 8, 512)
                flat_S_M_next = buf['S_M_next'].view(-1, 8, 512)
                flat_z = buf['z'].view(-1, 8, 512)
                
                intrinsic_rewards_flat = torch.zeros(flat_S_M.size(0), device=device)
                chunk_size = 8192
                
                for idx in range(0, flat_S_M.size(0), chunk_size):
                    end_idx = idx + chunk_size
                    z_ach_chunk = actor_net.inverse_model(flat_S_M[idx:end_idx], flat_S_M_next[idx:end_idx])
                    # Compute chunk similarity immediately to avoid massive memory buildup
                    sim_chunk = F.cosine_similarity(z_ach_chunk.float(), flat_z[idx:end_idx].float(), dim=-1).mean(dim=1)
                    intrinsic_rewards_flat[idx:end_idx] = sim_chunk
                
                # Reshape back to buffer dimensions
                intrinsic_rewards = intrinsic_rewards_flat.view(args.num_steps, args.num_envs, NUM_AGENTS)

                intrinsic_rewards.masked_fill_(~buf['masks'], 0.0)

                buf['intrinsic_rewards'].copy_(intrinsic_rewards)
                
        lastgaelam_ext, lastgaelam_int = 0, 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                next_ext_v = next_ext_value
                next_int_v = next_int_value
            else:
                nextnonterminal = 1.0 - buf['dones'][t]
                is_trunc = buf['truncations'][t] == 1.0
                next_ext_v = torch.where(is_trunc, buf['ext_terminal_values'][t], buf['ext_values'][t + 1])
                next_int_v = torch.where(is_trunc, buf['int_terminal_values'][t], buf['int_values'][t + 1])
                
            delta_ext = buf['extrinsic_rewards'][t] + args.gamma * next_ext_v * nextnonterminal - buf['ext_values'][t]
            buf['ext_advantages'][t] = lastgaelam_ext = delta_ext + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam_ext
            
            delta_int = buf['intrinsic_rewards'][t] + args.gamma * next_int_v * nextnonterminal - buf['int_values'][t]
            buf['int_advantages'][t] = lastgaelam_int = delta_int + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam_int

        buf['ext_returns'].copy_(buf['ext_advantages'] + buf['ext_values'])
        buf['int_returns'].copy_(buf['int_advantages'] + buf['int_values'])

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
    net = FeudalDiplomacyAgent(d_model=512, vocab_size=VOCAB_SIZE).to(device)
    actor_net = FeudalDiplomacyAgent(d_model=512, vocab_size=VOCAB_SIZE).to(device)
    bc_baseline_net = FeudalDiplomacyAgent(d_model=512, vocab_size=VOCAB_SIZE).to(device)

    NUM_HISTORY_NETS = 3
    history_nets = []
    
    temp_game = Game()
    provinces = sorted([p.upper() for p in list(temp_game.map.locs)])
    
    D_matrix = build_distance_matrix(provinces).to(device)
    net.D.copy_(D_matrix)
    actor_net.D.copy_(D_matrix)
    bc_baseline_net.D.copy_(D_matrix)
    
    # Initialize Historical Opponent Pool
    for _ in range(NUM_HISTORY_NETS):
        net_h = FeudalDiplomacyAgent(d_model=512, vocab_size=VOCAB_SIZE).to(device)
        net_h.D.copy_(D_matrix)
        net_h.eval()
        for param in net_h.parameters(): param.requires_grad = False
        net_h.share_memory()
        history_nets.append(net_h)
    
    if os.path.exists(args.bc_weights):
        bc_state_dict = torch.load(args.bc_weights, map_location=device)
        filtered_state_dict = {k: v for k, v in bc_state_dict.items() if 'inverse_model' not in k}
        net.load_state_dict(filtered_state_dict, strict=False)
        
        actor_net.load_state_dict(net.state_dict())
        bc_baseline_net.load_state_dict(filtered_state_dict, strict=False)
        
        for net_h in history_nets:
            net_h.load_state_dict(filtered_state_dict, strict=False)

        actor_net.eval()
        for param in actor_net.parameters(): param.requires_grad = False
            
        if global_rank == 0: 
            print("Successfully loaded pre-trained BC weights for policy initialization.")

    loaded_opt_state = None
    loaded_sched_state = None
    start_update = 1

    if args.resume_weights and os.path.exists(args.resume_weights):
        checkpoint = torch.load(args.resume_weights, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'], strict=False)
            actor_net.load_state_dict(checkpoint['model_state_dict'], strict=False)
            loaded_opt_state = checkpoint['optimizer_state_dict']
            loaded_sched_state = checkpoint.get('scheduler_state_dict')
            start_update = checkpoint.get('update', 0) + 1 
        else:
            net.load_state_dict(checkpoint, strict=False)
            actor_net.load_state_dict(checkpoint, strict=False)
        
            
        if global_rank == 0:
            print(f"Successfully resumed RL training from checkpoint: {args.resume_weights}")

    net = DDP(net, device_ids=[local_rank], find_unused_parameters=False, bucket_cap_mb=256, broadcast_buffers=False)

    manager_params = []
    worker_params = []
    for name, param in net.named_parameters():
        if 'worker' in name:
            worker_params.append(param)
        else:
            manager_params.append(param)

    optimizer = optim.Adam([
        {'params': manager_params, 'lr': args.lr},
        {'params': worker_params, 'lr': args.lr * 0.05}
    ], eps=1e-5, fused=True)
    
    def manager_warmup(update):
        warmup_updates = 10
        if update < warmup_updates:
            return float(max(1, update)) / float(warmup_updates)
        return 1.0

    def worker_warmup(update):
        unfreeze_update = 11
        warmup_updates = 10
        freeze_again_update = 600 
    
        if update < unfreeze_update:
            return 0.0 
            
        if update >= freeze_again_update:
            return 0.0
            
        active_steps = update - unfreeze_update + 1
        if active_steps < warmup_updates:
            return float(active_steps) / float(warmup_updates)
            
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda=[manager_warmup, worker_warmup])
    
    if loaded_opt_state is not None:
        optimizer.load_state_dict(loaded_opt_state)
        if global_rank == 0:
            print("Successfully restored Optimizer momentum and variance states.")

    if loaded_sched_state is not None:
        scheduler.load_state_dict(loaded_sched_state)
        if global_rank == 0:
            print("Successfully restored Learning Rate Scheduler state.")

    def create_buffer():
        """Creates a memory-pinned tensor buffer for experience collection."""
        return {
            'obs': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 82, 61), dtype=torch.int8, device=device),
            'actions': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 82), dtype=torch.int16, device=device),
            'logprobs': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 82), dtype=torch.float32, device=device),
            'extrinsic_rewards': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'intrinsic_rewards': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'dones': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'ext_values': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'int_values': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'masks': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.bool, device=device),
            'truncations': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'ext_terminal_values': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'int_terminal_values': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'sparse_masks': torch.full((args.num_steps, args.num_envs, NUM_AGENTS, 4000), -1, dtype=torch.int32, device=device),
            'ext_advantages': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'int_advantages': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'ext_returns': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'int_returns': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'S_M': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 8, 512), dtype=torch.bfloat16, device=device),
            'S_M_next': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 8, 512), dtype=torch.bfloat16, device=device),
            'z': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 8, 512), dtype=torch.bfloat16, device=device),
            'prev_h': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 8, 512), dtype=torch.bfloat16, device=device),
            'H': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 7, 7), dtype=torch.float32, device=device),
            'unit_penalties': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 82), dtype=torch.float32, device=device),
        }
        
    # Double-buffering architecture masks CPU environment latency behind GPU backpropagation
    buffers = {0: create_buffer(), 1: create_buffer()}
    
    free_buffers_queue = mp.Queue()
    ready_buffers_queue = mp.Queue()
    
    free_buffers_queue.put(0)
    free_buffers_queue.put(1)

    thread_stats = {"env_time": 0.0, "gpu_fwd_time": 0.0, "proposed": 0, "dropped": 0}

    actor_net.share_memory()
    bc_baseline_net.share_memory()

    worker_stats = torch.zeros(6, dtype=torch.float32, device=device).share_memory_()
    last_avg_ep_reward = 0.0
    agent_to_idx = {a: i for i, a in enumerate(possible_agents)}

    inference_stream = torch.cuda.Stream(device=device)
    actor_weights_lock = mp.Lock()

    if args.run_name:
        run_name = args.run_name
    else:
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

    rollout_process = mp.Process(target=rollout_worker, args=(
        local_rank, device, args, buffers, actor_net, bc_baseline_net, history_nets, ckpt_dir,
        free_buffers_queue, ready_buffers_queue, worker_stats,
        NUM_AGENTS, MAP_PROVINCES, VOCAB_SIZE, NONE_IDX, possible_agents, agent_to_idx,
        actor_weights_lock, start_update
    ))
    rollout_process.start()

    if global_rank == 0:
        writer = SummaryWriter(log_dir=f"/data/restanislao/tb_runs/{run_name}")
        wandb.init(
            project="diplomacy-ppo",
            name=run_name,
            id=run_name,
            resume="allow",
            config=vars(args),
            dir="/data/restanislao/wandb"
        )
        wandb.watch(net.module, log="all", log_freq=10)

    dist.barrier()

    for update in range(start_update, args.num_updates + 1):
        start_time = time.time()
        
        buffer_idx = ready_buffers_queue.get()
        full_update_start = time.time()
        
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

        total_active_steps = buf['masks'].sum().item()
        if total_active_steps > 0:
            avg_extrinsic_step_reward = buf['extrinsic_rewards'].sum().item() / total_active_steps
            avg_intrinsic_step_reward = buf['intrinsic_rewards'].sum().item() / total_active_steps
            avg_total_step_reward = (buf['extrinsic_rewards'].sum().item() + buf['intrinsic_rewards'].sum().item()) / total_active_steps
        else:
            avg_extrinsic_step_reward = 0.0
            avg_intrinsic_step_reward = 0.0
            avg_total_step_reward = 0.0
        
        valid_indices = torch.nonzero(buf['masks'].view(-1), as_tuple=True)[0]
        b_size = valid_indices.shape[0]
        
        # DDP min-batch synchronization to prevent NCCL hanging
        local_b_size = torch.tensor([b_size], dtype=torch.long, device=device)
        dist.all_reduce(local_b_size, op=dist.ReduceOp.MIN)
        min_b_size = local_b_size.item()

        if b_size > min_b_size:
            perm = torch.randperm(b_size, device=device)
            epoch_indices = valid_indices[perm][:min_b_size]
        else:
            epoch_indices = valid_indices[torch.randperm(b_size, device=device)]
            
        b_size = min_b_size

        local_stats = torch.zeros((NUM_AGENTS, 5), dtype=torch.float32, device=device)
        
        if b_size > 1:
            for a_idx in range(NUM_AGENTS):
                agent_mask = (epoch_indices % NUM_AGENTS) == a_idx
                agent_indices = epoch_indices[agent_mask]
                
                count = agent_indices.numel()
                if count > 0:
                    agent_ext_adv = buf['ext_advantages'].view(-1)[agent_indices]
                    agent_int_adv = buf['int_advantages'].view(-1)[agent_indices]
                    
                    local_stats[a_idx, 0] = agent_ext_adv.sum()
                    local_stats[a_idx, 1] = (agent_ext_adv ** 2).sum()
                    local_stats[a_idx, 2] = agent_int_adv.sum()
                    local_stats[a_idx, 3] = (agent_int_adv ** 2).sum()
                    local_stats[a_idx, 4] = count

        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

        for a_idx in range(NUM_AGENTS):
            agent_count = local_stats[a_idx, 4]
            if agent_count > 1:
                agent_mask = (epoch_indices % NUM_AGENTS) == a_idx
                agent_indices = epoch_indices[agent_mask]
                
                if agent_indices.numel() > 0:
                    agent_ext_adv = buf['ext_advantages'].view(-1)[agent_indices]
                    agent_int_adv = buf['int_advantages'].view(-1)[agent_indices]
                    
                    ext_mean = local_stats[a_idx, 0] / agent_count
                    ext_var = (local_stats[a_idx, 1] / agent_count) - (ext_mean ** 2)
                    ext_std = torch.sqrt(torch.clamp(ext_var, min=1e-8))
                    buf['ext_advantages'].view(-1)[agent_indices] = (agent_ext_adv - ext_mean) / (ext_std + 1e-8)
                    
                    int_mean = local_stats[a_idx, 2] / agent_count
                    int_var = (local_stats[a_idx, 3] / agent_count) - (int_mean ** 2)
                    int_std = torch.sqrt(torch.clamp(int_var, min=1e-8))
                    buf['int_advantages'].view(-1)[agent_indices] = (agent_int_adv - int_mean) / (int_std + 1e-8)

        t_update_start = time.time()
        net.train()

        SEQ_LEN = 16
        assert args.num_steps % SEQ_LEN == 0
        num_chunks = args.num_steps // SEQ_LEN

        valid_sequences = []
        for c in range(num_chunks):
            start_t = c * SEQ_LEN
            end_t = start_t + SEQ_LEN
            for env_idx in range(args.num_envs):
                for a_idx in range(NUM_AGENTS):
                    if buf['masks'][start_t:end_t, env_idx, a_idx].any():
                        valid_sequences.append((start_t, env_idx, a_idx))
        
        b_size_seqs = len(valid_sequences)
        
        local_b_size_seqs = torch.tensor([b_size_seqs], dtype=torch.long, device=device)
        dist.all_reduce(local_b_size_seqs, op=dist.ReduceOp.MIN)
        min_b_size_seqs = local_b_size_seqs.item()

        if b_size_seqs > min_b_size_seqs:
            perm_seq = torch.randperm(b_size_seqs, device=device)
            epoch_seq_indices = [valid_sequences[i] for i in perm_seq[:min_b_size_seqs].tolist()]
        else:
            epoch_seq_indices = valid_sequences
            
        b_size_seqs = min_b_size_seqs

        mb_size_seqs = 32
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
        warmup_end = 11
        if update < warmup_end:
            current_c_int = 0.05
        else:
            decay_progress = min(1.0, (update - warmup_end) / (args.num_updates * 0.8))
            current_c_int = max(0.001, 0.05 * (1.0 - decay_progress))
        track_steps = 0

        target_kl = 0.02
        global_avg_kl = 0.0

        for epoch in range(args.update_epochs):
            random.shuffle(epoch_seq_indices)

            start_indices = list(range(0, b_size_seqs, mb_size_seqs))
            if len(start_indices) > 0 and (b_size_seqs % mb_size_seqs != 0):
                start_indices = start_indices[:-1]
            epoch_kl_sum, epoch_kl_steps = 0.0, 0

            for step_idx, start in enumerate(start_indices):
                end = start + mb_size_seqs
                mb_seqs = epoch_seq_indices[start:end]
                
                start_ts = [s[0] for s in mb_seqs]
                envs = [s[1] for s in mb_seqs]
                agts = [s[2] for s in mb_seqs]

                is_last_batch = (step_idx + 1) == len(start_indices)
                sync_this_step = (step_idx + 1) % accum_steps == 0 or is_last_batch
                my_context = net.no_sync() if not sync_this_step else contextlib.nullcontext()
                
                with my_context:
                    all_ts = []
                    for t_offset in range(SEQ_LEN):
                        all_ts.extend([st + t_offset for st in start_ts])
                    
                    flat_envs = envs * SEQ_LEN
                    flat_agts = agts * SEQ_LEN
                    
                    seq_obs = buf['obs'][all_ts, flat_envs, flat_agts].view(SEQ_LEN, mb_size_seqs, 82, 61).to(dtype=torch.bfloat16)
                    seq_H = buf['H'][all_ts, flat_envs, flat_agts].view(SEQ_LEN, mb_size_seqs, 7, 7)
                    seq_z = buf['z'][all_ts, flat_envs, flat_agts].view(SEQ_LEN, mb_size_seqs, 8, 512).to(torch.bfloat16)
                    seq_S_M_next = buf['S_M_next'][all_ts, flat_envs, flat_agts].view(SEQ_LEN, mb_size_seqs, 8, 512)
                    seq_dones = buf['dones'][all_ts, flat_envs, flat_agts].view(SEQ_LEN, mb_size_seqs)
                    
                    h_0 = buf['prev_h'][start_ts, envs, agts].detach()

                    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        flat_bc_obs = seq_obs.view(SEQ_LEN * mb_size_seqs, 82, 61)
                        bc_z_zero = torch.zeros(SEQ_LEN * mb_size_seqs, 8, 512, device=device, dtype=torch.bfloat16)
                        flat_bc_logits, _ = bc_baseline_net.worker(flat_bc_obs, bc_z_zero, bc_baseline_net.D)
                        
                        flat_bc_logits = torch.nan_to_num(flat_bc_logits, nan=0.0, posinf=50.0, neginf=-50.0)
                        flat_bc_logits = torch.clamp(flat_bc_logits, min=-50.0, max=50.0)
                        seq_bc_logits = flat_bc_logits.view(SEQ_LEN, mb_size_seqs, 82, VOCAB_SIZE)
                    
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        out_S_M, out_pred_z, out_next_h, out_logits, out_ext_v, out_int_v, out_z_ach = net(
                            seq_obs, mb_H=seq_H, mb_prev_h=h_0, worker_z_target=seq_z, 
                            mb_S_M_next=seq_S_M_next, is_bptt=True, seq_dones=seq_dones
                        )
                    
                    mb_loss = torch.tensor(0.0, device=device)
                    seq_active_frames = 0

                    for t_offset in range(SEQ_LEN):
                        curr_ts = [st + t_offset for st in start_ts]
                        
                        t_act = buf['actions'][curr_ts, envs, agts].long()
                        t_logprobs = buf['logprobs'][curr_ts, envs, agts]
                        t_ext_adv = buf['ext_advantages'][curr_ts, envs, agts]
                        t_int_adv = buf['int_advantages'][curr_ts, envs, agts]
                        t_ext_ret = buf['ext_returns'][curr_ts, envs, agts]
                        t_int_ret = buf['int_returns'][curr_ts, envs, agts]
                        t_sparse = buf['sparse_masks'][curr_ts, envs, agts]
                        t_z = buf['z'][curr_ts, envs, agts]
                        t_masks = buf['masks'][curr_ts, envs, agts]
                        
                        if not t_masks.any():
                            continue

                        active_idx = torch.where(t_masks)[0]
                        
                        if active_idx.shape[0] > 0:
                            seq_active_frames += active_idx.shape[0]
                            
                            active_logits = out_logits[t_offset][active_idx]
                            active_predicted_z = out_pred_z[t_offset][active_idx]
                            active_z_achieved_raw = out_z_ach[t_offset][active_idx]
                            active_t_z = t_z[active_idx].to(torch.bfloat16)
                            active_ext_values_pred = out_ext_v[t_offset][active_idx]
                            active_int_values_pred = out_int_v[t_offset][active_idx]
                            
                            active_t_obs = buf['obs'][curr_ts, envs, agts].to(dtype=torch.bfloat16)[active_idx]
                            active_t_act = t_act[active_idx]
                            active_t_logprobs = t_logprobs[active_idx]
                            active_t_ext_adv = t_ext_adv[active_idx]
                            active_t_int_adv = t_int_adv[active_idx]
                            active_t_ext_ret = t_ext_ret[active_idx]
                            active_t_int_ret = t_int_ret[active_idx]
                            active_t_sparse = t_sparse[active_idx]
                            
                            active_logits = torch.nan_to_num(active_logits, nan=0.0, posinf=50.0, neginf=-50.0)
                            active_logits = torch.clamp(active_logits, min=-50.0, max=50.0)
                            z_achieved = active_z_achieved_raw.float()

                            intrinsic_reward = F.cosine_similarity(z_achieved, active_t_z.detach().float(), dim=-1)
                            epoch_intrinsic_reward_sum += intrinsic_reward.detach().mean().item()

                            flat_mb_z_det = active_t_z.view(-1, 512).float().detach()
                            flat_z_achieved = z_achieved.view(-1, 512)
                            flat_predicted_z = active_predicted_z.view(-1, 512).float()
                            
                            norm_pred_z = F.normalize(flat_predicted_z, p=2, dim=-1)
                            norm_z_ach = F.normalize(flat_z_achieved, p=2, dim=-1)
                            norm_t_z = F.normalize(flat_mb_z_det, p=2, dim=-1)

                            z_variance = active_z_achieved_raw.var(dim=0).mean().detach() if active_z_achieved_raw.size(0) > 1 else torch.tensor(0.0, device=device)
                            epoch_z_var_sum += z_variance.item()
                            
                            v_loss = F.huber_loss(active_ext_values_pred, active_t_ext_ret.float(), delta=10.0) + F.huber_loss(active_int_values_pred, active_t_int_ret.float(), delta=10.0)
                            
                            distance_penalty = F.mse_loss(norm_t_z.detach(), norm_z_ach.detach())
                            
                            temperature = 0.5
                            N_batch = norm_pred_z.size(0)
                            labels = torch.arange(N_batch, dtype=torch.long, device=device)
                            
                            logits_mgr = torch.matmul(norm_pred_z, norm_z_ach.detach().T) / temperature
                            logits_inv = torch.matmul(norm_z_ach, norm_t_z.T) / temperature
                            
                            manager_loss = F.cross_entropy(logits_mgr, labels)
                            inverse_model_loss = F.cross_entropy(logits_inv, labels)

                            cos_sim = F.cosine_similarity(norm_pred_z, norm_z_ach.detach(), dim=-1)
                            manager_pg_loss = -(active_t_ext_adv.detach() * cos_sim).mean()

                            is_active, packed_masks, padded_idx, max_act_live = rebuild_packed_masks(
                                active_t_sparse, MAP_PROVINCES, VOCAB_SIZE, NONE_IDX, device
                            )
                            
                            B_live, max_act_live = padded_idx.shape
                            
                            pg_loss = torch.tensor(0.0, device=device)
                            entropy = torch.tensor(0.0, device=device)
                            bc_kl_penalty = torch.tensor(0.0, device=device)
                            kl_divergence = torch.tensor(0.0, device=device)
                            
                            if max_act_live > 0:
                                b_idx_live = torch.arange(B_live, device=device).view(B_live, 1).expand(B_live, max_act_live)
                                
                                step_active_logits = active_logits[b_idx_live, padded_idx, :].float()
                                step_active_logits = step_active_logits.masked_fill(~packed_masks, -1e20)
                                
                                seq_lens = is_active.sum(dim=1)
                                valid_pack_mask = torch.arange(max_act_live, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
                                step_active_logits = step_active_logits.masked_fill(~valid_pack_mask.unsqueeze(-1), 0.0)
                                
                                dist_cat = Categorical(logits=step_active_logits)
                                packed_actions = torch.gather(active_t_act, 1, padded_idx)
                                new_logprobs = dist_cat.log_prob(packed_actions)

                                new_logprobs = torch.clamp(new_logprobs, min=-30.0)
                                old_logprobs_packed = torch.clamp(torch.gather(active_t_logprobs, 1, padded_idx), min=-30.0)

                                log_ratio = new_logprobs - old_logprobs_packed
                                ratio = torch.exp(log_ratio)

                                valid_ratio_mask = (packed_actions != NONE_IDX) & valid_pack_mask

                                if valid_ratio_mask.any():
                                    worker_adv = (active_t_ext_adv + (current_c_int * active_t_int_adv)).unsqueeze(1)
                                    
                                    surr1 = ratio * worker_adv
                                    surr2 = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * worker_adv
                                    
                                    pg_loss = -torch.min(surr1, surr2)[valid_ratio_mask].mean()
                                    entropy = dist_cat.entropy()[valid_ratio_mask].mean()
                                    
                                    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                        bc_active_logits = seq_bc_logits[t_offset][active_idx]

                                        bc_active_logits = bc_active_logits[b_idx_live, padded_idx, :].float()
                                        bc_active_logits = bc_active_logits.masked_fill(~packed_masks, -1e20)
                                        bc_active_logits = bc_active_logits.masked_fill(~valid_pack_mask.unsqueeze(-1), 0.0)
                                        
                                        bc_dist = Categorical(logits=bc_active_logits)
                                        bc_logprobs = bc_dist.log_prob(packed_actions)
                                        bc_logprobs = torch.clamp(bc_logprobs, min=-20.0)
                                        
                                    kl_divergence = 0.5 * (log_ratio ** 2)[valid_ratio_mask].mean()
                                    
                                    exact_kl = torch.distributions.kl.kl_divergence(dist_cat, bc_dist)
                                    raw_bc_kl = exact_kl[valid_pack_mask].mean()
                                    progress = min(1.0, update / 100.0) 
                                    dynamic_target_kl = 0.02 + (0.48 * progress) 
                                    
                                    bc_kl_penalty = F.relu(raw_bc_kl - dynamic_target_kl)

                            unscaled_loss = (pg_loss 
                                            - (args.ent_coef * entropy) 
                                            + (args.bc_kl_coef * bc_kl_penalty) 
                                            + (args.v_coef * v_loss)
                                            + (0.05 * manager_loss)
                                            + (0.05 * inverse_model_loss)
                                            + (0.1 * manager_pg_loss))
                                            
                            mb_loss = mb_loss + (unscaled_loss / SEQ_LEN)

                            epoch_pg_loss_sum += pg_loss.detach().item()
                            epoch_manager_loss_sum += manager_loss.detach().item()
                            epoch_inv_loss_sum += inverse_model_loss.detach().item()
                            epoch_feasibility_error_sum += distance_penalty.detach().item()
                            epoch_bc_kl_sum += bc_kl_penalty.detach().item() 
                            epoch_entropy_sum += entropy.detach().item()
                            epoch_total_loss_sum += unscaled_loss.detach().item()
                            track_steps += 1
                            
                            epoch_kl_sum += kl_divergence.detach()
                            epoch_kl_steps += 1

                    if seq_active_frames > 0:
                        current_block_start = (step_idx // accum_steps) * accum_steps
                        current_block_end = min(current_block_start + accum_steps, len(start_indices))
                        actual_accum_steps = current_block_end - current_block_start
                        
                        loss = mb_loss / actual_accum_steps
                        loss.backward()

                if sync_this_step:
                    inv_grad_norm = torch.nn.utils.clip_grad_norm_(net.module.inverse_model.parameters(), max_norm=1.0)
                    epoch_inv_grad_norm_sum += inv_grad_norm.detach().item() if isinstance(inv_grad_norm, torch.Tensor) else float(inv_grad_norm)
                    
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
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

        free_buffers_queue.put(buffer_idx)
        inference_stream.wait_stream(torch.cuda.current_stream())

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
        ], dtype=torch.float32, device=device)

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
            full_update_duration = time.time() - full_update_start
            global_steps = args.num_envs * args.num_steps * NUM_AGENTS * dist.get_world_size()
            sps = int(global_steps / total_time)

            illegal_rate = (illegal_dropped / max(1, proposed_actions)) * 100.0
            
            print(f"Update {update}/{args.num_updates} | SPS: {sps} | Extrinsic Step: {avg_extrinsic_step_reward:.2f} | Intrinsic Step: {avg_intrinsic_step_reward:.4f} | Total Step: {avg_total_step_reward:.2f}")
            print(f"  CPU Time: {env_step_time:.2f}s | GPU Fwd: {gpu_forward_time:.2f}s | Bwd: {update_time:.2f}s | Full Update: {full_update_duration:.2f}s")
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

            if update % 5 == 0:
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
            
        EVAL_FREQ = 5
        
        if update % EVAL_FREQ == 0:
            eval_start_time = time.time()
            torch.cuda.empty_cache()

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
                
                # Execute games for this power concurrently
                with concurrent.futures.ThreadPoolExecutor(max_workers=GAMES_PER_POWER) as executor:
                    futures = [
                        executor.submit(
                            evaluate_against_baseline,
                            net.module, bc_baseline_net, device, update, power, game_idx + 1, eval_dir
                        ) for game_idx in range(GAMES_PER_POWER)
                    ]
                    
                    for future in concurrent.futures.as_completed(futures):
                        sc = future.result()
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
                    
            torch.cuda.empty_cache()

    rollout_process.join()
    if global_rank == 0:
        writer.close()
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()