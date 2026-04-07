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
import contextlib
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import wandb
from dotenv import load_dotenv
from gymnasium.vector import AsyncVectorEnv

from diplomacy_helpers import DiplomacyTransformer, DiplomacyTransformerEnv, build_global_vocab

def rebuild_dense_mask(sparse_masks, num_provs, vocab_size, device):
    """Reconstructs a 3D dense boolean mask from memory-efficient 2D sparse indices."""
    batch_size = sparse_masks.size(0)
    valid_mask = sparse_masks != -1
    
    row_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * (num_provs * vocab_size)
    global_indices = sparse_masks + row_offsets
    valid_global_indices = global_indices[valid_mask]
    
    batch_mask_flat = torch.zeros(batch_size * num_provs * vocab_size, dtype=torch.bool, device=device)
    batch_mask_flat[valid_global_indices] = True
    
    return batch_mask_flat.view(batch_size, num_provs, vocab_size)

def worker(remote, parent_remote):
    """Background worker process for executing environment steps."""
    torch.set_num_threads(1)
    parent_remote.close()
    env = DiplomacyTransformerEnv(history_length=3) 
    
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step': 
                obs, rewards, terms, truncs, infos = env.step(data)
                
                env_is_done = len(terms) == 0 or all(terms.values()) or all(truncs.values()) or len(env.agents) == 0
                
                if env_is_done:
                    terminal_obs = obs
                    
                    obs, infos = env.reset()
                    
                    infos['__terminal_observation'] = terminal_obs

                remote.send((obs, rewards, terms, truncs, infos, env.agents))
            elif cmd == 'reset': 
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
            # mp.connection.wait() blocks until at least one pipe has data ready
            ready_remotes = mp.connection.wait(remotes_left.keys())
            
            for remote in ready_remotes:
                idx = remotes_left[remote]
                results[idx] = remote.recv() # Clear the buffer instantly
                del remotes_left[remote]     # Remove from the polling pool
                
        return results

    def reset(self):
        for remote in self.remotes: 
            remote.send(('reset', None))
        return self._gather_results()

    def step(self, actions_list):
        for remote, action_dict in zip(self.remotes, actions_list): 
            remote.send(('step', action_dict))
        return self._gather_results()
        
    def close(self):
        for remote in self.remotes: 
            remote.send(('close', None))
        for p in self.ps: 
            p.join()

def apply_top_k_mask(logits, k=3):
    """Strictly masks all but the exact top K highest values."""
    if logits.size(-1) <= k:
        return logits
        
    # Get the exact K indices
    _, top_k_indices = torch.topk(logits, k, dim=-1)
    
    # Build a strict mask for only those K indices
    strict_mask = torch.zeros_like(logits, dtype=torch.bool)
    strict_mask.scatter_(-1, top_k_indices, True)
    
    # Mask out everything else
    return logits.masked_fill(~strict_mask, -1e9)

def evaluate_and_save_game(net, device, update_num, save_dir="./eval_games"):
    """Runs a deterministic evaluation game and logs the full transcript."""
    os.makedirs(save_dir, exist_ok=True)
    env = DiplomacyTransformerEnv(history_length=3)
    obs, infos = env.reset()
    
    log_path = os.path.join(save_dir, f"eval_game_update_{update_num}.txt")
    
    with open(log_path, "w") as f:
        f.write(f"Evaluation Game - Update {update_num}\n")
        f.write("=" * 40 + "\n")
        
        step_count = 0
        while len(env.agents) > 0 and step_count < 150:
            phase_name = env.game.get_current_phase()
            f.write(f"\n--- Phase {phase_name} ---\n")
            active_agents = env.agents
            
            obs_tensor = torch.stack([torch.tensor(obs[a]) for a in active_agents]).to(device=device, dtype=torch.bfloat16)
            
            sparse_masks_tensor = torch.stack([torch.tensor(infos[a]['action_mask'], dtype=torch.long) for a in active_agents]).to(device)
            masks_tensor = rebuild_dense_mask(sparse_masks_tensor, env.num_provinces, env.vocab_size, device)
            
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    state_repr, _ = net.encode_state(obs_tensor)
                
                batch_size = obs_tensor.size(0)
                NONE_IDX = env.order_to_idx['NONE']
                
                is_active_mask = ~masks_tensor[:, :, NONE_IDX]
                num_active_per_batch = is_active_mask.sum(dim=1)
                max_decode_steps = num_active_per_batch.max().item()
                
                final_actions = torch.full((batch_size, env.num_provinces), NONE_IDX, dtype=torch.long, device=device)
                
                if max_decode_steps > 0:
                    padded_indices = torch.zeros((batch_size, max_decode_steps), dtype=torch.long, device=device)
                    step_mask = torch.zeros((batch_size, max_decode_steps), dtype=torch.bool, device=device)
                    
                    for b in range(batch_size):
                        valid_idx = torch.where(is_active_mask[b])[0]
                        count = len(valid_idx)
                        if count > 0:
                            padded_indices[b, :count] = valid_idx
                            step_mask[b, :count] = True

                    current_action = torch.zeros(batch_size, dtype=torch.long, device=device)
                    kv_cache = None
                    batch_indices = torch.arange(batch_size, device=device)
                    
                    b_idx_expand = batch_indices.unsqueeze(1)
                    active_states = state_repr[b_idx_expand, padded_indices, :]

                    for decode_idx in range(max_decode_steps):
                        prov_indices = padded_indices[:, decode_idx]
                        valid_step = step_mask[:, decode_idx]
                        
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            logits, kv_cache = net.decode_step(current_action, active_states, decode_idx, kv_cache)
                        
                        logits = logits.float()
                        prov_mask = masks_tensor[batch_indices, prov_indices, :]
                        logits = logits.masked_fill(~prov_mask, -1e9)
                        
                        # Deterministic sampling for evaluation
                        current_action = torch.argmax(logits, dim=-1)
                        final_actions[batch_indices[valid_step], prov_indices[valid_step]] = current_action[valid_step]
                
            action_dict = {a: final_actions[i].cpu().numpy() for i, a in enumerate(active_agents)}
            
            for idx, agent in enumerate(active_agents):
                f.write(f"{agent} Orders\n")
                agent_actions = final_actions[idx]
                orders_issued = False
                
                for prov_idx, order_idx in enumerate(agent_actions):
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
        for agent in env.possible_agents:
            scs = env.game.get_centers(agent)
            f.write(f"{agent}: {len(scs)}\n")
            
    print(f"  -> Saved evaluation game log to {log_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="PPO Training for Diplomacy")
    parser.add_argument("--num_envs", type=int, default=28, help="Number of parallel environments per GPU")
    parser.add_argument("--num_steps", type=int, default=1024, help="Number of steps per rollout")
    parser.add_argument("--num_updates", type=int, default=1000, help="Total number of PPO updates")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO policy clipping coefficient")
    parser.add_argument("--ent_coef", type=float, default=0.001, help="Entropy coefficient")
    parser.add_argument("--v_coef", type=float, default=0.1, help="Value function loss coefficient")
    parser.add_argument("--kl_coef", type=float, default=0.05, help="KL divergence penalty coefficient")
    parser.add_argument("--update_epochs", type=int, default=2, help="Number of epochs per PPO update")
    parser.add_argument("--bc_weights", type=str, default="diplomacy_transformer_bc.pth", help="Path to pre-trained Behavioral Cloning weights")
    parser.add_argument("--resume_weights", type=str, default=None, help="Path to RL checkpoint to resume training from")
    return parser.parse_args()

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
    HISTORY_LENGTH = 3

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

    vec_env = SubprocVecDiplomacy(num_envs=args.num_envs)
    
    # Model Initialization
    net = DiplomacyTransformer(num_provinces=MAP_PROVINCES, history_length=HISTORY_LENGTH, vocab_size=VOCAB_SIZE).to(device)
    bc_model = DiplomacyTransformer(num_provinces=MAP_PROVINCES, history_length=HISTORY_LENGTH, vocab_size=VOCAB_SIZE).to(device)
    actor_bc_model = DiplomacyTransformer(num_provinces=MAP_PROVINCES, history_length=HISTORY_LENGTH, vocab_size=VOCAB_SIZE).to(device)
    actor_net = DiplomacyTransformer(num_provinces=MAP_PROVINCES, history_length=HISTORY_LENGTH, vocab_size=VOCAB_SIZE).to(device)
    
    if os.path.exists(args.bc_weights):
        bc_state_dict = torch.load(args.bc_weights, map_location=device)
        net.load_state_dict(bc_state_dict, strict=False)
        bc_model.load_state_dict(bc_state_dict, strict=False)
        actor_bc_model.load_state_dict(bc_state_dict, strict=False)
        actor_net.load_state_dict(bc_state_dict, strict=False)
        
        # Freeze reference models
        bc_model.eval()
        for param in bc_model.parameters(): param.requires_grad = False
        actor_bc_model.eval()
        for param in actor_bc_model.parameters(): param.requires_grad = False
        actor_net.eval()
        for param in actor_net.parameters(): param.requires_grad = False
            
        if global_rank == 0: 
            print("Successfully loaded pre-trained BC weights for policy initialization.")

    if args.resume_weights and os.path.exists(args.resume_weights):
        rl_state_dict = torch.load(args.resume_weights, map_location=device)
        net.load_state_dict(rl_state_dict, strict=False)
        actor_net.load_state_dict(rl_state_dict, strict=False)
        
        if global_rank == 0:
            print(f"Successfully resumed RL training from checkpoint: {args.resume_weights}")

    # if global_rank == 0: 
    #     print("Compiling PyTorch models...")
    
    # net = torch.compile(net, dynamic=True)
    # bc_model = torch.compile(bc_model, dynamic=True)
    # actor_bc_model = torch.compile(actor_bc_model, dynamic=True)
    # actor_net = torch.compile(actor_net, dynamic=True)

    net = DDP(net, device_ids=[local_rank])
    optimizer = optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    def create_buffer():
        """Creates a memory-pinned tensor buffer for experience collection."""
        return {
            'obs': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, HISTORY_LENGTH, MAP_PROVINCES, 46), dtype=torch.bfloat16, device=device),
            'actions': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, MAP_PROVINCES), dtype=torch.long, device=device),
            'logprobs': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, MAP_PROVINCES), dtype=torch.float32, device=device),
            'rewards': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'dones': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'values': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'masks': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.bool, device=device),
            'sparse_masks': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, 2000), dtype=torch.int32, device=device),
            'top_k_indices': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, MAP_PROVINCES, 3), dtype=torch.long, device=device),
            'top_k_values': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS, MAP_PROVINCES, 3), dtype=torch.float32, device=device),
            'advantages': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device),
            'returns': torch.zeros((args.num_steps, args.num_envs, NUM_AGENTS), dtype=torch.float32, device=device)
        }
        
    # Double-buffering architecture masks CPU environment latency behind GPU backpropagation
    buffers = {0: create_buffer(), 1: create_buffer()}
    
    rollout_complete_event = threading.Event()
    update_complete_event = threading.Event()
    update_complete_event.set() 

    thread_stats = {"env_time": 0.0, "gpu_fwd_time": 0.0}
    agent_to_idx = {a: i for i, a in enumerate(possible_agents)}

    inference_stream = torch.cuda.Stream(device=device)

    def rollout_worker():
        """Background thread responsible for filling the experience buffer asynchronously."""
        torch.cuda.set_device(device) 
        buffer_idx = 0
        next_env_results = vec_env.reset()
        next_done = torch.zeros((args.num_envs, NUM_AGENTS), device=device)
        
        for update in range(1, args.num_updates + 1):
            update_complete_event.wait()
            update_complete_event.clear()

            buf = buffers[buffer_idx]
            with torch.cuda.stream(inference_stream):
                buf['masks'].zero_()
                buf['rewards'].zero_()
                
                env_step_time, gpu_forward_time = 0.0, 0.0
            
                for step in range(args.num_steps):
                    buf['dones'][step] = next_done
                    actions_to_send = [{} for _ in range(args.num_envs)]
                    
                    with torch.no_grad():
                        for i in range(args.num_envs):
                            if len(next_env_results[i]) == 3:
                                obs_dict, infos_dict, active_agents = next_env_results[i]
                            else:
                                obs_dict, step_rewards, terms, truncs, infos_dict, active_agents = next_env_results[i]
                                for a in possible_agents:
                                    buf['rewards'][step, i, agent_to_idx[a]] = step_rewards.get(a, 0.0)
                                    next_done[i, agent_to_idx[a]] = float(terms.get(a, False))
                            
                            for a in active_agents:
                                a_idx = agent_to_idx[a]
                                buf['masks'][step, i, a_idx] = True
                                buf['obs'][step, i, a_idx] = torch.tensor(obs_dict[a], dtype=torch.bfloat16, device=device)
                                buf['sparse_masks'][step, i, a_idx] = torch.tensor(infos_dict[a]['action_mask'], dtype=torch.long)

                        flat_obs = buf['obs'][step][buf['masks'][step]]
                        if flat_obs.shape[0] > 0:
                            t_gpu_start = time.time()
                            
                            active_sparse = buf['sparse_masks'][step][buf['masks'][step]]
                            active_masks = rebuild_dense_mask(active_sparse, MAP_PROVINCES, VOCAB_SIZE, device)
                            is_active_mask = ~active_masks[:, :, NONE_IDX]
                            num_active_per_batch = is_active_mask.sum(dim=1)
                            max_decode_steps = num_active_per_batch.max().item()
                            
                            if max_decode_steps > 0:
                                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                    state_repr, values = actor_net.encode_state(flat_obs.to(dtype=torch.bfloat16))
                                values = values.float()
                                
                                batch_size = flat_obs.size(0)
                                final_actions = torch.full((batch_size, MAP_PROVINCES), NONE_IDX, dtype=torch.long, device=device)
                                final_logprobs = torch.zeros((batch_size, MAP_PROVINCES), dtype=torch.float32, device=device)
                                final_top_k = torch.zeros((batch_size, MAP_PROVINCES, 3), dtype=torch.long, device=device)
                                final_top_k_vals = torch.zeros((batch_size, MAP_PROVINCES, 3), dtype=torch.float32, device=device)
                                
                                padded_indices = torch.zeros((batch_size, max_decode_steps), dtype=torch.long, device=device)
                                step_mask = torch.zeros((batch_size, max_decode_steps), dtype=torch.bool, device=device)
                                
                                for b in range(batch_size):
                                    valid_idx = torch.where(is_active_mask[b])[0]
                                    count = len(valid_idx)
                                    if count > 0:
                                        padded_indices[b, :count] = valid_idx
                                        step_mask[b, :count] = True

                                current_action = torch.zeros(batch_size, dtype=torch.long, device=device)
                                kv_cache, bc_kv_cache = None, None
                                batch_indices = torch.arange(batch_size, device=device)
                                
                                b_idx_expand = batch_indices.unsqueeze(1)
                                active_states = state_repr[b_idx_expand, padded_indices, :]
                                
                                # Evaluate BC states identically for KL Divergence checks
                                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                    bc_state_repr, _ = actor_bc_model.encode_state(flat_obs.to(dtype=torch.bfloat16))
                                bc_active_states = bc_state_repr[b_idx_expand, padded_indices, :]

                                for decode_idx in range(max_decode_steps):
                                    prov_indices = padded_indices[:, decode_idx]
                                    valid_step = step_mask[:, decode_idx]
                                    
                                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                        bc_logits, bc_kv_cache = actor_bc_model.decode_step(current_action, bc_active_states, decode_idx, bc_kv_cache)
                                        logits, kv_cache = actor_net.decode_step(current_action, active_states, decode_idx, kv_cache)
                                    
                                    prov_mask = active_masks[batch_indices, prov_indices, :]

                                    pure_logits = logits.float().masked_fill(~prov_mask, -1e9)
                                    true_dist = Categorical(logits=pure_logits)

                                    bc_probs = torch.softmax(bc_logits.float(), dim=-1)
                                    bc_approved_mask = bc_probs > 0.05 
                                    combined_mask = prov_mask & bc_approved_mask
                                    fallback_mask = combined_mask.sum(dim=-1) == 0
                                    combined_mask[fallback_mask] = prov_mask[fallback_mask]

                                    sample_logits = pure_logits.masked_fill(~combined_mask, -1e9)
                                    sample_logits = apply_top_k_mask(sample_logits, k=3)
                                    sample_dist = Categorical(logits=sample_logits)

                                    current_action = sample_dist.sample()
                                    step_log_probs = true_dist.log_prob(current_action)

                                    top_k_vals, top_k_idx = torch.topk(logits.float(), 3, dim=-1)
                                    
                                    final_actions[batch_indices[valid_step], prov_indices[valid_step]] = current_action[valid_step]
                                    final_logprobs[batch_indices[valid_step], prov_indices[valid_step]] = step_log_probs[valid_step]
                                    final_top_k[batch_indices[valid_step], prov_indices[valid_step]] = top_k_idx[valid_step]
                                    final_top_k_vals[batch_indices[valid_step], prov_indices[valid_step]] = top_k_vals[valid_step]
                                    
                                buf['values'][step][buf['masks'][step]] = values.squeeze(-1)
                                buf['logprobs'][step][buf['masks'][step]] = final_logprobs
                                
                                idx_counter = 0
                                for i in range(args.num_envs):
                                    for a in possible_agents:
                                        if buf['masks'][step, i, agent_to_idx[a]]:
                                            act_array = final_actions[idx_counter]
                                            buf['actions'][step, i, agent_to_idx[a]] = act_array
                                            buf['top_k_indices'][step, i, agent_to_idx[a]] = final_top_k[idx_counter]
                                            buf['top_k_values'][step, i, agent_to_idx[a]] = final_top_k_vals[idx_counter]
                                            actions_to_send[i][a] = act_array.cpu().numpy()
                                            idx_counter += 1
                                            
                            gpu_forward_time += (time.time() - t_gpu_start)

                    t_env_start = time.time()
                    next_env_results = vec_env.step(actions_to_send)
                    env_step_time += (time.time() - t_env_start)

            # Calculate GAE Advantages
            with torch.no_grad():
                next_value = torch.zeros((args.num_envs, NUM_AGENTS), device=device)
                obs_to_encode = []
                indices_to_update = []
                
                for i in range(args.num_envs):
                    infos_dict = next_env_results[i][4]
                    
                    if '__terminal_observation' in infos_dict:
                        obs_dict = infos_dict['__terminal_observation']
                        active_agents_list = [a for a in possible_agents if a in obs_dict]
                    else:
                        obs_dict = next_env_results[i][0]
                        active_agents_list = next_env_results[i][5]
                    
                    for a in active_agents_list:
                        if a in obs_dict:
                            obs_to_encode.append(torch.tensor(obs_dict[a], dtype=torch.bfloat16, device=device))
                            indices_to_update.append((i, agent_to_idx[a]))
                        
                if obs_to_encode:
                    obs_tensor = torch.stack(obs_to_encode)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        _, next_v = actor_net.encode_state(obs_tensor)
                    
                    for list_idx, (env_idx, agent_idx) in enumerate(indices_to_update):
                        next_value[env_idx, agent_idx] = next_v[list_idx].float().squeeze(-1)
                
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - buf['dones'][t + 1]
                    nextvalues = buf['values'][t + 1]
                delta = buf['rewards'][t] + args.gamma * nextvalues * nextnonterminal - buf['values'][t]
                buf['advantages'][t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            buf['returns'] = buf['advantages'] + buf['values']
            
            inference_stream.synchronize()

            thread_stats["env_time"] = env_step_time
            thread_stats["gpu_fwd_time"] = gpu_forward_time
        
            rollout_complete_event.set()
            buffer_idx = 1 - buffer_idx

    rollout_thread = threading.Thread(target=rollout_worker, daemon=True)
    rollout_thread.start()

    if global_rank == 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ppo_run_{timestamp}"
        
        base_dir = f"/data/restanislao/model_runs/{run_name}"
        ckpt_dir = os.path.join(base_dir, "checkpoints")
        eval_dir = os.path.join(base_dir, "eval_games")
        
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=f"/data/restanislao/tb_runs/{run_name}")
        wandb.init(
            project="diplomacy-ppo",
            name=run_name,
            config=vars(args),
            dir="/data/restanislao/wandb"
        )

    buffer_idx = 0
    for update in range(1, args.num_updates + 1):
        start_time = time.time()
        
        # Await experience buffer completion
        rollout_complete_event.wait()
        rollout_complete_event.clear()
        
        with torch.no_grad():
            for param, actor_param in zip(net.module.parameters(), actor_net.parameters()):
                actor_param.data.copy_(param)
        torch.cuda.current_stream().synchronize()

        buf = buffers[buffer_idx]
        env_step_time = thread_stats["env_time"]
        gpu_forward_time = thread_stats["gpu_fwd_time"]

        # Signal background thread to begin filling the alternate buffer
        update_complete_event.set()
        inference_stream.wait_stream(torch.cuda.current_stream())
        
        valid = buf['masks'].view(-1)
        valid_cpu = valid.cpu()
        flat_obs = buf['obs'].view(-1, HISTORY_LENGTH, MAP_PROVINCES, 46)[valid]
        flat_act = buf['actions'].view(-1, MAP_PROVINCES)[valid]
        flat_logprobs = buf['logprobs'].view(-1, MAP_PROVINCES)[valid]
        flat_adv = buf['advantages'].view(-1)[valid]
        flat_ret = buf['returns'].view(-1)[valid]
        flat_val = buf['values'].view(-1)[valid]
        flat_sparse_masks = buf['sparse_masks'].view(-1, 2000)[valid]
        flat_top_k = buf['top_k_indices'].view(-1, MAP_PROVINCES, 3)[valid]
        flat_top_k_vals = buf['top_k_values'].view(-1, MAP_PROVINCES, 3)[valid]

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
            flat_val = flat_val[perm][:min_b_size]
            flat_sparse_masks = flat_sparse_masks[perm][:min_b_size]
            b_size = min_b_size

        if flat_adv.shape[0] > 1:
            local_sum = flat_adv.sum()
            local_sq_sum = (flat_adv ** 2).sum()
            local_count = torch.tensor(flat_adv.shape[0], dtype=torch.float32, device=device)
        else:
            local_sum = torch.tensor(0.0, device=device)
            local_sq_sum = torch.tensor(0.0, device=device)
            local_count = torch.tensor(0.0, device=device)
        # Stack and sync across all GPUs
        stats = torch.stack([local_sum, local_sq_sum, local_count])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        global_sum, global_sq_sum, global_n = stats[0], stats[1], stats[2]

        if global_n > 1:
            global_mean = global_sum / global_n
            global_var = (global_sq_sum / global_n) - (global_mean ** 2)
            global_std = torch.sqrt(torch.clamp(global_var, min=1e-8))
            flat_adv = (flat_adv - global_mean) / (global_std + 1e-8)

        t_update_start = time.time()
        net.train()
        
        mb_size = 768
        accum_steps = 6
        indices = np.arange(b_size)
        optimizer.zero_grad() 

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
            epoch_top_k = flat_top_k[perm]
            epoch_top_k_vals = flat_top_k_vals[perm]

            start_indices = list(range(0, b_size, mb_size))
            
            epoch_kl_sum = 0.0
            epoch_kl_steps = 0

            for step_idx, start in enumerate(start_indices):
                end = start + mb_size
                
                mb_obs = epoch_obs[start:end].to(dtype=torch.bfloat16)
                mb_act = epoch_act[start:end]
                mb_logprobs = epoch_logprobs[start:end]
                mb_adv = epoch_adv[start:end]
                mb_ret = epoch_ret[start:end]
                mb_top_k = epoch_top_k[start:end]
                mb_top_k_vals = epoch_top_k_vals[start:end]
                
                mb_sparse_gpu = epoch_sparse[start:end].to(device)
                mb_masks_gpu = rebuild_dense_mask(mb_sparse_gpu, MAP_PROVINCES, VOCAB_SIZE, device)

                is_last_batch = (step_idx + 1) == len(start_indices)
                sync_this_step = (step_idx + 1) % accum_steps == 0 or is_last_batch

                my_context = net.no_sync() if not sync_this_step else contextlib.nullcontext()
                
                debug_print = (update == 1 and epoch == 0 and step_idx == 0 and global_rank == 0)

                with my_context:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        state_repr, new_val = net.module.encode_state(mb_obs)
                        
                        is_active_mask = ~mb_masks_gpu[:, :, NONE_IDX]
                        
                        batch_size_mb = mb_obs.size(0)
                        max_active = is_active_mask.sum(dim=1).max().item()

                        if max_active > 0:
                            # Parallel Full-Sequence Decode
                            active_logits_seq, padding_mask = net.module.decode_full(state_repr, mb_act, is_active_mask)
                            
                            # Pack targets and masks
                            padded_indices = torch.zeros((batch_size_mb, max_active), dtype=torch.long, device=device)
                            for b in range(batch_size_mb):
                                v_idx = torch.where(is_active_mask[b])[0]
                                if len(v_idx) > 0:
                                    padded_indices[b, :len(v_idx)] = v_idx
                                    
                            b_idx_expand = torch.arange(batch_size_mb, device=device).unsqueeze(1)
                            packed_targets = mb_act[b_idx_expand, padded_indices]
                            packed_masks = mb_masks_gpu[b_idx_expand, padded_indices, :]
                            
                            # Apply only the environment's legal action mask to maintain proper probability distributions
                            active_logits_seq = active_logits_seq.float().masked_fill(~packed_masks, -1e9)
                            
                            dist_cat = Categorical(logits=active_logits_seq)

                            logp_seq = dist_cat.log_prob(packed_targets)
                            entropy_seq = dist_cat.entropy()
                            if debug_print:
                                print("\n--- DEBUG: MASK RECONSTRUCTION ---")
                                b_idx, u_idx = 0, 0 # First batch, first unit
                                target_action = packed_targets[b_idx, u_idx].item()
                                is_action_masked = packed_masks[b_idx, u_idx, target_action].item()
                                print(f"Target Action Index: {target_action}")
                                print(f"Is Target Masked VALID (True)? {is_action_masked}")
                                print(f"Total Valid Actions in Mask: {packed_masks[b_idx, u_idx].sum().item()}")
                            
                            # Pack the old logprobs to match the active units
                            packed_old_logprobs = mb_logprobs[b_idx_expand, padded_indices]
                            
                            if debug_print:
                                # Grab the first sequence in the batch
                                print("\n--- DEBUG: LOGPROB ALIGNMENT ---")
                                print(f"Target Actions:    {packed_targets[0][:5]}")
                                print(f"Old Logprobs:      {packed_old_logprobs[0][:5]}")
                                print(f"New Logprobs:      {logp_seq[0][:5]}")
                                diff = (packed_old_logprobs - logp_seq).abs()
                                print(f"Max Diff in Batch: {diff.max().item():.4f}")
                                print(f"Mean Diff:         {diff.mean().item():.4f}")
                            
                            # Zero out padded units for all sequence metrics
                            valid_mask = ~padding_mask
                            masked_new_logprobs = logp_seq * valid_mask
                            masked_old_logprobs = packed_old_logprobs * valid_mask
                            joint_new_logprob = masked_new_logprobs.sum(dim=1)
                            joint_old_logprob = masked_old_logprobs.sum(dim=1)
                            joint_ratio = torch.exp(joint_new_logprob - joint_old_logprob)

                            pg_loss1 = -mb_adv * joint_ratio
                            pg_loss2 = -mb_adv * torch.clamp(joint_ratio, 1 - args.clip_coef, 1 + args.clip_coef)

                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                            unit_counts = valid_mask.sum(dim=1).clamp(min=1)
                            entropy = ((entropy_seq * valid_mask).sum(dim=1) / unit_counts).mean()

                            if debug_print:
                                print("\n--- DEBUG: PROBABILITY DISTRIBUTION ---")
                                # Convert logits to probabilities to see where the mass is
                                probs = torch.softmax(active_logits_seq[0, 0], dim=-1)
                                print(f"Top 5 Probs: {torch.topk(probs, 5).values}")
                                print(f"Target Prob: {probs[packed_targets[0, 0]]}")
                        
                            
                            with torch.no_grad():
                                kl_divergence = 0.5 * (joint_old_logprob - joint_new_logprob).pow(2).mean()
                        else:
                            pg_loss = torch.tensor(0.0, device=device)
                            entropy = torch.tensor(0.0, device=device)
                            kl_divergence = torch.tensor(0.0, device=device)

                    epoch_kl_sum += kl_divergence.item()
                    epoch_kl_steps += 1

                    new_val = new_val.float()
                    
                    # Calculate Value loss
                    v_loss = 0.5 * ((new_val.squeeze() - mb_ret) ** 2).mean()
                    
                    # Combine into final loss
                    loss = pg_loss - args.ent_coef * entropy + v_loss * args.v_coef + args.kl_coef * kl_divergence
                    
                    current_block_start = (step_idx // accum_steps) * accum_steps
                    current_block_end = min(current_block_start + accum_steps, len(start_indices))
                    actual_accum_steps = current_block_end - current_block_start
                    
                    loss = loss / actual_accum_steps
                    loss.backward()

                if sync_this_step:
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    optimizer.step()
                    optimizer.zero_grad()
                
                del mb_obs, mb_act, mb_masks_gpu, new_val
                if max_active > 0:
                    del active_logits_seq, dist_cat

            local_epoch_kl = epoch_kl_sum / max(1, epoch_kl_steps)
            epoch_kl_tensor = torch.tensor([local_epoch_kl], device=device)
            dist.all_reduce(epoch_kl_tensor, op=dist.ReduceOp.SUM)
            global_epoch_kl = epoch_kl_tensor.item() / dist.get_world_size()
            
            global_avg_kl = global_epoch_kl
            
            if global_epoch_kl > target_kl * 1.5:
                if global_rank == 0:
                    print(f"Early stopping triggered at epoch {epoch+1} due to high KL: {global_epoch_kl:.4f}")
                break

        if global_avg_kl > target_kl * 1.2:
            args.kl_coef *= 1.5
        elif global_avg_kl < target_kl * 0.8:
            args.kl_coef /= 1.5
            
        args.kl_coef = max(0.0001, min(5.0, args.kl_coef))

        update_time = time.time() - t_update_start
        buffer_idx = 1 - buffer_idx
        
        if global_rank == 0:
            total_time = time.time() - start_time
            global_steps = args.num_envs * args.num_steps * NUM_AGENTS * dist.get_world_size()
            sps = int(global_steps / total_time)  
            total_country_episodes = buf['dones'].sum().item()
            if total_country_episodes > 0:
                avg_country_return = buf['rewards'].sum().item() / total_country_episodes
            else:
                avg_country_return = buf['rewards'].sum().item() / (args.num_envs * NUM_AGENTS)
            
            print(f"Update {update}/{args.num_updates} | SPS: {sps} | Avg Country Return: {avg_country_return:.2f} | Loss: {loss.item():.4f}")
            print(f"  CPU Time: {env_step_time:.2f}s | GPU Fwd: {gpu_forward_time:.2f}s | Bwd: {update_time:.2f}s")
            
            writer.add_scalar("Perf/SPS", sps, update)
            writer.add_scalar("Reward/Avg_Country_Return", avg_country_return, update)
            writer.add_scalar("Loss/Policy_Loss", loss.item(), update)
            writer.add_scalar("Loss/Value_Loss", v_loss.item(), update)
            writer.add_scalar("Loss/Entropy", entropy.item(), update)

            wandb.log({
                "Perf/SPS": sps,
                "Reward/Avg_Country_Return": avg_country_return,
                "Loss/Policy_Loss": loss.item(),
                "Loss/Value_Loss": v_loss.item(),
                "Loss/Entropy": entropy.item(),
                "Loss/KL_Div": global_avg_kl,
                "global_step": update * global_steps,
            }, step=update)

            if update % 1 == 0:
                # Save checkpoint to the unique run folder
                ckpt_path = os.path.join(ckpt_dir, f"diplomacy_APPO_update_{update}.pth")
                torch.save(net.module.state_dict(), ckpt_path)
                print(f"  -> Saved checkpoint to {ckpt_path}")
                
                evaluate_and_save_game(net.module, device, update, save_dir=eval_dir)
                
                eval_log_path = os.path.join(eval_dir, f"eval_game_update_{update}.txt")
                if os.path.exists(eval_log_path):
                    eval_artifact = wandb.Artifact(
                        name=f"eval_transcript_update_{update}",
                        type="evaluation_log",
                        description=f"Full transcript of evaluation game at update {update}"
                    )
                    eval_artifact.add_file(eval_log_path)
                    wandb.log_artifact(eval_artifact)
        
    vec_env.close()
    if global_rank == 0:
        writer.close()
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()