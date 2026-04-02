import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from torch.utils.tensorboard import SummaryWriter
import datetime
from gymnasium.vector import AsyncVectorEnv
import wandb
from dotenv import load_dotenv
from diplomacy_helpers import DiplomacyTransformer, DiplomacyTransformerEnv
import contextlib

# --- FAST SPARSE-TO-DENSE RECONSTRUCTION ---
def rebuild_dense_mask(sparse_masks, num_provs, vocab_size, device):
    batch_size = sparse_masks.size(0)
    valid_mask = sparse_masks != -1
    
    row_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * (num_provs * vocab_size)
    global_indices = sparse_masks + row_offsets
    valid_global_indices = global_indices[valid_mask]
    
    batch_mask_flat = torch.zeros(batch_size * num_provs * vocab_size, dtype=torch.bool, device=device)
    batch_mask_flat[valid_global_indices] = True
    
    return batch_mask_flat.view(batch_size, num_provs, vocab_size)


def worker(remote, parent_remote):
    torch.set_num_threads(1)
    parent_remote.close()
    env = DiplomacyTransformerEnv(history_length=3) 
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step': 
                obs, rewards, terms, truncs, infos = env.step(data)
                if len(terms) == 0 or all(terms.values()) or len(env.agents) == 0:
                    obs, infos = env.reset()
                remote.send((obs, rewards, terms, truncs, infos, env.agents))
            elif cmd == 'reset': 
                remote.send((*env.reset(), env.agents))
            elif cmd == 'close':
                remote.close()
                break
        except EOFError: 
            break
        except Exception as e:
            print(f"Worker crashed {e}")
            remote.close()
            break

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

def evaluate_and_save_game(net, device, update_num, save_dir="./eval_games"):
    os.makedirs(save_dir, exist_ok=True)
    env = DiplomacyTransformerEnv(history_length=3)
    obs, infos = env.reset()
    
    log_path = os.path.join(save_dir, f"eval_game_KV_update_{update_num}.txt")
    
    with open(log_path, "w") as f:
        f.write(f"Evaluation Game - Update {update_num}\n")
        f.write("========================================\n")
        
        step_count = 0
        while len(env.agents) > 0 and step_count < 150:
            phase_name = env.game.get_current_phase()
            f.write(f"\n--- Phase {phase_name} ---\n")
            active_agents = env.agents
            
            obs_tensor = torch.stack([torch.tensor(obs[a]) for a in active_agents]).to(device=device, dtype=torch.bfloat16)
            
            # --- NEW: Rebuild Dense Mask ---
            sparse_masks_tensor = torch.stack([torch.tensor(infos[a]['action_mask'], dtype=torch.long) for a in active_agents]).to(device)
            masks_tensor = rebuild_dense_mask(sparse_masks_tensor, env.num_provinces, env.vocab_size, device)
            
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    state_repr, _ = net.encode_state(obs_tensor)
                
                batch_size = obs_tensor.size(0)
                NONE_IDX = env.order_to_idx['NONE']
                
                # --- IDENTIFY ACTIVE UNITS FOR EVAL ---
                is_active_mask = ~masks_tensor[:, :, NONE_IDX]
                num_active_per_batch = is_active_mask.sum(dim=1)
                max_decode_steps = num_active_per_batch.max().item()
                
                # 1. Initialize outputs with defaults
                final_actions = torch.full((batch_size, env.num_provinces), NONE_IDX, dtype=torch.long, device=device)
                
                # 2. Only decode if there are actual units to move
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

                    # 3. Decode ONLY up to the max number of units
                    for decode_idx in range(max_decode_steps):
                        prov_indices = padded_indices[:, decode_idx]
                        valid_step = step_mask[:, decode_idx]
                        
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            logits, kv_cache = net.decode_step(current_action, prov_indices, state_repr, kv_cache)
                        
                        logits = logits.float()
                        
                        # Grab the legal action mask for these specific provinces
                        prov_mask = masks_tensor[batch_indices, prov_indices, :]
                        logits = logits.masked_fill(~prov_mask, -1e4)
                        
                        # In eval, we take the best move (argmax) instead of random sampling
                        current_action = torch.argmax(logits, dim=-1)
                        
                        # 4. Scatter the results back into the 82-length tensor
                        final_actions[batch_indices[valid_step], prov_indices[valid_step]] = current_action[valid_step]
                
                actions = final_actions
            
            action_dict = {a: actions[i].cpu().numpy() for i, a in enumerate(active_agents)}
            
            for idx, agent in enumerate(active_agents):
                f.write(f"{agent} Orders\n")
                agent_actions = actions[idx]
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
            
        f.write("\n========================================\n")
        f.write("FINAL SUPPLY CENTER COUNTS\n")
        for agent in env.possible_agents:
            scs = env.game.get_centers(agent)
            f.write(f"{agent} {len(scs)}\n")
            
    print(f"  -> Saved evaluation game log to {log_path}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    torch.set_num_threads(1)

    NUM_ENVS = 28
    NUM_STEPS = 1024
    NUM_AGENTS = 7
    HISTORY_LENGTH = 3

    if global_rank == 0:
        print("--- STARTING NATIVE TRANSFORMER PPO ---")

        load_dotenv() # Loads the .env file
        wandb_key = os.environ.get("WANDB_KEY")
        if wandb_key:
            wandb.login(key=wandb_key)
        else:
            print("WARNING: WANDB_KEY not found in .env file!")

    if global_rank == 0:
        # Rank 0 takes the job of building the file
        from diplomacy_helpers import build_global_vocab
        build_global_vocab() 
        
    # Ranks 1, 2, and 3 will wait at this line until Rank 0 finishes writing the file!
    dist.barrier() 
    # -----------------------------------

    # Now all 4 ranks can safely initialize their environments.
    # Ranks 1, 2, and 3 will instantly load from the cache Rank 0 just built.
    dummy_env = DiplomacyTransformerEnv()
    possible_agents = dummy_env.possible_agents
    MAP_PROVINCES = dummy_env.num_provinces
    VOCAB_SIZE = dummy_env.vocab_size
    NONE_IDX = dummy_env.order_to_idx['NONE']
    del dummy_env

    if global_rank == 0:
        print(f"Detected {MAP_PROVINCES} provinces and {VOCAB_SIZE} actions")

    vec_env = SubprocVecDiplomacy(num_envs=NUM_ENVS)
    net = DiplomacyTransformer(num_provinces=MAP_PROVINCES, history_length=HISTORY_LENGTH, vocab_size=VOCAB_SIZE).to(device)
    bc_model = DiplomacyTransformer(num_provinces=MAP_PROVINCES, history_length=HISTORY_LENGTH, vocab_size=VOCAB_SIZE).to(device)
    
    bc_weights_path = "diplomacy_transformer_bc.pth"
    
    if os.path.exists(bc_weights_path):
        net.load_state_dict(torch.load(bc_weights_path, map_location=device), strict=False)
        bc_model.load_state_dict(torch.load(bc_weights_path, map_location=device), strict=False)
        
        bc_model.eval()
        for param in bc_model.parameters():
            param.requires_grad = False
            
        if global_rank == 0:
            print(f"Successfully loaded pre-trained weights.")
            
    # --- NEW: APPLY TORCH.COMPILE ---
    if global_rank == 0:
        print("Compiling models... (This will stall the first training step for 2-5 minutes while ROCm optimizes the graphs)")
    
    # net = torch.compile(net)
    # bc_model = torch.compile(bc_model)
    
    # Wrap in DDP *after* compiling
    net = DDP(net, device_ids=[local_rank])
    optimizer = optim.Adam(net.parameters(), lr=1e-5, eps=1e-5)

    # --- MEMORY FIX: Rollout Buffers ---
    b_obs = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS, HISTORY_LENGTH, MAP_PROVINCES, 46), dtype=torch.bool, device=device)
    b_actions = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS, MAP_PROVINCES), dtype=torch.long, device=device)
    b_logprobs = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS), dtype=torch.float32, device=device)
    b_rewards = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS), dtype=torch.float32, device=device)
    b_dones = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS), dtype=torch.float32, device=device)
    b_values = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS), dtype=torch.float32, device=device)
    b_masks = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS), dtype=torch.bool, device=device) 
    
    b_sparse_masks = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS, 2000), dtype=torch.int32, device=device)

    num_updates = 1000
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.0
    v_coef = 0.1
    kl_coef = 0.05
    update_epochs = 2

    agent_to_idx = {a: i for i, a in enumerate(possible_agents)}
    next_env_results = vec_env.reset()
    next_done = torch.zeros((NUM_ENVS, NUM_AGENTS), device=device)

    if global_rank == 0:
        os.makedirs("./checkpoints", exist_ok=True)
        
        # Generates a name like: ppo_run_20260326_123045
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ppo_run_{timestamp}"
        
        writer = SummaryWriter(log_dir=f"./runs/{run_name}")
        print(f"Logging TensorBoard data to: ./runs/{run_name}")

        wandb.init(
            project="diplomacy-ppo", # Give your project a name
            name=run_name,
            config={
                "num_envs": NUM_ENVS,
                "num_steps": NUM_STEPS,
                "num_agents": NUM_AGENTS,
                "history_length": HISTORY_LENGTH,
                "num_updates": num_updates,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_coef": clip_coef,
                "ent_coef": ent_coef,
                "v_coef": v_coef,
                "update_epochs": update_epochs
            }
        )

    for update in range(1, num_updates + 1):
        start_time = time.time()
        net.eval()
        

        env_step_time = 0.0
        gpu_forward_time = 0.0
        for step in range(NUM_STEPS):
            b_dones[step] = next_done
            actions_to_send = [{} for _ in range(NUM_ENVS)]
            
            with torch.no_grad():
                for i in range(NUM_ENVS):
                    obs_dict, infos_dict, active_agents = next_env_results[i] if len(next_env_results[i]) == 3 else (*next_env_results[i][0:2], next_env_results[i][-2], next_env_results[i][-1])[0:3]
                    if len(next_env_results[i]) == 6: 
                        obs_dict, step_rewards, terms, _, infos_dict, active_agents = next_env_results[i]
                        for a in possible_agents:
                            b_rewards[step-1, i, agent_to_idx[a]] = step_rewards.get(a, 0.0)
                            next_done[i, agent_to_idx[a]] = float(terms.get(a, False))
                    
                    for a in active_agents:
                        a_idx = agent_to_idx[a]
                        b_masks[step, i, a_idx] = True
                        b_obs[step, i, a_idx] = torch.tensor(obs_dict[a], dtype=torch.bool, device=device)
                        b_sparse_masks[step, i, a_idx] = torch.tensor(infos_dict[a]['action_mask'], dtype=torch.long)

                flat_obs = b_obs[step][b_masks[step]]
                if flat_obs.shape[0] > 0:
                    t_gpu_start = time.time()
                    # --- REBUILD DENSE MASK FOR GPU ---
                    active_sparse = b_sparse_masks[step][b_masks[step]]
                    active_masks = rebuild_dense_mask(active_sparse, MAP_PROVINCES, VOCAB_SIZE, device)
                    
                    # --- PHASE 2: IDENTIFY ACTIVE UNITS ---
                    is_active_mask = ~active_masks[:, :, NONE_IDX]
                    num_active_per_batch = is_active_mask.sum(dim=1)
                    max_decode_steps = num_active_per_batch.max().item()
                    
                    if max_decode_steps == 0:
                        continue
                        
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        # Encode the board once
                        state_repr, values = net.module.encode_state(flat_obs.to(dtype=torch.bfloat16))
                    values = values.float()
                    
                    # --- PHASE 3: THE UNIT-CENTRIC ROLLOUT LOOP ---
                    batch_size = flat_obs.size(0)
                    
                    # 1. Initialize outputs with defaults
                    final_actions = torch.full((batch_size, MAP_PROVINCES), NONE_IDX, dtype=torch.long, device=device)
                    final_logprobs = torch.zeros((batch_size, MAP_PROVINCES), dtype=torch.float32, device=device)
                    
                    # 2. Build the padded active index tensor
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

                    # 3. Decode ONLY up to the max number of units
                    # NEW: Add a separate KV cache for the BC model
                    bc_kv_cache = None 
                    
                    for decode_idx in range(max_decode_steps):
                        prov_indices = padded_indices[:, decode_idx]
                        valid_step = step_mask[:, decode_idx]
                        
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            # Standard PPO forward pass ONLY
                            logits, kv_cache = net.module.decode_step(current_action, prov_indices, state_repr, kv_cache)
                        
                        # Grab the strict game engine legal action mask
                        prov_mask = active_masks[batch_indices, prov_indices, :]
                        
                        logits = logits.float()
                        # Mask using the engine mask
                        logits = logits.masked_fill(~prov_mask, -1e4)
                        
                        dist_cat = Categorical(logits=logits)
                        current_action = dist_cat.sample()
                        step_log_probs = dist_cat.log_prob(current_action)
                        
                        # 4. Scatter the results back into the 82-length tensor ONLY if the step was valid
                        final_actions[batch_indices[valid_step], prov_indices[valid_step]] = current_action[valid_step]
                        final_logprobs[batch_indices[valid_step], prov_indices[valid_step]] = step_log_probs[valid_step]
                    # 5. Collapse logprobs to 1D for PPO and assign actions
                    b_values[step][b_masks[step]] = values.squeeze()
                    b_logprobs[step][b_masks[step]] = final_logprobs.sum(dim=1)
                    
                    idx_counter = 0
                    for i in range(NUM_ENVS):
                        for a in possible_agents:
                            if b_masks[step, i, agent_to_idx[a]]:
                                act_array = final_actions[idx_counter]
                                b_actions[step, i, agent_to_idx[a]] = act_array
                                actions_to_send[i][a] = act_array.cpu().numpy()
                                idx_counter += 1
                                
                    gpu_forward_time += (time.time() - t_gpu_start)

            t_env_start = time.time()
            next_env_results = vec_env.step(actions_to_send)
            env_step_time += (time.time() - t_env_start)
            
        with torch.no_grad():
            next_value = torch.zeros((NUM_ENVS, NUM_AGENTS), device=device)
            
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

        valid = b_masks.view(-1)
        valid_cpu = valid.cpu()
        flat_obs = b_obs.view(-1, HISTORY_LENGTH, MAP_PROVINCES, 46)[valid]
        flat_act = b_actions.view(-1, MAP_PROVINCES)[valid]
        flat_logprobs = b_logprobs.view(-1)[valid]
        flat_adv = advantages.view(-1)[valid]
        flat_ret = returns.view(-1)[valid]
        flat_val = b_values.view(-1)[valid]
        
        flat_sparse_masks = b_sparse_masks.view(-1, 2000)[valid_cpu]

        b_size = flat_obs.shape[0]
        
        # 1. Find the minimum batch size across all GPUs
        local_b_size = torch.tensor([b_size], dtype=torch.long, device=device)
        max_b_tensor = local_b_size.clone()
        dist.all_reduce(max_b_tensor, op=dist.ReduceOp.MAX)
        max_b_size = max_b_tensor.item()

        # 2. Create a validity mask (True for real data, False for padding)
        valid_batch_mask = torch.ones(max_b_size, dtype=torch.bool, device=device)

        # 3. Pad all tensors if this GPU has less data than the max
        if b_size < max_b_size:
            pad_len = max_b_size - b_size
            valid_batch_mask[b_size:] = False
            
            # Pad standard tensors with zeros
            flat_obs = torch.cat([flat_obs, torch.zeros((pad_len, *flat_obs.shape[1:]), dtype=flat_obs.dtype, device=device)])
            flat_logprobs = torch.cat([flat_logprobs, torch.zeros(pad_len, dtype=flat_logprobs.dtype, device=device)])
            flat_adv = torch.cat([flat_adv, torch.zeros(pad_len, dtype=flat_adv.dtype, device=device)])
            flat_ret = torch.cat([flat_ret, torch.zeros(pad_len, dtype=flat_ret.dtype, device=device)])
            flat_val = torch.cat([flat_val, torch.zeros(pad_len, dtype=flat_val.dtype, device=device)])
            
            # Pad actions with NONE_IDX so the network ignores them during the update loop
            flat_act = torch.cat([flat_act, torch.full((pad_len, *flat_act.shape[1:]), NONE_IDX, dtype=flat_act.dtype, device=device)])
            
            # Pad sparse masks with -1
            flat_sparse_masks = torch.cat([flat_sparse_masks, torch.full((pad_len, *flat_sparse_masks.shape[1:]), -1, dtype=flat_sparse_masks.dtype, device=device)])
            
            b_size = max_b_size 

        # Normalize advantages strictly over the valid elements
        valid_adv = flat_adv[valid_batch_mask]
        if valid_adv.shape[0] > 1:
            adv_mean = valid_adv.mean()
            adv_std = valid_adv.std()
            # Apply normalization to the whole tensor, dummy data will just become shifted (which is fine, it gets masked out later)
            flat_adv = (flat_adv - adv_mean) / (adv_std + 1e-8)

        t_update_start = time.time()
        net.train()
        b_size = flat_obs.shape[0]
        
        # --- GRADIENT ACCUMULATION SETUP ---
        mb_size = 1024
        accum_steps = 4  # Accumulate gradients over 4 minibatches before syncing
        indices = np.arange(b_size)

        # Ensure we start with a clean slate before the epoch loops begin
        optimizer.zero_grad() 

        for epoch in range(update_epochs):
            np.random.shuffle(indices)
            
            # Create a list of all start indices for the minibatches
            start_indices = list(range(0, b_size, mb_size))

            
            for step_idx, start in enumerate(start_indices):
                end = start + mb_size
                mb_idx = indices[start:end]
                
                # Inside the epoch loop, grab the validity mask for this minibatch
                mb_valid = valid_batch_mask[mb_idx]

                mb_obs = flat_obs[mb_idx].to(dtype=torch.bfloat16).contiguous()
                mb_act = flat_act[mb_idx].contiguous()
                mb_logprobs = flat_logprobs[mb_idx]
                mb_adv = flat_adv[mb_idx]
                mb_ret = flat_ret[mb_idx]
                
                # --- REBUILD DENSE MASK FOR UPDATE ---
                mb_sparse_gpu = flat_sparse_masks[mb_idx].to(device)
                mb_masks_gpu = rebuild_dense_mask(mb_sparse_gpu, MAP_PROVINCES, VOCAB_SIZE, device)

                # Are we syncing on this step? (Yes, if we hit accum_steps or the very last batch)
                is_last_batch = (step_idx + 1) == len(start_indices)
                sync_this_step = (step_idx + 1) % accum_steps == 0 or is_last_batch

                # --- DDP NO_SYNC CONTEXT MANAGER ---
                my_context = net.no_sync() if not sync_this_step else contextlib.nullcontext()
                
                with my_context:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        # 1. Encode states once (Value head uses this directly)
                        state_repr, new_val = net.module.encode_state(mb_obs)
                        
                        # 2. Rebuild padded indices to match the rollout sequence exactly
                        is_active_mask = (mb_act != NONE_IDX)
                        batch_size = mb_obs.size(0)
                        
                        # Flattened targets and masks (keeps PyTorch's native left-to-right order)
                        active_targets = mb_act[is_active_mask]
                        active_masks_gpu = mb_masks_gpu[is_active_mask]
                        
                        # Track which batch item each active unit belongs to (for scattering logprobs later)
                        batch_indices = torch.where(is_active_mask)[0]
                        
                        num_active_per_batch = is_active_mask.sum(dim=1)
                        max_active = num_active_per_batch.max().item()

                        # Default dummy tensors in case of a completely empty batch
                        active_logits = torch.zeros((0, VOCAB_SIZE), device=device, dtype=torch.bfloat16)
                        bc_active_logits = torch.zeros((0, VOCAB_SIZE), device=device, dtype=torch.bfloat16)

                        # 3. Only run the decoders if there are actual units to move
                        if max_active > 0:
                            padded_indices = torch.zeros((batch_size, max_active), dtype=torch.long, device=device)
                            step_mask = torch.zeros((batch_size, max_active), dtype=torch.bool, device=device)
                            
                            # Pack the active indices to the front of the sequence
                            for b in range(batch_size):
                                valid_idx = torch.where(is_active_mask[b])[0]
                                count = len(valid_idx)
                                if count > 0:
                                    padded_indices[b, :count] = valid_idx
                                    step_mask[b, :count] = True

                            b_idx_expand = torch.arange(batch_size, device=device).unsqueeze(1)
                            
                            # --- PPO MODEL PARALLEL DECODE ---
                            # Gather only the states and actions for active units
                            active_states = state_repr[b_idx_expand, padded_indices, :]
                            active_actions = mb_act[b_idx_expand, padded_indices]
                            
                            if max_active > 1:
                                action_emb = net.module.action_embedding(active_actions[:, :-1])
                                start_emb = net.module.start_token_embedding.expand(batch_size, 1, -1)
                                shifted_emb = torch.cat([start_emb, action_emb], dim=1)
                            else:
                                shifted_emb = net.module.start_token_embedding.expand(batch_size, 1, -1)
                                
                            decoder_input = shifted_emb + active_states
                            active_seq_hidden, _ = net.module.causal_decoder_block(decoder_input, kv_cache=None)
                            
                            # Boolean masking automatically flattens matching the active_targets order!
                            active_hidden = active_seq_hidden[step_mask] 
                            active_logits = net.module.action_head(active_hidden)
                            active_logits = active_logits.masked_fill(~active_masks_gpu, -1e4)

                            # --- BC MODEL PARALLEL DECODE (Frozen) ---
                            with torch.no_grad():
                                bc_state_repr, _ = bc_model.encode_state(mb_obs)
                                bc_active_states = bc_state_repr[b_idx_expand, padded_indices, :]
                                
                                if max_active > 1:
                                    bc_action_emb = bc_model.action_embedding(active_actions[:, :-1])
                                    bc_start_emb = bc_model.start_token_embedding.expand(batch_size, 1, -1)
                                    bc_shifted_emb = torch.cat([bc_start_emb, bc_action_emb], dim=1)
                                else:
                                    bc_shifted_emb = bc_model.start_token_embedding.expand(batch_size, 1, -1)
                                    
                                bc_decoder_input = bc_shifted_emb + bc_active_states
                                bc_seq_hidden, _ = bc_model.causal_decoder_block(bc_decoder_input, kv_cache=None)
                                
                                bc_active_hidden = bc_seq_hidden[step_mask]
                                bc_active_logits = bc_model.action_head(bc_active_hidden)
                                bc_active_logits = bc_active_logits.masked_fill(~active_masks_gpu, -1e4)
                                
                    # Cast back to 32-bit for stable distribution math
                    active_logits = active_logits.float()
                    bc_active_logits = bc_active_logits.float()
                    new_val = new_val.float()
                    
                    # 4. Calculate probabilities strictly for active units
                    if max_active > 0:
                        dist_cat = Categorical(logits=active_logits)
                        active_logp = dist_cat.log_prob(active_targets)
                        active_entropy = dist_cat.entropy()
                        
                        bc_dist_cat = Categorical(logits=bc_active_logits)
                        kl_div_active = torch.distributions.kl.kl_divergence(dist_cat, bc_dist_cat)
                    else:
                        active_logp = torch.zeros(0, device=device)
                        active_entropy = torch.zeros(0, device=device)
                        kl_div_active = torch.zeros(0, device=device)

                    # 5. Scatter the isolated math back into standard batch sizes
                    new_logp = torch.zeros(batch_size, device=device)
                    new_logp.scatter_add_(0, batch_indices, active_logp)
                    
                    entropy_sums = torch.zeros(batch_size, device=device)
                    entropy_sums.scatter_add_(0, batch_indices, active_entropy)
                    
                    kl_sums = torch.zeros(batch_size, device=device)
                    kl_sums.scatter_add_(0, batch_indices, kl_div_active)
                    
                    # --- NEW: MASKED PPO MATH ---
                    # We compute the means by summing and dividing by the number of valid items
                    valid_count = mb_valid.sum() + 1e-8 
                    
                    entropy = (entropy_sums * mb_valid.float()).sum() / valid_count
                    kl_divergence = (kl_sums * mb_valid.float()).sum() / valid_count

                    logratio = new_logp - mb_logprobs
                    ratio = logratio.exp()
                    
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    
                    # Maximize objective, mask out invalid, and average
                    pg_loss_unmasked = torch.max(pg_loss1, pg_loss2)
                    pg_loss = (pg_loss_unmasked * mb_valid.float()).sum() / valid_count
                    
                    v_loss_unmasked = 0.5 * ((new_val.view(-1) - mb_ret) ** 2)
                    v_loss = (v_loss_unmasked * mb_valid.float()).sum() / valid_count
                    
                    loss = pg_loss - ent_coef * entropy + v_loss * v_coef + kl_coef * kl_divergence
                    
                    loss = loss / accum_steps
                    loss.backward()

                # --- ONLY STEP AND ZERO IF WE SYNCED ---
                if sync_this_step:
                    for param in net.parameters():
                        if param.grad is not None:
                            # Operate directly on the gradient tensor, no .data
                            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                            param.grad /= dist.get_world_size()

                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Make sure to clear the new variables out of VRAM!
                del mb_obs, mb_act, mb_masks_gpu, new_val
                if max_active > 0:
                    del active_seq_hidden, bc_seq_hidden, active_logits, bc_active_logits, dist_cat, bc_dist_cat
                    
        update_time = time.time() - t_update_start

        if global_rank == 0:
            end_time = time.time()  
            total_time = end_time - start_time
            global_steps = NUM_ENVS * NUM_STEPS * NUM_AGENTS * dist.get_world_size()
            sps = int(global_steps / total_time)  
            avg_reward = b_rewards.sum() / (NUM_ENVS * NUM_AGENTS) 
            print(f"--- Timing Breakdown for Update {update} ---")
            print(f"  Total Time: {total_time:.2f}s")
            print(f"  CPU Env Step Time: {env_step_time:.2f}s ({(env_step_time/total_time)*100:.1f}%)")
            print(f"  GPU Forward Pass:  {gpu_forward_time:.2f}s ({(gpu_forward_time/total_time)*100:.1f}%)")
            print(f"  GPU Update Loop:   {update_time:.2f}s ({(update_time/total_time)*100:.1f}%)")
            print("--------------------------------------")
            print(f"Update {update}/{num_updates} | SPS {sps} | Avg Reward {avg_reward:.2f} | Loss {loss.item():.4f} | Ent {entropy.item():.4f}")   
            
            writer.add_scalar("Perf/SPS", sps, update)
            writer.add_scalar("Reward/Avg_Reward", avg_reward, update)
            writer.add_scalar("Loss/Policy_Loss", loss.item(), update)
            writer.add_scalar("Loss/Value_Loss", v_loss.item(), update)
            writer.add_scalar("Loss/Entropy", entropy.item(), update)

            # --- WANDB LOGGING ---
            wandb.log({
                "Perf/SPS": sps,
                "Reward/Avg_Reward": avg_reward,
                "Loss/Policy_Loss": loss.item(),
                "Loss/Value_Loss": v_loss.item(),
                "Loss/Entropy": entropy.item(),
                "Loss/KL_Div": kl_divergence.item(),
                "global_step": update * global_steps 
            }, step=update)

            if global_rank == 0:
                if update % 10 == 0:
                    ckpt_path = f"./checkpoints/diplomacy_ppo_KV_update_{update}.pth"
                    torch.save(net.module.state_dict(), ckpt_path)
                    print(f"  -> Saved checkpoint to {ckpt_path}")
                    
                    # Run the memory-heavy evaluation ONLY on GPU 0
                    evaluate_and_save_game(net.module, device, update)
                    
                    eval_log_path = f"./eval_games/eval_game_KV_update_{update}.txt"
                    if os.path.exists(eval_log_path):
                        wandb.save(eval_log_path)
        b_masks.zero_()
        b_rewards.zero_()
        
    vec_env.close()
    if global_rank == 0:
        writer.close()
        wandb.finish()
    dist.destroy_process_group()