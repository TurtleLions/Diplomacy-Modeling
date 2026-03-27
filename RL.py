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

from diplomacy_helpers import DiplomacyTransformer, DiplomacyTransformerEnv

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
            
            obs_tensor = torch.stack([torch.tensor(obs[a]) for a in active_agents]).to(device)
            
            # --- NEW: Rebuild Dense Mask ---
            sparse_masks_tensor = torch.stack([torch.tensor(infos[a]['action_mask'], dtype=torch.long) for a in active_agents]).to(device)
            masks_tensor = rebuild_dense_mask(sparse_masks_tensor, env.num_provinces, env.vocab_size, device)
            
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    state_repr, _ = net.encode_state(obs_tensor)
                
                batch_size = obs_tensor.size(0)
                current_action = torch.zeros(batch_size, dtype=torch.long, device=device)
                kv_cache = None
                actions_list = []
                
                for prov_idx in range(env.num_provinces):
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits, kv_cache = net.decode_step(current_action, prov_idx, state_repr, kv_cache)
                    
                    logits = logits.float()
                    prov_mask = masks_tensor[:, prov_idx, :]
                    logits = logits.masked_fill(~prov_mask, -1e4)
                    
                    # In eval, we take the best move (argmax) instead of random sampling
                    current_action = torch.argmax(logits, dim=-1)
                    actions_list.append(current_action)
                
                actions = torch.stack(actions_list, dim=1)
                
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
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    NUM_ENVS = 16 
    NUM_STEPS = 128
    NUM_AGENTS = 7
    HISTORY_LENGTH = 3

    if global_rank == 0:
        print("--- STARTING NATIVE TRANSFORMER PPO ---")

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
    del dummy_env

    if global_rank == 0:
        print(f"Detected {MAP_PROVINCES} provinces and {VOCAB_SIZE} actions")

    vec_env = SubprocVecDiplomacy(num_envs=NUM_ENVS)
    net = DiplomacyTransformer(num_provinces=MAP_PROVINCES, history_length=HISTORY_LENGTH, vocab_size=VOCAB_SIZE).to(device)
    
    bc_weights_path = "diplomacy_transformer_bc.pth"
    if os.path.exists(bc_weights_path):
        # We set strict=False just in case the new Value Head causes a warning
        # Since BC didn't train a value head, it will just initialize randomly
        net.load_state_dict(torch.load(bc_weights_path, map_location=device), strict=False)
        if global_rank == 0:
            print(f"Successfully loaded pre-trained weights from {bc_weights_path}")
    else:
        if global_rank == 0:
            print(f"WARNING: BC weights not found at {bc_weights_path}. Starting from scratch.")

    net = DDP(net, device_ids=[local_rank])
    optimizer = optim.Adam(net.parameters(), lr=1e-6, eps=1e-5)

    # --- MEMORY FIX: Rollout Buffers ---
    b_obs = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS, HISTORY_LENGTH, MAP_PROVINCES, 19), dtype=torch.float32, device=device)
    b_actions = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS, MAP_PROVINCES), dtype=torch.long, device=device)
    b_logprobs = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS), dtype=torch.float32, device=device)
    b_rewards = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS), dtype=torch.float32, device=device)
    b_dones = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS), dtype=torch.float32, device=device)
    b_values = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS), dtype=torch.float32, device=device)
    b_masks = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS), dtype=torch.bool, device=device) 
    
    b_sparse_masks = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_AGENTS, 1200), dtype=torch.long).pin_memory()

    num_updates = 1000
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.0
    v_coef = 0.1
    update_epochs = 4

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

    for update in range(1, num_updates + 1):
        start_time = time.time()
        net.eval()
        
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
                        b_obs[step, i, a_idx] = torch.tensor(obs_dict[a], device=device)
                        b_sparse_masks[step, i, a_idx] = torch.tensor(infos_dict[a]['action_mask'], dtype=torch.long)

                flat_obs = b_obs[step][b_masks[step]]
                if flat_obs.shape[0] > 0:
                    
                    # --- REBUILD DENSE MASK FOR GPU ---
                    active_sparse = b_sparse_masks[step][b_masks[step].cpu()].to(device)
                    active_masks = rebuild_dense_mask(active_sparse, MAP_PROVINCES, VOCAB_SIZE, device)
                    
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        # Encode the board once
                        state_repr, values = net.module.encode_state(flat_obs)
                    values = values.float()
                    
                    batch_size = flat_obs.size(0)
                    current_action = torch.zeros(batch_size, dtype=torch.long, device=device)
                    kv_cache = None
                    
                    sampled_actions_list = []
                    log_probs_list = []
                    
                    # Decode 81 times sequentially
                    for prov_idx in range(MAP_PROVINCES):
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            logits, kv_cache = net.module.decode_step(current_action, prov_idx, state_repr, kv_cache)
                        
                        logits = logits.float()
                        
                        # Apply the mask for THIS specific province
                        prov_mask = active_masks[:, prov_idx, :]
                        logits = logits.masked_fill(~prov_mask, -1e4)
                        
                        dist_cat = Categorical(logits=logits)
                        current_action = dist_cat.sample()
                        
                        sampled_actions_list.append(current_action)
                        log_probs_list.append(dist_cat.log_prob(current_action))
                    
                    sampled_actions = torch.stack(sampled_actions_list, dim=1)
                    # We keep log_p as a 2D tensor [batch, 81]. The .sum(dim=1) happens on the next line in your code!
                    log_p = torch.stack(log_probs_list, dim=1)
                    
                    b_values[step][b_masks[step]] = values.squeeze()
                    b_logprobs[step][b_masks[step]] = log_p.sum(dim=1) 
                    
                    idx_counter = 0
                    for i in range(NUM_ENVS):
                        for a in possible_agents:
                            if b_masks[step, i, agent_to_idx[a]]:
                                act_array = sampled_actions[idx_counter]
                                b_actions[step, i, agent_to_idx[a]] = act_array
                                actions_to_send[i][a] = act_array.cpu().numpy()
                                idx_counter += 1

            next_env_results = vec_env.step(actions_to_send)
            
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
        flat_obs = b_obs.view(-1, HISTORY_LENGTH, MAP_PROVINCES, 19)[valid]
        flat_act = b_actions.view(-1, MAP_PROVINCES)[valid]
        flat_logprobs = b_logprobs.view(-1)[valid]
        flat_adv = advantages.view(-1)[valid]
        flat_ret = returns.view(-1)[valid]
        flat_val = b_values.view(-1)[valid]
        
        valid_cpu = valid.cpu()
        flat_sparse_masks = b_sparse_masks.view(-1, 1200)[valid_cpu]

        if flat_adv.shape[0] > 1:
            flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        net.train()
        b_size = flat_obs.shape[0]
        mb_size = 512
        indices = np.arange(b_size)

        for epoch in range(update_epochs):
            np.random.shuffle(indices)
            for start in range(0, b_size, mb_size):
                end = start + mb_size
                mb_idx = indices[start:end]
                
                mb_obs = flat_obs[mb_idx].contiguous()
                mb_act = flat_act[mb_idx].contiguous()
                mb_logprobs = flat_logprobs[mb_idx]
                mb_adv = flat_adv[mb_idx]
                mb_ret = flat_ret[mb_idx]
                
                # --- REBUILD DENSE MASK FOR UPDATE ---
                mb_sparse_gpu = flat_sparse_masks[mb_idx].to(device)
                mb_masks_gpu = rebuild_dense_mask(mb_sparse_gpu, MAP_PROVINCES, VOCAB_SIZE, device)

                optimizer.zero_grad() 

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, new_val = net(mb_obs, mb_act)
                
                # ADD THESE TWO LINES TO CAST BACK TO 32-BIT
                logits = logits.float()
                new_val = new_val.float()
                
                logits = logits.masked_fill(~mb_masks_gpu, -1e4)
                    
                dist_cat = Categorical(logits=logits)
                new_logp = dist_cat.log_prob(mb_act).sum(dim=1)
                entropy = dist_cat.entropy().sum(dim=1).mean()

                logratio = new_logp - mb_logprobs
                ratio = logratio.exp()
                
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                v_loss = 0.5 * ((new_val.squeeze() - mb_ret) ** 2).mean()
                loss = pg_loss - ent_coef * entropy + v_loss * v_coef
                
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()

        if global_rank == 0:
            end_time = time.time()  
            global_steps = NUM_ENVS * NUM_STEPS * NUM_AGENTS * dist.get_world_size()
            sps = int(global_steps / (end_time - start_time))  
            avg_reward = b_rewards.sum() / (NUM_ENVS * NUM_AGENTS) 
            
            print(f"Update {update}/{num_updates} | SPS {sps} | Avg Reward {avg_reward:.2f} | Loss {loss.item():.4f} | Ent {entropy.item():.4f}")   
            writer.add_scalar("Perf/SPS", sps, update)
            writer.add_scalar("Reward/Avg_Reward", avg_reward, update)
            writer.add_scalar("Loss/Policy_Loss", loss.item(), update)
            writer.add_scalar("Loss/Value_Loss", v_loss.item(), update)
            writer.add_scalar("Loss/Entropy", entropy.item(), update)

            if update % 50 == 0:
                ckpt_path = f"./checkpoints/diplomacy_ppo_KV_update_{update}.pth"
                torch.save(net.module.state_dict(), ckpt_path)
                print(f"  -> Saved checkpoint to {ckpt_path}")
                evaluate_and_save_game(net.module, device, update)

        b_masks.zero_()
        b_rewards.zero_()
        
    vec_env.close()
    if global_rank == 0:
        writer.close()
    dist.destroy_process_group()