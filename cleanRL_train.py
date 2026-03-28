import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import wandb
import concurrent.futures
import multiprocessing as mp

# Import your pristine environment and model
from diplomacy_helpers import DiplomacyTransformerEnv, DiplomacyTransformer
from RL import rebuild_dense_mask

# --- 1. HYPERPARAMETERS & SETUP ---
NUM_ENVS = 24              # Uses 24 of your 28 CPU cores
NUM_STEPS = 150            # Steps per environment before updating
BATCH_SIZE = NUM_ENVS * NUM_STEPS # Total transitions per agent (~3600)
MINIBATCH_SIZE = 512       # Memory-safe chunk for 64GB GPU
UPDATE_EPOCHS = 4          # How many times to loop over the batch
LEARNING_RATE = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _worker(remote, parent_remote):
    """This function runs entirely inside the isolated CPU core."""
    parent_remote.close()
    # Instantiate the environment INSIDE the worker to avoid Pickling errors
    env = DiplomacyTransformerEnv(history_length=3)
    
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs_dict, rewards_dict, terms_dict, truncs_dict, infos_dict = env.step(data)
                # Only send back standard dictionaries and numpy arrays (easily pickleable)
                remote.send((obs_dict, rewards_dict, terms_dict, infos_dict))
            elif cmd == 'reset':
                obs_dict, infos_dict = env.reset()
                remote.send((obs_dict, infos_dict))
            elif cmd == 'close':
                remote.close()
                break
        except Exception as e:
            print(f"Worker Error: {e}")
            remote.send(("ERROR", e))
            break

class VectorizedDiplomacy:
    """Manages the communication pipes to the isolated CPU workers."""
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = [mp.Process(target=_worker, args=(work_remote, remote))
                          for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        
        for p in self.processes:
            p.daemon = True # Ensure background cores die if the main script crashes
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        
        obs_dicts = [res[0] for res in results]
        infos_dicts = [res[1] for res in results]
        return obs_dicts, infos_dicts

    def step(self, actions_list):
        for remote, action in zip(self.remotes, actions_list):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        
        obs = [res[0] for res in results]
        rewards = [res[1] for res in results]
        terms = [res[2] for res in results]
        infos = [res[3] for res in results]
        return obs, rewards, terms, infos
        
    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()

# --- 3. EVALUATION & GAME LOGGER ---
def evaluate_and_save_game(model, vocab_size, num_provinces, update_num):
    """Plays a single deterministic game and saves the orders to read."""
    os.makedirs("eval_games", exist_ok=True)
    filepath = f"eval_games/game_update_{update_num:04d}.txt"
    
    eval_env = DiplomacyTransformerEnv(history_length=3)
    obs_dict, infos_dict = eval_env.reset()
    
    with open(filepath, "w") as f:
        f.write(f"--- DIPLOMACY EVALUATION: UPDATE {update_num} ---\n\n")
        
        while eval_env.agents:
            phase = eval_env.game.get_current_phase()
            f.write(f"=== PHASE: {phase} ===\n")
            
            actions_to_send = {}
            for agent in eval_env.agents:
                agent_obs = torch.tensor(obs_dict[agent], dtype=torch.float32, device=device).unsqueeze(0)
                
                # FIX: Add .unsqueeze(0) to fake a batch dimension
                sparse_mask = torch.tensor(infos_dict[agent]['action_mask'], device=device).unsqueeze(0)
                dense_mask = rebuild_dense_mask(sparse_mask, num_provinces, vocab_size, device)
                
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    state_repr, _ = model.encode_state(agent_obs)
                    current_action = torch.zeros(1, dtype=torch.long, device=device)
                    kv_cache = None
                    actions = []
                    
                    for prov_idx in range(num_provinces):
                        logits, kv_cache = model.decode_step(current_action, prov_idx, state_repr, kv_cache)
                        
                        # FIX: Pull the mask correctly for the batch
                        prov_mask = dense_mask[:, prov_idx, :]
                        logits = logits.float().masked_fill(~prov_mask, -1e9)
                        
                        # Deterministic: take the max probability (argmax) instead of sampling
                        current_action = torch.argmax(logits, dim=-1)
                        actions.append(current_action)
                
                final_actions = torch.stack(actions, dim=1).squeeze(0).cpu().numpy()
                actions_to_send[agent] = final_actions
                
                # Write orders to file
                f.write(f"{agent} Orders:\n")
                for order_idx in final_actions:
                    order_str = eval_env.idx_to_order[order_idx]
                    if order_str != 'NONE':
                        f.write(f"  - {order_str}\n")
            
            f.write("\n")
            obs_dict, _, terms_dict, _, infos_dict = eval_env.step(actions_to_send)
    
    print(f"[*] Evaluation game saved to {filepath}")

# --- 4. THE CLEANRL PPO ALGORITHM ---
def main():
    # Force AMD hardware context cleanly
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        os.environ["ROCR_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

    wandb.init(
        project="Diplomacy-Transformer",
        entity="turtlelions-uc-san-diego",
        name="CleanRL_Sync_Run_01",
        config={"lr": LEARNING_RATE, "batch_size": BATCH_SIZE, "num_envs": NUM_ENVS}
    )

    envs = VectorizedDiplomacy(NUM_ENVS)
    dummy_env = DiplomacyTransformerEnv() # Just to grab dimensions
    
    model = DiplomacyTransformer(
        input_dim=25, 
        num_provinces=dummy_env.num_provinces, 
        history_length=3, 
        vocab_size=dummy_env.vocab_size
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)

    num_updates = 10000
    global_step = 0

    for update in range(1, num_updates + 1):
        start_time = time.time()
        
        # Tensor allocations for the rollout
        b_obs = []
        b_actions = []
        b_logprobs = []
        b_rewards = []
        b_dones = []
        b_values = []
        
        # Logging metrics
        epoch_rewards = []
        epoch_sc_counts = []
        
        obs_dicts, infos_dicts = envs.reset()
        
        # --- ROLLOUT PHASE ---
        model.eval()
        for step in range(NUM_STEPS):
            actions_list_for_envs = []
            
            # We process all 24 envs sequentially on the GPU to utilize batching
            # (In PyTorch, a batch of 24 is much faster than 24 individual passes)
            for i in range(NUM_ENVS):
                env_actions = {}
                active_agents = [a for a in dummy_env.possible_agents if a in obs_dicts[i]]
                
                for agent in active_agents:
                    agent_obs = torch.tensor(obs_dicts[i][agent], dtype=torch.float32, device=device).unsqueeze(0)
                    
                    # FIX: Add .unsqueeze(0) here too
                    sparse_mask = torch.tensor(infos_dicts[i][agent]['action_mask'], device=device).unsqueeze(0)
                    dense_mask = rebuild_dense_mask(sparse_mask, dummy_env.num_provinces, dummy_env.vocab_size, device)
                    
                    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        state_repr, state_value = model.encode_state(agent_obs)
                        current_action = torch.zeros(1, dtype=torch.long, device=device)
                        kv_cache = None
                        actions, logprobs = [], []
                        
                        for prov_idx in range(dummy_env.num_provinces):
                            logits, kv_cache = model.decode_step(current_action, prov_idx, state_repr, kv_cache)
                            
                            # FIX: Pull the mask correctly for the batch
                            prov_mask = dense_mask[:, prov_idx, :]
                            logits = logits.float().masked_fill(~prov_mask, -1e9)
                            
                            dist = Categorical(logits=logits)
                            current_action = dist.sample()
                            
                            actions.append(current_action)
                            logprobs.append(dist.log_prob(current_action))

                    final_actions = torch.stack(actions, dim=1)
                    final_logprobs = torch.stack(logprobs, dim=1).sum(dim=1)
                    
                    env_actions[agent] = final_actions.squeeze(0).cpu().numpy()
                    
                    # Store data
                    b_obs.append(agent_obs)
                    b_actions.append(final_actions)
                    b_logprobs.append(final_logprobs)
                    b_values.append(state_value.squeeze(-1))
                
                actions_list_for_envs.append(env_actions)
                
            # Step all 24 environments on the CPU simultaneously
            next_obs_dicts, rewards_dicts, terms_dicts, next_infos_dicts = envs.step(actions_list_for_envs)
            
            for i in range(NUM_ENVS):
                active_agents = [a for a in dummy_env.possible_agents if a in obs_dicts[i]]
                for agent in active_agents:
                    reward = rewards_dicts[i].get(agent, 0.0)
                    done = terms_dicts[i].get(agent, False)
                    
                    b_rewards.append(torch.tensor([reward], device=device))
                    b_dones.append(torch.tensor([1.0 if done else 0.0], device=device))
                    
                    epoch_rewards.append(reward)
                    
                    # Log SC Count if the year updated
                    sc_count = len(envs.envs[i].game.get_centers(agent))
                    epoch_sc_counts.append(sc_count)

            obs_dicts = next_obs_dicts
            infos_dicts = next_infos_dicts
            global_step += NUM_ENVS * len(dummy_env.possible_agents)

        # Flatten buffers
        b_obs = torch.cat(b_obs)
        b_actions = torch.cat(b_actions)
        b_logprobs = torch.cat(b_logprobs)
        b_rewards = torch.cat(b_rewards)
        b_dones = torch.cat(b_dones)
        b_values = torch.cat(b_values)

        # --- GAE CALCULATION ---
        with torch.no_grad():
            # A rough estimate for next value since agents die/terminate asynchronously
            next_value = torch.zeros(1, device=device) 
            advantages = torch.zeros_like(b_rewards)
            lastgaelam = 0
            for t in reversed(range(len(b_rewards))):
                if t == len(b_rewards) - 1:
                    nextnonterminal = 1.0 - b_dones[-1]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - b_dones[t]
                    nextvalues = b_values[t + 1]
                delta = b_rewards[t] + GAMMA * nextvalues * nextnonterminal - b_values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + b_values

        # --- OPTIMIZATION PHASE ---
        model.train()
        b_inds = np.arange(len(b_obs))
        clipfracs = []
        
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_obs), MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                # Teacher Forcing Fast Pass
                action_logits, new_values = model(b_obs[mb_inds], b_actions[mb_inds])
                new_values = new_values.squeeze(-1)
                
                dist = Categorical(logits=action_logits)
                new_logprobs = dist.log_prob(b_actions[mb_inds]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                logratio = new_logprobs - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # PPO Clipping
                mb_advantages = advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_values - returns[mb_inds]) ** 2).mean()
                loss = pg_loss + 0.5 * v_loss - ENT_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        # --- LOGGING & EVALUATION ---
        sps = int((NUM_ENVS * NUM_STEPS) / (time.time() - start_time))
        avg_reward = np.mean(epoch_rewards)
        avg_scs = np.mean(epoch_sc_counts)

        print(f"Update {update:04d} | SPS: {sps} | Loss: {loss.item():.4f} | Avg Reward: {avg_reward:.4f} | Avg SCs: {avg_scs:.2f}")
        
        wandb.log({
            "charts/SPS": sps,
            "charts/learning_rate": LEARNING_RATE,
            "losses/policy_loss": pg_loss.item(),
            "losses/value_loss": v_loss.item(),
            "losses/entropy": entropy.item(),
            "rewards/avg_reward": avg_reward,
            "game/avg_supply_centers": avg_scs
        }, step=global_step)

        # Every 50 updates, save a game and save the model
        if update % 50 == 0:
            evaluate_and_save_game(model, dummy_env.vocab_size, dummy_env.num_provinces, update)
            torch.save(model.state_dict(), f"diplomacy_model_update_{update}.pth")

if __name__ == "__main__":
    main()