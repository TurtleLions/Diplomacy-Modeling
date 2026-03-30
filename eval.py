import os
import torch
import argparse
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

def evaluate_and_save_game(checkpoint_path, save_dir="./eval_games", max_steps=150):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")

    # 1. Initialize Environment
    env = DiplomacyTransformerEnv(history_length=3)
    obs, infos = env.reset()

    # 2. Initialize Model
    net = DiplomacyTransformer(
        num_provinces=env.num_provinces, 
        history_length=3, 
        vocab_size=env.vocab_size
    ).to(device)

    # 3. Load Weights
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Allow strict=False in case the value head mismatches
        net.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        print(f"Successfully loaded weights from {checkpoint_path}")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}. Playing with random weights.")
    
    net.eval()

    # Determine log name based on checkpoint
    ckpt_name = os.path.basename(checkpoint_path).split('.')[0] if checkpoint_path else "random_weights"
    log_path = os.path.join(save_dir, f"eval_game_{ckpt_name}.txt")

    # 4. Run Game Loop
    print(f"Starting evaluation game... Logging to {log_path}")
    with open(log_path, "w") as f:
        f.write(f"Evaluation Game - Model: {ckpt_name}\n")
        f.write("========================================\n")
        
        step_count = 0
        while len(env.agents) > 0 and step_count < max_steps:
            phase_name = env.game.get_current_phase()
            f.write(f"\n--- Phase {phase_name} ---\n")
            active_agents = env.agents
            
            obs_tensor = torch.stack([torch.tensor(obs[a]) for a in active_agents]).to(device)
            sparse_masks_tensor = torch.stack([torch.tensor(infos[a]['action_mask'], dtype=torch.long) for a in active_agents]).to(device)
            masks_tensor = rebuild_dense_mask(sparse_masks_tensor, env.num_provinces, env.vocab_size, device)
            
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    state_repr, _ = net.encode_state(obs_tensor)
                
                batch_size = obs_tensor.size(0)
                current_action = torch.zeros(batch_size, dtype=torch.long, device=device)
                kv_cache = None
                actions_list = []
                
                for prov_idx in range(env.num_provinces):
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        logits, kv_cache = net.decode_step(current_action, prov_idx, state_repr, kv_cache)
                    
                    logits = logits.float()
                    prov_mask = masks_tensor[:, prov_idx, :]
                    logits = logits.masked_fill(~prov_mask, -1e4)
                    
                    # Greedy selection for evaluation
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
            
    print(f"Game complete! Log saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Diplomacy Transformer.")
    parser.add_argument("--checkpoint", type=str, default="diplomacy_transformer_bc.pth", 
                        help="Path to the .pth model checkpoint.")
    parser.add_argument("--out_dir", type=str, default="./eval_games", 
                        help="Directory to save the game text log.")
    parser.add_argument("--max_steps", type=int, default=150, 
                        help="Maximum number of phases to simulate.")
    
    args = parser.parse_args()
    
    evaluate_and_save_game(
        checkpoint_path=args.checkpoint, 
        save_dir=args.out_dir,
        max_steps=args.max_steps
    )