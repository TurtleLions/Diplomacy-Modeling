import sys
import os
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.train import run_rl
import torch
import numpy as np
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
from sf2_env import SF2DiplomacyEnv
from sf2_model import DiplomacySF2Model

from dotenv import load_dotenv

def make_diplomacy_env(full_env_name, cfg=None, env_config=None, render_mode=None, **kwargs):
    return SF2DiplomacyEnv(full_env_name, cfg, env_config, render_mode, **kwargs)

def make_custom_model(cfg, obs_space, action_space):
    return DiplomacySF2Model(cfg, obs_space, action_space)

def main():
    load_dotenv()  # Loads variables from your .env file
    wandb_key = os.getenv("WANDB_KEY")
    if wandb_key:
        # WandB automatically authenticates if this specific OS variable is set
        os.environ["WANDB_API_KEY"] = wandb_key
    else:
        print("WARNING: WANDB_KEY not found in .env file. WandB might fail to sync.")
    # 1. Register Environment & Model Customizations
    register_env("diplomacy_transformer_v0", make_diplomacy_env)
    global_model_factory().register_actor_critic_factory(make_custom_model)
    
    # 2. Hardcode the base hardware and model config
    # FIXED: Added [1:] to sys.argv to ignore the script name
    argv = sys.argv[1:] + [
        "--env=diplomacy_transformer_v0",
        "--experiment=diplomacy_run_06_async",
        "--train_dir=/data/restanislao/sf2_runs",
        "--save_every_sec=600", # Save every 30 minutes instead of every 2 mins
        "--keep_checkpoints=100",  # Only keep the newest weights per policy
        
        # --- WANDB INTEGRATION ---
        "--with_wandb=True",
        "--wandb_project=Diplomacy-Transformer",
        "--wandb_user=turtlelions-uc-san-diego",
        "--wandb_group=PBT_Run_01",
        
        # --- CLUSTER SCALING ---
        "--num_workers=24",          
        "--num_envs_per_worker=2",   
        "--device=gpu",              
        
        # --- SAFETY OVERRIDES ---
        "--heartbeat_interval=60",             # Give the GPU more time to breathe
        "--heartbeat_reporting_interval=600",  # Don't kill the script unless 10 minutes pass without a ping
        

        # --- PBT (MULTI-GPU EVOLUTION) ---
        "--num_policies=1",          
        "--with_pbt=False",
        "--pbt_period_env_steps=500000",   
        "--pbt_start_mutation=2000000",    
        "--pbt_replace_fraction=0.3",      
        "--pbt_mutation_rate=0.15",        
        "--pbt_target_objective=episode_extra_stats/supply_centers",
        "--pbt_optimize_gamma=False",
        
        # How much to multiply the hyperparameters by when mutating (e.g., lr * 1.1)
        "--pbt_perturb_min=1.05", # 5% minimum mutation
        "--pbt_perturb_max=1.15", # 15% maximum mutation
        
        # --- PPO & BATCHING ---
        "--rollout=64",             
        "--batch_size=512",         
        "--num_epochs=2",            
        "--learning_rate=1e-5",
        "--use_rnn=False",
        "--max_grad_norm=1.0",

        # --- TRAINING LENGTH ---
        # Run for 100 million environment steps before shutting down
        "--train_for_env_steps=50000000",
        
        # --- CUSTOM TRANSFORMER VARIABLES ---
        "--vocab_size=22231",        
        "--history_length=3",
        "--adaptive_stddev=True",
        "--continuous_tanh_scale=0.0"

    ]
    
    parser, partial_cfg = parse_sf_args(argv=argv)
    
    # Add custom arguments to the parser so SF2 doesn't complain about them
    parser.add_argument("--vocab_size", type=int, default=22231)
    parser.add_argument("--history_length", type=int, default=3)
    
    cfg = parse_full_cfg(parser, argv)
    
    # 3. Ignite SF2
    status = run_rl(cfg)
    return status

if __name__ == '__main__':
    main()