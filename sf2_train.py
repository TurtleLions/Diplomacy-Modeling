import sys
import os
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.train import run_rl

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
        "--experiment=diplomacy_run_01_async",
        "--train_dir=./sf2_runs",
        
        # --- WANDB INTEGRATION ---
        "--with_wandb=True",
        "--wandb_project=Diplomacy-Transformer",
        "--wandb_user=turtlelions-uc-san-diego",
        "--wandb_group=PBT_Run_01",
        
        # --- CLUSTER SCALING ---
        "--num_workers=24",          
        "--num_envs_per_worker=4",   
        "--device=gpu",              
        
        # --- PBT (MULTI-GPU EVOLUTION) ---
        "--num_policies=4",          
        "--with_pbt=True",
        "--pbt_period_env_steps=500000",   
        "--pbt_start_mutation=2000000",    
        "--pbt_replace_fraction=0.3",      
        "--pbt_mutation_rate=0.15",        
        "--pbt_optimize_gamma=False",      
        
        # --- PPO & BATCHING ---
        "--rollout=128",             
        "--batch_size=2048",         
        "--num_batches_per_epoch=64", 
        "--num_epochs=4",            
        "--learning_rate=1e-5",
        
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