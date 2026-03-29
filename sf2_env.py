import gymnasium as gym
import numpy as np
from diplomacy_helpers import DiplomacyTransformerEnv

class SF2DiplomacyEnv(gym.Env):
    def __init__(self, full_env_name, cfg=None, env_config=None, render_mode=None, **kwargs):
        self.env = DiplomacyTransformerEnv(history_length=3)
        self.num_agents = len(self.env.possible_agents)
        self.is_multiagent = True
        
        self.vocab_size = self.env.vocab_size
        self.num_provinces = self.env.num_provinces
        
        # SF2 Dict Space mapping
        self.observation_space = gym.spaces.Dict({
            "obs": gym.spaces.Box(low=0.0, high=1.0, shape=(3, self.num_provinces, 25), dtype=np.float32),
            "action_mask": gym.spaces.Box(low=-1, high=self.vocab_size, shape=(1200,), dtype=np.int32)
        })
        
        # Action space for a single agent: 82 discrete choices
        self.action_space = gym.spaces.Box(
            low=0, 
            high=self.vocab_size - 1, 
            shape=(self.num_provinces,), 
            dtype=np.int64
        )

    def _format_outputs(self, obs_dict, rewards_dict, terms_dict, infos_dict):
        obs_list, reward_list, done_list, info_list = [], [], [], []
        
        # 1. Determine if the global game is over
        # If any alive agent triggers a termination, the whole board is resetting.
        is_global_done = any(terms_dict.values()) or self.env.step_count >= 150
        
        for agent in self.env.possible_agents:
            info = {}
            if agent in self.env.agents:
                # Agent is alive
                agent_obs = {
                    "obs": obs_dict[agent].astype(np.float32),
                    "action_mask": infos_dict[agent]['action_mask'].astype(np.int32)
                }
                reward = float(rewards_dict.get(agent, 0.0))
                done = terms_dict.get(agent, False)
                
                if done: # Game ended while they were alive
                    sc_count = len(self.env.game.get_centers(agent))
                    info["episode_extra_stats"] = {
                        "supply_centers": sc_count
                    }
            else:
                # Agent is eliminated (Zombie State)
                agent_obs = {
                    "obs": np.zeros(self.observation_space['obs'].shape, dtype=np.float32),
                    "action_mask": np.full(1200, -1, dtype=np.int32)
                }
                reward = 0.0
                
                # IMPORTANT: Only flag as done when the whole board resets!
                done = is_global_done 
                
                if done: # Game ended, they finished with nothing
                    info["episode_extra_stats"] = {
                        "supply_centers": 0 
                    }

            obs_list.append(agent_obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            
        return obs_list, reward_list, done_list, [False]*self.num_agents, info_list

    def reset(self, **kwargs):
        obs_dict, infos_dict = self.env.reset()
        rewards_dict = {a: 0.0 for a in self.env.agents}
        terms_dict = {a: False for a in self.env.agents}
        
        obs_list, _, _, _, info_list = self._format_outputs(obs_dict, rewards_dict, terms_dict, infos_dict)
        return obs_list, info_list

    def step(self, action_list):
        # Map SF2's list of actions back to PettingZoo's dictionary format
        action_dict = {}
        for i, agent in enumerate(self.env.possible_agents):
            if agent in self.env.agents: 
                action_dict[agent] = action_list[i]
                
        obs_dict, rewards_dict, terms_dict, truncs_dict, infos_dict = self.env.step(action_dict)
             
        # Format the outputs of the turn that just finished
        obs_list, reward_list, done_list, trunc_list, info_list = self._format_outputs(obs_dict, rewards_dict, terms_dict, infos_dict)
        
        # --- THE AUTO-RESET FIX ---
        # If every agent is reporting that the game is over, wipe the board!
        if all(done_list):
            new_obs_dict, new_infos_dict = self.env.reset()
            
            # Format the completely fresh 1901 starting board
            dummy_rewards = {a: 0.0 for a in self.env.possible_agents}
            dummy_terms = {a: False for a in self.env.possible_agents}
            new_obs_list, _, _, _, _ = self._format_outputs(new_obs_dict, dummy_rewards, dummy_terms, new_infos_dict)
            
            # Swap out the observations!
            # Sample Factory will receive the "Game Over" rewards and stats, 
            # but the actual observations passed to the neural network will be a brand new game.
            obs_list = new_obs_list
        # --------------------------
        
        return obs_list, reward_list, done_list, trunc_list, info_list