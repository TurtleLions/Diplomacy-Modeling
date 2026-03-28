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
        # SF2 requires LISTS of length `num_agents`, maintaining a strict agent order
        obs_list, reward_list, done_list, info_list = [], [], [], []
        
        for agent in self.env.possible_agents:
            if agent in self.env.agents:
                # Agent is alive
                agent_obs = {
                    "obs": obs_dict[agent].astype(np.float32),
                    "action_mask": infos_dict[agent]['action_mask'].astype(np.int32)
                }
                reward = float(rewards_dict.get(agent, 0.0))
                done = terms_dict.get(agent, False)
                
                # --- NEW: CUSTOM TENSORBOARD LOGGING ---
                sc_count = len(self.env.game.get_centers(agent))
                info = {
                    "episode_extra_stats": {
                        "supply_centers": sc_count
                    }
                }
                # ---------------------------------------
            else:
                # Agent is eliminated
                agent_obs = {
                    "obs": np.zeros(self.observation_space['obs'].shape, dtype=np.float32),
                    "action_mask": np.full(1200, -1, dtype=np.int32)
                }
                reward = 0.0
                done = True
                info = {} # Dead agents don't log stats

            obs_list.append(agent_obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            
        # SF2 unpacks: obs, rewards, dones, truncations, infos
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
             
        return self._format_outputs(obs_dict, rewards_dict, terms_dict, infos_dict)