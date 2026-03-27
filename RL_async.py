import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs import EnvBase, ParallelEnv
from torchrl.envs.utils import check_env_specs
from torchrl.collectors import MultiSyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.data import (
    Composite,
    UnboundedContinuous,
    UnboundedDiscrete,
    Categorical  # <-- The new class for discrete/boolean spaces
)
from diplomacy_helpers import DiplomacyTransformer, DiplomacyTransformerEnv

# You will need to wrap your existing env to output TensorDicts.
class TorchRLDiplomacyEnv(EnvBase):
    def __init__(self, history_length=3, device="cpu"):
        super().__init__(device=device)
        self.inner_env = DiplomacyTransformerEnv(history_length=history_length)
        
        # Dynamically pull constants directly from your custom engine
        self.num_agents = len(self.inner_env.possible_agents)
        self.history_length = history_length
        self.num_provs = self.inner_env.num_provinces  # <--- This fixes the 81 vs 82 mismatch!
        self.vocab_size = self.inner_env.vocab_size
        
        # Peek at a dummy observation to get the exact feature dimension (usually 16)
        dummy_obs, _ = self.inner_env.reset()
        first_agent = list(dummy_obs.keys())[0]
        self.obs_dim = dummy_obs[first_agent].shape[-1] 
        
        self.agent_names = self.inner_env.possible_agents 
        
        self._make_specs()

    def _make_specs(self):
        # 1. Observation Spec
        self.observation_spec = Composite({
            "agents": Composite({
                "observation": UnboundedContinuous(
                    shape=(self.num_agents, self.history_length, self.num_provs, self.obs_dim),
                    dtype=torch.float32,
                ),
                # The sparse mask max size is 1200
                "action_mask": UnboundedDiscrete(
                    shape=(self.num_agents, self.vocab_size), 
                    dtype=torch.int64,
                )
            })
        })

        # 2. Action Spec (81 provinces)
        self.action_spec = Composite({
            "agents": Composite({
                "action": Categorical(
                    n=self.vocab_size, # <-- Bounds random actions to 0-1199
                    shape=(self.num_agents, self.num_provs),
                    dtype=torch.int64,
                )
            })
        })

        # 3. Reward Spec
        self.reward_spec = Composite({
            "agents": Composite({
                "reward": UnboundedContinuous(
                    shape=(self.num_agents, 1),
                    dtype=torch.float32,
                )
            })
        })

        # 4. Done Spec (TorchRL needs both root-level and agent-level dones)
        self.done_spec = Composite({
            "done": Categorical(n=2, shape=(1,), dtype=torch.bool),
            "terminated": Categorical(n=2, shape=(1,), dtype=torch.bool),
            "agents": Composite({
                "done": Categorical(n=2, shape=(self.num_agents, 1), dtype=torch.bool),
                "terminated": Categorical(n=2, shape=(self.num_agents, 1), dtype=torch.bool),
            })
        })

    # --- FORMATTING HELPERS (Dict <-> Tensor) ---
    def _format_obs(self, obs_dict):
        obs_list = []
        for agent in self.agent_names:
            if agent in obs_dict:
                obs_list.append(torch.tensor(obs_dict[agent], dtype=torch.float32))
            else: # Pad eliminated agents with zeros
                obs_list.append(torch.zeros((self.history_length, self.num_provs, self.obs_dim), dtype=torch.float32))
        return torch.stack(obs_list)

    def _format_masks(self, infos_dict):
        mask_list = []
        for agent in self.agent_names:
            if agent in infos_dict and 'action_mask' in infos_dict[agent]:
                mask_list.append(torch.tensor(infos_dict[agent]['action_mask'], dtype=torch.int64))
            else: # Pad eliminated agents with -1 (your mask rebuild ignores -1)
                mask_list.append(torch.full((self.vocab_size,), -1, dtype=torch.int64))
        return torch.stack(mask_list)

    def _format_rewards(self, reward_dict):
        rew_list = []
        for agent in self.agent_names:
            rew_list.append(torch.tensor([reward_dict.get(agent, 0.0)], dtype=torch.float32))
        return torch.stack(rew_list)
        
    def _format_dones(self, terms_dict):
        done_list = []
        for agent in self.agent_names:
            # If agent isn't in dict, assume they are done
            done_list.append(torch.tensor([terms_dict.get(agent, True)], dtype=torch.bool))
        return torch.stack(done_list)

    def _unformat_actions(self, action_tensor):
        action_dict = {}
        for i, agent in enumerate(self.agent_names):
            # Only pass actions to the inner env if the agent is still alive
            if agent in self.inner_env.agents: 
                action_dict[agent] = action_tensor[i]
        return action_dict

    # --- CORE OVERRIDES ---
    def _reset(self, tensordict=None):
        obs, infos = self.inner_env.reset()
        
        return TensorDict({
            "agents": {
                "observation": self._format_obs(obs),
                "action_mask": self._format_masks(infos)
            },
            "done": torch.zeros(1, dtype=torch.bool),
            "terminated": torch.zeros(1, dtype=torch.bool),
        }, batch_size=[])

    def _step(self, tensordict):
        actions = tensordict["agents", "action"].cpu().numpy()
        action_dict = self._unformat_actions(actions)
        
        obs, rewards, terms, truncs, infos = self.inner_env.step(action_dict)
        
        agent_dones = self._format_dones(terms)
        global_done = torch.tensor([all(terms.values()) or len(self.inner_env.agents) == 0], dtype=torch.bool)
        
        return TensorDict({
            "agents": {
                "observation": self._format_obs(obs),
                "action_mask": self._format_masks(infos),
                "reward": self._format_rewards(rewards),
                "done": agent_dones,
                "terminated": agent_dones,
            },
            "done": global_done,
            "terminated": global_done,
        }, batch_size=[])

    def _set_seed(self, seed):
        pass # Handle inner env seeding if necessary

def rebuild_dense_mask(sparse_masks, num_provs, vocab_size, device):
    batch_size = sparse_masks.size(0)
    valid_mask = sparse_masks != -1
    
    row_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * (num_provs * vocab_size)
    global_indices = sparse_masks + row_offsets
    valid_global_indices = global_indices[valid_mask]
    
    batch_mask_flat = torch.zeros(batch_size * num_provs * vocab_size, dtype=torch.bool, device=device)
    batch_mask_flat[valid_global_indices] = True
    
    return batch_mask_flat.view(batch_size, num_provs, vocab_size)

class DiplomacyAutoregressiveActor(nn.Module):
    def __init__(self, base_net, num_provs, vocab_size):
        super().__init__()
        self.net = base_net
        self.num_provs = num_provs
        self.vocab_size = vocab_size

    def forward(self, obs, sparse_masks):
        # 1. Rebuild dense mask
        device = obs.device
        batch_size = obs.size(0)
        masks_tensor = rebuild_dense_mask(sparse_masks, self.num_provs, self.vocab_size, device)
        
        # 2. Encode State
        state_repr, _ = self.net.encode_state(obs)
        
        current_action = torch.zeros(batch_size, dtype=torch.long, device=device)
        kv_cache = None
        
        sampled_actions = []
        log_probs = []
        
        # 3. Autoregressive Loop
        for prov_idx in range(self.num_provs):
            logits, kv_cache = self.net.decode_step(current_action, prov_idx, state_repr, kv_cache)
            prov_mask = masks_tensor[:, prov_idx, :]
            logits = logits.masked_fill(~prov_mask, -1e4)
            
            dist = torch.distributions.Categorical(logits=logits)
            current_action = dist.sample()
            
            sampled_actions.append(current_action)
            log_probs.append(dist.log_prob(current_action))
            
        actions = torch.stack(sampled_actions, dim=1)
        # Sum log probs across the 81 provinces to get the total sequence log_prob
        total_log_prob = torch.stack(log_probs, dim=1).sum(dim=1) 
        
        return actions, total_log_prob

class DiplomacyValue(nn.Module):
    def __init__(self, base_net):
        super().__init__()
        self.net = base_net
        
    def forward(self, obs):
        _, values = self.net.encode_state(obs)
        return values

from torchrl.modules import ProbabilisticActor, ValueOperator
from tensordict.nn import TensorDictModule

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    NUM_ENVS = 16
    NUM_STEPS = 128
    BATCH_SIZE = NUM_ENVS * NUM_STEPS
    MINI_BATCH_SIZE = 512
    EPOCHS = 4

    # 1. Initialize Environments Asynchronously
    make_env = lambda: TorchRLDiplomacyEnv(history_length=3, device="cpu")
    parallel_env = ParallelEnv(NUM_ENVS, make_env)

    # dynamically get sizes from a dummy env so we don't hardcode 81 or 1200
    dummy_env = DiplomacyTransformerEnv()
    MAP_PROVINCES = dummy_env.num_provinces
    VOCAB_SIZE = dummy_env.vocab_size

    # 2. Initialize Networks
    base_net = DiplomacyTransformer(
        num_provinces=MAP_PROVINCES, 
        history_length=3, 
        vocab_size=VOCAB_SIZE
    ).to(device)
    
    actor_module = DiplomacyAutoregressiveActor(
        base_net, 
        num_provs=MAP_PROVINCES, 
        vocab_size=VOCAB_SIZE
    )
    value_module = DiplomacyValue(base_net)

    # 3. Setup Async Data Collector
    # This replaces your massive manual zero-tensors and rollout loop
    collector = MultiSyncDataCollector(
        create_env_fn=[make_env] * NUM_ENVS, # Creates environments across processes
        policy=actor,
        frames_per_batch=BATCH_SIZE,
        total_frames=BATCH_SIZE * 1000,
        device=device,
        storing_device=device,
    )

    # 4. Setup PPO Loss and GAE
    adv_module = GAE(
        gamma=0.99, 
        lmbda=0.95, 
        value_network=value, 
        average_gae=True
    )
    
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=value,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coef=0.0,
        loss_critic_type="l2"
    )
    
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=1e-6, eps=1e-5)

    # 5. The Training Loop
    for i, tensordict_data in enumerate(collector):
        # tensordict_data contains a full batch of rollouts gathered asynchronously!
        
        # Calculate Advantages
        with torch.no_grad():
            adv_module(tensordict_data)

        # Setup Replay Buffer for Mini-batching
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=BATCH_SIZE),
            sampler=SamplerWithoutReplacement(),
            batch_size=MINI_BATCH_SIZE,
        )
        replay_buffer.extend(tensordict_data)

        # Epoch loop
        for epoch in range(EPOCHS):
            for _ in range(BATCH_SIZE // MINI_BATCH_SIZE):
                subdata = replay_buffer.sample()
                
                # TorchRL handles the PPO ratio, clipping, and value loss automatically
                loss_vals = loss_module(subdata)
                
                loss_value = (
                    loss_vals["loss_objective"] + 
                    loss_vals["loss_critic"] * 0.1 - 
                    loss_vals["entropy"] * 0.0
                )

                loss_value.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()

        print(f"Update {i} complete. Policy Loss: {loss_vals['loss_objective'].item()}")

if __name__ == "__main__":
    main()