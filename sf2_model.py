import torch
import torch.nn as nn
from torch.distributions import Categorical
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.algo.utils.tensor_dict import TensorDict
import os

from diplomacy_helpers import DiplomacyTransformer
from RL import rebuild_dense_mask  

class DiplomacyAutoregressiveDistribution:
    """A native PyTorch distribution object that Sample Factory can interact with natively."""
    def __init__(self, transformer, state_repr, dense_mask):
        self.transformer = transformer
        self.state_repr = state_repr
        self.dense_mask = dense_mask
        self.logits = None

    def sample(self):
        batch_size = self.state_repr.size(0)
        device = self.state_repr.device
        current_action = torch.zeros(batch_size, dtype=torch.long, device=device)
        kv_cache = None
        actions_list = []

        for prov_idx in range(self.transformer.num_provinces):
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, kv_cache = self.transformer.decode_step(current_action, prov_idx, self.state_repr, kv_cache)
            logits = logits.float()
            prov_mask = self.dense_mask[:, prov_idx, :]
            logits = logits.masked_fill(~prov_mask, -1e4)

            dist = Categorical(logits=logits)
            current_action = dist.sample()
            actions_list.append(current_action)

        return torch.stack(actions_list, dim=1)

    def log_prob(self, actions):
        actions = actions.long()
        # When SF2 calculates losses, it passes the true actions here for teacher forcing
        if self.logits is None:
            self.logits = self.transformer.decode_full(self.state_repr, actions)
            self.logits = self.logits.masked_fill(~self.dense_mask, -1e4)

        dist = Categorical(logits=self.logits)
        return dist.log_prob(actions).sum(dim=-1)

    def entropy(self):
        # Entropy relies on the logits calculated during the log_prob teacher-forced pass
        if self.logits is None:
            return torch.zeros(self.state_repr.size(0), device=self.state_repr.device)
        dist = Categorical(logits=self.logits)
        return dist.entropy().sum(dim=-1)

    def kl_divergence(self, other):
        # Sample Factory's core PPO math uses log_prob_actions (which is working).
        # We return zeros here to satisfy the stats logger, bypass the Continuous dummy,
        # and avoid calculating millions of redundant logits.
        batch_size = self.state_repr.size(0)
        return torch.zeros(batch_size, device=self.state_repr.device)

class DiplomacyActionParameterization(nn.Module):
    """Forces Sample Factory to use our custom distribution instead of its defaults."""
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def get_action_distribution(self, action_logits):
        """
        Sample Factory calls this during PPO updates to rebuild the 'old' policy from the buffer.
        We stored the raw logits in the buffer during forward_tail!
        """
        batch_size = action_logits.size(0)
        num_provinces = 82
        
        class DummyOldDist:
            def __init__(self, raw_logits, v_size):
                self.logits = raw_logits.view(batch_size, num_provinces, v_size)

        return DummyOldDist(action_logits, self.vocab_size)


class DiplomacySF2Model(ActorCritic):
    def __init__(self, cfg, obs_space, action_space):
        # 1. Let Sample Factory do its default initialization first
        super().__init__(obs_space, action_space, cfg)
        
        # 2. IMMEDIATELY overwrite the default parameterization with ours
        self.action_parameterization = DiplomacyActionParameterization(cfg.vocab_size)
        
        self.is_rnn = False
        self.num_provinces = 82
        self.vocab_size = cfg.vocab_size 
        self.history_length = cfg.history_length
        
        self.transformer = DiplomacyTransformer(
            input_dim=25, 
            num_provinces=self.num_provinces, 
            history_length=self.history_length, 
            vocab_size=self.vocab_size
        )

        bc_weights_path = "diplomacy_transformer_bc.pth"
        if os.path.exists(bc_weights_path):
            state_dict = torch.load(bc_weights_path, map_location="cpu")
            self.transformer.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded BC weights from {bc_weights_path}")
        else:
            print("WARNING: BC weights not found. Starting from random initialization.")

        self._dist = None

    def device_for_input_tensor(self, input_tensor_name):
        return next(self.transformer.parameters()).device

    def type_for_input_tensor(self, input_tensor_name):
        if input_tensor_name == "action_mask":
            return torch.int32
        return torch.float32

    def forward(self, normalized_obs_dict, rnn_states, values_only=False):
        """Used strictly by Inference Workers during rollout (Thread-safe)."""
        obs = normalized_obs_dict["obs"]
        action_mask = normalized_obs_dict["action_mask"]
        
        state_repr, state_value = self.transformer.encode_state(obs)
        state_value = state_value.squeeze(-1)
        
        result = TensorDict({"values": state_value, "new_rnn_states": rnn_states})
        if values_only:
            return result
            
        dense_mask = rebuild_dense_mask(action_mask, self.num_provinces, self.vocab_size, obs.device)
        self._dist = DiplomacyAutoregressiveDistribution(self.transformer, state_repr, dense_mask)
        
        # Satisfy SF2 Box space requirement
        result["action_logits"] = torch.zeros(obs.size(0), self.num_provinces * 2, device=obs.device)
        
        if "actions" not in normalized_obs_dict:
            actions = self._dist.sample()
            result["actions"] = actions
            result["log_prob_actions"] = self._dist.log_prob(actions)
            
        return result

    def forward_head(self, normalized_obs_dict, **kwargs):
        """Used by the Learner. Single-threaded, so saving state here is safe."""
        # Save action mask for the tail to use during the teacher-forced loss calculation
        self._learner_action_mask = normalized_obs_dict["action_mask"]
        
        # SF2 MUST get a tensor back here so it can calculate batch size with .size(0)
        return normalized_obs_dict["obs"]

    def forward_core(self, head_output, rnn_states, **kwargs):
        obs = head_output
        state_repr, state_value = self.transformer.encode_state(obs)
        
        # Save the value for the tail (safe because the Learner is single-threaded)
        self._learner_state_value = state_value
        
        # Return ONLY the state_repr tensor so SF2 can safely call .shape[0] on it
        return state_repr, rnn_states

    def forward_tail(self, core_output, values_only, sample_actions, **kwargs):
        state_repr = core_output
        state_value = self._learner_state_value.squeeze(-1)

        result = TensorDict({"values": state_value})
        if values_only:
            return result

        sparse_mask = self._learner_action_mask
        dense_mask = rebuild_dense_mask(sparse_mask, self.num_provinces, self.vocab_size, state_repr.device)
        
        self._dist = DiplomacyAutoregressiveDistribution(self.transformer, state_repr, dense_mask)

        if sample_actions:
            actions = self._dist.sample()
            result["actions"] = actions
            result["log_prob_actions"] = self._dist.log_prob(actions)

        # MANDATORY FOR MEMORY: Return a tiny dummy tensor to satisfy the Box shape.
        # Do not return self._dist.logits, as it will overflow the buffer and crash the workers.
        result["action_logits"] = torch.zeros(state_repr.size(0), self.num_provinces * 2, device=state_repr.device)

        return result

    def action_distribution(self):
        """SF2 calls this to compute losses; we return our heavily customized class."""
        return self._dist