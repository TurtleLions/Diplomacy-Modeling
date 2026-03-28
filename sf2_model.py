import torch
import torch.nn as nn
from torch.distributions import Categorical
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.algo.utils.tensor_dict import TensorDict
import os
import math

from diplomacy_helpers import DiplomacyTransformer
from RL import rebuild_dense_mask  # Import your fast GPU mask builder

class DummyDist:
    """Satisfies SF2's need for a distribution object during loss calculation."""
    def __init__(self, log_prob, entropy_val):
        self.log_p = log_prob
        self.ent = entropy_val
        
    def log_prob(self, actions):
        return self.log_p
        
    def entropy(self):
        return self.ent

    def kl_divergence(self, other):
        import sys
        try:
            # Peek 1 level up into SF2's _calculate_losses function
            frame = sys._getframe(1)
            if 'mb' in frame.f_locals:
                mb = frame.f_locals['mb']
                # Grab the exact old log probabilities recorded during the CPU rollout
                old_log_p = mb.log_prob_actions
                return old_log_p - self.log_p
        except Exception:
            pass
            
        # Absolute fallback: if the frame hack fails, return 0 so the engine never crashes
        return torch.zeros_like(self.log_p)

    def to(self, *args, **kwargs):
        return self

class DiplomacySF2Model(ActorCritic):
    def __init__(self, cfg, obs_space, action_space):
        super().__init__(obs_space, action_space, cfg)
        
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
            os.environ["ROCR_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

        self.num_provinces = 82
        self.vocab_size = cfg.vocab_size 
        self.history_length = cfg.history_length
        
        self.transformer = DiplomacyTransformer(
            input_dim=25, 
            num_provinces=self.num_provinces, 
            history_length=self.history_length, 
            vocab_size=self.vocab_size
        )

        # --- NEW: LOAD BEHAVIORAL CLONING WEIGHTS ---
        bc_weights_path = "diplomacy_transformer_bc.pth"
        if os.path.exists(bc_weights_path):
            # Load to CPU first; SF2 will automatically distribute to GPUs later
            state_dict = torch.load(bc_weights_path, map_location="cpu")
            # strict=False because BC has no Value Head, so SF2 will randomly initialize it
            self.transformer.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded BC weights from {bc_weights_path}")
        else:
            print("WARNING: BC weights not found. Starting from random initialization.")
        # --------------------------------------------

    def device_for_input_tensor(self, input_tensor_name):
        """
        SF2 normally checks self.encoders[0] to find the device.
        Since we bypassed standard SF2 encoders, we must override this
        to point directly to our custom transformer's device.
        """
        # Return the device of the first parameter in our custom transformer
        return next(self.transformer.parameters()).device

    def type_for_input_tensor(self, input_tensor_name):
        """
        SF2 also checks self.encoders[0] to find the expected data type.
        We override this to map our dictionary keys to the correct PyTorch dtypes.
        """
        if input_tensor_name == "action_mask":
            return torch.int32
        
        # Default to standard float32 for the main 'obs' tensor
        return torch.float32

    def forward(self, normalized_obs_dict, rnn_states, values_only=False):
        
        obs = normalized_obs_dict["obs"]
        batch_size = obs.size(0)
        device = obs.device

        if values_only:
            _, state_value = self.transformer.encode_state(obs)
            state_value = state_value.squeeze(-1)  # <--- ADD THIS SQUEEZE
            return TensorDict({"values": state_value, "new_rnn_states": rnn_states})

        state_repr, state_value = self.transformer.encode_state(obs)
        state_value = state_value.squeeze(-1)  # <--- ADD THIS SQUEEZE
        
        sparse_mask = normalized_obs_dict["action_mask"]
        dense_mask = rebuild_dense_mask(sparse_mask, self.num_provinces, self.vocab_size, device)

        # --- TRAINING PHASE (THE LEARNER) ---
        if "actions" in normalized_obs_dict:
            targets = normalized_obs_dict["actions"]
            
            logits, _ = self.transformer(obs, targets)
            logits = logits.view(batch_size, self.num_provinces, self.vocab_size)
            logits = logits.masked_fill(~dense_mask, -1e4)
            
            dist = Categorical(logits=logits)
            log_prob_per_prov = dist.log_prob(targets) # Shape: [batch, 82]
            log_prob_actions = log_prob_per_prov.sum(dim=-1)
            
            # NEW: Calculate real entropy for PPO
            entropy_actions = dist.entropy().sum(dim=-1) 
            
            # NEW: Save the dummy directly to the model instance
            self.dummy_dist = DummyDist(log_prob_actions, entropy_actions)
            
            log_std = -log_prob_per_prov - 0.5 * math.log(2 * math.pi)
            fake_action_logits = torch.cat([targets.float(), log_std], dim=-1)
            
            result = TensorDict({
                "values": state_value,
                "action_logits": fake_action_logits,
                "log_prob_actions": log_prob_actions,
                "new_rnn_states": rnn_states
                # (Removed the dummy from here)
            })
            return result

        # --- ROLLOUT PHASE (THE CPU ACTORS) ---
        current_action = torch.zeros(batch_size, dtype=torch.long, device=device)
        kv_cache = None
        actions_list, logprobs_list, entropy_list = [], [], []
        
        for prov_idx in range(self.num_provinces):
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, kv_cache = self.transformer.decode_step(current_action, prov_idx, state_repr, kv_cache)
            
            logits = logits.float()
            prov_mask = dense_mask[:, prov_idx, :]
            logits = logits.masked_fill(~prov_mask, -1e4)
            
            dist = Categorical(logits=logits)
            current_action = dist.sample()
            
            actions_list.append(current_action)
            logprobs_list.append(dist.log_prob(current_action))
            entropy_list.append(dist.entropy())

        actions = torch.stack(actions_list, dim=1)
        log_prob_per_prov = torch.stack(logprobs_list, dim=1)
        log_prob_actions = log_prob_per_prov.sum(dim=1)

        entropy_actions = torch.stack(entropy_list, dim=1).sum(dim=1)
        self.dummy_dist = DummyDist(log_prob_actions, entropy_actions)

        # THE GAUSSIAN BRIDGE HACK (Rollout):
        log_std = -log_prob_per_prov - 0.5 * math.log(2 * math.pi)
        fake_action_logits = torch.cat([actions.float(), log_std], dim=-1)

        result = TensorDict({
            "values": state_value,
            "actions": actions,
            "action_logits": fake_action_logits, 
            "log_prob_actions": log_prob_actions,
            "new_rnn_states": rnn_states
        })
        return result
        
    def forward_head(self, normalized_obs_dict, **kwargs):
        self._temp_saved_dict = normalized_obs_dict
        return normalized_obs_dict["obs"]

    def forward_core(self, head_output, rnn_states, **kwargs):
        return head_output, rnn_states

    def forward_tail(self, core_output, values_only=False, sample_actions=True, **kwargs):
        # The Ultimate Hack: Peek into Sample Factory's memory to grab the target actions
        try:
            # Go 1 level up the execution stack to SF2's _calculate_losses function
            frame = sys._getframe(1) 
            if 'mb' in frame.f_locals:
                mb = frame.f_locals['mb']
                
                # Extract the actions and inject them into our saved dictionary
                if hasattr(mb, 'actions'):
                    self._temp_saved_dict["actions"] = mb.actions
                elif 'actions' in mb:
                    self._temp_saved_dict["actions"] = mb['actions']
        except Exception:
            pass # Fail gracefully if the frame isn't what we expect
            
        # Run our monolithic forward pass. 
        # Because "actions" is now safely in the dict, it will trigger Teacher Forcing!
        results = self.forward(self._temp_saved_dict, rnn_states=None, values_only=values_only)

        self._temp_saved_dict = None
        return results

    def action_distribution(self):
        """Override SF2's internal distribution fetcher to return our custom Dummy."""
        return self.dummy_dist
        