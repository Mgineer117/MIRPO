from typing import Any, Mapping, Optional, Sequence, Union
import copy
import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.multi_agents.torch import MultiAgent
from skrl.resources.schedulers.torch import KLAdaptiveLR

# fmt: off
MIRPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 
    "learning_epochs": 8,           
    "mini_batches": 2,              

    "discount_factor": 0.99,        
    "lambda": 0.95,                 
    "cost_discount_factor": 0.99,   # Discount for cost stream
    "cost_lambda": 0.95,            # GAE lambda for cost stream
    "safety_conservatism": 0.5,     # The lambda multiplier for cost advantage (IRPO adaptation)

    "learning_rate": 1e-3,                  
    "learning_rate_scheduler": None,        
    "learning_rate_scheduler_kwargs": {},   

    "state_preprocessor": None,             
    "state_preprocessor_kwargs": {},        
    "shared_state_preprocessor": None,      
    "shared_state_preprocessor_kwargs": {}, 
    "value_preprocessor": None,             
    "value_preprocessor_kwargs": {},        

    "random_timesteps": 0,          
    "learning_starts": 0,           

    "grad_norm_clip": 0.5,              
    "ratio_clip": 0.2,                  
    "value_clip": 0.2,                  
    "clip_predicted_values": False,     

    "entropy_loss_scale": 0.01,     
    "value_loss_scale": 1.0,        
    "cost_value_loss_scale": 1.0,   # Loss scale for the intrinsic cost critic

    "kl_threshold": 0,              
    "rewards_shaper": None,         
    "time_limit_bootstrap": False,  
    "mixed_precision": False,       
    "experiment": {
        "directory": "",            
        "experiment_name": "",      
        "write_interval": "auto",   
        "checkpoint_interval": "auto",      
        "store_separately": False,          
        "wandb": False,             
        "wandb_kwargs": {}          
    }
}
# fmt: on

class MIRPO(MultiAgent):
    def __init__(self, possible_agents: Sequence[str], models: Mapping[str, Model], **kwargs) -> None:
        _cfg = copy.deepcopy(MIRPO_DEFAULT_CONFIG)
        _cfg.update(kwargs.pop("cfg", {}))
        super().__init__(possible_agents=possible_agents, models=models, cfg=_cfg, **kwargs)

        self.shared_observation_spaces = kwargs.get("shared_observation_spaces")

        # Extract Models (Now expecting a cost_value model as well)
        self.policies = {uid: self.models[uid].get("policy", None) for uid in self.possible_agents}
        self.values = {uid: self.models[uid].get("value", None) for uid in self.possible_agents}
        self.cost_values = {uid: self.models[uid].get("cost_value", None) for uid in self.possible_agents}

        for uid in self.possible_agents:
            self.checkpoint_modules[uid]["policy"] = self.policies[uid]
            self.checkpoint_modules[uid]["value"] = self.values[uid]
            self.checkpoint_modules[uid]["cost_value"] = self.cost_values[uid]

        # Config extraction (abbreviated for clarity, mirroring your setup)
        self._learning_epochs = self._as_dict(self.cfg["learning_epochs"])
        self._mini_batches = self._as_dict(self.cfg["mini_batches"])
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0
        self._safety_conservatism = self._as_dict(self.cfg["safety_conservatism"])
        self._value_loss_scale = self._as_dict(self.cfg["value_loss_scale"])
        self._cost_value_loss_scale = self._as_dict(self.cfg["cost_value_loss_scale"])
        self._entropy_loss_scale = self._as_dict(self.cfg["entropy_loss_scale"])
        self._ratio_clip = self._as_dict(self.cfg["ratio_clip"])
        self._mixed_precision = self.cfg["mixed_precision"]
        self._device_type = torch.device(self.device).type

        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # Setup optimizers for policy, value, and cost_value
        self.optimizers = {}
        for uid in self.possible_agents:
            params = list(self.policies[uid].parameters())
            if self.values[uid] is not self.policies[uid]:
                params += list(self.values[uid].parameters())
            if self.cost_values[uid] is not self.policies[uid] and self.cost_values[uid] is not self.values[uid]:
                params += list(self.cost_values[uid].parameters())
            
            lr = self.cfg["learning_rate"] if isinstance(self.cfg["learning_rate"], float) else self.cfg["learning_rate"][uid]
            self.optimizers[uid] = torch.optim.Adam(params, lr=lr)
            self.checkpoint_modules[uid]["optimizer"] = self.optimizers[uid]

            # Preprocessors (Assuming default empty for brevity in this snippet)
            self._state_preprocessor = {uid: self._empty_preprocessor for uid in self.possible_agents}
            self._shared_state_preprocessor = {uid: self._empty_preprocessor for uid in self.possible_agents}
            self._value_preprocessor = {uid: self._empty_preprocessor for uid in self.possible_agents}

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        if self.memories:
            for uid in self.possible_agents:
                # Standard tensors
                self.memories[uid].create_tensor(name="states", size=self.observation_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="shared_states", size=self.shared_observation_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="actions", size=self.action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="rewards", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="terminated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="truncated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="log_prob", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="values", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="returns", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="advantages", size=1, dtype=torch.float32)
                
                # IRPO Cost Tensors
                self.memories[uid].create_tensor(name="costs", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="cost_values", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="cost_returns", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="cost_advantages", size=1, dtype=torch.float32)

                self._tensors_names = [
                    "states", "shared_states", "actions", "log_prob", 
                    "values", "returns", "advantages",
                    "cost_values", "cost_returns", "cost_advantages"
                ]

        self._current_log_prob = {}
        self._current_shared_next_states = []

    def act(self, states: Mapping[str, torch.Tensor], timestep: int, timesteps: int) -> torch.Tensor:
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            data = [self.policies[uid].act({"states": states[uid]}, role="policy") for uid in self.possible_agents]
            actions = {uid: d[0] for uid, d in zip(self.possible_agents, data)}
            log_prob = {uid: d[1] for uid, d in zip(self.possible_agents, data)}
            outputs = {uid: d[2] for uid, d in zip(self.possible_agents, data)}
            self._current_log_prob = log_prob
        return actions, log_prob, outputs

    def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps) -> None:
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

        if self.memories:
            shared_states = infos.get("shared_states", states)
            self._current_shared_next_states = infos.get("shared_next_states", next_states)
            
            # Extract costs from the IsaacLab environment infos dictionary
            costs = infos.get("costs", {uid: torch.zeros_like(rewards[uid]) for uid in self.possible_agents})

            for uid in self.possible_agents:
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    values, _, _ = self.values[uid].act({"states": shared_states[uid] if isinstance(shared_states, dict) else shared_states}, role="value")
                    cost_values, _, _ = self.cost_values[uid].act({"states": shared_states[uid] if isinstance(shared_states, dict) else shared_states}, role="cost_value")

                self.memories[uid].add_samples(
                    states=states[uid],
                    actions=actions[uid],
                    rewards=rewards[uid],
                    costs=costs[uid],
                    next_states=next_states[uid],
                    terminated=terminated[uid],
                    truncated=truncated[uid],
                    log_prob=self._current_log_prob[uid],
                    values=values,
                    cost_values=cost_values,
                    shared_states=shared_states[uid] if isinstance(shared_states, dict) else shared_states,
                )

    def _update(self, timestep: int, timesteps: int) -> None:
        def compute_gae(rewards, dones, values, next_values, discount_factor, lambda_coefficient):
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]
            for i in reversed(range(memory_size)):
                nxt_val = values[i + 1] if i < memory_size - 1 else next_values
                advantage = rewards[i] - values[i] + discount_factor * not_dones[i] * (nxt_val + lambda_coefficient * advantage)
                advantages[i] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        for uid in self.possible_agents:
            policy, value, cost_value = self.policies[uid], self.values[uid], self.cost_values[uid]
            memory = self.memories[uid]

            with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                last_values, _, _ = value.act({"states": self._current_shared_next_states[uid]}, role="value")
                last_cost_values, _, _ = cost_value.act({"states": self._current_shared_next_states[uid]}, role="cost_value")

            dones = memory.get_tensor_by_name("terminated") | memory.get_tensor_by_name("truncated")
            
            # Standard Extrinsic GAE
            returns, advantages = compute_gae(
                rewards=memory.get_tensor_by_name("rewards"), dones=dones,
                values=memory.get_tensor_by_name("values"), next_values=last_values,
                discount_factor=self.cfg["discount_factor"], lambda_coefficient=self.cfg["lambda"]
            )
            
            # Intrinsic Cost GAE
            cost_returns, cost_advantages = compute_gae(
                rewards=memory.get_tensor_by_name("costs"), dones=dones,
                values=memory.get_tensor_by_name("cost_values"), next_values=last_cost_values,
                discount_factor=self.cfg.get("cost_discount_factor", 0.99), lambda_coefficient=self.cfg.get("cost_lambda", 0.95)
            )

            memory.set_tensor_by_name("returns", returns)
            memory.set_tensor_by_name("advantages", advantages)
            memory.set_tensor_by_name("cost_returns", cost_returns)
            memory.set_tensor_by_name("cost_advantages", cost_advantages)

            sampled_batches = memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches[uid])

            for epoch in range(self._learning_epochs[uid]):
                for batch in sampled_batches:
                    (sampled_states, sampled_shared_states, sampled_actions, sampled_log_prob, 
                     sampled_values, sampled_returns, sampled_advantages,
                     sampled_cost_values, sampled_cost_returns, sampled_cost_advantages) = batch

                    with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                        _, next_log_prob, _ = policy.act({"states": sampled_states, "taken_actions": sampled_actions}, role="policy")

                        # Combine Advantages: Extrinsic - (lambda * Intrinsic/Cost)
                        combined_advantages = sampled_advantages - (self._safety_conservatism[uid] * sampled_cost_advantages)
                        
                        ratio = torch.exp(next_log_prob - sampled_log_prob)
                        surrogate = combined_advantages * ratio
                        surrogate_clipped = combined_advantages * torch.clip(ratio, 1.0 - self._ratio_clip[uid], 1.0 + self._ratio_clip[uid])
                        policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                        entropy_loss = -self._entropy_loss_scale[uid] * policy.get_entropy(role="policy").mean()

                        predicted_values, _, _ = value.act({"states": sampled_shared_states}, role="value")
                        value_loss = self._value_loss_scale[uid] * F.mse_loss(sampled_returns, predicted_values)

                        predicted_cost_values, _, _ = cost_value.act({"states": sampled_shared_states}, role="cost_value")
                        cost_value_loss = self._cost_value_loss_scale[uid] * F.mse_loss(sampled_cost_returns, predicted_cost_values)

                        total_loss = policy_loss + entropy_loss + value_loss + cost_value_loss

                    self.optimizers[uid].zero_grad()
                    self.scaler.scale(total_loss).backward()
                    self.scaler.step(self.optimizers[uid])
                    self.scaler.update()

            self.track_data(f"Loss / Policy loss ({uid})", policy_loss.item())
            self.track_data(f"Loss / Value loss ({uid})", value_loss.item())
            self.track_data(f"Loss / Cost Value loss ({uid})", cost_value_loss.item())

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")
        super().post_interaction(timestep, timesteps)