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


# fmt: off
ICPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 
    "learning_epochs": 8,           # Only applied to the Value / Cost Value networks
    "mini_batches": 2,              # Only applied to the Value / Cost Value networks

    "discount_factor": 0.99,        
    "lambda": 0.95,                 
    "cost_discount_factor": 0.99,   
    "cost_lambda": 0.95,            

    "learning_rate": 1e-3,          # Applied to critics only
    "learning_rate_scheduler": None,        
    "learning_rate_scheduler_kwargs": {},   

    "state_preprocessor": None,             
    "state_preprocessor_kwargs": {},        
    "value_preprocessor": None,             
    "value_preprocessor_kwargs": {},        

    "random_timesteps": 0,          
    "learning_starts": 0,           

    "value_loss_scale": 1.0,        
    "cost_value_loss_scale": 1.0,   
    "clip_predicted_values": False,
    "value_clip": 0.2,
    "grad_norm_clip": 0.5,          # Added gradient clipping default

    # --- CPO Specific Parameters ---
    "cost_limit": 10,                 # Constraint threshold (d_c)
    "kl_threshold": 0.01,           # Trust region size
    "cg_damping": 0.1,              # Damping for Hessian-vector product
    "cg_steps": 10,                 # Number of steps for Conjugate Gradients
    "backtrack_iters": 10,          # Max iterations for line search
    "backtrack_coeff": 0.8,         # Line search decay coefficient
    # -------------------------------

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


class ICPO(MultiAgent):
    def __init__(
        self,
        possible_agents: Sequence[str],
        models: Mapping[str, Model],
        memories: Optional[Mapping[str, Memory]] = None,
        observation_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
        action_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Independent Constrained Policy Optimization (ICPO)

        :param possible_agents: Name of all possible agents the environment could generate
        :type possible_agents: list of str
        :param models: Models used by the agents.
                       External keys are environment agents' names. Internal keys are the models required by the algorithm
        :type models: nested dictionary of skrl.models.torch.Model
        :param memories: Memories to storage the transitions.
        :type memories: dictionary of skrl.memory.torch.Memory, optional
        :param observation_spaces: Observation/state spaces or shapes (default: ``None``)
        :type observation_spaces: dictionary of int, sequence of int or gymnasium.Space, optional
        :param action_spaces: Action spaces or shapes (default: ``None``)
        :type action_spaces: dictionary of int, sequence of int or gymnasium.Space, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
        _cfg = copy.deepcopy(ICPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policies = {uid: self.models[uid].get("policy", None) for uid in self.possible_agents}
        self.values = {uid: self.models[uid].get("value", None) for uid in self.possible_agents}
        self.cost_values = {uid: self.models[uid].get("cost_value", None) for uid in self.possible_agents}

        for uid in self.possible_agents:
            # checkpoint models
            self.checkpoint_modules[uid]["policy"] = self.policies[uid]
            self.checkpoint_modules[uid]["value"] = self.values[uid]
            self.checkpoint_modules[uid]["cost_value"] = self.cost_values[uid]

            # broadcast models' parameters in distributed runs
            if config.torch.is_distributed:
                logger.info(f"Broadcasting models' parameters")
                if self.policies[uid] is not None:
                    self.policies[uid].broadcast_parameters()
                if self.values[uid] is not None:
                    self.values[uid].broadcast_parameters()
                if self.cost_values[uid] is not None:
                    self.cost_values[uid].broadcast_parameters()

        # configuration
        self._learning_epochs = self._as_dict(self.cfg["learning_epochs"])
        self._mini_batches = self._as_dict(self.cfg["mini_batches"])
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self._as_dict(self.cfg["grad_norm_clip"])
        self._value_clip = self._as_dict(self.cfg["value_clip"])
        self._clip_predicted_values = self._as_dict(self.cfg["clip_predicted_values"])

        self._value_loss_scale = self._as_dict(self.cfg["value_loss_scale"])
        self._cost_value_loss_scale = self._as_dict(self.cfg.get("cost_value_loss_scale", 1.0))

        self._learning_rate = self._as_dict(self.cfg["learning_rate"])
        self._learning_rate_scheduler = self._as_dict(self.cfg["learning_rate_scheduler"])
        self._learning_rate_scheduler_kwargs = self._as_dict(self.cfg["learning_rate_scheduler_kwargs"])

        self._state_preprocessor = self._as_dict(self.cfg["state_preprocessor"])
        self._state_preprocessor_kwargs = self._as_dict(self.cfg["state_preprocessor_kwargs"])
        self._value_preprocessor = self._as_dict(self.cfg["value_preprocessor"])
        self._value_preprocessor_kwargs = self._as_dict(self.cfg["value_preprocessor_kwargs"])

        self._discount_factor = self._as_dict(self.cfg["discount_factor"])
        self._lambda = self._as_dict(self.cfg["lambda"])
        self._cost_discount_factor = self._as_dict(self.cfg.get("cost_discount_factor", 0.99))
        self._cost_lambda = self._as_dict(self.cfg.get("cost_lambda", 0.95))

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self._as_dict(self.cfg["time_limit_bootstrap"])

        # --- ICPO Specific Extractions ---
        self._cost_limit = self._as_dict(self.cfg["cost_limit"])
        self._kl_threshold = self._as_dict(self.cfg["kl_threshold"]) # Replaces IPPO's kl_threshold intent
        self._cg_damping = self._as_dict(self.cfg["cg_damping"])
        self._cg_steps = self._as_dict(self.cfg["cg_steps"])
        self._backtrack_iters = self._as_dict(self.cfg["backtrack_iters"])
        self._backtrack_coeff = self._as_dict(self.cfg["backtrack_coeff"])

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler (Critics only)
        self.optimizers = {}
        self.schedulers = {}

        for uid in self.possible_agents:
            value = self.values[uid]
            cost_value = self.cost_values[uid]
            
            if value is not None and cost_value is not None:
                # ICPO only uses Adam for the value networks. Policy is updated via Trust Region.
                optimizer = torch.optim.Adam(
                    itertools.chain(value.parameters(), cost_value.parameters()), lr=self._learning_rate[uid]
                )
                self.optimizers[uid] = optimizer
                if self._learning_rate_scheduler[uid] is not None:
                    self.schedulers[uid] = self._learning_rate_scheduler[uid](
                        optimizer, **self._learning_rate_scheduler_kwargs[uid]
                    )

            self.checkpoint_modules[uid]["optimizer"] = self.optimizers[uid]

            # set up preprocessors
            if self._state_preprocessor[uid] is not None:
                self._state_preprocessor[uid] = self._state_preprocessor[uid](**self._state_preprocessor_kwargs[uid])
                self.checkpoint_modules[uid]["state_preprocessor"] = self._state_preprocessor[uid]
            else:
                self._state_preprocessor[uid] = self._empty_preprocessor

            if self._value_preprocessor[uid] is not None:
                self._value_preprocessor[uid] = self._value_preprocessor[uid](**self._value_preprocessor_kwargs[uid])
                self.checkpoint_modules[uid]["value_preprocessor"] = self._value_preprocessor[uid]
            else:
                self._value_preprocessor[uid] = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        if self.memories:
            for uid in self.possible_agents:
                # Standard IPPO Tensors
                self.memories[uid].create_tensor(name="states", size=self.observation_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="actions", size=self.action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="rewards", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="terminated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="truncated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="log_prob", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="values", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="returns", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="advantages", size=1, dtype=torch.float32)
                
                # CPO Cost Tensors
                self.memories[uid].create_tensor(name="costs", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="cost_values", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="cost_returns", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="cost_advantages", size=1, dtype=torch.float32)

                self._tensors_names = [
                    "states", "actions", "log_prob", "values", "returns", "advantages",
                    "costs", "cost_values", "cost_returns", "cost_advantages"
                ]

        self._current_log_prob = {}
        self._current_next_states = {}

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
            self._current_next_states = next_states
            costs = infos.get("costs", {uid: torch.zeros_like(rewards[uid]) for uid in self.possible_agents})

            for uid in self.possible_agents:
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    values, _, _ = self.values[uid].act({"states": states[uid]}, role="value")
                    cost_values, _, _ = self.cost_values[uid].act({"states": states[uid]}, role="cost_value")

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
                    cost_values=cost_values
                )
    
    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _flat_params(self, model):
        return torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])

    def _set_flat_params(self, model, flat_params):
        pointer = 0
        for p in model.parameters():
            if p.requires_grad:
                num_param = p.numel()
                p.data.copy_(flat_params[pointer : pointer + num_param].view_as(p))
                pointer += num_param

    def _compute_kl(self, old_policy, new_policy, states):
        with torch.no_grad():
            old_policy.act({"states": states}, role="policy")
            old_dist = old_policy.distribution(role="policy")
        new_policy.act({"states": states}, role="policy")
        new_dist = new_policy.distribution(role="policy")
        return torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()

    def _conjugate_gradients(self, Av_func, b, nsteps):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        increments = 0
        decrements = 0
        new_rdotr = rdotr

        for i in range(nsteps):
            Avp = Av_func(p)
            pAvp = torch.dot(p, Avp)
            if pAvp <= 1e-8:
                break
            alpha = rdotr / (pAvp + 1e-8)
            x += alpha * p
            r -= alpha * Avp
            new_rdotr = torch.dot(r, r)
            
            if new_rdotr > rdotr: 
                increments += 1
            else: 
                decrements += 1
                
            if new_rdotr < 1e-8:
                break
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
            
        consistency_ratio = increments / (decrements + 1e-8)
        return x, new_rdotr, consistency_ratio

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
            dones = memory.get_tensor_by_name("terminated") | memory.get_tensor_by_name("truncated")

            # 1. Compute Dual GAE
            with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                last_values, _, _ = value.act({"states": self._current_next_states[uid]}, role="value")
                last_cost_values, _, _ = cost_value.act({"states": self._current_next_states[uid]}, role="cost_value")

            returns, advantages = compute_gae(
                memory.get_tensor_by_name("rewards"), dones, memory.get_tensor_by_name("values"), last_values,
                self.cfg["discount_factor"], self.cfg["lambda"]
            )
            cost_returns, cost_advantages = compute_gae(
                memory.get_tensor_by_name("costs"), dones, memory.get_tensor_by_name("cost_values"), last_cost_values,
                self.cfg.get("cost_discount_factor", 0.99), self.cfg.get("cost_lambda", 0.95)
            )

            memory.set_tensor_by_name("returns", returns)
            memory.set_tensor_by_name("advantages", advantages)
            memory.set_tensor_by_name("cost_returns", cost_returns)
            memory.set_tensor_by_name("cost_advantages", cost_advantages)

            # --- TRACKING VARIABLES ---
            cumulative_value_loss = 0.0
            cumulative_cost_value_loss = 0.0

            # 2. Update Critics via Mini-batches (Standard SGD/Adam)
            sampled_batches = memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches[uid])
            for epoch in range(self._learning_epochs[uid]):
                for batch in sampled_batches:
                    s_states, _, _, s_values, s_returns, _, _, s_cost_values, s_cost_returns, _ = batch
                    
                    with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                        p_values, _, _ = value.act({"states": s_states}, role="value")
                        p_cost_values, _, _ = cost_value.act({"states": s_states}, role="cost_value")
                        
                        # --- IMPLEMENTED PREDICTED VALUE CLIPPING ---
                        if self._clip_predicted_values[uid]:
                            p_values = s_values + torch.clip(
                                p_values - s_values, min=-self._value_clip[uid], max=self._value_clip[uid]
                            )
                            p_cost_values = s_cost_values + torch.clip(
                                p_cost_values - s_cost_values, min=-self._value_clip[uid], max=self._value_clip[uid]
                            )

                        value_loss = self._value_loss_scale[uid] * F.mse_loss(s_returns, p_values)
                        cost_value_loss = self.cfg.get("cost_value_loss_scale", 1.0) * F.mse_loss(s_cost_returns, p_cost_values)
                        total_critic_loss = value_loss + cost_value_loss

                    self.optimizers[uid].zero_grad()
                    self.scaler.scale(total_critic_loss).backward()
                    
                    # --- IMPLEMENTED GRADIENT NORM CLIPPING ---
                    if self._grad_norm_clip[uid] > 0:
                        self.scaler.unscale_(self.optimizers[uid])
                        nn.utils.clip_grad_norm_(
                            itertools.chain(value.parameters(), cost_value.parameters()), 
                            self._grad_norm_clip[uid]
                        )

                    self.scaler.step(self.optimizers[uid])
                    self.scaler.update()

                    cumulative_value_loss += value_loss.item()
                    cumulative_cost_value_loss += cost_value_loss.item()

            # 3. Full-Batch CPO Policy Update
            # Flatten tensors from (rollouts, num_envs, dim) to (rollouts * num_envs, dim)
            full_states = memory.get_tensor_by_name("states").flatten(0, 1)
            full_actions = memory.get_tensor_by_name("actions").flatten(0, 1)
            old_logprobs = memory.get_tensor_by_name("log_prob").flatten(0, 1)
            adv = memory.get_tensor_by_name("advantages").flatten(0, 1)
            c_adv = memory.get_tensor_by_name("cost_advantages").flatten(0, 1)

            old_policy = copy.deepcopy(policy)
            policy_params = [p for p in policy.parameters() if p.requires_grad]

            def get_objective(adv_tensor, volatile=False):
                with torch.set_grad_enabled(not volatile):
                    _, logprobs, _ = policy.act({"states": full_states, "taken_actions": full_actions}, role="policy")
                    ratio = torch.exp(logprobs - old_logprobs)
                    return (adv_tensor * ratio).mean()

            def Hv(v):
                kl = self._compute_kl(old_policy, policy, full_states)
                grads = torch.autograd.grad(kl, policy_params, create_graph=True)
                flat_grads = torch.cat([g.view(-1) for g in grads])
                g_v = (flat_grads * v).sum()
                hv = torch.autograd.grad(g_v, policy_params)
                flat_hv = torch.cat([h.contiguous().view(-1) for h in hv])
                return flat_hv + self._cg_damping[uid] * v

            # Gradients
            loss = -get_objective(adv)  # Negative because we want to maximize objective
            cost_loss = get_objective(c_adv)

            loss_grad = torch.cat([g.view(-1) for g in torch.autograd.grad(loss, policy_params)]).detach()
            cost_loss_grad = torch.cat([g.view(-1) for g in torch.autograd.grad(cost_loss, policy_params)]).detach()

            # Step Directions & Metrics
            stepdir, cg_r_error, r_consistency = self._conjugate_gradients(Hv, -loss_grad, self._cg_steps[uid])
            cost_stepdir, cg_c_error, c_consistency = self._conjugate_gradients(Hv, -cost_loss_grad, self._cg_steps[uid])

            # Dual Optimization Math
            q = -loss_grad.dot(stepdir)
            r = loss_grad.dot(cost_stepdir)
            s = -cost_loss_grad.dot(cost_stepdir)

            # Use the GAE-computed discounted returns instead of the raw step costs
            current_cost = memory.get_tensor_by_name("cost_returns").mean() 
            cc = current_cost - self._cost_limit[uid]

            A = torch.sqrt(torch.clamp((q - (r**2) / (s + 1e-8)) / (self._kl_threshold[uid] - (cc**2) / (s + 1e-8)), min=1e-8))
            B = torch.sqrt(torch.clamp(q / self._kl_threshold[uid], min=1e-8))

            if cc > 0: 
                opt_lam = torch.max(r / (cc + 1e-8), A)
            else: 
                opt_lam = torch.max(r / (cc + 1e-8), B) if r / (cc - 1e-8) > B else B

            opt_lam = opt_lam.clamp(min=1e-8)
            opt_nu = torch.max((opt_lam * cc - r) / (s + 1e-8), torch.zeros_like(r))

            # Feasibility Check
            if ((cc**2) / (s + 1e-8) - self._kl_threshold[uid]) > 0 and cc > 0:
                opt_stepdir = torch.sqrt(2 * self._kl_threshold[uid] / (s + 1e-8)) * cost_stepdir
            else:
                opt_stepdir = (stepdir - opt_nu * cost_stepdir) / opt_lam

            # Line Search Analytics
            expected_improve = -loss_grad.dot(opt_stepdir)
            expected_cost_change = -cost_loss_grad.dot(opt_stepdir)
            
            old_flat_params = self._flat_params(policy)
            fval = get_objective(adv, volatile=True).item()
            actual_improve = 0.0
            
            for i in range(self._backtrack_iters[uid]):
                fraction = self._backtrack_coeff[uid] ** i
                self._set_flat_params(policy, old_flat_params + fraction * opt_stepdir)

                kl = self._compute_kl(old_policy, policy, full_states)
                new_fval = get_objective(adv, volatile=True).item()
                actual_improve = new_fval - fval

                if kl <= self._kl_threshold[uid] and actual_improve > 0:
                    break
            else:
                self._set_flat_params(policy, old_flat_params) # Revert if failed
            
            # Critic Losses
            self.track_data(f"Loss / Value loss ({uid})", cumulative_value_loss / (self._learning_epochs[uid] * self._mini_batches[uid]))
            self.track_data(f"Loss / Cost Value loss ({uid})", cumulative_cost_value_loss / (self._learning_epochs[uid] * self._mini_batches[uid]))
            
            # RL Analytics
            self.track_data(f"RL_analytics / expected_improve ({uid})", expected_improve.item())
            self.track_data(f"RL_analytics / expected_cost_change ({uid})", expected_cost_change.item())
            self.track_data(f"RL_analytics / actual_improve ({uid})", actual_improve)
            
            self.track_data(f"RL_analytics / kl_divergence ({uid})", kl.item())
            self.track_data(f"RL_analytics / cost_violation ({uid})", cc.item())
            
            self.track_data(f"RL_analytics / opt_nu ({uid})", opt_nu.item())
            self.track_data(f"RL_analytics / opt_lambda ({uid})", opt_lam.item())
            
            self.track_data(f"RL_analytics / cg_r_error ({uid})", cg_r_error.item())
            self.track_data(f"RL_analytics / cg_c_error ({uid})", cg_c_error.item())
            self.track_data(f"RL_analytics / cg_r_consistency ({uid})", r_consistency)
            self.track_data(f"RL_analytics / cg_c_consistency ({uid})", c_consistency)
            
            self.track_data(f"RL_analytics / num_backtracks ({uid})", i)
            self.track_data(f"RL_analytics / cost_limit ({uid})", self._cost_limit[uid])

            self.track_data(f"RL_analytics / std_dev ({uid})", policy.distribution(role="policy").stddev.mean().item())

            # Environment Tracking
            self.track_data(f"Environment / avg_rewards ({uid})", memory.get_tensor_by_name("rewards").mean().item())
            self.track_data(f"Environment / avg_costs ({uid})", current_cost.item())

            if self._learning_rate_scheduler[uid]:
                self.track_data(f"Learning / Learning rate ({uid})", self.schedulers[uid].get_last_lr()[0])
