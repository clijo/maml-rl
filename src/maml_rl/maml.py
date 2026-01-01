from collections import OrderedDict
from typing import Mapping, Tuple, Dict

import torch
from torch import nn
from torch.func import functional_call, vmap, grad
from torchrl.modules import TanhNormal, ValueOperator
from tensordict.nn import TensorDictSequential
from tensordict import TensorDict

from maml_rl.utils.optimization import (
    parameters_to_vector,
    vector_to_parameters,
    conjugate_gradients,
)


def _get_log_prob(
    dist: TanhNormal, action: torch.Tensor, batch_dims: int
) -> torch.Tensor:
    """Helper to compute and correctly shape log probabilities."""
    log_prob = dist.log_prob(action)
    if log_prob.ndim > batch_dims:
        log_prob = log_prob.sum(-1, keepdim=True)
    else:
        log_prob = log_prob.unsqueeze(-1)
    return log_prob


def _dist_from_params(
    policy_model: TensorDictSequential,
    params: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    td: TensorDict,
) -> Tuple[TanhNormal, TensorDict]:
    """Compute policy distribution from functional parameters."""
    td_out = functional_call(policy_model, (params, buffers), (td,))
    loc = td_out.get("loc")
    scale = td_out.get("scale")
    dist = TanhNormal(loc, scale)
    return dist, td_out


def inner_update_vpg(
    policy_model: TensorDictSequential,
    params: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    support_td: TensorDict,
    alpha: float = 0.1,
) -> OrderedDict[str, torch.Tensor]:
    """Single VPG inner step for a SINGLE task."""

    def compute_loss(p):
        dist, _ = _dist_from_params(policy_model, p, buffers, support_td)
        log_prob = _get_log_prob(dist, support_td["action"], support_td.batch_dims)
        advantage = support_td["advantage"].detach()
        return -(log_prob * advantage).mean()

    grads = grad(compute_loss)(params)
    updated_params = OrderedDict(
        (name, param - alpha * g)
        for (name, param), g in zip(params.items(), grads.values())
    )
    return updated_params


def inner_update_value(
    value_module: ValueOperator,
    value_params: Mapping[str, torch.Tensor],
    value_buffers: Mapping[str, torch.Tensor],
    support_td: TensorDict,
    alpha: float = 0.1,
) -> OrderedDict[str, torch.Tensor]:
    """Single value function inner update for a SINGLE task."""

    def compute_loss(p):
        td_out = functional_call(value_module, (p, value_buffers), (support_td,))
        value_pred = td_out.get("state_value")
        value_target = support_td.get("value_target")
        if value_target is None:
            value_target = support_td.get("advantage", 0.0) + value_pred.detach()
        return ((value_pred - value_target.detach()) ** 2).mean()

    grads = grad(compute_loss)(value_params)
    updated_params = OrderedDict(
        (name, param - alpha * g)
        for (name, param), g in zip(value_params.items(), grads.values())
    )
    return updated_params


def outer_loss_ppo(
    policy_model: TensorDictSequential,
    adapted_params: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    query_td: TensorDict,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.0,
) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """PPO loss for a SINGLE task using its specific adapted parameters."""
    dist, _ = _dist_from_params(policy_model, adapted_params, buffers, query_td)
    log_prob = _get_log_prob(dist, query_td["action"], query_td.batch_dims)

    old_log_prob = query_td.get("action_log_prob")
    if old_log_prob is None:
        old_log_prob = query_td.get("sample_log_prob", query_td.get("log_prob"))

    if old_log_prob is None:
        raise KeyError("No old log-prob found in query TensorDict.")

    old_log_prob = old_log_prob.detach()
    advantage = query_td["advantage"].detach()
    ratio = torch.exp(log_prob - old_log_prob)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage

    policy_loss = -torch.min(surr1, surr2).mean()

    try:
        entropy = dist.entropy().mean()
    except Exception:
        entropy = torch.zeros((), device=log_prob.device)

    loss = policy_loss - entropy_coef * entropy

    info = {
        "policy_loss": policy_loss.detach(),
        "entropy": entropy.detach(),
        "ratio_mean": ratio.detach().mean(),
    }
    return loss, info


def _adapt_over_tasks(
    policy_model,
    policy_params,
    policy_buffers,
    support_td,
    inner_lr,
    inner_steps,
):
    """
    Runs inner loop adaptation for a batch of tasks.
    Returns: Batched OrderedDict of adapted parameters.
    """

    def inner_loop(params, buffers, td):
        curr = params
        for _ in range(inner_steps):
            curr = inner_update_vpg(policy_model, curr, buffers, td, inner_lr)
        return curr

    # vmap over batch dim (0) of support_td
    # params and buffers are shared (None)
    adapted_list = vmap(inner_loop, (None, None, 0))(
        policy_params, policy_buffers, support_td
    )
    
    # Reconstruct OrderedDict with batched tensors
    adapted_params = OrderedDict(
        (name, adapted_list[name]) for name in policy_params.keys()
    )
    return adapted_params


def outer_step_trpo(
    policy_model: TensorDictSequential,
    policy_params: OrderedDict[str, torch.Tensor],
    policy_buffers: OrderedDict[str, torch.Tensor],
    support_td: TensorDict,
    query_td: TensorDict,
    inner_lr: float,
    inner_steps: int,
    max_kl: float = 1e-2,
    damping: float = 1e-2,
    cg_iters: int = 10,
    line_search_max_steps: int = 10,
    line_search_backtrack_ratio: float = 0.5,
) -> Tuple[OrderedDict[str, torch.Tensor], Dict]:
    """
    Perform a single TRPO outer step.
    Updates policy_params in-place (or returns new ones).
    """
    
    # 1. Compute Old Distributions (Reference)
    with torch.no_grad():
        old_adapted_params = _adapt_over_tasks(
            policy_model, policy_params, policy_buffers, support_td, inner_lr, inner_steps
        )
        # Compute old probs/dists on query set for KL
        # We process task-by-task via vmap
        def get_dist(p, t):
            d, _ = _dist_from_params(policy_model, p, policy_buffers, t)
            return d
        
        # This returns a batch of distributions
        # Note: TanhNormal might not be perfectly vmap-friendly for storage, 
        # so we store the loc/scale parameters instead if needed, but let's try direct use.
        # Actually, for KL, we need the parameters of the old distribution.
        
        def get_dist_params(p, t):
            out = functional_call(policy_model, (p, policy_buffers), (t,))
            return out.get("loc"), out.get("scale")

        old_locs, old_scales = vmap(get_dist_params, (0, 0))(old_adapted_params, query_td)
        # Detach them as they are fixed references
        old_locs = old_locs.detach()
        old_scales = old_scales.detach()

    # 2. Define Objective and KL functions w.r.t. Meta-Parameters
    # We work with flattened parameters for CG/Line Search
    flat_params = parameters_to_vector(policy_params)
    
    def get_objective(flat_p):
        """Compute the surrogated return (TRPO objective)."""
        curr_params = vector_to_parameters(flat_p, policy_params)
        adapted_params = _adapt_over_tasks(
            policy_model, curr_params, policy_buffers, support_td, inner_lr, inner_steps
        )
        
        # Compute surrogate loss
        def compute_task_surrogate(p_adapted, t_query, old_loc, old_scale):
            dist, _ = _dist_from_params(policy_model, p_adapted, policy_buffers, t_query)
            log_prob = _get_log_prob(dist, t_query["action"], t_query.batch_dims)
            
            # Reconstruct old dist for ratio
            old_dist = TanhNormal(old_loc, old_scale)
            old_log_prob = _get_log_prob(old_dist, t_query["action"], t_query.batch_dims)
            
            ratio = torch.exp(log_prob - old_log_prob)
            adv = t_query["advantage"]
            return (ratio * adv).mean()

        surrogates = vmap(compute_task_surrogate, (0, 0, 0, 0))(
            adapted_params, query_td, old_locs, old_scales
        )
        return surrogates.mean() # Maximize this

    def get_kl(flat_p):
        """Compute mean KL divergence between old and new adapted policies."""
        curr_params = vector_to_parameters(flat_p, policy_params)
        adapted_params = _adapt_over_tasks(
            policy_model, curr_params, policy_buffers, support_td, inner_lr, inner_steps
        )

        def compute_task_kl(p_adapted, t_query, old_loc, old_scale):
            dist, _ = _dist_from_params(policy_model, p_adapted, policy_buffers, t_query)
            old_dist = TanhNormal(old_loc, old_scale)
            # KL(Old || New)
            return torch.distributions.kl_divergence(old_dist, dist).mean()
        
        kls = vmap(compute_task_kl, (0, 0, 0, 0))(
            adapted_params, query_td, old_locs, old_scales
        )
        return kls.mean()

    # 3. Compute Gradient of Objective
    # We need to retain graph for FVP if we used a double-backward approach,
    # but here we use a separate function for FVP or re-compute.
    # Standard TRPO: grad of objective, then FVP of KL.
    
    loss = get_objective(flat_params) # We want to MAXIMIZE objective, so minimize -objective
    # Note: TRPO usually maximizes objective. 
    grads = torch.autograd.grad(loss, flat_params, retain_graph=True)[0]
    
    # 4. Fisher Vector Product
    # FVP(v) = grad( (grad(KL) * v).sum() )
    # We can use autograd.grad on the KL function.
    
    def fisher_vector_product(v):
        kl = get_kl(flat_params)
        kl_grad = torch.autograd.grad(kl, flat_params, create_graph=True)[0]
        kl_v = (kl_grad * v).sum()
        fvp = torch.autograd.grad(kl_v, flat_params)[0]
        return fvp + damping * v

    # 5. Conjugate Gradient
    step_dir = conjugate_gradients(fisher_vector_product, grads, nsteps=cg_iters)
    
    # 6. Line Search
    # We have step direction `step_dir` (x in Hx=g).
    # Step size scaling: beta = sqrt( 2 * max_kl / (xHx) )
    shs = 0.5 * (step_dir * fisher_vector_product(step_dir)).sum(0, keepdim=True)
    lm = torch.sqrt(shs / max_kl)
    full_step = step_dir / lm[0]
    
    # In case of numerical instability
    if torch.isnan(full_step).any():
        print("NaN in TRPO step direction. Skipping update.")
        return policy_params, {"trpo_kl": 0.0, "trpo_surr": 0.0}

    old_objective = loss.item()
    
    final_step = None
    expected_improve = (grads * full_step).sum().item()
    
    # Backtracking
    for i in range(line_search_max_steps):
        step_frac = line_search_backtrack_ratio ** i
        step = full_step * step_frac
        new_params_flat = flat_params + step
        
        with torch.no_grad():
            new_objective = get_objective(new_params_flat).item()
            new_kl = get_kl(new_params_flat).item()
            
        # Check acceptance
        # 1. Improvement > 0 (or satisfying Armijo condition if strict)
        # 2. KL < max_kl * (something slightly > 1 for tolerance?)
        
        if new_objective > old_objective and new_kl <= max_kl * 1.5:
             final_step = step
             break
    
    if final_step is None:
        print("TRPO line search failed.")
        return policy_params, {"trpo_kl": 0.0, "trpo_surr": old_objective}
    
    new_params = vector_to_parameters(flat_params + final_step, policy_params)
    
    return new_params, {
        "trpo_kl": new_kl, 
        "trpo_surr": new_objective,
        "trpo_improvement": new_objective - old_objective
    }


class FunctionalPolicy(nn.Module):
    """
    Policy wrapper to use adapted parameters inside env.rollout.
    Automatically handles batched parameters via vmap.
    """

    def __init__(
        self,
        policy_model: TensorDictSequential,
        params: Mapping[str, torch.Tensor],
        buffers: Mapping[str, torch.Tensor],
    ):
        super().__init__()
        self.policy_model = policy_model
        self.params = params
        self.buffers = buffers
        self._is_batched = self._detect_batching()

    def _detect_batching(self) -> bool:
        """Detect if parameters are batched (one extra dimension for task batch)."""
        if not self.params:
            return False
        
        # Use arbitrary key
        name, param = next(iter(self.params.items()))
        try:
            model_param = self.policy_model.get_parameter(name)
        except AttributeError:
            model_param = None
            # Fallback scan
            for n, p in self.policy_model.named_parameters():
                if n == name:
                    model_param = p
                    break

        if model_param is None:
             # Just assume if not found it might be batched if dim is high?
             # Safer: assume not batched if we can't verify. 
             # But MAML code ensures names match.
             return False

        return param.ndim > model_param.ndim

    def forward(self, td: TensorDict) -> TensorDict:
        """
        Forward pass with functional parameters.
        """
        if self._is_batched:
            # vmap over the batch dimension (0) of params and input td
            # td is [batch_size, ...] matching params batch size

            # Helper to bind buffers (None)
            def call_single(p, t):
                return functional_call(self.policy_model, (p, self.buffers), (t,))

            td_out = vmap(call_single, (0, 0))(self.params, td)
        else:
            td_out = functional_call(
                self.policy_model, (self.params, self.buffers), (td,)
            )

        loc = td_out.get("loc")
        scale = td_out.get("scale")
        dist = TanhNormal(loc, scale)

        action = dist.rsample()
        td_out.set("action", action)

        log_prob = _get_log_prob(dist, action, td_out.batch_dims)
        td_out.set("action_log_prob", log_prob)

        return td_out


__all__ = [
    "inner_update_vpg",
    "inner_update_value",
    "outer_loss_ppo",
    "outer_step_trpo",
    "FunctionalPolicy",
]