from collections import OrderedDict
from typing import Mapping, Tuple

import torch
from torch import nn
from torch.func import functional_call, vmap, grad
from torchrl.modules import TanhNormal, ValueOperator
from tensordict.nn import TensorDictSequential
from tensordict import TensorDict


def _get_log_prob(
    dist: TanhNormal, action: torch.Tensor, batch_dims: int
) -> torch.Tensor:
    """Helper to compute and correctly shape log probabilities."""
    log_prob = dist.log_prob(action)
    # If log_prob has more dimensions than batch_dims, sum over event dimensions (last dim)
    if log_prob.ndim > batch_dims:
        log_prob = log_prob.sum(-1, keepdim=True)
    else:
        # If dimensions match, just unsqueeze to align with expected [Batch, 1] shape
        log_prob = log_prob.unsqueeze(-1)
    return log_prob


def _dist_from_params(
    policy_model: TensorDictSequential,
    params: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    td: TensorDict,
) -> Tuple[TanhNormal, TensorDict]:
    """
    Compute policy distribution from functional parameters.

    Args:
        policy_model: Policy model template
        params: Functional parameters for the policy
        buffers: Functional buffers for the policy
        td: Input TensorDict with observations

    Returns:
        Tuple of (distribution, output TensorDict with loc/scale)
    """
    td_out = functional_call(policy_model, (params, buffers), (td,))
    loc = td_out.get("loc")
    scale = td_out.get("scale")
    # Use default bounds from torchrl's TanhNormal
    dist = TanhNormal(loc, scale)
    return dist, td_out


def inner_update_vpg(
    policy_model: TensorDictSequential,
    params: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    support_td: TensorDict,
    alpha: float = 0.1,
) -> OrderedDict[str, torch.Tensor]:
    """
    Single VPG inner step for a SINGLE task.
    This function is designed to be vmapped over a batch of tasks.
    """

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
    """
    Single value function inner update for a SINGLE task.
    """

    def compute_loss(p):
        td_out = functional_call(value_module, (p, value_buffers), (support_td,))
        value_pred = td_out.get("state_value")

        # Value target should be in support_td (computed via GAE)
        value_target = support_td.get("value_target")
        if value_target is None:
            # Fallback: reconstruction
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
    """
    PPO loss for a SINGLE task using its specific adapted parameters.
    """
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

    # Entropy (approximated or placeholder if TanhNormal doesn't implement it fully differentiable)
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

        # Compare first param dim with model param dim
        name, param = next(iter(self.params.items()))
        try:
            model_param = self.policy_model.get_parameter(name)
        except AttributeError:
            model_param = None

        if model_param is None:
            # Fallback to iteration
            model_param = next(self.policy_model.parameters())

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
    "FunctionalPolicy",
]
