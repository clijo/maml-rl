import torch
from tensordict import TensorDict
from torchrl.modules import ValueOperator
from torchrl.objectives.value.functional import generalized_advantage_estimate
from typing import Optional


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """
    Normalize advantages across the time dimension.

    Args:
        advantages: Advantage tensor of shape [batch, time, 1] or [time, 1]

    Returns:
        Normalized advantages with zero mean and unit variance
    """
    if advantages.numel() <= 1:
        return advantages

    # Normalize over the last two dimensions if 3D (batch, time, 1) or all if smaller
    # Typically advantages are [Batch, Time, 1] or [Time, 1]
    # We want to normalize over the Time dimension usually.

    if advantages.ndim >= 2:
        # Normalize over time dimension (dim=-2)
        std, mean = torch.std_mean(advantages, dim=-2, keepdim=True)
    else:
        std, mean = torch.std_mean(advantages)

    return (advantages - mean) / std.clamp(min=1e-8)


def add_gae(
    td: TensorDict,
    value_module: Optional[ValueOperator] = None,
    gamma: float = 0.99,
    lam: float = 0.95,
    compute_grads: bool = False,
) -> TensorDict:
    """
    Compute GAE(lambda) advantages and value targets in-place on a rollout TensorDict.
    Assumes standard TorchRL rollout structure with "next" key.

    Args:
        td: TensorDict with rollout data
        value_module: Value function module. If None, assumes "state_value" is already in td.
        gamma: Discount factor
        lam: GAE lambda parameter
        compute_grads: If False, use no_grad for value computation (default: False)

    Returns:
        Modified TensorDict with "advantage" and "value_target" added
    """
    # 1. Compute values if module provided
    if value_module is not None:
        if compute_grads:
            value_module(td)
            value_module(td.get("next"))
        else:
            with torch.no_grad():
                value_module(td)
                value_module(td.get("next"))

    # 2. Extract tensors
    td_next = td.get("next")
    rewards = td_next.get("reward").float()

    # Ensure values exist
    if "state_value" not in td.keys():
        raise KeyError(
            "state_value not found in td. Provide value_module or pre-compute values."
        )
    if "state_value" not in td_next.keys():
        raise KeyError(
            "state_value not found in td['next']. Provide value_module or pre-compute values."
        )

    values = td.get("state_value").float()
    next_values = td_next.get("state_value").float()

    # Handle done/terminated/truncated
    dones = td_next.get("done", None)
    if dones is None:
        dones = td_next.get("terminated", None)
        truncated = td_next.get("truncated", None)
        if dones is not None and truncated is not None:
            dones = torch.maximum(dones, truncated)
        elif dones is None and truncated is not None:
            dones = truncated

    if dones is None:
        dones = torch.zeros_like(rewards, dtype=torch.bool)
    else:
        dones = dones.bool()

    # 3. Compute GAE
    # generalized_advantage_estimate expects [..., time, 1] tensors by default (time_dim=-2)
    adv, val_target = generalized_advantage_estimate(
        gamma, lam, values, next_values, rewards, dones, time_dim=-2
    )

    # 4. Store results
    if not compute_grads:
        adv = adv.detach()
        val_target = val_target.detach()

    td.set("advantage", adv)
    td.set("value_target", val_target)

    return td


__all__ = ["add_gae", "normalize_advantages"]
