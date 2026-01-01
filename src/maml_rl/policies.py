from collections import OrderedDict
from typing import Iterable, Tuple

from torch import nn
from torchrl.modules import (
    NormalParamExtractor,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from tensordict.nn import TensorDictModule, TensorDictSequential


def _mlp(input_dim: int, hidden_sizes: Iterable[int], output_dim: int) -> nn.Sequential:
    """Build a multi-layer perceptron with ReLU activations."""
    layers = []
    last = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


def build_actor_critic(
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Tuple[int, ...] = (128, 128),
) -> Tuple[ProbabilisticActor, TensorDictSequential, ValueOperator]:
    """
    Build a Gaussian actor (TanhNormal) and a value head for PPO/VPG.

    Returns:
        actor: ProbabilisticActor that samples actions and writes action/log_prob.
        policy_model: TensorDictSequential producing loc/scale (used for functional calls).
        value_module: ValueOperator producing state_value.
    """
    policy_backbone = _mlp(obs_dim, hidden_sizes, 2 * act_dim)

    # Initialize policy weights
    for layer in policy_backbone[:-1]:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=2**0.5)
            nn.init.constant_(layer.bias, 0.0)
    # Policy output layer (small gain for near-zero initial mean/action)
    nn.init.orthogonal_(policy_backbone[-1].weight, gain=0.01)
    nn.init.constant_(policy_backbone[-1].bias, 0.0)

    policy_model = TensorDictSequential(
        TensorDictModule(policy_backbone, in_keys=["observation"], out_keys=["param"]),
        TensorDictModule(
            NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]
        ),
    )

    actor = ProbabilisticActor(
        module=policy_model,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )

    value_net = _mlp(obs_dim, hidden_sizes, 1)

    # Initialize value weights
    for layer in value_net[:-1]:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=2**0.5)
            nn.init.constant_(layer.bias, 0.0)
    # Value output layer
    nn.init.orthogonal_(value_net[-1].weight, gain=1.0)
    nn.init.constant_(value_net[-1].bias, 0.0)

    value_module = ValueOperator(
        value_net, in_keys=["observation"], out_keys=["state_value"]
    )
    return actor, policy_model, value_module


def params_and_buffers(module: nn.Module):
    """Return ordered dicts of params and buffers for functional_call."""
    return OrderedDict(module.named_parameters()), OrderedDict(module.named_buffers())


__all__ = ["build_actor_critic", "params_and_buffers"]
