import torch
import pytest
from tensordict import TensorDict
from torch.func import vmap

from maml_rl.maml import (
    FunctionalPolicy,
    inner_update_vpg,
    inner_update_value,
    outer_loss_ppo,
    _dist_from_params,
)

from maml_rl.policies import build_actor_critic, params_and_buffers


def _make_policy(obs_dim=5, act_dim=2):
    actor, policy_model, value_module = build_actor_critic(obs_dim, act_dim)
    params, buffers = params_and_buffers(policy_model)
    return actor, policy_model, value_module, params, buffers


def test_inner_update_vpg_produces_params():
    torch.manual_seed(0)
    _, policy_model, _, params, buffers = _make_policy()
    batch = 6
    obs_dim, act_dim = 5, 2
    support_td = TensorDict(
        {
            "observation": torch.randn(batch, obs_dim),
            "action": torch.randn(batch, act_dim).clamp(-0.5, 0.5),
            "advantage": torch.randn(batch, 1),
        },
        batch_size=[batch],
    )
    adapted = inner_update_vpg(policy_model, params, buffers, support_td, alpha=0.1)
    assert set(adapted.keys()) == set(params.keys())
    # params should change
    first_key = next(iter(params))
    assert not torch.equal(adapted[first_key], params[first_key])


def test_inner_update_value_produces_params():
    torch.manual_seed(0)
    _, _, value_module, _, _ = _make_policy()
    value_params, value_buffers = params_and_buffers(value_module)

    batch = 6
    obs_dim = 5
    support_td = TensorDict(
        {
            "observation": torch.randn(batch, obs_dim),
            "value_target": torch.randn(batch, 1),
        },
        batch_size=[batch],
    )

    adapted = inner_update_value(
        value_module, value_params, value_buffers, support_td, alpha=0.1
    )
    assert set(adapted.keys()) == set(value_params.keys())
    # params should change
    first_key = next(iter(value_params))
    assert not torch.equal(adapted[first_key], value_params[first_key])


def test_inner_update_value_fallback():
    """Test inner_update_value fallback when value_target is missing."""
    torch.manual_seed(0)
    _, _, value_module, _, _ = _make_policy()
    value_params, value_buffers = params_and_buffers(value_module)

    batch = 6
    obs_dim = 5
    # No value_target provided
    support_td = TensorDict(
        {
            "observation": torch.randn(batch, obs_dim),
            "advantage": torch.randn(batch, 1),  # Used in fallback
        },
        batch_size=[batch],
    )

    adapted = inner_update_value(
        value_module, value_params, value_buffers, support_td, alpha=0.1
    )
    # Just check it runs and returns params
    assert set(adapted.keys()) == set(value_params.keys())
    first_key = next(iter(value_params))
    assert not torch.equal(adapted[first_key], value_params[first_key])


def test_vmap_inner_update():
    """Test that inner update works when vmapped over tasks."""
    torch.manual_seed(0)
    _, policy_model, _, params, buffers = _make_policy()

    num_tasks = 3
    batch_per_task = 5
    obs_dim, act_dim = 5, 2

    # Create support data for multiple tasks: [num_tasks, batch_per_task]
    support_td = TensorDict(
        {
            "observation": torch.randn(num_tasks, batch_per_task, obs_dim),
            "action": torch.randn(num_tasks, batch_per_task, act_dim).clamp(-0.5, 0.5),
            "advantage": torch.randn(num_tasks, batch_per_task, 1),
        },
        batch_size=[num_tasks, batch_per_task],
    )

    def inner_update_single(p, b, td):
        return inner_update_vpg(policy_model, p, b, td, alpha=0.1)

    # vmap over tasks (dim 0 of support_td)
    # params and buffers are shared (None)
    adapted_params_list = vmap(inner_update_single, in_dims=(None, None, 0))(
        params, buffers, support_td
    )

    # Check output structure
    first_key = next(iter(params))
    # Output should have shape [num_tasks, *param_shape]
    assert adapted_params_list[first_key].shape == (num_tasks, *params[first_key].shape)


def test_outer_loss_ppo_runs():
    torch.manual_seed(0)
    obs_dim, act_dim = 5, 2
    _, policy_model, _, params, buffers = _make_policy(obs_dim, act_dim)
    batch = 4

    # Build a query tensordict with self-consistent action/log_prob
    td_base = TensorDict(
        {"observation": torch.randn(batch, obs_dim)}, batch_size=[batch]
    )
    # Get distribution to sample valid actions
    td_out = policy_model(td_base.clone())
    loc, scale = td_out["loc"], td_out["scale"]
    dist = torch.distributions.Independent(torch.distributions.Normal(loc, scale), 1)
    action = dist.rsample()
    log_prob = dist.log_prob(action)
    if log_prob.ndim == 1:
        log_prob = log_prob.unsqueeze(-1)

    query_td = TensorDict(
        {
            "observation": td_base["observation"],
            "action": action,
            "action_log_prob": log_prob,
            "advantage": torch.randn(batch, 1),
        },
        batch_size=[batch],
    )

    loss, info = outer_loss_ppo(
        policy_model, params, buffers, query_td, clip_eps=0.2, entropy_coef=0.01
    )
    assert loss.requires_grad
    assert "policy_loss" in info and "ratio_mean" in info
    assert "entropy" in info
    assert not torch.isnan(info["entropy"])


def test_outer_loss_ppo_missing_log_prob_error():
    """Test outer_loss_ppo raises KeyError if old log-probs are missing."""
    obs_dim, act_dim = 5, 2
    _, policy_model, _, params, buffers = _make_policy(obs_dim, act_dim)
    query_td = TensorDict(
        {
            "observation": torch.randn(4, obs_dim),
            "action": torch.randn(4, act_dim),
            "advantage": torch.randn(4, 1),
            # Missing action_log_prob
        },
        batch_size=[4],
    )
    with pytest.raises(KeyError, match="No old log-prob found"):
        outer_loss_ppo(policy_model, params, buffers, query_td)


def test_functional_policy_writes_log_prob():
    _, policy_model, _, params, buffers = _make_policy()
    td = TensorDict({"observation": torch.randn(3, 5)}, batch_size=[3])
    fpol = FunctionalPolicy(policy_model, params, buffers)
    out = fpol(td)
    assert "action" in out.keys()
    assert "action_log_prob" in out.keys()


def test_functional_policy_batched_params():
    """Test FunctionalPolicy with task-batched parameters (e.g. from vmap inner loop)."""
    torch.manual_seed(0)
    _, policy_model, _, params, buffers = _make_policy()

    num_tasks = 4
    batch_size = 5
    obs_dim = 5

    # Fake batched parameters: [num_tasks, ...]
    batched_params = {}
    for k, v in params.items():
        # Expand and add some noise so they are different
        batched_params[k] = (
            v.unsqueeze(0).expand(num_tasks, *v.shape)
            + torch.randn(num_tasks, *v.shape) * 0.01
        )

    fpol = FunctionalPolicy(policy_model, batched_params, buffers)

    # Input TD must match batch structure: [num_tasks, batch_size, ...]
    td = TensorDict(
        {"observation": torch.randn(num_tasks, batch_size, obs_dim)},
        batch_size=[num_tasks, batch_size],
    )

    out = fpol(td)

    assert "action" in out.keys()
    assert out["action"].shape == (num_tasks, batch_size, 2)  # act_dim=2 default
    assert "action_log_prob" in out.keys()


def test_functional_policy_empty_params():
    """Test FunctionalPolicy with empty parameters."""
    _, policy_model, _, _, buffers = _make_policy()
    fpol = FunctionalPolicy(policy_model, {}, buffers)
    assert fpol._is_batched is False


def test_dist_from_params_logic():
    """Test the _dist_from_params helper directly."""
    _, policy_model, _, params, buffers = _make_policy()
    td = TensorDict({"observation": torch.randn(2, 5)}, batch_size=[2])
    dist, td_out = _dist_from_params(policy_model, params, buffers, td)
    assert isinstance(dist, torch.distributions.Distribution)
    assert "loc" in td_out.keys()
