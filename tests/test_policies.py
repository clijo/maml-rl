import torch
from torch import nn
from tensordict import TensorDict

from maml_rl.policies import build_actor_critic, params_and_buffers


def test_build_actor_critic_forward():
    obs_dim, act_dim = 5, 3
    actor, policy_model, value_module = build_actor_critic(obs_dim, act_dim)

    batch = 7
    td = TensorDict({"observation": torch.randn(batch, obs_dim)}, batch_size=[batch])

    # policy_model produces loc/scale
    td_params = policy_model(td.clone())
    assert td_params["loc"].shape == (batch, act_dim)
    assert td_params["scale"].shape == (batch, act_dim)

    # actor samples action and log_prob
    td_action = actor(td.clone())
    assert "action" in td_action.keys()
    assert (
        "sample_log_prob" in td_action.keys() or "action_log_prob" in td_action.keys()
    )
    assert td_action["action"].shape[-1] == act_dim

    # value head
    td_value = value_module(td.clone())
    assert "state_value" in td_value.keys()
    assert td_value["state_value"].shape == (batch, 1)


def test_params_and_buffers_extraction():
    model = nn.Sequential(nn.Linear(5, 5), nn.ReLU(), nn.Linear(5, 1))
    params, buffers = params_and_buffers(model)

    # Check that all parameters are captured
    assert len(params) == 4  # weight+bias for 2 layers
    assert isinstance(params, dict)

    # Verify values match
    for name, param in model.named_parameters():
        assert torch.equal(params[name], param)

    # Add a buffer to check extraction
    model.register_buffer("running_mean", torch.ones(5))
    params, buffers = params_and_buffers(model)
    assert "running_mean" in buffers
    assert torch.equal(buffers["running_mean"], model.running_mean)
