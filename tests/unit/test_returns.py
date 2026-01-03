import torch
import pytest
from tensordict import TensorDict
from torchrl.modules import ValueOperator
from unittest.mock import MagicMock

from maml_rl.utils.returns import add_gae, normalize_advantages


class SimpleValue(ValueOperator):
    def __init__(self):
        # Create a dummy module that just returns 0.0 or fixed values
        # We need a module that conforms to ValueOperator interface mostly
        module = torch.nn.Linear(1, 1)  # Dummy
        super().__init__(module, in_keys=["observation"], out_keys=["state_value"])

    def forward(self, td):
        # Override to provide deterministic values for testing
        obs = td.get("observation")
        # Let's say value is just 0.5 everywhere for simplicity,
        # or we can write specific values into td before calling if we want.
        # Here we just blindly write 0.0 if not present, or keep what's there?
        # The add_gae function calls this. Let's make it write a known constant.
        batch = td.batch_size
        if "state_value" not in td.keys():
            value = torch.zeros(*batch, 1, device=obs.device, dtype=obs.dtype)
            td.set("state_value", value)
        return td


def test_normalize_advantages():
    # Case 1: [Time, 1]
    adv = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(-1)
    norm_adv = normalize_advantages(adv)
    assert torch.isclose(norm_adv.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(norm_adv.std(), torch.tensor(1.0), atol=1e-5)

    # Case 2: [Batch, Time, 1]
    adv = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unsqueeze(-1)  # [2, 3, 1]
    norm_adv = normalize_advantages(adv)
    # Mean/Std should be over dim 1 (Time)
    assert torch.isclose(norm_adv.mean(dim=1), torch.zeros(2, 1), atol=1e-5).all()
    assert torch.isclose(norm_adv.std(dim=1), torch.ones(2, 1), atol=1e-5).all()


def test_add_gae_correctness():
    # Construct a small trajectory where we can manually calculate GAE
    # Gamma = 0.99, Lambda = 0.95
    gamma = 0.99
    lam = 0.95

    T = 3
    # Rewards: [1, 1, 1]
    # Values: [0, 0, 0] (simplifies things)
    # Next Values: [0, 0, 0]
    # Dones: [0, 0, 0]

    # Delta_t = r_t + gamma * V(s_{t+1}) * (1-d) - V(s_t)
    # Since V=0, Delta_t = r_t = 1

    # GAE_t = Delta_t + gamma * lam * (1-d) * GAE_{t+1}
    # GAE_2 = 1
    # GAE_1 = 1 + 0.99 * 0.95 * 1 = 1 + 0.9405 = 1.9405
    # GAE_0 = 1 + 0.99 * 0.95 * 1.9405 = 1 + 0.9405 * 1.9405 = 2.82504

    td = TensorDict(
        {
            "observation": torch.zeros(T, 1),
            "state_value": torch.zeros(T, 1),  # V(s_t)
            "next": TensorDict(
                {
                    "observation": torch.zeros(T, 1),
                    "reward": torch.ones(T, 1),
                    "done": torch.zeros(T, 1).bool(),  # Must be bool or int for new GAE
                    "state_value": torch.zeros(
                        T, 1
                    ),  # V(s_{t+1}) - normally computed by module
                },
                batch_size=[T],
            ),
        },
        batch_size=[T],
    )

    # We mock the value module to strictly not change existing values
    # because we manually set them above.
    value_module = MagicMock()
    value_module.side_effect = lambda x: x

    # Need to mock the call inside add_gae which re-computes values.
    # Actually add_gae computes values using the module.
    # Let's use a module that returns fixed 0s.
    class ZeroValue(torch.nn.Module):
        def forward(self, td):
            batch = td.batch_size
            td.set("state_value", torch.zeros(*batch, 1))
            return td

    val_mod = ZeroValue()

    out = add_gae(
        td,
        val_mod,
        gamma=gamma,
        lam=lam,
    )

    adv = out["advantage"]
    assert torch.isclose(adv[2], torch.tensor(1.0)).item()
    assert torch.isclose(adv[1], torch.tensor(1.0 + gamma * lam)).item()
    assert torch.isclose(
        adv[0], torch.tensor(1.0 + gamma * lam * (1.0 + gamma * lam))
    ).item()

    # Also check value targets: Target = Advantage + Value (Value=0 here)
    assert torch.allclose(out["value_target"], adv)


def test_add_gae_compute_grads():
    """Test add_gae with compute_grads=True."""
    # Setup simple TD
    td = TensorDict(
        {
            "observation": torch.randn(5, 4),
            "next": TensorDict(
                {
                    "observation": torch.randn(5, 4),
                    "reward": torch.randn(5, 1),
                    "done": torch.zeros(5, 1, dtype=torch.bool),
                },
                batch_size=[5],
            ),
        },
        batch_size=[5],
    )

    # Simple value module
    module = torch.nn.Linear(4, 1)
    val_op = ValueOperator(module, in_keys=["observation"], out_keys=["state_value"])

    # Run with compute_grads=True
    out = add_gae(td.clone(), val_op, compute_grads=True)

    # Check if gradients can flow back to module parameters
    loss = out["advantage"].sum()
    loss.backward()

    assert module.weight.grad is not None
    assert torch.norm(module.weight.grad) > 0


def test_add_gae_dones_logic():
    """Test logic for terminated/truncated handling in add_gae."""
    batch = 5
    td = TensorDict(
        {
            "observation": torch.randn(batch, 4),
            "state_value": torch.randn(batch, 1),
            "next": TensorDict(
                {
                    "observation": torch.randn(batch, 4),
                    "state_value": torch.randn(batch, 1),
                    "reward": torch.randn(batch, 1),
                    # No "done" key
                    "terminated": torch.tensor(
                        [1, 0, 0, 0, 0], dtype=torch.bool
                    ).unsqueeze(-1),
                    "truncated": torch.tensor(
                        [0, 1, 0, 0, 0], dtype=torch.bool
                    ).unsqueeze(-1),
                },
                batch_size=[batch],
            ),
        },
        batch_size=[batch],
    )

    # No value module needed as values are pre-filled
    out = add_gae(td)
    # Should run without error
    assert "advantage" in out.keys()

    # Check if dones were correctly inferred implicitly?
    # We can't easily check the internal 'dones' variable, but we can check if
    # advantage calculation respected the done flag (advantage should be different for done steps)
    # But simpler: ensure it ran.


def test_add_gae_missing_value_error():
    """Test that add_gae raises KeyError if values are missing."""
    td = TensorDict(
        {
            "observation": torch.randn(5, 4),
            "next": TensorDict(
                {
                    "observation": torch.randn(5, 4),
                    "reward": torch.randn(5, 1),
                },
                batch_size=[5],
            ),
        },
        batch_size=[5],
    )

    with pytest.raises(KeyError, match="state_value not found"):
        add_gae(td, value_module=None)


def test_add_gae_missing_next_state_value_error():
    """Test that add_gae raises KeyError if next state_value is missing."""
    td = TensorDict(
        {
            "observation": torch.randn(5, 4),
            "state_value": torch.randn(5, 1),
            "next": TensorDict(
                {
                    "observation": torch.randn(5, 4),
                    "reward": torch.randn(5, 1),
                    # Missing state_value here
                },
                batch_size=[5],
            ),
        },
        batch_size=[5],
    )
    with pytest.raises(KeyError, match=r"state_value not found in td\['next'\]"):
        add_gae(td, value_module=None)


def test_normalize_advantages_1d():
    """Test normalize_advantages with 1D tensor."""
    adv = torch.tensor([1.0, 2.0, 3.0])
    norm = normalize_advantages(adv)
    assert torch.isclose(norm.mean(), torch.tensor(0.0), atol=1e-5)


def test_add_gae_truncated_only():
    """Test add_gae with only truncated flag."""
    batch = 5
    td = TensorDict(
        {
            "observation": torch.randn(batch, 4),
            "state_value": torch.randn(batch, 1),
            "next": TensorDict(
                {
                    "observation": torch.randn(batch, 4),
                    "state_value": torch.randn(batch, 1),
                    "reward": torch.randn(batch, 1),
                    "truncated": torch.ones(batch, 1, dtype=torch.bool),
                },
                batch_size=[batch],
            ),
        },
        batch_size=[batch],
    )
    out = add_gae(td)
    assert "advantage" in out.keys()


def test_add_gae_no_dones_fallback():
    """Test add_gae with no done/terminated/truncated flags."""
    batch = 5
    td = TensorDict(
        {
            "observation": torch.randn(batch, 4),
            "state_value": torch.randn(batch, 1),
            "next": TensorDict(
                {
                    "observation": torch.randn(batch, 4),
                    "state_value": torch.randn(batch, 1),
                    "reward": torch.randn(batch, 1),
                },
                batch_size=[batch],
            ),
        },
        batch_size=[batch],
    )
    out = add_gae(td)
    assert "advantage" in out.keys()


def test_normalize_advantages_edge_cases():
    """Test normalize_advantages with edge cases like single element."""
    adv = torch.tensor([1.5]).unsqueeze(-1)  # [1, 1]
    norm = normalize_advantages(adv)
    # Should return as is because numel <= 1 (actually numel=1 here)
    assert torch.equal(norm, adv)
