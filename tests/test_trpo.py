import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.modules import TanhNormal
from tensordict.nn import TensorDictSequential, TensorDictModule, NormalParamExtractor
from collections import OrderedDict

from maml_rl.maml import outer_step_trpo
from maml_rl.policies import params_and_buffers

def test_trpo_outer_step_mechanics():
    """
    Test that outer_step_trpo:
    1. Runs without crashing.
    2. Updates parameters.
    3. Respects KL constraints (heuristically).
    """
    torch.manual_seed(42)
    
    # 1. Setup minimal Policy
    obs_dim = 2
    act_dim = 1
    net = nn.Sequential(nn.Linear(obs_dim, 4), nn.ReLU(), nn.Linear(4, 2 * act_dim))
    policy_module = TensorDictModule(net, in_keys=["observation"], out_keys=["param"])
    extractor_module = TensorDictModule(
        NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]
    )
    policy_model = TensorDictSequential(policy_module, extractor_module)
    
    policy_params, policy_buffers = params_and_buffers(policy_model)
    initial_params_flat = torch.cat([p.view(-1) for p in policy_params.values()]).clone()
    
    # 2. Mock Data (2 tasks)
    num_tasks = 2
    steps = 10
    
    support_td = TensorDict({
        "observation": torch.randn(num_tasks, steps, obs_dim),
        "action": torch.randn(num_tasks, steps, act_dim),
        "advantage": torch.randn(num_tasks, steps, 1),
    }, batch_size=[num_tasks, steps])
    
    query_td = TensorDict({
        "observation": torch.randn(num_tasks, steps, obs_dim),
        "action": torch.randn(num_tasks, steps, act_dim),
        "advantage": torch.randn(num_tasks, steps, 1),
    }, batch_size=[num_tasks, steps])
    
    # 3. Run TRPO Step
    new_params, info = outer_step_trpo(
        policy_model=policy_model,
        policy_params=policy_params,
        policy_buffers=policy_buffers,
        support_td=support_td,
        query_td=query_td,
        inner_lr=0.1,
        inner_steps=1,
        max_kl=0.01,
        damping=0.1,
    )
    
    # 4. Verifications
    assert isinstance(new_params, OrderedDict)
    assert "trpo_kl" in info
    assert "trpo_surr" in info
    
    new_params_flat = torch.cat([p.view(-1) for p in new_params.values()])
    
    # Check that parameters moved
    assert not torch.allclose(initial_params_flat, new_params_flat, atol=1e-6)
    
    # Check KL is within bounds (with 1.5x tolerance as in code)
    assert info["trpo_kl"] <= 0.01 * 1.51
    
    print(f"TRPO Test Passed. KL: {info['trpo_kl']:.6f}, Improvement: {info.get('trpo_improvement', 0):.6f}")

if __name__ == "__main__":
    test_trpo_outer_step_mechanics()
