
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.modules import ValueOperator, TanhNormal
from tensordict.nn import TensorDictSequential, TensorDictModule, NormalParamExtractor
from collections import OrderedDict
from torch.func import vmap, functional_call

from maml_rl.maml import inner_update_vpg, inner_update_value, outer_loss_ppo
from maml_rl.policies import params_and_buffers

def test_ppo_optimization_loop():
    """
    Integration test to verify the Meta-PPO optimization loop mechanics:
    1. Re-adaptation inside the loop.
    2. Gradient flow.
    3. Parameter updates.
    """
    # 1. Setup minimal Policy & Value
    # Simple 1-layer MLP for policy
    net = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 2))
    policy_module = TensorDictModule(net, in_keys=["observation"], out_keys=["param"])
    extractor = NormalParamExtractor()
    extractor_module = TensorDictModule(extractor, in_keys=["param"], out_keys=["loc", "scale"])
    policy_model = TensorDictSequential(policy_module, extractor_module)
    
    # Simple Value
    val_net = nn.Sequential(nn.Linear(2, 1))
    value_module = ValueOperator(val_net, in_keys=["observation"], out_keys=["state_value"])

    # 2. Mock Data (Batch of 2 tasks)
    batch_size = 2
    time_steps = 5
    
    # Support data
    support_td = TensorDict({
        "observation": torch.randn(batch_size, time_steps, 2),
        "action": torch.randn(batch_size, time_steps, 1),
        "advantage": torch.randn(batch_size, time_steps, 1),
        "value_target": torch.randn(batch_size, time_steps, 1),
        "sample_log_prob": torch.randn(batch_size, time_steps, 1), # For old log prob
    }, batch_size=[batch_size, time_steps])
    
    # Query data
    query_td = TensorDict({
        "observation": torch.randn(batch_size, time_steps, 2),
        "action": torch.randn(batch_size, time_steps, 1),
        "advantage": torch.randn(batch_size, time_steps, 1),
        "value_target": torch.randn(batch_size, time_steps, 1),
        "action_log_prob": torch.randn(batch_size, time_steps, 1), # Old log prob
        "state_value": torch.randn(batch_size, time_steps, 1), # For value clipping
    }, batch_size=[batch_size, time_steps])

    # 3. Optimization Loop Simulation
    optimizer = torch.optim.SGD(list(policy_model.parameters()) + list(value_module.parameters()), lr=0.1)
    
    initial_param = next(policy_model.parameters()).clone()

    # Config params
    ppo_epochs = 2
    clip_eps = 0.2
    inner_lr = 0.01
    
    for _ in range(ppo_epochs):
        # A. Get current params
        curr_policy_params, curr_policy_buffers = params_and_buffers(policy_model)
        curr_value_params, curr_value_buffers = params_and_buffers(value_module)
        
        # B. Re-adapt Policy
        def inner_update_single(params, buffers, data):
            # One step update
            return inner_update_vpg(policy_model, params, buffers, data, inner_lr)

        adapted_params_list = vmap(inner_update_single, (None, None, 0))(
            curr_policy_params, curr_policy_buffers, support_td
        )
        adapted_params = OrderedDict((k, v) for k, v in adapted_params_list.items())

        # C. Re-adapt Value
        def inner_update_val_single(params, buffers, data):
             return inner_update_value(value_module, params, buffers, data, inner_lr)
             
        adapted_val_params_list = vmap(inner_update_val_single, (None, None, 0))(
            curr_value_params, curr_value_buffers, support_td
        )
        adapted_val_params = OrderedDict((k, v) for k, v in adapted_val_params_list.items())

        # D. Compute Losses
        losses, infos = vmap(outer_loss_ppo, (None, 0, None, 0, None, None))(
            policy_model, adapted_params, curr_policy_buffers, query_td, clip_eps, 0.0
        )
        policy_loss = losses.mean()
        
        # Value Loss
        def compute_val_loss(params, buffers, td):
            out = functional_call(value_module, (params, buffers), (td,))
            pred = out.get("state_value")
            target = td.get("value_target")
            return ((pred - target)**2).mean()

        val_losses = vmap(compute_val_loss, (0, None, 0))(
            adapted_val_params, curr_value_buffers, query_td
        )
        value_loss = val_losses.mean()
        
        total_loss = policy_loss + 0.5 * value_loss
        
        # E. Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # 4. Verification
    # Check if parameters actually changed
    final_param = next(policy_model.parameters())
    assert not torch.allclose(initial_param, final_param), "Parameters did not update!"
    print("Integration test passed: Meta-PPO loop runs and updates parameters.")

if __name__ == "__main__":
    test_ppo_optimization_loop()
