import os
import torch
import numpy as np
import gymnasium as gym
from tensordict import TensorDict
from maml_rl.envs.mars_lander import MarsLanderEnv
from maml_rl.policies import build_actor_critic, params_and_buffers
from maml_rl.maml import inner_update_vpg
from collections import OrderedDict

def evaluate_checkpoint():
    checkpoint_path = "checkpoints/run_20260107_133527/checkpoint_1300.pt"
    device = torch.device("cpu")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 1. Setup Environment & Task (Easy)
    task = {
        "gravity": 3.711,
        "wind": [0.0, 0.0],
        "target_x": 2500.0,
        "target_y": 100.0,
        "landing_width": 2000.0,
        "start_x": 2500.0,
        "start_y": 2500.0,
        "start_hs": 0.0,
        "start_rotate": 0.0,
        "difficulty": 0.0
    }
    env = MarsLanderEnv(task=task)
    
    # 2. Setup Policy
    obs_dim = 9
    act_dim = 2
    actor, policy_model, _ = build_actor_critic(obs_dim, act_dim, hidden_sizes=[100, 100])
    policy_model.load_state_dict(checkpoint["policy_state_dict"])
    
    print("\n--- Evaluation (Prior Policy) ---")
    
    successes = 0
    
    for i in range(5):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_rew = 0
        landed = False
        crashed = False
        
        step_count = 0
        while not (done or truncated):
             with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32)
                # Wrap in TensorDict
                input_td = TensorDict({"observation": obs_t}, batch_size=[])
                
                # Use actor directly (weights are already loaded in policy_model which actor wraps)
                output_td = actor(input_td)
                
                # Deterministic or Stochastic?
                # output_td has "loc", "scale", "action"
                # "action" is sampled.
                # If we want deterministic, use "loc".
                # But typical RL evaluation uses deterministic mean.
                
                action = output_td["loc"].squeeze(0) if i == 0 else output_td["action"].squeeze(0)
                
             obs, reward, done, truncated, _ = env.step(action.numpy())
             ep_rew += reward
             step_count += 1
             
             if done:
                 if reward >= 100: landed = True
                 else: crashed = True
        
        status = "LANDED" if landed else "CRASHED"
        if landed: successes += 1
        print(f"Ep {i}: Reward={ep_rew:.2f} Steps={step_count} Status={status}")

    print(f"\nSuccess Rate (Prior): {successes}/5")

if __name__ == "__main__":
    evaluate_checkpoint()
