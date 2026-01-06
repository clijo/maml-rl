import numpy as np
from maml_rl.envs.mars_lander import MarsLanderEnv

def run_scenario(name, action_fn, steps=500):
    env = MarsLanderEnv()
    # Fixed start for reproducibility: x=2500, y=2500, target=Center
    env.reset(options={"task": {"target_x": 3500, "target_y": 100, "gravity": 3.711, "wind": [0,0]}})
    # Override state
    env.state = np.array([2500.0, 2500.0, 0.0, 0.0, 500.0, 0.0, 0.0], dtype=np.float32)
    
    total_reward = 0.0
    for t in range(steps):
        action = action_fn(t, env.state)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term:
            print(f"[{name}] Terminated at step {t}. Reward: {reward:.2f}")
            break
    print(f"[{name}] Total Return: {total_reward:.2f}\n")

def action_drop(t, state):
    return np.array([0.0, -1.0]) # Power off

def action_hover(t, state):
    # Simple P controller to hover
    x, y, hs, vs, fuel, rot, power = state
    target_vs = 0.0
    err = target_vs - vs
    act_power = 0.1 if err > 0 else -0.1
    return np.array([0.0, act_power])

def action_land_perfect(t, state):
    # Mock perfect landing? Hard to script without physics solver.
    # Just try to slow down near bottom.
    x, y, hs, vs, fuel, rot, power = state
    if y < 500:
        return np.array([0.0, 1.0]) # Full power
    return np.array([0.0, 0.0]) # Glide

print("--- REWARD DEBUG ---\n")
run_scenario("Free Fall", action_drop)
run_scenario("Hover Attempt", action_hover)
# run_scenario("Soft Fall", action_land_perfect)
