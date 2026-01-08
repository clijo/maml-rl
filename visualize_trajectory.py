
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tensordict import TensorDict
from maml_rl.envs.mars_lander import MarsLanderEnv
from maml_rl.policies import build_actor_critic

def visualize_checkpoint():
    # Find latest checkpoint in the specific run directory
    run_dir = "checkpoints/run_20260107_155906"
    checkpoints = [f for f in os.listdir(run_dir) if f.endswith(".pt") and "checkpoint" in f]
    if not checkpoints:
        print("No checkpoints found.")
        return
        
    # Sort by iteration number
    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    latest_checkpoint = checkpoints[-1]
    checkpoint_path = os.path.join(run_dir, latest_checkpoint)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    
    # 1. Setup Environment & Task
    # Difficulty 0.1 params (Approximate)
    task = {
        "gravity": 3.711,
        "wind": [0.0, 0.0],
        "target_x": 2500.0, # Center
        "target_y": 100.0,
        "landing_width": 2000.0,
        "start_x": 2500.0,
        "start_y": 2500.0,
        "start_hs": 0.0,
        "start_rotate": 0.0,
        "difficulty": 0.1
    }
    env = MarsLanderEnv(task=task)
    
    # 2. Setup Policy
    obs_dim = 9
    act_dim = 2
    actor, policy_model, _ = build_actor_critic(obs_dim, act_dim, hidden_sizes=[100, 100])
    policy_model.load_state_dict(checkpoint["policy_state_dict"])
    
    print("\n--- Running Trajectory ---")
    obs, _ = env.reset()
    done = False
    truncated = False
    
    xs = []
    ys = []
    
    step = 0
    while not (done or truncated):
         # Extract X, Y from obs for plotting (un-normalized if needed)
         # Obs: [x, y, hs, vs, fuel, rotate, power, dist_x, dist_y]
         # Normalized in env? MarsLanderEnv returns normalized obs usually?
         # Let's check env code. The env returns normalized if bounds are set?
         # Step method: x = obs[0] * MAP_WIDTH. 
         # So obs is normalized 0-1.
         
         x_norm = obs[0]
         y_norm = obs[1]
         xs.append(x_norm * 7000.0) # MAP_WIDTH
         ys.append(y_norm * 3000.0) # MAP_HEIGHT
         
         with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32)
            input_td = TensorDict({"observation": obs_t}, batch_size=[])
            output_td = actor(input_td)
            action = output_td["loc"].squeeze(0) # Deterministic
            
         obs, reward, done, truncated, _ = env.step(action.numpy())
         step += 1
    
    print(f"Episode finished in {step} steps.")
    
    # 3. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, label="Lander Trajectory", marker='o', markersize=3)
    
    # Draw Ground
    plt.axhline(y=100, color='brown', linestyle='-', linewidth=2, label="Ground")
    
    # Draw Landing Zone
    target_x = task["target_x"]
    width = task["landing_width"]
    plt.plot([target_x - width/2, target_x + width/2], [100, 100], color='green', linewidth=5, label="Landing Zone")
    
    # Draw Start
    plt.plot(task["start_x"], task["start_y"], 'r*', markersize=15, label="Start")
    
    plt.xlim(0, 7000)
    plt.ylim(0, 3000)
    plt.title(f"Mars Lander Trajectory (Iter ~{latest_checkpoint})")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = "trajectory_viz.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    visualize_checkpoint()
