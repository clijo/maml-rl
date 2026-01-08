import numpy as np
from maml_rl.envs.mars_lander import MarsLanderEnv

def verify_env():
    print("Verifying Mars Lander Environment...")
    
    # 1. Sample Tasks (Easy)
    tasks = MarsLanderEnv.sample_tasks(num_tasks=5, difficulty=0.0, randomize_physics=False, randomize_start=False, randomize_landing=True)
    
    print(f"\nSampled {len(tasks)} tasks (Difficulty 0.0):")
    for i, t in enumerate(tasks):
        print(f"Task {i}: Target X={t['target_x']:.1f}, Width={t['landing_width']:.1f}, Start X={t['start_x']:.1f}")
        # Verify Start X is fixed 2500, Target X is close (offset small for easy)
        assert t['start_x'] == 2500.0, "Start X should be fixed"
        
    # 2. Run Episode
    env = MarsLanderEnv(task=tasks[0])
    obs, _ = env.reset()
    print("\nInitial Observation:", obs)
    
    total_reward = 0
    steps = 0
    done = False
    
    print("\nRunning Simulation...")
    while not done and steps < 100:
        # Action: No rotation, some power
        action = np.array([0.0, 0.5], dtype=np.float32) 
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if steps % 10 == 0:
            print(f"Step {steps}: Reward={reward:.4f}, Obs={obs}")
            
    print(f"\nEpisode Finished. Steps: {steps}, Total Reward: {total_reward:.4f}, Done: {done}")

if __name__ == "__main__":
    verify_env()
