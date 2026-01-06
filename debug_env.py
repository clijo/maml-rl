
from maml_rl.envs.mars_lander import MarsLanderEnv

def test_env():
    print("Sampling tasks...")
    tasks = MarsLanderEnv.sample_tasks(1, difficulty=0.1)
    print(f"Task: {tasks[0]}")
    
    print("Creating Env...")
    env = MarsLanderEnv(task=tasks[0])
    
    print("Resetting...")
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")
    print(f"Obs: {obs}")
    
    print("Stepping...")
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step Reward: {reward}")
    print("Success!")

if __name__ == "__main__":
    test_env()
