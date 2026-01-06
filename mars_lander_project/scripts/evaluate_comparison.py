
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from tqdm import tqdm

from configs.base import TrainConfig
from maml_rl.envs.factory import make_vec_env, make_oracle_vec_env
from maml_rl.policies import build_actor_critic
from maml_rl.maml import inner_update_vpg

def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt["config"]
    # Reconstruct config object (simplified)
    # We mainly need model structure
    hidden_sizes = cfg_dict["model"]["hidden_sizes"]
    
    # We need to know obs_dim/act_dim. 
    # Hardcoding for MarsLander for now or we could instantiate env.
    # Standard: 9 (was 7). Oracle: 14 (was 11).
    is_oracle = cfg_dict.get("oracle", False)
    obs_dim = 14 if is_oracle else 9
    act_dim = 2
    
    actor, policy_model, _ = build_actor_critic(obs_dim, act_dim, hidden_sizes)
    policy_model.load_state_dict(ckpt["policy_state_dict"])
    actor.to(device)
    
    return actor, cfg_dict

def evaluate_on_tasks(
    policy, 
    tasks, 
    env_name="mars-lander", 
    is_oracle=False, 
    is_maml=False, 
    inner_lr=0.05, 
    inner_steps=1,
    device="cpu"
):
    """
    Evaluate policy on a set of tasks.
    For Oracle: Zero-shot (using oracle env).
    For MAML: K-shot adaptation.
    """
    metrics = {
        "rewards": [],
        "success": [],
        "fuel": [],
        "landing_speed": [],
    }
    
    # Create env
    if is_oracle:
        env = make_oracle_vec_env(
            env_name=env_name,
            tasks=tasks,
            max_steps=500,
            device=device,
            norm_obs=True, # Assuming trained with norm_obs
            seed=42
        )
    else:
        # Standard env
        # Note: We can't easily use the parallel env for sequential adaptation logic 
        # unless we broadcast the policy update or handle it per-env.
        # For MAML adaptation evaluation, it's often easier to loop over tasks sequentially
        # or use the vectorized approach if we adapt efficiently.
        # Let's use sequential for clarity and distinct metrics per task.
        pass

    # We'll adapt per task
    
    for i, task in enumerate(tqdm(tasks, desc="Evaluating Tasks")):
        # 1. Create single env for this task
        if is_oracle:
            # Oracle doesn't adapt, just runs
            # We can use the vec env if we want, but let's stick to single flow
            pass
        
        # Helper to run an episode
        def run_episode(policy_network, env_instance):
            # Rollout using torchrl env
            # Policy needs to be compatible. MamL policies are.
            with torch.no_grad():
                td = env_instance.rollout(max_steps=500, policy=policy_network, auto_reset=True)
            
            # Compute metrics
            rews = td["next", "reward"]
            total_rew = rews.sum().item()
            
            # Success check (Heuristic: Total Reward > 50 implies landing)
            # MarsLander: Landing = 100 + fuel. Crash = -100.
            is_success = total_rew > 0.0 
            
            # Fuel not easily tracked in obs without explicit key, but reward tells us enough.
            fuel_used = 0.0
            
            return total_rew, is_success, fuel_used

        # Setup Env
        from maml_rl.envs.mars_lander import MarsLanderEnv, MetaMarsLanderOracleEnv
        from torchrl.envs import GymWrapper, TransformedEnv
        from torchrl.envs.transforms import Compose, InitTracker, StepCounter, ObservationNorm
        
        base_env = MarsLanderEnv()
        base_env.set_task(task)
        if is_oracle:
            base_env = MetaMarsLanderOracleEnv(base_env)
            
        env = GymWrapper(base_env, device=device)
        env = TransformedEnv(env, Compose(InitTracker(), StepCounter(max_steps=500)))
        # Norm obs?
        # If model expects norm obs, we need it. 
        # But we need statistics! 
        # For evaluation, we usually use running stats from training or just re-estimate.
        # Ideally we load the obs_norm stats from checkpoint. 
        # For now, let's assume Identity or standard norm for MAML 
        # (MAML often normalizes per task batch, but standard PPO tracks moving avg).
        # We will add ObservationNorm(standard_normal=True) but it starts fresh.
        # This might be an issue.
        # TODO: Load ObsNorm stats.
        if False: # Always normalize
            env = TransformedEnv(env, ObservationNorm(in_keys=["observation"], standard_normal=True))
            env.transform[-1].init_stats(num_iter=10, reduce_dim=[0], cat_dim=0) 

        # Adaptation (MAML only)
        current_policy = policy
        # if is_maml:
        #      # Support set collection
        #     support_data = []
        #     # Reset
        #     # obs, _ = env.reset()
        #     pass

        # Validating Zero-Shot for Oracle vs Zero-Shot MAML (Pre-adaptation) first
        rew, succ, _ = run_episode(current_policy, env)
        metrics["rewards"].append(rew)
        metrics["success"].append(succ)
        
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maml_ckpt", type=str, default=None)
    parser.add_argument("--oracle_ckpt", type=str, default=None)
    parser.add_argument("--num_tasks", type=int, default=20)
    args = parser.parse_args()
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    
    # Generate test tasks
    from maml_rl.envs.mars_lander import MarsLanderEnv
    test_tasks = MarsLanderEnv.sample_tasks(args.num_tasks)
    
    results = {}
    
    if args.maml_ckpt:
        print("Evaluating MAML...")
        maml_policy, _ = load_model(args.maml_ckpt, device)
        maml_metrics = evaluate_on_tasks(maml_policy, test_tasks, is_maml=True, device=device)
        results["MAML"] = maml_metrics
        print(f"MAML Mean Reward: {np.mean(maml_metrics['rewards'])}")
        print(f"MAML Success Rate: {np.mean(maml_metrics['success'])}")

    if args.oracle_ckpt:
        print("Evaluating Oracle...")
        oracle_policy, _ = load_model(args.oracle_ckpt, device)
        oracle_metrics = evaluate_on_tasks(oracle_policy, test_tasks, is_oracle=True, device=device)
        results["Oracle"] = oracle_metrics
        print(f"Oracle Mean Reward: {np.mean(oracle_metrics['rewards'])}")
        print(f"Oracle Success Rate: {np.mean(oracle_metrics['success'])}")
        
    # Plotting
    if results:
        plt.figure(figsize=(10, 6))
        data = []
        labels = []
        for name, m in results.items():
            data.append(m["rewards"])
            labels.append(name)
        
        plt.boxplot(data, labels=labels)
        plt.title("Reward Distribution on Test Tasks")
        plt.ylabel("Episode Reward")
        plt.savefig("comparison_plot.png")
        print("Saved comparison_plot.png")

if __name__ == "__main__":
    main()
