"""MAML evaluation against baselines."""

import json
import os
from collections import OrderedDict

import numpy as np
import torch
import wandb
from torch.func import vmap

from configs.base import TrainConfig
from maml_rl.envs.factory import make_vec_env
from maml_rl.maml import FunctionalPolicy, inner_update_vpg
from maml_rl.policies import (
    AnalyticalNavigationOracle,
    build_actor_critic,
    params_and_buffers,
)
from maml_rl.utils.returns import add_gae, normalize_advantages


def evaluate(
    cfg: TrainConfig,
    device: torch.device,
    checkpoint_path: str,
    pretrained_checkpoint_path: str = None,
):
    """
    Evaluate MAML on new tasks against baselines (Random Init, Oracle, Pretrained).
    """
    if not checkpoint_path:
        raise ValueError("Must provide --checkpoint for evaluation mode.")

    # Try to load config from checkpoint dir
    ckpt_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(config_path):
        print(f"Loading task config from {config_path}...")
        with open(config_path, "r") as f:
            # We just print that we found it, but we rely on passed cfg for system compatibility
            pass

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create environment
    tasks, env = make_vec_env(
        env_name=cfg.env.name,
        num_tasks=cfg.env.num_tasks,  # Use num_tasks from config for batch size
        task_low=cfg.env.task_low,
        task_high=cfg.env.task_high,
        max_steps=cfg.env.max_steps,
        device=device,
        norm_obs=cfg.env.norm_obs,
        seed=cfg.seed + 1000,  # Different seed for evaluation
    )

    if cfg.env.norm_obs:
        # Ideally we should load stats from checkpoint, but for now we re-initialize
        # or we could assume the checkpoint contained them if we saved them.
        # The current save logic doesn't save env stats.
        # For fair evaluation if using NormObs, we should really save/load them.
        # But as a fallback, we init stats on the new eval tasks.
        print("Initializing observation normalization statistics for eval...")
        env.transform.init_stats(num_iter=5, reduce_dim=[0, 1], cat_dim=0)

    # Build model
    obs_spec = env.observation_spec["observation"]
    act_spec = env.action_spec
    if hasattr(act_spec, "keys") and "action" in act_spec.keys():
        act_spec = act_spec["action"]

    obs_dim = obs_spec.shape[-1]
    act_dim = act_spec.shape[-1] if act_spec.shape else 1

    # MAML Model
    maml_actor, maml_policy, maml_value = build_actor_critic(
        obs_dim, act_dim, hidden_sizes=cfg.model.hidden_sizes
    )
    maml_actor.to(device)
    maml_value.to(device)
    maml_policy.load_state_dict(checkpoint["policy_state_dict"])
    maml_value.load_state_dict(checkpoint["value_state_dict"])
    print("MAML model loaded.")

    # Pretrained Baseline (if provided)
    pretrained_policy = None
    pretrained_value = None
    if pretrained_checkpoint_path:
        print(f"Loading Pretrained Baseline from {pretrained_checkpoint_path}...")
        pt_checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
        _, pretrained_policy, pretrained_value = build_actor_critic(
            obs_dim, act_dim, hidden_sizes=cfg.model.hidden_sizes
        )
        pretrained_policy.to(device)
        pretrained_value.to(device)
        pretrained_policy.load_state_dict(pt_checkpoint["policy_state_dict"])
        pretrained_value.load_state_dict(pt_checkpoint["value_state_dict"])
        print("Pretrained Baseline loaded.")

    # Random Init Model (Baseline B)
    rand_actor, rand_policy, rand_value = build_actor_critic(
        obs_dim, act_dim, hidden_sizes=cfg.model.hidden_sizes
    )
    rand_actor.to(device)
    rand_value.to(device)
    print("Random Init model created.")

    # Oracle Model (if applicable)
    oracle_policy = None
    if cfg.env.name == "navigation":
        goals = np.stack([t["goal"] for t in tasks])
        goals_tensor = torch.tensor(goals, device=device, dtype=torch.float32)
        oracle_policy = AnalyticalNavigationOracle(goals_tensor, device=device)
        print("Analytical Oracle prepared.")

    # 2. Evaluation Helper
    def run_adaptation(name, policy_model, value_module, initial_params):
        results = {}

        # Reset env to ensure fair comparison from same start state
        # Note: Tasks are fixed in the env construction, so we just reset state.
        # But to be safe and cleaner, we reset with the original logic if possible,
        # though ParallelEnv auto-resets.
        # We explicitly reset here to start fresh for this model.
        env.reset()

        # Step 0 (Pre-adaptation)
        policy_params, policy_buffers = params_and_buffers(policy_model)
        current_params_list = policy_params

        # 1. Measure Pre-adaptation Performance (Eval/Query Set)
        # Use FULL max_steps for proper evaluation metric
        with torch.no_grad():
            step0_eval_td = env.rollout(
                max_steps=cfg.env.max_steps,
                policy=FunctionalPolicy(
                    policy_model, current_params_list, policy_buffers
                ),
                break_when_any_done=False,
                auto_reset=True,
            )
        rew0 = step0_eval_td.get(("next", "reward")).mean(dim=1).cpu().numpy()
        results[0] = (rew0.mean(), rew0.std())
        print(f"[{name}] Step 0: {rew0.mean():.3f} +/- {rew0.std():.3f}")

        # 2. Collect Support Data for Adaptation (Support Set)
        # Use rollout_steps (e.g. 40) for adaptation data
        with torch.no_grad():
            support_td = env.rollout(
                max_steps=cfg.rollout_steps,
                policy=FunctionalPolicy(
                    policy_model, current_params_list, policy_buffers
                ),
                break_when_any_done=False,
                auto_reset=True,
            )
            support_td = support_td.to(torch.float32)

        # Adaptation Loop
        support_td = add_gae(
            support_td, value_module, gamma=cfg.gamma, lam=cfg.lam, compute_grads=False
        )
        if cfg.inner.advantage_norm:
            support_td.set("advantage", normalize_advantages(support_td["advantage"]))

        def inner_step(params, buffers, data):
            # Inner loop VPG update
            return inner_update_vpg(policy_model, params, buffers, data, cfg.inner.lr)

        for step_k in range(1, cfg.inner.num_steps + 1):
            # Update params using SUPPORT data
            current_params_list = vmap(
                inner_step, in_dims=(None if step_k == 1 else 0, None, 0)
            )(current_params_list, policy_buffers, support_td)

            adapted_params = OrderedDict(
                (n, current_params_list[n]) for n in policy_params.keys()
            )

            # Evaluate on QUERY data (Full max_steps)
            with torch.no_grad():
                td_k = env.rollout(
                    max_steps=cfg.env.max_steps,
                    policy=FunctionalPolicy(
                        policy_model, adapted_params, policy_buffers
                    ),
                    break_when_any_done=False,
                    auto_reset=True,
                )

            rew_k = td_k.get(("next", "reward")).mean(dim=1).cpu().numpy()
            results[step_k] = (rew_k.mean(), rew_k.std())
            print(f"[{name}] Step {step_k}: {rew_k.mean():.3f} +/- {rew_k.std():.3f}")

        return results

    # 3. Run Comparisons
    print("\n--- Evaluating MAML ---")
    maml_results = run_adaptation("MAML", maml_policy, maml_value, None)

    pretrained_results = {}
    if pretrained_policy:
        print("\n--- Evaluating Pretrained Baseline ---")
        pretrained_results = run_adaptation(
            "Pretrained", pretrained_policy, pretrained_value, None
        )

    print("\n--- Evaluating Random Init (Baseline) ---")
    rand_results = run_adaptation("Random", rand_policy, rand_value, None)

    oracle_results = {}
    if oracle_policy:
        print("\n--- Evaluating Oracle ---")
        with torch.no_grad():
            oracle_td = env.rollout(
                max_steps=cfg.env.max_steps,
                policy=oracle_policy,
                break_when_any_done=False,
                auto_reset=True,
            )
        rew_oracle = oracle_td.get(("next", "reward")).mean(dim=1).cpu().numpy()
        oracle_results[0] = (rew_oracle.mean(), rew_oracle.std())
        print(f"[Oracle] Reward: {rew_oracle.mean():.3f} +/- {rew_oracle.std():.3f}")

    # 4. Logging and Summary
    if wandb.run:
        for step, (mean, std) in maml_results.items():
            wandb.log(
                {f"eval/MAML_step_{step}_mean": mean, f"eval/MAML_step_{step}_std": std}
            )
        for step, (mean, std) in rand_results.items():
            wandb.log(
                {
                    f"eval/Random_step_{step}_mean": mean,
                    f"eval/Random_step_{step}_std": std,
                }
            )
        if pretrained_results:
            for step, (mean, std) in pretrained_results.items():
                wandb.log(
                    {
                        f"eval/Pretrained_step_{step}_mean": mean,
                        f"eval/Pretrained_step_{step}_std": std,
                    }
                )
        if oracle_policy:
            wandb.log(
                {
                    "eval/Oracle_mean": oracle_results[0][0],
                    "eval/Oracle_std": oracle_results[0][1],
                }
            )

    print("\n=== Final Summary ===")
    header = f"{'Step':<5} | {'MAML':<20} | {'Random':<20}"
    if pretrained_policy:
        header += f" | {'Pretrained':<20}"
    header += f" | {'Oracle':<20}"
    print(header)
    print("-" * len(header))

    steps = sorted(maml_results.keys())
    for s in steps:
        m_mean, m_std = maml_results[s]
        r_mean, r_std = rand_results[s]
        o_str = (
            f"{oracle_results[0][0]:.2f} +/- {oracle_results[0][1]:.2f}"
            if oracle_policy
            else "N/A"
        )

        row = f"{s:<5} | {m_mean:.2f} +/- {m_std:.2f}      | {r_mean:.2f} +/- {r_std:.2f}      "
        if pretrained_policy:
            p_mean, p_std = pretrained_results.get(s, (0.0, 0.0))
            row += f"| {p_mean:.2f} +/- {p_std:.2f}      "
        row += f"| {o_str}"
        print(row)
