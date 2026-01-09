"""MAML evaluation against baselines."""

from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import wandb
from torch.func import vmap

from configs.base import TrainConfig
from maml_rl.envs.factory import make_vec_env, sample_tasks, ENV_REGISTRY
from maml_rl.maml import FunctionalPolicy, inner_update_vpg, inner_update_value
from maml_rl.policies import (
    build_actor_critic,
    params_and_buffers,
)
from maml_rl.utils.returns import add_gae, normalize_advantages


def evaluate(
    cfg: TrainConfig,
    device: torch.device,
    checkpoint_path: str,
    pretrained_checkpoint_path: str = None,
    oracle_checkpoint_path: str = None,
):
    """Evaluate MAML on new tasks against baselines (Random Init, Oracle, Pretrained)."""
    if not checkpoint_path:
        raise ValueError("Must provide --checkpoint for evaluation mode.")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create environment
    tasks = sample_tasks(
        env_name=cfg.env.name,
        num_tasks=cfg.env.num_tasks,
        task_low=cfg.env.task_low,
        task_high=cfg.env.task_high,
    )
    env = make_vec_env(
        env_name=cfg.env.name,
        tasks=tasks,
        max_steps=cfg.env.max_steps,
        device=str(device),
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

    # Oracle Model (via MetaEnv protocol)
    env_cls = ENV_REGISTRY[cfg.env.name]
    oracle_policy = env_cls.get_oracle(
        tasks, device, checkpoint_path=oracle_checkpoint_path
    )
    if oracle_policy is not None:
        print("Oracle policy loaded.")
    elif oracle_checkpoint_path:
        print("Warning: Oracle checkpoint provided but failed to load.")
    else:
        print("No oracle checkpoint provided, skipping oracle evaluation.")

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
        value_params, value_buffers = params_and_buffers(
            value_module
        )  # Get value params

        # 1. Measure Pre-adaptation Performance (Eval/Query Set)
        # Use FULL max_steps for proper evaluation metric
        with torch.no_grad():
            step0_eval_td = env.rollout(
                max_steps=cfg.env.max_steps,
                policy=FunctionalPolicy(policy_model, policy_params, policy_buffers),
                break_when_any_done=False,
                auto_reset=True,
            )
        rew0 = step0_eval_td.get(("next", "reward")).mean(dim=1).cpu().numpy()
        results[0] = (rew0.mean(), rew0.std())
        print(f"[{name}] Step 0: {rew0.mean():.3f} +/- {rew0.std():.3f}")

        # Adaptation Loop: collect fresh support data after each gradient step
        # This matches the MAML paper's test-time procedure
        current_params = policy_params
        current_value_params = value_params  # Initialize current value params
        params_are_batched = False

        for step_k in range(1, cfg.inner.num_steps + 1):
            # Collect support data using CURRENT adapted policy
            with torch.no_grad():
                support_td = env.rollout(
                    max_steps=cfg.rollout_steps,
                    policy=FunctionalPolicy(
                        policy_model, current_params, policy_buffers
                    ),
                    break_when_any_done=False,
                    auto_reset=True,
                )
                support_td = support_td.to(torch.float32)

            support_td = add_gae(
                support_td,
                value_module,
                gamma=cfg.gamma,
                lam=cfg.lam,
                compute_grads=False,
            )
            if cfg.inner.advantage_norm:
                support_td.set(
                    "advantage", normalize_advantages(support_td["advantage"])
                )

            # Determine learning rate for this step
            current_lr = cfg.inner.lr
            if step_k == 1 and cfg.inner.first_step_lr is not None:
                current_lr = cfg.inner.first_step_lr

            # Single gradient step
            def inner_one_step(params, buffers, data):
                return inner_update_vpg(policy_model, params, buffers, data, current_lr)

            # Define value update step
            def inner_update_value_single(params, buffers, data):
                curr = params
                for _ in range(cfg.inner.num_steps):
                    curr = inner_update_value(
                        value_module, curr, buffers, data, current_lr
                    )
                return curr

            # After first step, params are batched [num_tasks, ...], so change in_dims
            params_in_dim = 0 if params_are_batched else None

            # 1. Adapt Policy
            adapted_params_batched = vmap(
                inner_one_step, in_dims=(params_in_dim, None, 0)
            )(current_params, policy_buffers, support_td)

            adapted_params = OrderedDict(
                (n, adapted_params_batched[n]) for n in policy_params.keys()
            )

            # 2. Adapt Value Function
            adapted_value_batched = vmap(
                inner_update_value_single, in_dims=(params_in_dim, None, 0)
            )(current_value_params, value_buffers, support_td)

            adapted_value_params = OrderedDict(
                (n, adapted_value_batched[n]) for n in value_params.keys()
            )

            # Update current_params for next iteration (now batched)
            current_params = adapted_params
            current_value_params = adapted_value_params
            params_are_batched = True

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

    # Close main env to release file descriptors before creating oracle env
    env.close()

    oracle_results = {}
    if oracle_policy:
        print("\n--- Evaluating Oracle ---")
        # Oracle policy expects observations with task params - create oracle env
        from maml_rl.envs.factory import make_oracle_vec_env

        oracle_env = make_oracle_vec_env(
            env_name=cfg.env.name,
            tasks=tasks,
            max_steps=cfg.env.max_steps,
            device=str(device),
            norm_obs=cfg.env.norm_obs,
            seed=cfg.seed + 2000,
        )
        try:
            if cfg.env.norm_obs:
                oracle_env.transform.init_stats(
                    num_iter=5, reduce_dim=[0, 1], cat_dim=0
                )

            oracle_env.reset()
            with torch.no_grad():
                oracle_td = oracle_env.rollout(
                    max_steps=cfg.env.max_steps,
                    policy=oracle_policy,
                    break_when_any_done=False,
                    auto_reset=True,
                )
            rew_oracle = oracle_td.get(("next", "reward")).mean(dim=1).cpu().numpy()
            oracle_results[0] = (rew_oracle.mean(), rew_oracle.std())
            print(
                f"[Oracle] Reward: {rew_oracle.mean():.3f} +/- {rew_oracle.std():.3f}"
            )
        finally:
            oracle_env.close()

    # 4. Logging and Summary
    if wandb.run:
        # 1. Log a Table for easy custom plotting / raw data inspection
        table_data = []

        def add_to_data(results_dict, model_name):
            for s, (m, std) in results_dict.items():
                table_data.append([int(s), model_name, float(m), float(std)])

        add_to_data(maml_results, "MAML")
        add_to_data(rand_results, "Random")
        if pretrained_results:
            add_to_data(pretrained_results, "Pretrained")
        if oracle_policy:
            # Oracle is constant across steps, but we add it for comparison logic
            o_mean, o_std = oracle_results[0]
            for s in maml_results.keys():
                table_data.append([int(s), "Oracle", float(o_mean), float(o_std)])

        table = wandb.Table(
            data=table_data,
            columns=["inner_step", "model", "mean_reward", "std_reward"],
        )
        wandb.log({"eval/results_table": table})

        # 2. Log Scalars as a series (Curve)
        # We iterate through steps and log all models for that step at once.
        # This creates a single "run" timeline where x-axis = time (or we use a custom x-axis)

        sorted_steps = sorted(maml_results.keys())
        for step in sorted_steps:
            log_data = {
                "inner_step": int(step),
                "eval/MAML_mean": float(maml_results[step][0]),
                "eval/MAML_std": float(maml_results[step][1]),
                "eval/Random_mean": float(rand_results[step][0]),
                "eval/Random_std": float(rand_results[step][1]),
            }

            if pretrained_results and step in pretrained_results:
                log_data["eval/Pretrained_mean"] = float(pretrained_results[step][0])
                log_data["eval/Pretrained_std"] = float(pretrained_results[step][1])

            if oracle_policy:
                # Oracle is constant/scalar, but logging it at every step makes it a line on the chart
                log_data["eval/Oracle_mean"] = float(oracle_results[0][0])
                log_data["eval/Oracle_std"] = float(oracle_results[0][1])

            wandb.log(log_data)

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

    # 5. Generate and save plot
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate Standard Error of Mean (so bands are 95% CI roughly)
    import numpy as np

    num_tasks = cfg.env.num_tasks
    sqrt_n = np.sqrt(num_tasks)

    # Extract data for plotting
    steps = sorted(maml_results.keys())
    maml_means = [maml_results[s][0] for s in steps]
    maml_sems = [maml_results[s][1] / sqrt_n for s in steps]
    rand_means = [rand_results[s][0] for s in steps]
    rand_sems = [rand_results[s][1] / sqrt_n for s in steps]

    # Plot MAML
    ax.plot(steps, maml_means, "o-", label="MAML", linewidth=2, markersize=8)
    ax.fill_between(
        steps,
        [m - s for m, s in zip(maml_means, maml_sems)],
        [m + s for m, s in zip(maml_means, maml_sems)],
        alpha=0.2,
    )

    # Plot Random
    ax.plot(steps, rand_means, "s--", label="Random Init", linewidth=2, markersize=8)
    ax.fill_between(
        steps,
        [m - s for m, s in zip(rand_means, rand_sems)],
        [m + s for m, s in zip(rand_means, rand_sems)],
        alpha=0.2,
    )

    # Plot Pretrained (if available)
    if pretrained_results:
        pt_means = [pretrained_results[s][0] for s in steps]
        pt_sems = [pretrained_results[s][1] / sqrt_n for s in steps]
        ax.plot(steps, pt_means, "^-.", label="Pretrained", linewidth=2, markersize=8)
        ax.fill_between(
            steps,
            [m - s for m, s in zip(pt_means, pt_sems)],
            [m + s for m, s in zip(pt_means, pt_sems)],
            alpha=0.2,
        )

    # Plot Oracle (horizontal line since it doesn't adapt)
    if oracle_results:
        o_mean, o_std = oracle_results[0]
        o_sem = o_std / sqrt_n
        ax.axhline(y=o_mean, color="green", linestyle=":", linewidth=2, label="Oracle")
        ax.fill_between(steps, o_mean - o_sem, o_mean + o_sem, color="green", alpha=0.1)

    ax.set_xlabel("Gradient Steps", fontsize=12)
    ax.set_ylabel("Mean Return", fontsize=12)
    ax.set_title(f"MAML Evaluation (Mean +/- SEM over {num_tasks} tasks)", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(steps)

    plt.tight_layout()
    plot_path = output_dir / "eval_results.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {plot_path}")

    if wandb.run is not None:
        wandb.finish()
