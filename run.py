import wandb
from dataclasses import asdict
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from jsonargparse import ActionConfigFile, ArgumentParser
from torch.func import vmap, functional_call
from torch.optim import Adam

from configs.base import TrainConfig
from maml_rl.envs.factory import make_vec_env, sample_tasks
from maml_rl.maml import (
    FunctionalPolicy,
    inner_update_vpg,
    inner_update_value,
    outer_loss_ppo,
    outer_step_trpo,
)
from maml_rl.policies import build_actor_critic, params_and_buffers
from maml_rl.utils.returns import add_gae, normalize_advantages


def enable_tf32():
    """
    Enable TensorFloat32 matmul on Ampere+ GPUs for faster float32 GEMMs.
    No effect on older GPUs or CPU.
    """
    if not torch.cuda.is_available():
        return
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:  # Ampere or newer
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled (float32 matmul precision = high).")


def wandb_setup(cfg):
    if not cfg.wandb.enable:
        return None
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        name=cfg.wandb.name,
        config=asdict(cfg),
    )
    return run


def _get_device(device_str: str) -> torch.device:
    """
    Get torch device from string specification.

    Args:
        device_str: Device string ("auto", "cuda", "mps", or "cpu")

    Returns:
        torch.device object
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def main():
    """
    Run MAML on Ant Goal Velocity with VPG inner loop and PPO or TRPO outer loop.

    Example:
        python run.py --cfg configs/config.yaml
    Override any field via CLI, e.g.:
        python run.py --cfg configs/config.yaml --config.outer.lr 0.0001
    """
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile, help="Path to YAML config")
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "eval"], help="Run mode"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to MAML checkpoint for evaluation",
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default=None,
        help="Path to Pretrained (Baseline) checkpoint",
    )
    parser.add_class_arguments(TrainConfig, "config")
    args = parser.parse_args()
    cfg: TrainConfig = parser.instantiate_classes(args).config

    enable_tf32()
    wandb_setup(cfg)

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = _get_device(cfg.env.device)
    print(f"Device: {device}")
    print(f"Algorithm: {cfg.algorithm.upper()}")

    if args.mode == "train":
        train(cfg, device)
    else:
        evaluate(cfg, device, args.checkpoint, args.pretrained_checkpoint)


def train(cfg: TrainConfig, device: torch.device):
    """
    Run MAML training loop.
    """
    # Create environment once and reuse it
    tasks, env = make_vec_env(
        env_name=cfg.env.name,
        num_tasks=cfg.env.num_tasks,
        task_low=cfg.env.task_low,
        task_high=cfg.env.task_high,
        max_steps=cfg.env.max_steps,
        device=device,
        norm_obs=cfg.env.norm_obs,
        seed=cfg.seed,
    )

    if cfg.env.norm_obs:
        print("Initializing observation normalization statistics...")
        env.transform.init_stats(num_iter=5, reduce_dim=[0, 1], cat_dim=0)
        print(f"Obs norm loc mean: {env.transform.loc.mean():.4f}")

    # Determine dimensions from spec
    obs_spec = env.observation_spec["observation"]
    act_spec = env.action_spec
    if hasattr(act_spec, "keys") and "action" in act_spec.keys():
        act_spec = act_spec["action"]

    obs_dim = obs_spec.shape[-1]
    act_dim = act_spec.shape[-1] if act_spec.shape else 1

    actor, policy_model, value_module = build_actor_critic(
        obs_dim, act_dim, hidden_sizes=cfg.model.hidden_sizes
    )
    actor.to(device)
    value_module.to(device)

    # Optimizer setup based on algorithm
    if cfg.algorithm == "trpo":
        # TRPO handles policy update manually; optimizer only for value function
        optimizer = Adam(value_module.parameters(), lr=cfg.outer.lr)
    else:
        # PPO updates both policy and value
        optimizer = Adam(
            list(policy_model.parameters()) + list(value_module.parameters()),
            lr=cfg.outer.lr,
        )

    for iteration in range(1, cfg.num_iterations + 1):
        # Update tasks in existing env for better performance
        if iteration > 1:
            tasks = sample_tasks(
                env_name=cfg.env.name,
                num_tasks=cfg.env.num_tasks,
                task_low=cfg.env.task_low,
                task_high=cfg.env.task_high,
            )
            # Update tasks via reset options
            env.reset(options=[{"task": t} for t in tasks])
            env.set_seed(cfg.seed + iteration)

        policy_params, policy_buffers = params_and_buffers(policy_model)
        value_params, value_buffers = params_and_buffers(value_module)

        # 1. Collect Support Data (all tasks in parallel)
        with torch.no_grad():
            support_td = env.rollout(
                max_steps=cfg.rollout_steps,
                policy=actor,
                break_when_any_done=False,
                auto_reset=True,
            )
            support_td = support_td.to(torch.float32)
        # support_td shape: [num_tasks, rollout_steps]

        # Compute GAE with shared value function (no gradients needed here)
        support_td = add_gae(
            support_td,
            value_module,
            gamma=cfg.gamma,
            lam=cfg.lam,
            compute_grads=False,
        )

        # Normalize advantages if requested
        if cfg.inner.advantage_norm:
            support_td.set("advantage", normalize_advantages(support_td["advantage"]))

        # 2. Inner Loop Adaptation (Vectorized over tasks)
        num_tasks = support_td.shape[0]
        if num_tasks != cfg.env.num_tasks:
            raise ValueError(
                f"Support data batch size {num_tasks} doesn't match "
                f"num_tasks {cfg.env.num_tasks}"
            )

        def inner_update_single(params, buffers, support_td_single):
            current_params = params
            for _ in range(cfg.inner.num_steps):
                current_params = inner_update_vpg(
                    policy_model,
                    current_params,
                    buffers,
                    support_td_single,
                    cfg.inner.lr,
                )
            return current_params

        def inner_update_value_single(params, buffers, support_td_single):
            current_params = params
            for _ in range(cfg.inner.num_steps):
                current_params = inner_update_value(
                    value_module,
                    current_params,
                    buffers,
                    support_td_single,
                    cfg.inner.lr,
                )
            return current_params

        if cfg.inner.num_steps > 0:
            adapted_params_list = vmap(inner_update_single, in_dims=(None, None, 0))(
                policy_params, policy_buffers, support_td
            )

            adapted_params = OrderedDict(
                (name, adapted_params_list[name]) for name in policy_params.keys()
            )

            adapted_value_params_list = vmap(
                inner_update_value_single, in_dims=(None, None, 0)
            )(value_params, value_buffers, support_td)
            adapted_value_params = OrderedDict(
                (name, adapted_value_params_list[name]) for name in value_params.keys()
            )

            # 3. Collect Query Data (using Adapted Parameters)
            query_policy = FunctionalPolicy(
                policy_model, adapted_params, policy_buffers
            )

            with torch.no_grad():
                query_td = env.rollout(
                    max_steps=cfg.rollout_steps,
                    policy=query_policy,
                    break_when_any_done=False,
                    auto_reset=True,
                )
                query_td = query_td.to(torch.float32)

            def compute_adapted_values(params, buffers, td):
                functional_call(value_module, (params, buffers), (td,))
                functional_call(value_module, (params, buffers), (td.get("next"),))
                return td

            with torch.no_grad():
                query_td = vmap(compute_adapted_values, (0, None, 0))(
                    adapted_value_params, value_buffers, query_td
                )

            query_td = add_gae(
                query_td,
                value_module=None,
                gamma=cfg.gamma,
                lam=cfg.lam,
                compute_grads=False,
            )

            if cfg.outer.advantage_norm:
                query_td.set("advantage", normalize_advantages(query_td["advantage"]))

        else:
            # Optimization for Pretraining / Standard RL (num_steps=0)
            # No adaptation -> Reuse support data
            adapted_params = policy_params
            adapted_value_params = value_params
            query_td = support_td

            # support_td already has GAE computed and advantages normalized (if enabled)

        # 4. Outer Loop Update

        if cfg.algorithm == "trpo":
            # TRPO Update (Policy Only)
            new_params, trpo_info = outer_step_trpo(
                policy_model=policy_model,
                policy_params=policy_params,
                policy_buffers=policy_buffers,
                support_td=support_td,
                query_td=query_td,
                inner_lr=cfg.inner.lr,
                inner_steps=cfg.inner.num_steps,
                max_kl=cfg.trpo.max_kl,
                damping=cfg.trpo.damping,
                cg_iters=cfg.trpo.cg_iters,
                line_search_max_steps=cfg.trpo.line_search_max_steps,
                line_search_backtrack_ratio=cfg.trpo.line_search_backtrack_ratio,
            )

            # Apply new params to model
            for name, param in new_params.items():
                # We need to update the model parameters in-place
                # Since policy_model is TensorDictSequential, we access via get_parameter or modify via named_parameters
                # But functional parameters are disconnected. We need to copy back.
                # The safest way for standard nn.Module is:
                model_param = policy_model.get_parameter(name)
                model_param.data.copy_(param.data)

            # Value Function Update (Standard Adam)
            # We iterate a few times for value function to keep up
            curr_value_params, curr_value_buffers = params_and_buffers(value_module)

            def compute_value_loss_single(params, buffers, td):
                td_out = functional_call(value_module, (params, buffers), (td,))
                pred = td_out.get("state_value")
                target = td.get("value_target")
                return F.mse_loss(pred, target)

            # Re-adapt value params for the loss computation
            # We do this for a few epochs (same as PPO epochs usually or fixed 5)
            value_epochs = cfg.outer.ppo_epochs

            for _ in range(value_epochs):
                # Re-adapt value params
                adapted_val_list = vmap(
                    inner_update_value_single, in_dims=(None, None, 0)
                )(curr_value_params, curr_value_buffers, support_td)

                adapted_val = OrderedDict(
                    (n, adapted_val_list[n]) for n in curr_value_params.keys()
                )

                # Compute loss
                losses = vmap(compute_value_loss_single, (0, None, 0))(
                    adapted_val, curr_value_buffers, query_td
                )
                value_loss = losses.mean()

                optimizer.zero_grad()
                value_loss.backward()
                optimizer.step()

            total_loss = trpo_info["trpo_surr"]  # Just for logging
            policy_loss = trpo_info["trpo_surr"]
            avg_ratio = 1.0  # Placeholder
            avg_entropy = 0.0  # Placeholder
            grad_norm = 0.0  # Placeholder

            # Update log dict
            trpo_logs = trpo_info

        else:
            # PPO Update (Policy + Value)
            for ppo_epoch in range(cfg.outer.ppo_epochs):
                curr_policy_params, curr_policy_buffers = params_and_buffers(
                    policy_model
                )
                curr_value_params, curr_value_buffers = params_and_buffers(value_module)

                adapted_params_list = vmap(
                    inner_update_single, in_dims=(None, None, 0)
                )(curr_policy_params, curr_policy_buffers, support_td)

                adapted_params = OrderedDict(
                    (name, adapted_params_list[name])
                    for name in curr_policy_params.keys()
                )

                adapted_value_params_list = vmap(
                    inner_update_value_single, in_dims=(None, None, 0)
                )(curr_value_params, curr_value_buffers, support_td)

                adapted_value_params = OrderedDict(
                    (name, adapted_value_params_list[name])
                    for name in curr_value_params.keys()
                )

                losses, infos = vmap(outer_loss_ppo, (None, 0, None, 0, None, None))(
                    policy_model,
                    adapted_params,
                    curr_policy_buffers,
                    query_td,
                    cfg.outer.clip_eps,
                    cfg.outer.entropy_coef,
                )
                policy_loss = losses.mean()

                def compute_value_loss_single(params, buffers, td):
                    td_out = functional_call(value_module, (params, buffers), (td,))
                    pred = td_out.get("state_value")
                    target = td.get("value_target")

                    if cfg.outer.clip_value_loss:
                        old_pred = td.get("state_value").detach()
                        pred_clipped = old_pred + torch.clamp(
                            pred - old_pred,
                            -cfg.outer.clip_eps,
                            cfg.outer.clip_eps,
                        )
                        loss1 = (pred - target).pow(2)
                        loss2 = (pred_clipped - target).pow(2)
                        return torch.max(loss1, loss2).mean()
                    else:
                        return F.mse_loss(pred, target)

                value_losses = vmap(compute_value_loss_single, (0, None, 0))(
                    adapted_value_params, curr_value_buffers, query_td
                )
                value_loss = value_losses.mean()

                total_loss = policy_loss + cfg.outer.value_coef * value_loss

                optimizer.zero_grad()
                total_loss.backward()

                params_to_clip = list(policy_model.parameters()) + list(
                    value_module.parameters()
                )
                if cfg.outer.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        params_to_clip,
                        cfg.outer.max_grad_norm,
                    )
                else:
                    total_norm = 0.0
                    for p in params_to_clip:
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    grad_norm = total_norm**0.5

                optimizer.step()

            avg_ratio = infos["ratio_mean"].mean().item()
            avg_entropy = infos["entropy"].mean().item()
            mean_task_policy_loss = infos["policy_loss"].mean().item()
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
            trpo_logs = {}

        # Common Metrics
        avg_support_reward = support_td.get(("next", "reward")).mean().item()
        avg_query_reward = query_td.get(("next", "reward")).mean().item()

        # Print
        log_str = (
            f"[iter {iteration}] "
            f"loss={total_loss if isinstance(total_loss, float) else total_loss.item():.3f} "
            f"rew_support={avg_support_reward:.3f} "
            f"rew_query={avg_query_reward:.3f}"
        )
        if cfg.algorithm == "trpo":
            log_str += f" kl={trpo_logs.get('trpo_kl', 0):.4f}"

        print(log_str)

        if wandb.run is not None:
            log_data = {
                "iteration": iteration,
                "reward/support_avg": avg_support_reward,
                "reward/query_avg": avg_query_reward,
                "optimizer/outer_lr": optimizer.param_groups[0]["lr"],
            }
            if cfg.algorithm == "ppo":
                log_data.update(
                    {
                        "loss/total": total_loss.item(),
                        "loss/policy": policy_loss.item(),
                        "loss/value": value_loss.item(),
                        "ppo/ratio_mean": avg_ratio,
                        "ppo/entropy_mean": avg_entropy,
                        "optimizer/grad_norm": grad_norm,
                    }
                )
            elif cfg.algorithm == "trpo":
                log_data.update(
                    {
                        "loss/trpo_surr": trpo_logs.get("trpo_surr", 0),
                        "loss/value": value_loss.item(),
                        "trpo/kl": trpo_logs.get("trpo_kl", 0),
                        "trpo/improvement": trpo_logs.get("trpo_improvement", 0),
                    }
                )

            wandb.log(log_data, step=iteration)

    # Save Model
    import os
    from datetime import datetime

    run_name = (
        cfg.wandb.name
        if cfg.wandb.name
        else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    save_dir = os.path.join("checkpoints", run_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "model.pt")
    torch.save(
        {
            "policy_state_dict": policy_model.state_dict(),
            "value_state_dict": value_module.state_dict(),
            "config": asdict(cfg),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )

    import json

    # Custom encoder to handle potential non-serializable objects if any (though asdict usually produces dicts)
    class ConfigEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)
            return super().default(obj)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=4, cls=ConfigEncoder)

    print(f"Model saved to {save_path}")

    if wandb.run is not None:
        wandb.finish()


def evaluate(
    cfg: TrainConfig,
    device: torch.device,
    checkpoint_path: str,
    pretrained_checkpoint_path: str = None,
):
    """
    Evaluate MAML on new tasks against baselines (Random Init, Oracle, Pretrained).
    """
    import json
    import os
    from maml_rl.policies import AnalyticalNavigationOracle, RandomPolicy

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

        # Step 0 (Pre-adaptation)
        policy_params, policy_buffers = params_and_buffers(policy_model)

        current_params_list = policy_params

        with torch.no_grad():
            step0_td = env.rollout(
                max_steps=cfg.rollout_steps,
                policy=FunctionalPolicy(
                    policy_model, current_params_list, policy_buffers
                ),
                break_when_any_done=False,
                auto_reset=True,
            )

        rew0 = step0_td.get(("next", "reward")).mean(dim=1).cpu().numpy()
        results[0] = (rew0.mean(), rew0.std())
        print(f"[{name}] Step 0: {rew0.mean():.3f} +/- {rew0.std():.3f}")

        # Adaptation Loop
        support_td = step0_td.to(torch.float32)
        support_td = add_gae(
            support_td, value_module, gamma=cfg.gamma, lam=cfg.lam, compute_grads=False
        )
        if cfg.inner.advantage_norm:
            support_td.set("advantage", normalize_advantages(support_td["advantage"]))

        def inner_step(params, buffers, data):
            return inner_update_vpg(policy_model, params, buffers, data, cfg.inner.lr)

        for step_k in range(1, cfg.inner.num_steps + 1):
            current_params_list = vmap(
                inner_step, in_dims=(None if step_k == 1 else 0, None, 0)
            )(current_params_list, policy_buffers, support_td)

            adapted_params = OrderedDict(
                (n, current_params_list[n]) for n in policy_params.keys()
            )

            with torch.no_grad():
                td_k = env.rollout(
                    max_steps=cfg.rollout_steps,
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
                max_steps=cfg.rollout_steps,
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


if __name__ == "__main__":
    main()
