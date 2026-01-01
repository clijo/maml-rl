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
    Run MAML on Ant Goal Velocity with VPG inner loop and PPO outer loop.

    Example:
        python run.py --cfg configs/config.yaml
    Override any field via CLI, e.g.:
        python run.py --cfg configs/config.yaml --config.outer.lr 0.0001
    """
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile, help="Path to YAML config")
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
        # env is TransformedEnv, its transform is ObservationNorm (or Compose containing it)
        # Use a few rollouts to get initial stats.
        # Since it's a ParallelEnv, each rollout gives num_tasks * rollout_steps samples.
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
    actor.to(device)  # Shares parameters with policy_model and value_module backbone
    value_module.to(device)

    # if wandb.run is not None:
    #     wandb.watch(policy_model, log="all", log_freq=10)
    #     wandb.watch(value_module, log="all", log_freq=10)

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
        # Validate batch dimensions
        num_tasks = support_td.shape[0]
        if num_tasks != cfg.env.num_tasks:
            raise ValueError(
                f"Support data batch size {num_tasks} doesn't match "
                f"num_tasks {cfg.env.num_tasks}"
            )

        # Vectorized inner update: vmap over tasks (dimension 0)
        # policy_model is shared (None), params/buffers are shared (None),
        # support_td is batched (0), alpha is constant (None)
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

        # vmap returns a dict where each value is batched [num_tasks, ...]
        # We need to extract each parameter and stack them
        # Pass params and buffers explicitly to vmap so they are traced correctly
        adapted_params_list = vmap(inner_update_single, in_dims=(None, None, 0))(
            policy_params, policy_buffers, support_td
        )

        # vmap on OrderedDict returns a dict, convert back to OrderedDict
        # with batched tensors
        adapted_params = OrderedDict(
            (name, adapted_params_list[name]) for name in policy_params.keys()
        )

        # Adapt value function for each task
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

        adapted_value_params_list = vmap(
            inner_update_value_single, in_dims=(None, None, 0)
        )(value_params, value_buffers, support_td)
        adapted_value_params = OrderedDict(
            (name, adapted_value_params_list[name]) for name in value_params.keys()
        )

        # 3. Collect Query Data (using Adapted Parameters)
        # FunctionalPolicy now handles the batched adapted_params automatically via vmap
        query_policy = FunctionalPolicy(policy_model, adapted_params, policy_buffers)

        with torch.no_grad():
            query_td = env.rollout(
                max_steps=cfg.rollout_steps,
                policy=query_policy,
                break_when_any_done=False,
                auto_reset=True,
            )
            query_td = query_td.to(torch.float32)
        # query_td shape: [num_tasks, rollout_steps]

        # Compute GAE for query data using adapted value functions
        # We need to use functional calls with adapted value params
        def compute_adapted_values(params, buffers, td):
            # Compute for current step
            functional_call(value_module, (params, buffers), (td,))
            # Compute for next step
            functional_call(value_module, (params, buffers), (td.get("next"),))
            return td

        with torch.no_grad():
            query_td = vmap(compute_adapted_values, (0, None, 0))(
                adapted_value_params, value_buffers, query_td
            )

        query_td = add_gae(
            query_td,
            value_module=None,  # Use pre-computed values in td
            gamma=cfg.gamma,
            lam=cfg.lam,
            compute_grads=False,
        )

        # Normalize advantages if requested
        if cfg.outer.advantage_norm:
            query_td.set("advantage", normalize_advantages(query_td["advantage"]))

        # Validate query batch dimensions
        query_num_tasks = query_td.shape[0]
        if query_num_tasks != num_tasks:
            raise ValueError(
                f"Query data batch size {query_num_tasks} doesn't match "
                f"support batch size {num_tasks}"
            )

        # 4. Outer Loop Loss (Vectorized PPO)
        # We iterate multiple times over the same batch of data (PPO epochs)
        # For each epoch, we must re-adapt the parameters starting from the *current* meta-parameters
        # to properly compute gradients through the inner loop.

        for ppo_epoch in range(cfg.outer.ppo_epochs):
            # Re-extract current meta-parameters
            curr_policy_params, curr_policy_buffers = params_and_buffers(policy_model)
            curr_value_params, curr_value_buffers = params_and_buffers(value_module)

            # A. Re-Adapt Policy (vmap over tasks)
            adapted_params_list = vmap(
                inner_update_single, in_dims=(None, None, 0)
            )(curr_policy_params, curr_policy_buffers, support_td)

            adapted_params = OrderedDict(
                (name, adapted_params_list[name]) for name in curr_policy_params.keys()
            )

            # B. Re-Adapt Value Function (vmap over tasks)
            adapted_value_params_list = vmap(
                inner_update_value_single, in_dims=(None, None, 0)
            )(curr_value_params, curr_value_buffers, support_td)

            adapted_value_params = OrderedDict(
                (name, adapted_value_params_list[name])
                for name in curr_value_params.keys()
            )

            # C. Compute Policy Loss
            # vmap outer_loss_ppo over adapted_params (0) and query_td (0)
            losses, infos = vmap(outer_loss_ppo, (None, 0, None, 0, None, None))(
                policy_model,
                adapted_params,
                curr_policy_buffers,
                query_td,
                cfg.outer.clip_eps,
                cfg.outer.entropy_coef,
            )
            policy_loss = losses.mean()

            # D. Compute Value Loss
            # We need to predict values using the *current* adapted value params on the *query* data
            def compute_value_loss_single(params, buffers, td):
                td_out = functional_call(value_module, (params, buffers), (td,))
                pred = td_out.get("state_value")
                target = td.get("value_target")
                
                if cfg.outer.clip_value_loss:
                    # We need old_value from the original collection for clipping
                    # Assuming 'state_value' in query_td is the "old" value from collection
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

        # Metrics from the LAST epoch
        avg_support_reward = support_td.get(("next", "reward")).mean().item()
        std_support_reward = support_td.get(("next", "reward")).std().item()

        avg_query_reward = query_td.get(("next", "reward")).mean().item()
        std_query_reward = query_td.get(("next", "reward")).std().item()

        avg_ratio = infos["ratio_mean"].mean().item()
        avg_entropy = infos["entropy"].mean().item()
        mean_task_policy_loss = infos["policy_loss"].mean().item()

        # Ensure grad_norm is a float
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()

        print(
            f"[iter {iteration}] "
            f"loss={total_loss.item():.3f} "
            f"grad_norm={grad_norm:.3f} "
            f"rew_support={avg_support_reward:.3f} "
            f"rew_query={avg_query_reward:.3f}"
        )
        if wandb.run is not None:
            log_data = {
                "iteration": iteration,
                "loss/total": total_loss.item(),
                "loss/policy": policy_loss.item(),
                "loss/value": value_loss.item(),
                "ppo/ratio_mean": avg_ratio,
                "ppo/entropy_mean": avg_entropy,
                "ppo/policy_loss_mean": mean_task_policy_loss,
                "reward/support_avg": avg_support_reward,
                "reward/support_std": std_support_reward,
                "reward/query_avg": avg_query_reward,
                "reward/query_std": std_query_reward,
                "optimizer/outer_lr": optimizer.param_groups[0]["lr"],
                "optimizer/grad_norm": grad_norm,
            }
            wandb.log(log_data, step=iteration)

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
