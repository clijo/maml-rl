"""MAML training loop."""

import json
import os
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime

import torch
import torch.nn.functional as F
import wandb
from torch.func import functional_call, vmap
from torch.optim import Adam

from configs.base import TrainConfig
from maml_rl.envs.factory import make_vec_env, make_oracle_vec_env, sample_tasks
from maml_rl.maml import (
    FunctionalPolicy,
    inner_update_value,
    inner_update_vpg,
    outer_loss_ppo,
    outer_step_trpo,
)
from maml_rl.policies import build_actor_critic, params_and_buffers
from maml_rl.utils.returns import add_gae, normalize_advantages


def _expand_params_for_tasks(params: OrderedDict, num_tasks: int) -> OrderedDict:
    """Expand params to match vmap output shape [num_tasks, ...] for num_steps=0 case."""
    return OrderedDict(
        (name, p.unsqueeze(0).expand(num_tasks, *p.shape)) for name, p in params.items()
    )


def train(cfg: TrainConfig, device: torch.device):
    """
    Run MAML training loop.
    """
    # Sample tasks first
    tasks = sample_tasks(
        env_name=cfg.env.name,
        num_tasks=cfg.env.num_tasks,
        task_low=cfg.env.task_low,
        task_high=cfg.env.task_high,
    )

    # Create environment (oracle or standard)
    if cfg.oracle:
        print("Oracle mode: using environment with task params in observation")
        env = make_oracle_vec_env(
            env_name=cfg.env.name,
            tasks=tasks,
            max_steps=cfg.env.max_steps,
            device=str(device),
            norm_obs=cfg.env.norm_obs,
            seed=cfg.seed,
        )
    else:
        _, env = make_vec_env(
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

    # Best model tracking
    best_query_reward = float("-inf")
    best_policy_state = None
    best_value_state = None
    best_iteration = 0

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
                # Re-adapt value params (skip vmap when num_steps=0)
                if cfg.inner.num_steps > 0:
                    adapted_val_list = vmap(
                        inner_update_value_single, in_dims=(None, None, 0)
                    )(curr_value_params, curr_value_buffers, support_td)

                    adapted_val = OrderedDict(
                        (n, adapted_val_list[n]) for n in curr_value_params.keys()
                    )
                else:
                    adapted_val = _expand_params_for_tasks(curr_value_params, num_tasks)

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

                # Skip vmap when num_steps=0 (no inner adaptation needed)
                if cfg.inner.num_steps > 0:
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
                else:
                    adapted_params = _expand_params_for_tasks(
                        curr_policy_params, num_tasks
                    )
                    adapted_value_params = _expand_params_for_tasks(
                        curr_value_params, num_tasks
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

            # Additional PPO logging
            infos["mean_task_policy_loss"] = mean_task_policy_loss

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

        # Track best model
        if avg_query_reward > best_query_reward:
            best_query_reward = avg_query_reward
            best_policy_state = {
                k: v.clone() for k, v in policy_model.state_dict().items()
            }
            best_value_state = {
                k: v.clone() for k, v in value_module.state_dict().items()
            }
            best_iteration = iteration

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
                        "trpo/expected_improvement": trpo_logs.get(
                            "trpo_expected_improvement", 0
                        ),
                    }
                )

            wandb.log(log_data, step=iteration)

    # Save Best Model
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
            "policy_state_dict": best_policy_state,
            "value_state_dict": best_value_state,
            "config": asdict(cfg),
            "best_query_reward": best_query_reward,
            "best_iteration": best_iteration,
        },
        save_path,
    )

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=4)

    print(
        f"Best model (iter {best_iteration}, reward {best_query_reward:.3f}) saved to {save_path}"
    )

    if wandb.run is not None:
        wandb.finish()
