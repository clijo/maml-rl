from typing import Any, Mapping, Tuple, List
import torch
from torchrl.envs import EnvBase

from maml_rl.envs.ant import make_ant_vec_env, sample_ant_tasks
from maml_rl.envs.navigation import make_navigation_vec_env, sample_navigation_tasks


def sample_tasks(
    env_name: str,
    num_tasks: int,
    task_low: float,
    task_high: float,
) -> List[Mapping[str, Any]]:
    """Sample tasks for the specified environment."""
    if env_name == "ant":
        return sample_ant_tasks(num_tasks, low=task_low, high=task_high)
    elif env_name == "navigation":
        return sample_navigation_tasks(num_tasks, low=task_low, high=task_high)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")


def make_vec_env(
    env_name: str,
    num_tasks: int,
    task_low: float,
    task_high: float,
    max_steps: int,
    device: str,
    norm_obs: bool,
    seed: int,
) -> Tuple[List[Mapping[str, Any]], EnvBase]:
    """
    Factory function to create a vectorized environment and sample tasks.

    Returns:
        tasks: List of task specifications
        env: The vectorized environment (ParallelEnv wrapped in transforms)
    """
    if env_name == "ant":
        tasks = sample_ant_tasks(num_tasks, low=task_low, high=task_high)
        env = make_ant_vec_env(
            tasks,
            device=device,
            max_steps=max_steps,
            norm_obs=norm_obs,
        )
    elif env_name == "navigation":
        tasks = sample_navigation_tasks(num_tasks, low=task_low, high=task_high)
        env = make_navigation_vec_env(
            tasks,
            device=device,
            max_steps=max_steps,
            norm_obs=norm_obs,
        )
    else:
        raise ValueError(f"Unknown environment name: {env_name}")

    env.set_seed(seed)
    return tasks, env
