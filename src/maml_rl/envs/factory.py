from typing import Any, Dict, List, Mapping, Tuple, Type

from torchrl.envs import EnvBase

from maml_rl.envs.base import MetaEnv
from maml_rl.envs.ant import MetaAntGoalVelEnv
from maml_rl.envs.navigation import Navigation2DEnv


# Registry of available meta-learning environments.
# To add a new environment, simply add it to this dict.
ENV_REGISTRY: Dict[str, Type[MetaEnv]] = {
    "navigation": Navigation2DEnv,
    "ant": MetaAntGoalVelEnv,
}


def sample_tasks(
    env_name: str,
    num_tasks: int,
    task_low: float,
    task_high: float,
) -> List[Mapping[str, Any]]:
    """Sample tasks for the specified environment."""
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment name: {env_name}")
    return ENV_REGISTRY[env_name].sample_tasks(num_tasks, low=task_low, high=task_high)


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
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment name: {env_name}")

    env_cls = ENV_REGISTRY[env_name]
    tasks = env_cls.sample_tasks(num_tasks, low=task_low, high=task_high)
    env = env_cls.make_vec_env(
        tasks,
        device=device,
        max_steps=max_steps,
        norm_obs=norm_obs,
    )
    env.set_seed(seed)
    return tasks, env
