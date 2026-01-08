from typing import Any, List, Mapping, Tuple, Type

from torchrl.envs import EnvBase

from maml_rl.envs.base import MetaEnv
# from maml_rl.envs.ant import MetaAntGoalVelEnv
# from maml_rl.envs.navigation import Navigation2DEnv
from maml_rl.envs.mars_lander import MarsLanderEnv

# Registry of available meta-learning environments.
# To add a new environment, add it to this dict.
ENV_REGISTRY = {
    "mars-lander": MarsLanderEnv,
}


def sample_tasks(
    env_name: str,
    num_tasks: int,
    task_low: float,
    task_high: float,
    difficulty: float = 0.0,
) -> List[Mapping[str, Any]]:
    """Sample tasks for the specified environment."""
    env_cls = _get_env_cls(env_name)
    # User Request: Fixed Start, Fixed Physics, Variable Landing
    return env_cls.sample_tasks(
        num_tasks, 
        low=task_low, 
        high=task_high, 
        difficulty=difficulty,
        randomize_physics=False,
        randomize_start=False,
        randomize_landing=True
    )


def make_vec_env(
    env_name: str,
    num_tasks: int,
    task_low: float,
    task_high: float,
    max_steps: int,
    device: str,
    norm_obs: bool,
    seed: int,
    difficulty: float = 0.0,
) -> Tuple[List[Mapping[str, Any]], EnvBase]:
    """Create a vectorized environment and sample tasks.

    Returns:
        tasks: List of task specifications
        env: The vectorized environment
    """
    env_cls = _get_env_cls(env_name)
    tasks = env_cls.sample_tasks(
        num_tasks, 
        low=task_low, 
        high=task_high, 
        difficulty=difficulty,
        randomize_physics=False,
        randomize_start=False,
        randomize_landing=True
    )
    env = env_cls.make_vec_env(
        tasks,
        device=device,
        max_steps=max_steps,
        norm_obs=norm_obs,
    )
    env.set_seed(seed)
    return tasks, env


def make_oracle_vec_env(
    env_name: str,
    tasks: List[Mapping[str, Any]],
    max_steps: int,
    device: str,
    norm_obs: bool,
    seed: int,
) -> EnvBase:
    """Create a vectorized environment with task info in observations.

    Used for training oracle policies that condition on task parameters.
    """
    env_cls = _get_env_cls(env_name)
    env = env_cls.make_oracle_vec_env(
        tasks,
        device=device,
        max_steps=max_steps,
        norm_obs=norm_obs,
    )
    env.set_seed(seed)
    return env


def get_task_obs_dim(env_name: str) -> int:
    """Get the dimension of task parameters appended to oracle observations."""
    env_cls = _get_env_cls(env_name)
    return env_cls.get_task_obs_dim()


def _get_env_cls(env_name: str) -> Type[MetaEnv]:
    """Get environment class by name."""
    if env_name not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment: {env_name}. Available: {list(ENV_REGISTRY.keys())}"
        )
    return ENV_REGISTRY[env_name]
