from typing import Any, List, Mapping, Optional, Protocol, runtime_checkable

import torch
from torch import nn
from torchrl.envs import EnvBase


@runtime_checkable
class MetaEnv(Protocol):
    """Protocol for meta-learning environments.

    Any environment used for MAML training must implement:
    - set_task: Configure the environment for a specific task
    - sample_tasks: Generate a list of task specifications
    - make_vec_env: Create a vectorized environment for given tasks
    - get_task_obs_dim: Return dimension of task info for oracle observations
    - make_oracle_vec_env: Create env with task info in observations (for oracle)
    """

    def set_task(self, task: Mapping[str, Any]) -> None:
        """Set the task for this environment."""
        ...

    @staticmethod
    def sample_tasks(
        num_tasks: int, low: float, high: float
    ) -> List[Mapping[str, Any]]:
        """Sample task specifications for this environment."""
        ...

    @staticmethod
    def make_vec_env(
        tasks: List[Mapping[str, Any]],
        device: str,
        max_steps: int,
        norm_obs: bool,
    ) -> EnvBase:
        """Create a vectorized environment for the given tasks."""
        ...

    @staticmethod
    def get_task_obs_dim() -> int:
        """Return the dimension of task parameters appended to oracle observations.

        This is used to create the extended observation space for oracle policies.
        For example:
        - 2D Navigation: 2 (goal x, y)
        - Ant Velocity: 1 (goal velocity)
        - Ant Direction: 1 (goal direction)
        """
        ...

    @staticmethod
    def make_oracle_vec_env(
        tasks: List[Mapping[str, Any]],
        device: str,
        max_steps: int,
        norm_obs: bool,
    ) -> EnvBase:
        """Create a vectorized environment with task info in observations.

        Oracle environments append task parameters to observations,
        allowing a policy to condition directly on the task.
        """
        ...

    @staticmethod
    def get_oracle(
        tasks: List[Mapping[str, Any]],
        device: torch.device,
        checkpoint_path: Optional[str] = None,
    ) -> Optional[nn.Module]:
        """Return an oracle policy for these tasks.

        Args:
            tasks: List of task specifications
            device: Device to load the policy on
            checkpoint_path: Path to pretrained oracle checkpoint

        Returns:
            A policy module trained on oracle observations, or None if unavailable.
        """
        return None
