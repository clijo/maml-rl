from typing import Any, List, Mapping, Protocol, runtime_checkable

from torchrl.envs import EnvBase


@runtime_checkable
class MetaEnv(Protocol):
    """Protocol for meta-learning environments.

    Any environment used for MAML training must implement:
    - set_task: Configure the environment for a specific task
    - sample_tasks: Generate a list of task specifications (static method)
    - make_vec_env: Create a vectorized environment for given tasks (static method)
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
