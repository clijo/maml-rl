from collections.abc import Sequence
from typing import Any, List, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import (
    Compose,
    InitTracker,
    StepCounter,
    ObservationNorm,
    CatTensors,
)

from maml_rl.envs.navigation_vectorized import Navigation2DVectorized


class Navigation2DEnv(gym.Env):
    """2D Navigation environment as described in the MAML paper.

    A point agent must move to different goal positions in 2D,
    randomly chosen for each task within a unit square.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
        )
        # Actions in [-1, 1], scaled to [-0.1, 0.1] in step()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)

        self._goal = np.array([0.5, 0.5], dtype=np.float64)
        self._state = np.zeros(2, dtype=np.float64)

    def set_task(self, task: Mapping[str, Any]) -> None:
        self._goal = np.array(task["goal"], dtype=np.float64)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Mapping] = None
    ) -> Tuple[np.ndarray, Mapping]:
        super().reset(seed=seed)
        if options is not None and "task" in options:
            self.set_task(options["task"])
        self._state = np.zeros(2, dtype=np.float64)
        return self._state.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Mapping]:
        action = np.clip(action, -1.0, 1.0).astype(np.float64) * 0.1
        self._state = self._state + action
        dist = np.linalg.norm(self._state - self._goal)
        reward = -(dist**2)
        terminated = bool(dist < 0.01)
        return self._state.copy(), float(reward), terminated, False, {"error": dist}

    @staticmethod
    def sample_tasks(
        num_tasks: int, low: float = -0.5, high: float = 0.5
    ) -> List[Mapping[str, np.ndarray]]:
        """Sample goal positions within a square."""
        return Navigation2DVectorized.sample_tasks(num_tasks, low, high)

    @staticmethod
    def get_task_obs_dim() -> int:
        """Oracle observations include 2D goal position."""
        return 2

    @staticmethod
    def make_vec_env(
        tasks: Sequence[Mapping[str, np.ndarray]],
        device: str = "cpu",
        max_steps: int = 100,
        norm_obs: bool = False,
    ):
        """Create a vectorized Navigation environment."""
        return _make_nav_vectorized_env(
            tasks, device, max_steps, norm_obs, oracle=False
        )

    @staticmethod
    def make_oracle_vec_env(
        tasks: Sequence[Mapping[str, np.ndarray]],
        device: str = "cpu",
        max_steps: int = 100,
        norm_obs: bool = False,
    ):
        """Create a vectorized Navigation environment with oracle observations."""
        return _make_nav_vectorized_env(tasks, device, max_steps, norm_obs, oracle=True)

    @staticmethod
    def get_oracle(
        tasks: Sequence[Mapping[str, np.ndarray]],
        device: torch.device,
        checkpoint_path: Optional[str] = None,
    ) -> Optional[nn.Module]:
        """Load pretrained oracle policy from checkpoint."""
        if checkpoint_path is None:
            return None

        from maml_rl.policies import build_actor_critic

        # Oracle obs = position (2) + goal (2)
        oracle_obs_dim = 2 + Navigation2DEnv.get_task_obs_dim()
        act_dim = 2

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        # Get hidden_sizes from checkpoint config, fall back to direct key, then default
        if "config" in checkpoint and "model" in checkpoint["config"]:
            hidden_sizes = checkpoint["config"]["model"].get("hidden_sizes", (100, 100))
        else:
            hidden_sizes = checkpoint.get("hidden_sizes", (100, 100))
        oracle_actor, oracle_policy_model, _ = build_actor_critic(
            oracle_obs_dim, act_dim, hidden_sizes=hidden_sizes
        )
        oracle_policy_model.load_state_dict(checkpoint["policy_state_dict"])
        oracle_actor.to(device)
        return oracle_actor


class Navigation2DOracleWrapper(gym.Wrapper):
    """Wrapper that appends goal position to observations for oracle training."""

    def __init__(self, env: Navigation2DEnv):
        super().__init__(env)
        # Observation = [position, goal] = 4D
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment_obs(obs), reward, terminated, truncated, info

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """Append goal position to observation."""
        return np.concatenate([obs, self.env._goal])


def _make_nav_vectorized_env(
    tasks: Sequence[Mapping[str, np.ndarray]],
    device: str,
    max_steps: int,
    norm_obs: bool,
    oracle: bool,
):
    """Create vectorized Navigation environment."""
    num_tasks = len(tasks)

    # Base vectorized env
    env = Navigation2DVectorized(num_tasks=num_tasks, device=device)

    # Set the specific tasks (goals)
    # The vectorized env expects list of dicts, same format
    env.set_task(tasks)

    # Transforms
    transforms = [
        InitTracker(),
        StepCounter(max_steps=max_steps),
    ]

    if oracle:
        # We need to append goal to observation.
        # Navigation2DVectorized returns "observation" and "goals".
        # We use CatTensors to concat them into "observation".
        transforms.append(
            CatTensors(
                in_keys=["observation", "goals"],
                out_key="observation",
                dim=-1,
                del_keys=False,
            )
        )

    # Compose transforms
    env = TransformedEnv(env, Compose(*transforms))

    if norm_obs:
        # Note: Vectorized envs need standard_normal=True usually
        obs_norm = ObservationNorm(in_keys=["observation"], standard_normal=True)
        env = TransformedEnv(env, obs_norm)
        # Initialize stats if needed? Usually training loop does it.

    return env
