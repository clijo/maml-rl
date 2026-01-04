from collections.abc import Sequence
from typing import Any, List, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torchrl.envs import GymWrapper, ParallelEnv, TransformedEnv
from torchrl.envs.transforms import (
    Compose,
    InitTracker,
    StepCounter,
    DoubleToFloat,
    ObservationNorm,
)


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
        goals = np.random.uniform(low, high, size=(num_tasks, 2))
        return [{"goal": goal.astype(np.float64)} for goal in goals]

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
        """Create a parallel Navigation vector environment."""
        return _make_nav_parallel_env(tasks, device, max_steps, norm_obs, oracle=False)

    @staticmethod
    def make_oracle_vec_env(
        tasks: Sequence[Mapping[str, np.ndarray]],
        device: str = "cpu",
        max_steps: int = 100,
        norm_obs: bool = False,
    ):
        """Create a parallel Navigation environment with goal in observations."""
        return _make_nav_parallel_env(tasks, device, max_steps, norm_obs, oracle=True)

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
        _, oracle_policy, _ = build_actor_critic(
            oracle_obs_dim, act_dim, hidden_sizes=hidden_sizes
        )
        oracle_policy.load_state_dict(checkpoint["policy_state_dict"])
        oracle_policy.to(device)
        return oracle_policy


class Navigation2DOracleEnv(gym.Wrapper):
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


def _make_nav_parallel_env(
    tasks: Sequence[Mapping[str, np.ndarray]],
    device: str,
    max_steps: int,
    norm_obs: bool,
    oracle: bool,
):
    """Create parallel Navigation environment, optionally with oracle observations."""

    def make_single_env(task):
        base_env = Navigation2DEnv()
        base_env.set_task(task)

        if oracle:
            base_env = Navigation2DOracleEnv(base_env)

        env = GymWrapper(base_env, device=device)
        env = TransformedEnv(
            env,
            Compose(
                InitTracker(),
                StepCounter(max_steps=max_steps),
                DoubleToFloat(in_keys=["observation"]),
            ),
        )
        return env

    env_fn_list = [lambda t=task: make_single_env(t) for task in tasks]
    env = ParallelEnv(num_workers=len(tasks), create_env_fn=env_fn_list)

    if norm_obs:
        obs_norm = ObservationNorm(in_keys=["observation"], standard_normal=True)
        env = TransformedEnv(env, obs_norm)

    return env
