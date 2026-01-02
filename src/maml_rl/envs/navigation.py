from collections.abc import Sequence
from typing import List, Mapping, Optional, Tuple, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from torchrl.envs import GymWrapper, ParallelEnv, TransformedEnv
from torchrl.envs.transforms import (
    Compose,
    InitTracker,
    StepCounter,
    DoubleToFloat,
    ObservationNorm,
)


class Navigation2DEnv(gym.Env):
    """
    2D Navigation environment as described in the MAML paper.

    A point agent must move to different goal positions in 2D, randomly chosen
    for each task within a unit square.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        # Observation is the current 2D position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
        )

        # Actions correspond to velocity commands clipped to be in range [-0.1, 0.1]
        # We define the action space as [-1, 1] to match TanhNormal policy output,
        # and scale it inside step().
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)

        self._task = {"goal": np.array([0.5, 0.5], dtype=np.float64)}
        self._goal = self._task["goal"]
        self._state = np.zeros(2, dtype=np.float64)
        self._max_episode_steps = 100  # Horizon H = 100

    def set_task(self, task: Mapping[str, Any]) -> None:
        self._task = task
        self._goal = np.array(task["goal"], dtype=np.float64)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Mapping] = None
    ) -> Tuple[np.ndarray, Mapping]:
        super().reset(seed=seed)
        if options is not None and "task" in options:
            self.set_task(options["task"])

        # Agent typically starts at (0, 0) in this benchmark
        self._state = np.zeros(2, dtype=np.float64)

        return self._state.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Mapping]:
        # Action is in [-1, 1], scale to [-0.1, 0.1]
        action = np.clip(action, -1.0, 1.0).astype(np.float64) * 0.1

        self._state = self._state + action

        dist = np.linalg.norm(self._state - self._goal)

        # Reward is the negative squared distance to the goal
        reward = -(dist**2)

        # Terminate when within 0.01 of the goal
        terminated = bool(dist < 0.01)
        truncated = False  # Handled by StepCounter wrapper usually, but defining here strictly per env logic is fine.

        return self._state.copy(), float(reward), terminated, truncated, {"error": dist}


def sample_navigation_tasks(
    num_tasks: int, low: float = -0.5, high: float = 0.5
) -> List[Mapping[str, np.ndarray]]:
    """
    Sample tasks for the 2D Navigation environment.
    Goals are randomly chosen within a unit square.

    The paper says "within a unit square".
    Usually interpreted as [-0.5, 0.5] x [-0.5, 0.5] or [0, 1] x [0, 1].
    Given the agent starts at (0,0), [-0.5, 0.5] creates goals in all directions.
    """
    # Using uniform distribution for the unit square
    goals = np.random.uniform(low, high, size=(num_tasks, 2))
    return [{"goal": goal.astype(np.float64)} for goal in goals]


def make_navigation_env(
    task: Mapping[str, np.ndarray],
    device: str = "cpu",
    max_steps: int = 100,
):
    """Instantiate a single-task Navigation env wrapped for TorchRL."""
    base_env = Navigation2DEnv()
    base_env.set_task(task)
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


def make_navigation_vec_env(
    tasks: Sequence[Mapping[str, np.ndarray]],
    device: str = "cpu",
    max_steps: int = 100,
    norm_obs: bool = False,
):
    """Create a parallel Navigation vector environment with fixed tasks."""
    env_fn_list = [
        lambda t=task: make_navigation_env(t, device=device, max_steps=max_steps)
        for task in tasks
    ]
    env = ParallelEnv(num_workers=len(tasks), create_env_fn=env_fn_list)

    if norm_obs:
        obs_norm = ObservationNorm(in_keys=["observation"], standard_normal=True)
        env = TransformedEnv(env, obs_norm)

    return env
