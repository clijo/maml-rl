from collections.abc import Sequence
from typing import List, Mapping, Optional, Tuple

import numpy as np
from torchrl.envs import GymWrapper, ParallelEnv, TransformedEnv
from torchrl.envs.transforms import (
    Compose,
    InitTracker,
    StepCounter,
    DoubleToFloat,
    ObservationNorm,
)
from gymnasium.envs.mujoco.ant_v4 import AntEnv  # gymnasium >=0.28


class MetaAntGoalVelEnv(AntEnv):
    """
    Ant with a target forward velocity.

    Reward: -|v_actual - v_goal| + ctrl_cost (ctrl_cost is negative in Gym)
    """

    def __init__(self, terminate_when_unhealthy: bool = False, **kwargs):
        super().__init__(terminate_when_unhealthy=terminate_when_unhealthy, **kwargs)
        self._goal_vel = 0.0

    def set_task(self, task: Mapping[str, float]) -> None:
        self._goal_vel = float(task["velocity"])

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Mapping] = None
    ) -> Tuple[np.ndarray, Mapping]:
        if options is not None and "task" in options:
            self.set_task(options["task"])
        return super().reset(seed=seed, options=options)

    def sample_tasks(self, num_tasks: int) -> List[Mapping[str, float]]:
        velocities = np.random.uniform(0.0, 3.0, (num_tasks,))
        return [{"velocity": float(v)} for v in velocities]

    def step(self, action):
        observation, _, terminated, truncated, info = super().step(action)
        x_velocity = info["x_velocity"]
        forward_reward = -1.0 * np.abs(x_velocity - self._goal_vel)
        ctrl_cost = info["reward_ctrl"]
        # Include healthy reward to encourage survival
        healthy_reward = info.get("reward_survive", 1.0)
        reward = forward_reward + ctrl_cost + healthy_reward
        return observation, reward, terminated, truncated, info


def sample_ant_tasks(
    num_tasks: int, low: float = 0.0, high: float = 3.0
) -> List[Mapping[str, float]]:
    """Sample target velocities for Ant tasks."""
    velocities = np.random.uniform(low, high, size=(num_tasks,))
    return [{"velocity": float(v)} for v in velocities]


def make_ant_env(
    task: Mapping[str, float],
    device: str = "cpu",
    render_mode: Optional[str] = None,
    max_steps: int = 200,
):
    """Instantiate a single-task Ant env wrapped for TorchRL."""
    base_env = MetaAntGoalVelEnv(render_mode=render_mode)
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


def make_ant_vec_env(
    tasks: Sequence[Mapping[str, float]],
    device: str = "cpu",
    max_steps: int = 200,
    norm_obs: bool = True,
):
    """Create a parallel Ant GoalVel vector environment with fixed tasks."""
    env_fn_list = [
        lambda t=task: make_ant_env(t, device=device, max_steps=max_steps)
        for task in tasks
    ]
    env = ParallelEnv(num_workers=len(tasks), create_env_fn=env_fn_list)
    if norm_obs:
        obs_norm = ObservationNorm(in_keys=["observation"], standard_normal=True)
        env = TransformedEnv(env, obs_norm)
    return env
