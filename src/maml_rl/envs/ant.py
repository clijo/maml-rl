from collections.abc import Sequence
from typing import Any, List, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from torch import nn
from torchrl.envs import GymWrapper, ParallelEnv, TransformedEnv
from torchrl.envs.transforms import (
    Compose,
    InitTracker,
    StepCounter,
    DoubleToFloat,
    ObservationNorm,
)


class MetaAntGoalVelEnv(AntEnv):
    """Ant with a target forward velocity.

    Reward: -|v_actual - v_goal| + ctrl_cost + healthy_reward
    """

    def __init__(self, terminate_when_unhealthy: bool = False, **kwargs):
        super().__init__(terminate_when_unhealthy=terminate_when_unhealthy, **kwargs)
        self._goal_vel = 0.0

    def set_task(self, task: Mapping[str, Any]) -> None:
        self._goal_vel = float(task["velocity"])

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Mapping] = None
    ) -> Tuple[np.ndarray, Mapping]:
        if options is not None and "task" in options:
            self.set_task(options["task"])
        return super().reset(seed=seed, options=options)

    def step(self, action):
        observation, _, terminated, truncated, info = super().step(action)
        x_velocity = info["x_velocity"]
        forward_reward = -1.0 * np.abs(x_velocity - self._goal_vel)
        ctrl_cost = info["reward_ctrl"]
        healthy_reward = info.get("reward_survive", 1.0)
        reward = forward_reward + ctrl_cost + healthy_reward
        return observation, reward, terminated, truncated, info

    @staticmethod
    def sample_tasks(
        num_tasks: int, low: float = 0.0, high: float = 3.0
    ) -> List[Mapping[str, float]]:
        """Sample target velocities for Ant tasks."""
        velocities = np.random.uniform(low, high, size=(num_tasks,))
        return [{"velocity": float(v)} for v in velocities]

    @staticmethod
    def get_task_obs_dim() -> int:
        """Oracle observations include 1D goal velocity."""
        return 1

    @staticmethod
    def make_vec_env(
        tasks: Sequence[Mapping[str, float]],
        device: str = "cpu",
        max_steps: int = 200,
        norm_obs: bool = True,
    ):
        """Create a parallel Ant GoalVel vector environment."""
        return _make_ant_parallel_env(tasks, device, max_steps, norm_obs, oracle=False)

    @staticmethod
    def make_oracle_vec_env(
        tasks: Sequence[Mapping[str, float]],
        device: str = "cpu",
        max_steps: int = 200,
        norm_obs: bool = True,
    ):
        """Create a parallel Ant environment with goal velocity in observations."""
        return _make_ant_parallel_env(tasks, device, max_steps, norm_obs, oracle=True)

    @staticmethod
    def get_oracle(
        tasks: Sequence[Mapping[str, float]],
        device: torch.device,
        checkpoint_path: Optional[str] = None,
    ) -> Optional[nn.Module]:
        """Load pretrained oracle policy from checkpoint."""
        if checkpoint_path is None:
            return None

        from maml_rl.policies import build_actor_critic

        # Oracle obs = standard obs + goal_vel
        # Standard Ant observation is 27-dim (from AntEnv)
        oracle_obs_dim = 27 + MetaAntGoalVelEnv.get_task_obs_dim()
        act_dim = 8  # Ant has 8-dimensional action

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        # Get hidden_sizes from checkpoint config, fall back to direct key, then default
        if "config" in checkpoint and "model" in checkpoint["config"]:
            hidden_sizes = checkpoint["config"]["model"].get("hidden_sizes", (100, 100))
        else:
            hidden_sizes = checkpoint.get("hidden_sizes", (100, 100))
        oracle_actor, oracle_policy_model, _ = build_actor_critic(
            oracle_obs_dim,
            act_dim,
            hidden_sizes=hidden_sizes,
        )
        oracle_policy_model.load_state_dict(checkpoint["policy_state_dict"])
        oracle_actor.to(device)
        return oracle_actor


class MetaAntGoalVelOracleEnv(gym.Wrapper):
    """Wrapper that appends goal velocity to observations for oracle training."""

    def __init__(self, env: MetaAntGoalVelEnv):
        super().__init__(env)
        base_obs = env.observation_space
        oracle_dim = base_obs.shape[0] + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(oracle_dim,), dtype=np.float64
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment_obs(obs), reward, terminated, truncated, info

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """Append goal velocity to observation."""
        return np.append(obs, self.env._goal_vel)


def _make_ant_parallel_env(
    tasks: Sequence[Mapping[str, float]],
    device: str,
    max_steps: int,
    norm_obs: bool,
    oracle: bool,
):
    """Create parallel Ant environment, optionally with oracle observations."""

    def make_single_env(task):
        base_env = MetaAntGoalVelEnv()
        base_env.set_task(task)

        if oracle:
            base_env = MetaAntGoalVelOracleEnv(base_env)

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
