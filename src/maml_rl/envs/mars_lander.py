from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

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


class MarsLanderEnv(gym.Env):
    """
    Mars Lander Environment for MAML.
    Based on the CodinGame puzzle "Mars Lander".
    
    State: [x, y, h_speed, v_speed, fuel, rotate, power]
    Actions: [rotate, power] (Continuous for RL adaptability)
    """
    
    # Constants
    GRAVITY_MARS = 3.711
    MAP_WIDTH = 7000
    MAP_HEIGHT = 3000
    LANDING_TARGET_WIDTH = 1000
    
    def __init__(self, task: Dict[str, Any]|None = None):
        super().__init__()
        
        # Default task parameters
        self._task = task if task else {}
        self.gravity = self._task.get("gravity", self.GRAVITY_MARS)
        self.wind = np.array(self._task.get("wind", [0.0, 0.0]))
        self.target_x = self._task.get("target_x", self.MAP_WIDTH // 2)
        self.target_y = self._task.get("target_y", 100) # Slightly above ground for flat landing zone
        self.landing_width = self._task.get("landing_width", 1000.0)
        
        # Physics State
        self.state = None
        self.steps_beyond_done = None
        
        # Action Space: [rotate_change, power_change]
        # We normalize inputs to [-1, 1] for RL
        # Rotate change: -15 to +15 degrees
        # Power change: -1 to +1
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Observation Space (normalized)
        # [x/W, y/H, hs/100, vs/100, fuel/2000, rotate/90, power/4]
        # Also include target info? For now, we assume implicit knowledge or it's part of the state if we want variable targets to be visible.
        # But for MAML, the task is often hidden.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Apply task if provided in options (common pattern in MAML envs)
        if options and "task" in options:
            self.set_task(options["task"])
            
        # Initial State Generation
        # Use task start params if available
        start_y_task = self._task.get("start_y", None)
        
        start_x = self.np_random.uniform(2000, 5000)
        
        if start_y_task:
             start_y = start_y_task
        else:
             start_y = self.np_random.uniform(2000, 2800)
             
        # Use task start x if available
        start_x_task = self._task.get("start_x", None)
        if start_x_task:
            start_x = start_x_task
        else:
            start_x = self.np_random.uniform(2000, 5000)
            
        start_hs = self.np_random.uniform(-50, 50)
        start_vs = self.np_random.uniform(-50, 0) # Moving down
        start_fuel = self.np_random.uniform(500, 2000)
        start_rotate = self.np_random.uniform(-90, 90)
        start_power = 0.0
        
        self.state = np.array([
            start_x, start_y, start_hs, start_vs, 
            start_fuel, start_rotate, start_power
        ], dtype=np.float32)
        
        self.steps_beyond_done = None
        
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        x, y, hs, vs, fuel, rotate, power = self.state
        
        # Decode actions
        # Action is normalized [-1, 1]
        # rotate_change in [-15, 15]
        # power_change in [-1, 1]
        act_rotate = np.clip(action[0], -1.0, 1.0) * 15.0
        act_power = np.clip(action[1], -1.0, 1.0)
        
        # Update Control State
        # Rotate is clamped to [-90, 90]
        rotate = np.clip(rotate + act_rotate, -90.0, 90.0)
        
        # Power is clamped to [0, 4]
        # Note: CodinGame allows integer steps -1, 0, +1. Here we allow continuous change.
        # We model power change as direct delta for conceptual simplicity in optimization
        power = np.clip(power + act_power, 0.0, 4.0)
        
        # Fuel consumption
        fuel -= power
        if fuel < 0:
            fuel = 0
            power = 0 # Engine cuts off
            
        # Physics Update
        # Convert degrees to radians
        rad_rot = np.deg2rad(rotate)
        
        # Acceleration
        # Thrust vector
        # BOOSTED THRUST: Multiplier to ensure feasibility against high gravity
        # If max gravity is 9.0, we need max thrust > 9.0.
        # Original: Power [0,4] -> Thrust [0,4].
        # New: Power [0,4] * 3.0 -> Thrust [0, 12].
        THRUST_MULT = 3.0
        
        ax_thrust = -power * THRUST_MULT * np.sin(rad_rot)
        ay_thrust = power * THRUST_MULT * np.cos(rad_rot)
        
        # Total acceleration
        ax = ax_thrust + self.wind[0]
        ay = ay_thrust - self.gravity + self.wind[1]
        
        # Symplectic Euler (or similar integration often used in simple games)
        # Position update based on previous velocity + half accel (Verlet-like) or simple Euler.
        # CodinGame standard:
        # x += hs + 0.5 * ax
        # y += vs + 0.5 * ay
        # hs += ax
        # vs += ay
        
        x_new = x + hs + 0.5 * ax
        y_new = y + vs + 0.5 * ay
        hs_new = hs + ax
        vs_new = vs + ay
        
        # Update State
        self.state = np.array([
            x_new, y_new, hs_new, vs_new, fuel, rotate, power
        ], dtype=np.float32)
        
        # Reward & Termination Calculation
        terminated = False
        truncated = False
        reward = 0.0 # No explicit time penalty, rely on potential shaping + gamma

        # Helper for potential: Distance to target + Velocity penalty
        # We want to minimize distance and landing velocity
        def compute_potential(px, py, phs, pvs):
            dist_x = np.abs(px - self.target_x)
            dist_y = np.abs(py - self.target_y) # Distance to ground target
            dist = np.sqrt(dist_x**2 + dist_y**2)
            # Normalized potential
            # Return positive potential so that gamma*phi' - phi is negative for staying still (time penalty)
            # Max dist ~ 8000. 
            return 100.0 - (dist / 100.0) 

        # Previous potential
        phi_old = compute_potential(x, y, hs, vs)
        phi_new = compute_potential(x_new, y_new, hs_new, vs_new)

        # Shaping reward (Gamma should be close to 1, e.g. 0.999)
        shaping = 0.999 * phi_new - phi_old
        reward += shaping

            # Check Crash/Landing
        if y_new <= 0 or x_new < 0 or x_new > self.MAP_WIDTH:
            terminated = True
            
            # Distance to landing target center
            dist_x = np.abs(x_new - self.target_x)
            
            landing_conditions_met = (
                y_new <= 0 and # Touched ground
                dist_x < (self.landing_width / 2.0) and # Inside variable zone
                np.abs(vs_new) <= 40.0 and # Vertical speed limit
                np.abs(hs_new) <= 20.0 and # Horizontal speed limit
                np.abs(rotate) <= 10.0 # Angle limit (flat)
            )
            
            if landing_conditions_met:
                reward += 100.0 + fuel * 0.01 # Bonus for saving fuel (reduced multiplier)
            else:
                # Crash penalty
                reward -= 1000.0 
                # Shaping: penalized for high speed impact
                reward -= np.abs(vs_new) + np.abs(hs_new)
                reward -= dist_x * 0.1
        
        return self._get_obs(), reward, terminated, truncated, {}
        
    def _get_obs(self):
        # Normalize observation
        x, y, hs, vs, fuel, rotate, power = self.state
        
        # Features
        dist_target_x = x - self.target_x
        dist_ground = y
        
        obs = np.array([
            x / self.MAP_WIDTH,
            y / self.MAP_HEIGHT,
            hs / 100.0, 
            vs / 100.0,
            fuel / 2000.0,
            rotate / 90.0,
            power / 4.0,
            dist_target_x / self.MAP_WIDTH, # Normalized distance to target center
            dist_ground / self.MAP_HEIGHT # Normalized distance to ground
        ], dtype=np.float32)
        
        # If Oracle, append task parameters to observation
        if self._oracle:
            # Task: [gravity, wind_x, wind_y, target_x, landing_width]
            task_obs = np.array([
                self.gravity, 
                self.wind[0], 
                self.wind[1],
                self.target_x / self.MAP_WIDTH,
                self.landing_width / 2000.0 # Normalize width approx
            ], dtype=np.float32)
            return np.concatenate([obs, task_obs]).astype(np.float32)
            
        return obs

    def set_task(self, task: Dict[str, Any]):
        self._task = task
        self.gravity = task.get("gravity", self.gravity)
        self.wind = np.array(task.get("wind", self.wind))
        self.target_x = task.get("target_x", self.target_x)
        self.target_y = task.get("target_y", self.target_y)
        self.landing_width = task.get("landing_width", 1000.0)

    @staticmethod
    def sample_tasks(num_tasks: int, difficulty: float = 0.1, **kwargs) -> List[Dict[str, Any]]:
        """
        Generate a set of tasks with difficulty scaling.
        difficulty: 0.0 (Easy) to 1.0 (Hard)
        
        Easy: 
        - Start Altitude ~200m
        - Gravity ~3.711
        - Target directly under or very close to start x.
        
        Hard: 
        - Start Altitude ~3000m
        - Gravity varies
        - Target far from start x.
        """
        tasks = []
        for _ in range(num_tasks):
            # Scale parameters by difficulty
            
            # Gravity
            grav_center = 3.711
            grav_spread = 5.0 * difficulty
            gravity = np.clip(np.random.uniform(grav_center - grav_spread, grav_center + grav_spread), 1.6, 9.0)
            
            # Wind
            wind_scale = 10.0 * difficulty
            wind = [
                np.random.uniform(-wind_scale, wind_scale),
                np.random.uniform(-wind_scale * 0.5, wind_scale * 0.5)
            ]
            
            # Start Altitude
            min_alt = 200.0 + (1800.0 * difficulty)
            max_alt = 400.0 + (2600.0 * difficulty)
            start_y = np.random.uniform(min_alt, max_alt)
            
            # Start X (randomize to cover map)
            start_x = np.random.uniform(1000, 6000)
            
            # Target X placement relative to Start X based on Difficulty
            # Easy (0.0): Offset 0. 
            # Hard (1.0): Offset up to 3000m (half map width).
            max_offset = 500.0 + (2500.0 * difficulty)
            offset = np.random.uniform(-max_offset, max_offset)
            target_x = np.clip(start_x + offset, 500, 6500) # Keep within map bounds
            
            # Landing Width scaling
            # Easy: Wide (1000m - 2000m)
            # Hard: Narrow (200m - 500m)
            # Linearly interpolate
            min_width = 800.0 - (600.0 * difficulty)
            max_width = 2000.0 - (1000.0 * difficulty)
            landing_width = np.random.uniform(min_width, max_width)
            
            task = {
                "gravity": gravity,
                "wind": wind,
                "target_x": target_x,
                "landing_width": landing_width, # Variable width
                "start_y": start_y,
                "start_x": start_x, 
                "difficulty": difficulty
            }
            tasks.append(task)
        return tasks

    @staticmethod
    def get_task_obs_dim() -> int:
        """Oracle observations include gravity (1), wind (2), target_x (1), landing_width(1)."""
        return 5

    @staticmethod
    def make_vec_env(
        tasks: Sequence[Dict[str, Any]],
        device: str = "cpu",
        max_steps: int = 500,
        norm_obs: bool = True,
    ):
        """Create a parallel Mars Lander vector environment."""
        return _make_mars_parallel_env(tasks, device, max_steps, norm_obs, oracle=False)

    @staticmethod
    def make_oracle_vec_env(
        tasks: Sequence[Dict[str, Any]],
        device: str = "cpu",
        max_steps: int = 500,
        norm_obs: bool = True,
    ):
        """Create a parallel Mars Lander environment with task info in observations."""
        return _make_mars_parallel_env(tasks, device, max_steps, norm_obs, oracle=True)

    @staticmethod
    def get_oracle(
        tasks: Sequence[Dict[str, Any]],
        device: torch.device,
        checkpoint_path: Optional[str] = None,
    ) -> Optional[nn.Module]:
        """Load pretrained oracle policy from checkpoint."""
        if checkpoint_path is None:
            return None

        from maml_rl.policies import build_actor_critic

        # Oracle obs = standard obs + task info
        obs_dim = 9 + MarsLanderEnv.get_task_obs_dim()
        act_dim = 2

        checkpoint = torch.load(checkpoint_path, map_location=device)
        _, oracle_policy, _ = build_actor_critic(
            obs_dim,
            act_dim,
            hidden_sizes=checkpoint.get("hidden_sizes", (128, 128)),
        )
        oracle_policy.load_state_dict(checkpoint["policy_state_dict"])
        oracle_policy.to(device)
        return oracle_policy


class MetaMarsLanderOracleEnv(gym.Wrapper):
    """Wrapper that appends task info to observations for oracle training."""

    def __init__(self, env: MarsLanderEnv):
        super().__init__(env)
        base_obs = env.observation_space
        oracle_dim = base_obs.shape[0] + MarsLanderEnv.get_task_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(oracle_dim,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment_obs(obs), reward, terminated, truncated, info

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """Append task info (gravity, wind, target_x) to observation."""
        task = self.env._task
        # Default values if not set (should be set by reset options)
        gravity = task.get("gravity", self.env.GRAVITY_MARS)
        wind = task.get("wind", [0.0, 0.0])
        target_x = task.get("target_x", self.env.MAP_WIDTH // 2)
        
        task_info = np.array([gravity, wind[0], wind[1], target_x], dtype=np.float32)
        
        # Normalize task info to be roughly in similar range as obs if possible, 
        # or just append raw. Raw is usually fine for neural nets if not huge.
        # Wind is small, gravity ~ [1.6, 10], target_x ~ [2500, 4500].
        # Target X needs normalization!
        task_info[3] = task_info[3] / self.env.MAP_WIDTH
        
        return np.concatenate([obs, task_info])


def _make_mars_parallel_env(
    tasks: Sequence[Dict[str, Any]],
    device: str,
    max_steps: int,
    norm_obs: bool,
    oracle: bool,
):
    """Create parallel Mars Lander environment, optionally with oracle observations."""

    def make_single_env(task):
        base_env = MarsLanderEnv()
        base_env.set_task(task)

        if oracle:
            base_env = MetaMarsLanderOracleEnv(base_env)

        env = GymWrapper(base_env, device=device)
        env = TransformedEnv(
            env,
            Compose(
                InitTracker(),
                StepCounter(max_steps=max_steps),
            ),
        )
        return env

    env_fn_list = [lambda t=task: make_single_env(t) for task in tasks]
    # Use SerialEnv to save RAM (1 process) vs ParallelEnv (N processes)
    # Since we have 20 tasks, ParallelEnv is too heavy on RAM/CPU context switching.
    env = ParallelEnv(num_workers=len(tasks), create_env_fn=env_fn_list)

    if norm_obs:
        obs_norm = ObservationNorm(in_keys=["observation"], standard_normal=True)
        env = TransformedEnv(env, obs_norm)

    return env

