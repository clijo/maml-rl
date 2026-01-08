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
        
        # Load C++ Module
        try:
            # Check if we can import directly (if in path) or via package
            try:
                import mars_lander_cpp
            except ImportError:
                from maml_rl.envs import mars_lander_cpp
        except ImportError:
             raise ImportError("Could not import mars_lander_cpp. Did you compile the extension?")
            
        self._cpp_env = mars_lander_cpp.MarsLander()
        
        # Default task parameters
        self._task = task if task else {}
        self.set_task(self._task)
        
        # Action Space: [rotate_change, power_change]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Observation Space
        # [x, y, hs, vs, rotate, power]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
    
    def set_task(self, task: Dict[str, Any]):
        self._task = task.copy()
        self.gravity = self._task.get("gravity", self.GRAVITY_MARS)
        self.target_x = self._task.get("target_x", self.MAP_WIDTH // 2)
        self.target_y = self._task.get("target_y", 100.0)
        self.landing_width = self._task.get("landing_width", self.LANDING_TARGET_WIDTH)

    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        if options and "task" in options:
            self.set_task(options["task"])
            
        # Call C++ Reset
        task_float = {k: float(v) for k, v in self._task.items() if isinstance(v, (int, float))}
        if "wind" in self._task:
            task_float["wind_power"] = float(self._task["wind"][0]) 
            
        obs = self._cpp_env.reset(seed if seed else 0, task_float)
        
        # Initialize Shaping
        self.prev_potential = self._compute_potential(obs[0] * self.MAP_WIDTH, obs[1] * self.MAP_HEIGHT)
        
        return np.array(obs, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Call C++ Step (pure physics + sparse reward)
        act_list = action.tolist()
        obs, cpp_reward, done, truncated, info = self._cpp_env.step(act_list)
        obs = np.array(obs, dtype=np.float32)
        
        # --- REWARD SHAPING ---
        reward = 0.0
        
        # Unpack State
        # Obs: [x, y, hs, vs, rotate, power]
        x = obs[0] * self.MAP_WIDTH
        y = obs[1] * self.MAP_HEIGHT
        hs = obs[2] * 100.0
        vs = obs[3] * 100.0
        # Fuel removed from obs, but we can track it internally if needed or ignore it.
        # fuel = ... (not in obs anymore)
        power = obs[5] * 4.0
        
        # Current Distances
        dist_x = np.abs(x - self.target_x)
        dist_y = np.abs(y - self.target_y) 
        
        # POTENTIAL SHAPING
        # Guide the lander towards [target_x, target_y] with [0,0] velocity
        # Potential Phi = - (alpha * dist + beta * velocity)
        # We maximize potential (get closer to 0).
        # Reward = Phi_new - Phi_old
        
        dist_norm = np.sqrt(dist_x**2 + dist_y**2) / 3000.0 # Normalize roughly to 0-3
        vel_norm = np.sqrt(hs**2 + vs**2) / 100.0 # Normalize roughly to 0-2 (100m/s = 1)
        
        potential = - (1.0 * dist_norm + 0.1 * vel_norm)
        
        if hasattr(self, 'prev_potential'):
            reward += 10.0 * (potential - self.prev_potential)
            
        self.prev_potential = potential
        
        # 1. Step Cost (Fuel) - REMOVED per user request/optimization
        # reward -= 0.0 power # Free to use fuel to learn
        
        # 2. Terminal Rewards
        if done:
            if cpp_reward > 0.5:
                # SUCCESS (Landed)
                reward += 1000.0 # Huge bonus found!
                # No Fuel Bonus requested
            elif cpp_reward < 0.5:
                # CRASH
                reward -= 100.0
                
                # Crash Velocity Penalty (still important)
                reward -= 1.0 * np.abs(hs)
                reward -= 1.0 * np.abs(vs)
                
                # Distance on Ground Penalty
                reward -= 1.0 * (dist_x / 100.0)
            
        if truncated and not done:
             # Timeout - still in air implies failed to land
             # Penalty proportional to distance
            reward -= 100.0 + (dist_norm * 100.0)

        return obs, reward, done, truncated, info

    def _compute_potential(self, x, y):
        # Used for initialization in Reset
        dist_x = np.abs(x - self.target_x)
        dist_y = np.abs(y - self.target_y)
        hs = 0.0 # Assumed start
        vs = 0.0
        
        dist_norm = np.sqrt(dist_x**2 + dist_y**2) / 3000.0
        vel_norm = 0.0
        return - (1.0 * dist_norm + 0.1 * vel_norm) 

    @staticmethod
    def sample_tasks(
        num_tasks: int, 
        difficulty: float = 0.1, 
        randomize_physics: bool = False,
        randomize_start: bool = False,
        randomize_landing: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate tasks varying ONLY the landing site.
        Difficulty scales the distance/size of the landing site.
        """
        tasks = []
        for _ in range(num_tasks):
            # Fixed Physics (Earth-like Mars)
            gravity = 3.711
            wind = [0.0, 0.0]
            
            # Fixed Start (Center of Map)
            start_x = 3500.0
            start_y = 2500.0
            start_hs = 0.0
            start_rotate = 0.0
            
            # Variable Landing Site (Spatially Diverse)
            # Map Width = 7000. Margins ~500.
            target_x = np.random.uniform(500.0, 6500.0)
            
            # Landing Width
            # Fixed at 1000m
            landing_width = 1000.0
            
            target_y = 100.0 
            
            task = {
                "gravity": 3.711,
                "wind": [0.0, 0.0],
                "target_x": target_x,
                "target_y": target_y,
                "landing_width": landing_width,
                "start_y": start_y,
                "start_x": start_x, 
                "start_hs": start_hs,
                "start_rotate": start_rotate,
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
        # Note: factory.py calls sample_tasks with difficulty=0.1 in the caller (run.py or main)
        # or we check where sample_tasks is called.
        # Actually factory.py doesn't call sample_tasks, run.py does via factory.
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
        obs_dim = 6 + MarsLanderEnv.get_task_obs_dim()
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
        landing_width = task.get("landing_width", 1000.0)
        
        task_info = np.array([gravity, wind[0], wind[1], target_x, landing_width], dtype=np.float32)
        
        # Normalize task info to be roughly in similar range as obs if possible, 
        # or just append raw. Raw is usually fine for neural nets if not huge.
        # Wind is small, gravity ~ [1.6, 10], target_x ~ [2500, 4500].
        # Target X needs normalization!
        task_info[3] = task_info[3] / self.env.MAP_WIDTH
        task_info[4] = task_info[4] / 2000.0 # Normalize width approx
        
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

