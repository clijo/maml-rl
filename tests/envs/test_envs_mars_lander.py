
import pytest
import numpy as np
from maml_rl.envs.mars_lander import MarsLanderEnv

class TestMarsLanderEnv:
    def test_initialization(self):
        env = MarsLanderEnv()
        assert env.action_space.shape == (2,)
        assert env.observation_space.shape == (7,)

    def test_reset(self):
        env = MarsLanderEnv()
        obs, info = env.reset(seed=42)
        assert len(obs) == 7
        assert isinstance(obs, np.ndarray)

    def test_step(self):
        env = MarsLanderEnv()
        env.reset(seed=42)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert len(next_obs) == 7
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)

    def test_physics_gravity(self):
        """Verify that higher gravity causes faster fall."""
        env = MarsLanderEnv()
        
        # Idle action (no thrust)
        # Power mapped from [-1, 1] to [0, 4], so -1 is 0 power.
        action_idle = np.array([-1.0, -1.0]) 

        # Case 1: Standard Gravity
        env.reset(seed=42, options={'task': {'gravity': 3.711, 'wind': [0, 0]}})
        # Force state to known value
        env.state = np.array([2500, 2500, 0, 0, 500, 0, 0], dtype=np.float32)
        env.step(action_idle)
        v_speed_normal = env.state[3]

        # Case 2: Heavy Gravity
        env.reset(seed=42, options={'task': {'gravity': 20.0, 'wind': [0, 0]}})
        env.state = np.array([2500, 2500, 0, 0, 500, 0, 0], dtype=np.float32)
        env.step(action_idle)
        v_speed_heavy = env.state[3]

        # Heavy gravity -> More negative vertical speed
        assert v_speed_heavy < v_speed_normal
        
        # Case 3: Simple Gravity
        env.reset(seed=42, options={'task': {'gravity': 100.0, 'wind': [0, 0]}})
        env.state = np.array([0, 1000, 0, 0, 500, 0, 0], dtype=np.float32)
        env.step(action_idle)
        v_speed = env.state[3]
        h_speed = env.state[2]
        
        assert abs(v_speed - -100.0) < 1e-3  # Should fall at approx -100 m/s after one step
        assert h_speed == 0.0  # Falling down 
        
        # Case 4: Zero Gravity + Wind
        env.reset(seed=42, options={'task': {'gravity': 0.0, 'wind': [5.0, 0.0]}})
        env.state = np.array([0, 1000, 0, 0, 500, 0, 0], dtype=np.float32)
        env.step(action_idle)
        v_speed = env.state[3]
        h_speed = env.state[2]
        
        assert v_speed == 0.0  # No vertical acceleration
        assert abs(h_speed - 5.0) < 1e-3  #

    def test_sample_tasks(self):
        num_tasks = 5
        tasks = MarsLanderEnv.sample_tasks(num_tasks)
        assert len(tasks) == num_tasks
        for task in tasks:
            assert "gravity" in task
            assert "wind" in task
            assert "target_x" in task
