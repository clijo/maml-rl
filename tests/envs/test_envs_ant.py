import pytest
import numpy as np
from unittest.mock import patch

from maml_rl.envs.ant import MetaAntGoalVelEnv

# Check availability of AntEnv
try:
    from gymnasium.envs.mujoco.ant_v4 import AntEnv  # noqa: F401

    ANT_AVAILABLE = True
except (ImportError, Exception):
    ANT_AVAILABLE = False


@pytest.mark.skipif(not ANT_AVAILABLE, reason="AntEnv/mujoco not available")
class TestAntEnv:
    def test_sample_tasks(self):
        num_tasks = 10
        tasks = MetaAntGoalVelEnv.sample_tasks(num_tasks, low=0.0, high=3.0)
        assert len(tasks) == num_tasks
        for task in tasks:
            assert "velocity" in task
            assert 0.0 <= task["velocity"] <= 3.0

    def test_meta_ant_goal_vel_env_logic(self):
        # Let's try to mock the step to return a specific velocity in info
        with patch("gymnasium.envs.mujoco.ant_v4.AntEnv.step") as mock_step:
            # mock return: obs, reward, terminated, truncated, info
            # info needs x_velocity and reward_ctrl
            mock_step.return_value = (
                np.zeros(27),
                1.0,
                False,
                False,
                {"x_velocity": 1.5, "reward_ctrl": -0.1},
            )

            env = MetaAntGoalVelEnv()
            task = {"velocity": 1.0}
            env.set_task(task)

            obs, reward, terminated, truncated, info = env.step(np.zeros(8))

            # Expected reward: -|1.5 - 1.0| + (-0.1) + 1.0 = -0.5 - 0.1 + 1.0 = 0.4
            assert np.isclose(reward, 0.4)
            assert info["x_velocity"] == 1.5

    def test_meta_ant_goal_vel_env_reset_with_options(self):
        env = MetaAntGoalVelEnv()
        task = {"velocity": 2.5}
        env.reset(options={"task": task})
        assert env._goal_vel == 2.5

    def test_make_vec_env(self):
        tasks = MetaAntGoalVelEnv.sample_tasks(2)
        # By default norm_obs=True, so we must initialize it
        env = MetaAntGoalVelEnv.make_vec_env(tasks, device="cpu", max_steps=10)

        # Access the transform to initialize it
        env.transform.init_stats(num_iter=1, reduce_dim=[0, 1], cat_dim=0)

        td = env.reset()
        assert td.shape[0] == 2
        assert "observation" in td.keys()
        env.close()
