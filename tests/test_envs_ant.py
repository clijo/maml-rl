import pytest
import numpy as np
from unittest.mock import patch

from maml_rl.envs.ant import (
    make_ant_env,
    sample_ant_tasks,
    make_ant_vec_env,
    MetaAntGoalVelEnv,
)

# Check availability of AntEnv
try:
    from gymnasium.envs.mujoco.ant_v4 import AntEnv  # noqa: F401

    ANT_AVAILABLE = True
except (ImportError, Exception):
    ANT_AVAILABLE = False


@pytest.mark.skipif(not ANT_AVAILABLE, reason="AntEnv/mujoco not available")
class TestAntEnv:
    def test_sample_ant_tasks(self):
        num_tasks = 10
        tasks = sample_ant_tasks(num_tasks, low=0.0, high=3.0)
        assert len(tasks) == num_tasks
        for task in tasks:
            assert "velocity" in task
            assert 0.0 <= task["velocity"] <= 3.0

    def test_meta_ant_goal_vel_env_logic(self):
        # We can test the logic without full physics if we mock the super().step
        # But since we have mujoco, we can try running it.
        # If running fails due to missing GL/libraries, we might need to mock.

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

            # Expected reward: -|1.5 - 1.0| + (-0.1) = -0.5 - 0.1 = -0.6
            assert np.isclose(reward, -0.6)
            assert info["x_velocity"] == 1.5

    def test_meta_ant_goal_vel_env_reset_with_options(self):
        env = MetaAntGoalVelEnv()
        task = {"velocity": 2.5}
        env.reset(options={"task": task})
        assert env._goal_vel == 2.5

    def test_make_ant_env_instantiates(self):
        task = sample_ant_tasks(1)[0]
        env = make_ant_env(task, device="cpu", max_steps=10)
        td = env.reset()
        assert "observation" in td.keys()
        assert env.action_spec is not None
        env.close()

    def test_make_ant_vec_env(self):
        tasks = sample_ant_tasks(2)
        # Mocking to avoid heavy physics initialization if possible,
        # but integration test is good too.
        # We'll rely on skipping if full mujoco fails.
        env = make_ant_vec_env(tasks, device="cpu", max_steps=10)
        td = env.reset()
        assert td.shape[0] == 2
        assert "observation" in td.keys()
        env.close()
