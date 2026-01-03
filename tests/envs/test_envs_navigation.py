import numpy as np

from maml_rl.envs.navigation import (
    make_navigation_env,
    sample_navigation_tasks,
    make_navigation_vec_env,
    Navigation2DEnv,
)


class TestNavigationEnv:
    def test_sample_navigation_tasks(self):
        num_tasks = 10
        tasks = sample_navigation_tasks(num_tasks, low=-0.5, high=0.5)
        assert len(tasks) == num_tasks
        for task in tasks:
            assert "goal" in task
            goal = task["goal"]
            assert goal.shape == (2,)
            assert np.all(goal >= -0.5)
            assert np.all(goal <= 0.5)

    def test_navigation_env_logic(self):
        env = Navigation2DEnv()

        # Set a specific goal
        target_goal = np.array([0.5, 0.5], dtype=np.float64)
        env.set_task({"goal": target_goal})
        env.reset()

        # Check initial state (should be 0,0)
        assert np.allclose(env._state, np.zeros(2))

        # 1. Test step with action within limits
        # Input 0.5 becomes 0.05 after scaling
        action = np.array([0.5, 0.5], dtype=np.float64)
        state, reward, terminated, truncated, info = env.step(action)

        # New state should be [0.05, 0.05]
        expected_state = np.array([0.05, 0.05], dtype=np.float64)
        assert np.allclose(state, expected_state)

        # Distance to goal [0.5, 0.5] is sqrt(0.45^2 + 0.45^2)
        # Reward is -distance^2
        dist_sq = (0.5 - 0.05) ** 2 + (0.5 - 0.05) ** 2
        expected_reward = -dist_sq
        assert np.isclose(reward, expected_reward)
        assert not terminated

        # 2. Test step with action exceeding limits (clipping)
        # Reset to known state
        env.reset()
        # Input [2.0, -5.0] should be clipped to [1.0, -1.0], then scaled to [0.1, -0.1]
        action_large = np.array([2.0, -5.0], dtype=np.float64)
        state, _, _, _, _ = env.step(action_large)

        # Should be clipped to [0.1, -0.1]
        expected_state_clipped = np.array([0.1, -0.1], dtype=np.float64)
        assert np.allclose(state, expected_state_clipped)

        # 3. Test termination condition
        # Set state very close to goal
        env._state = np.array([0.499, 0.499], dtype=np.float64)
        # Distance to [0.5, 0.5] is sqrt(0.001^2 + 0.001^2) approx 0.0014 < 0.01

        # Take a tiny step that keeps it within range or 0 step
        state, reward, terminated, truncated, _ = env.step(
            np.zeros(2, dtype=np.float64)
        )
        assert terminated

    def test_navigation_env_reset_with_options(self):
        env = Navigation2DEnv()
        task = {"goal": np.array([0.1, 0.2], dtype=np.float64)}
        env.reset(options={"task": task})
        assert np.allclose(env._goal, task["goal"])

    def test_make_navigation_env_instantiates(self):
        task = sample_navigation_tasks(1)[0]
        env = make_navigation_env(task, device="cpu", max_steps=10)
        td = env.reset()
        assert "observation" in td.keys()
        assert env.action_spec is not None
        env.close()

    def test_make_navigation_vec_env(self):
        tasks = sample_navigation_tasks(2)
        # norm_obs=False for navigation usually, but testing if it works
        env = make_navigation_vec_env(tasks, device="cpu", max_steps=10, norm_obs=True)

        # Access the transform to initialize it
        env.transform.init_stats(num_iter=1, reduce_dim=[0, 1], cat_dim=0)

        td = env.reset()
        assert td.shape[0] == 2
        assert "observation" in td.keys()
        env.close()
