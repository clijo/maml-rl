import pytest
from maml_rl.envs.factory import make_vec_env, sample_tasks

# Check availability of AntEnv
try:
    from gymnasium.envs.mujoco.ant_v4 import AntEnv  # noqa: F401

    ANT_AVAILABLE = True
except (ImportError, Exception):
    ANT_AVAILABLE = False


@pytest.mark.skipif(not ANT_AVAILABLE, reason="AntEnv/mujoco not available")
class TestEnvFactory:
    def test_sample_tasks_ant(self):
        tasks = sample_tasks("ant", num_tasks=5, task_low=0.0, task_high=1.0)
        assert len(tasks) == 5
        assert all("velocity" in t for t in tasks)

    def test_make_vec_env_ant(self):
        tasks, env = make_vec_env(
            env_name="ant",
            num_tasks=2,
            task_low=0.0,
            task_high=3.0,
            max_steps=10,
            device="cpu",
            norm_obs=True,
            seed=42,
        )
        assert len(tasks) == 2
        # Must initialize if norm_obs=True
        env.transform.init_stats(num_iter=1, reduce_dim=[0, 1], cat_dim=0)

        td = env.reset()
        assert td.shape[0] == 2
        env.close()

    def test_factory_unknown_env(self):
        with pytest.raises(ValueError, match="Unknown environment"):
            sample_tasks("unknown", 1, 0, 1)

        with pytest.raises(ValueError, match="Unknown environment"):
            make_vec_env("unknown", 1, 0, 1, 10, "cpu", True, 0)
