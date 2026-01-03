import sys
import os
import subprocess
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def test_eval_mode_runs():
    """
    Test that eval mode runs without error using a dummy checkpoint.
    """
    # 1. Generate dummy checkpoints
    checkpoint_dir = "checkpoints/test_eval"
    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
    pretrained_path = os.path.join(checkpoint_dir, "pretrained.pt")
    config_path = os.path.join(checkpoint_dir, "config.json")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # We need a real-ish state dict for loading to work
    from maml_rl.policies import build_actor_critic
    from configs.base import TrainConfig
    from dataclasses import asdict
    import json

    # Use Navigation dims (Obs=2, Act=2)
    obs_dim = 2
    act_dim = 2
    _, policy_model, value_module = build_actor_critic(obs_dim, act_dim)

    cfg = TrainConfig()
    cfg.env.name = "navigation"  # Switch to navigation for Oracle test

    state_dict = {
        "policy_state_dict": policy_model.state_dict(),
        "value_state_dict": value_module.state_dict(),
        "config": asdict(cfg),
        "optimizer_state_dict": {},
    }

    torch.save(state_dict, checkpoint_path)
    torch.save(state_dict, pretrained_path)  # Save same model as pretrained for testing

    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f)

    # 2. Run eval mode
    cmd = [
        "uv",
        "run",
        "run.py",
        "--mode",
        "eval",
        "--checkpoint",
        checkpoint_path,
        "--pretrained_checkpoint",
        pretrained_path,
        "--cfg.env.name",
        "navigation",
        "--cfg.env.num_tasks",
        "2",
        "--cfg.rollout_steps",
        "10",
        "--cfg.wandb.enable",
        "false",
        "--cfg.env.device",
        "cpu",
        "--cfg.inner.num_steps",
        "2",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check if it finished successfully
    assert result.returncode == 0, f"Eval mode failed with stderr: {result.stderr}"
    assert "Loading checkpoint from" in result.stdout
    assert "Evaluating MAML" in result.stdout
    assert "Evaluating Pretrained Baseline" in result.stdout
    assert "Evaluating Random Init" in result.stdout
    assert "Evaluating Oracle" in result.stdout
    assert "Final Summary" in result.stdout
    assert "Pretrained" in result.stdout  # Check header


def test_train_mode_no_regression():
    """
    Test that train mode still runs for at least one iteration and saves config.json.
    """
    import shutil

    # Use a unique seed or dir to avoid collision if running parallel
    cmd = [
        "uv",
        "run",
        "run.py",
        "--mode",
        "train",
        "--cfg.num_iterations",
        "1",
        "--cfg.env.num_tasks",
        "2",
        "--cfg.rollout_steps",
        "10",
        "--cfg.wandb.enable",
        "false",
        "--cfg.env.device",
        "cpu",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Train mode failed with stderr: {result.stderr}"
    assert "[iter 1]" in result.stdout
    assert "Model saved to" in result.stdout

    # Extract save path from output to check for config.json
    # "Model saved to checkpoints/run_YYYYMMDD_HHMMSS/model.pt"
    import re

    match = re.search(r"Model saved to (.*model\.pt)", result.stdout)
    if match:
        ckpt_path = match.group(1)
        ckpt_dir = os.path.dirname(ckpt_path)
        assert os.path.exists(os.path.join(ckpt_dir, "config.json"))

        # Cleanup
        shutil.rmtree(ckpt_dir)


def test_train_mode_num_steps_zero():
    """
    Test that train mode runs correctly with num_steps=0 (Optimization path).
    """
    cmd = [
        "uv",
        "run",
        "run.py",
        "--mode",
        "train",
        "--cfg.num_iterations",
        "1",
        "--cfg.env.num_tasks",
        "2",
        "--cfg.rollout_steps",
        "10",
        "--cfg.wandb.enable",
        "false",
        "--cfg.env.device",
        "cpu",
        "--cfg.inner.num_steps",
        "0",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Train mode (steps=0) failed with stderr: {result.stderr}"
    )
    assert "[iter 1]" in result.stdout
    assert "Model saved to" in result.stdout
