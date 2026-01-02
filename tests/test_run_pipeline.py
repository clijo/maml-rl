import subprocess
import os
import shutil
import sys
import pytest


def test_unified_training_pipeline():
    # Use specific names to locate the folders easily
    run_name_maml = "test_unified_pipeline_maml"
    run_name_baseline = "test_unified_pipeline_baseline"

    # Cleanup beforehand if exists
    ckpt_dir_maml = os.path.join("checkpoints", run_name_maml)
    ckpt_dir_baseline = os.path.join("checkpoints", run_name_baseline)

    if os.path.exists(ckpt_dir_maml):
        shutil.rmtree(ckpt_dir_maml)
    if os.path.exists(ckpt_dir_baseline):
        shutil.rmtree(ckpt_dir_baseline)

    # 1. Train Baseline (Pretrained)
    cmd_baseline = [
        sys.executable,
        "run.py",
        "--mode",
        "train",
        "--config.num_iterations",
        "1",
        "--config.rollout_steps",
        "10",
        "--config.env.num_tasks",
        "2",
        "--config.inner.num_steps",
        "0",  # Baseline mode
        "--config.wandb.enable",
        "false",
        "--config.wandb.name",
        run_name_baseline,
        "--config.env.device",
        "cpu",
    ]

    print(f"Running Baseline Training: {' '.join(cmd_baseline)}")
    result_b = subprocess.run(cmd_baseline, capture_output=True, text=True)
    assert result_b.returncode == 0, f"Baseline Training failed: {result_b.stderr}"
    assert os.path.exists(os.path.join(ckpt_dir_baseline, "model.pt")), (
        "Baseline model.pt not created"
    )

    # 2. Train MAML
    cmd_maml = [
        sys.executable,
        "run.py",
        "--mode",
        "train",
        "--config.num_iterations",
        "1",
        "--config.rollout_steps",
        "10",
        "--config.env.num_tasks",
        "2",
        "--config.inner.num_steps",
        "1",  # MAML mode
        "--config.wandb.enable",
        "false",
        "--config.wandb.name",
        run_name_maml,
        "--config.env.device",
        "cpu",
    ]

    print(f"Running MAML Training: {' '.join(cmd_maml)}")
    result_m = subprocess.run(cmd_maml, capture_output=True, text=True)
    assert result_m.returncode == 0, f"MAML Training failed: {result_m.stderr}"
    assert os.path.exists(os.path.join(ckpt_dir_maml, "model.pt")), (
        "MAML model.pt not created"
    )

    print("Training phase verified. Files created.")

    # 3. Run evaluation (Compare MAML vs Baseline)
    cmd_eval = [
        sys.executable,
        "run.py",
        "--mode",
        "eval",
        "--checkpoint",
        os.path.join(ckpt_dir_maml, "model.pt"),
        "--pretrained_checkpoint",
        os.path.join(ckpt_dir_baseline, "model.pt"),
        "--config.wandb.enable",
        "false",
        "--config.env.device",
        "cpu",
        "--config.env.num_tasks",
        "2",
        "--config.rollout_steps",
        "10",
    ]

    print(f"Running eval command: {' '.join(cmd_eval)}")
    result_eval = subprocess.run(cmd_eval, capture_output=True, text=True)

    if result_eval.returncode != 0:
        print(result_eval.stdout)
        print(result_eval.stderr)
    assert result_eval.returncode == 0, f"Evaluation failed: {result_eval.stderr}"

    # Check if it detected baseline and ran comparison
    assert "Evaluating MAML" in result_eval.stdout
    assert "Evaluating Pretrained Baseline" in result_eval.stdout

    print("Evaluation phase verified.")

    # Cleanup
    if os.path.exists(ckpt_dir_maml):
        shutil.rmtree(ckpt_dir_maml)
    if os.path.exists(ckpt_dir_baseline):
        shutil.rmtree(ckpt_dir_baseline)


if __name__ == "__main__":
    test_unified_training_pipeline()
