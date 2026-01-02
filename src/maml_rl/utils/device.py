"""Device and environment utilities."""

from dataclasses import asdict

import torch
import wandb


def enable_tf32():
    """
    Enable TensorFloat32 matmul on Ampere+ GPUs for faster float32 GEMMs.
    No effect on older GPUs or CPU.
    """
    if not torch.cuda.is_available():
        return
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:  # Ampere or newer
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled (float32 matmul precision = high).")


def get_device(device_str: str) -> torch.device:
    """
    Get torch device from string specification.

    Args:
        device_str: Device string ("auto", "cuda", "mps", or "cpu")

    Returns:
        torch.device object
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def wandb_setup(cfg):
    """Initialize wandb logging if enabled."""
    if not cfg.wandb.enable:
        return None
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        name=cfg.wandb.name,
        config=asdict(cfg),
    )
    return run
