"""
MAML-RL entry point.

Run MAML with VPG inner loop and PPO or TRPO outer loop.

Example:
    uv run python run.py --cfg configs/config.yaml
Override any field via CLI, e.g.:
    uv run python run.py --cfg configs/config.yaml --config.outer.lr 0.0001
"""

import random

import numpy as np
import torch
from jsonargparse import ArgumentParser

from configs.base import TrainConfig
from maml_rl.evaluation import evaluate
from maml_rl.training import train
from maml_rl.utils.device import enable_tf32, get_device, wandb_setup


def main():
    """
    Run MAML with VPG inner loop and PPO or TRPO outer loop.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "eval"], help="Run mode"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to MAML checkpoint for evaluation",
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default=None,
        help="Path to Pretrained (Baseline) checkpoint",
    )
    parser.add_argument(
        "--oracle_checkpoint",
        type=str,
        default=None,
        help="Path to Oracle checkpoint (trained on oracle observations)",
    )
    parser.add_class_arguments(TrainConfig, "cfg")
    args = parser.parse_args()
    cfg: TrainConfig = parser.instantiate_classes(args).cfg

    enable_tf32()
    wandb_setup(cfg)

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = get_device(cfg.env.device)
    print(f"Device: {device}")
    print(f"Algorithm: {cfg.algorithm.upper()}")

    if args.mode == "train":
        train(cfg, device, args.checkpoint)
    else:
        evaluate(
            cfg,
            device,
            args.checkpoint,
            args.pretrained_checkpoint,
            args.oracle_checkpoint,
        )


if __name__ == "__main__":
    main()
