# MAML-RL

An implementation of Model-Agnostic Meta-Learning (MAML) for Reinforcement Learning using PyTorch and TorchRL.

## Overview

This codebase implements the MAML algorithm for continuous control tasks. It uses `torch.func` and `tensordict` to efficiently handle meta-gradient computation via stateless functional calls and vectorization (`vmap`).

## Key Features

-   **Functional Architecture**: Utilizes `torch.func.functional_call` for stateless model execution, enabling seamless meta-optimization without manual gradient tape management.
-   **Vectorized Environments**: Supports batched task execution using `tensordict.vmap` for efficient data collection.
-   **Inner Loop**: Vanilla Policy Gradient (VPG) with differentiable functional updates, fully compatible with `vmap` over tasks.
-   **Outer Loop**: Proximal Policy Optimization (PPO) for stable meta-training of the initial policy parameters.
-   **Value Estimation**: Generalized Advantage Estimation (GAE) with a meta-learned value function.

## Usage

Run the meta-training loop:

```bash
uv run run.py --cfg configs/config.yaml
```

Override configuration parameters via CLI:

```bash
uv run run.py --cfg configs/config.yaml --config.outer.lr 0.0001
```

## Structure

-   `src/maml_rl/maml.py`: Core MAML algorithm (inner loop VPG, outer loop PPO, functional policy wrapper).
-   `src/maml_rl/envs/`: Environment definitions (Ant).
-   `configs/`: Configuration files (YAML and dataclasses).
-   `run.py`: Entry point for training.
