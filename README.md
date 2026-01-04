# MAML-RL

Implementation of Model-Agnostic Meta-Learning (MAML) for Reinforcement Learning using PyTorch and TorchRL.

## Overview

This codebase implements MAML for continuous control tasks with:
- **Inner Loop**: Vanilla Policy Gradient (VPG) with differentiable updates
- **Outer Loop**: PPO or TRPO for stable meta-training
- **Functional Architecture**: `torch.func` for stateless execution and `vmap` for task parallelization

## Installation

```bash
uv sync
```

## Quick Start

```bash
# Train with TRPO (default)
uv run python run.py --cfg configs/config.yaml

# Train with PPO
uv run python run.py --cfg configs/config.yaml --cfg.algorithm ppo

# Evaluate
uv run python run.py --mode eval --checkpoint checkpoints/model.pt --cfg configs/nav_eval.yaml
```

Override any config parameter via CLI:

```bash
uv run python run.py --cfg configs/config.yaml --cfg.outer.lr 0.0001 --cfg.inner.num_steps 3
```

## Try Navigation environment

```bash
# 1. run MAML
uv run python run.py --cfg configs/nav.yaml --cfg.wandb.name maml_nav

# 2. train pretrained baseline
uv run python run.py --cfg configs/nav.yaml --cfg.inner.num_steps 0 --cfg.wandb.name pretrained_nav

# 3. train oracle
uv run python run.py --cfg configs/nav.yaml  --cfg.oracle true  --cfg.inner.num_steps 0 --cfg.wandb.name oracle_nav

# 4. run evaluation
uv run python run.py --mode eval --checkpoint checkpoints/maml_nav/model.pt --pretrained_checkpoint checkpoints/pretrained_nav/model.pt --oracle_checkpoint checkpoints/oracle_nav/model.pt --cfg configs/nav_eval.yaml --cfg.wandb.name eval_nav
```

## Environments

| Environment | Task | Obs Dim | Act Dim |
|-------------|------|---------|---------|
| `navigation` | Reach 2D goal position | 2 | 2 |
| `ant` | Match target velocity | 27 | 8 |

## Project Structure

```
src/maml_rl/
├── maml.py          # Inner loop (VPG), outer loop (PPO/TRPO), FunctionalPolicy
├── training.py      # Meta-training loop
├── evaluation.py    # Evaluation against baselines
├── policies.py      # Actor-critic architecture
├── envs/
│   ├── base.py      # MetaEnv protocol
│   ├── factory.py   # Environment registry
│   ├── navigation.py
│   └── ant.py
└── utils/
    ├── returns.py      # GAE computation
    ├── device.py       # Device/WandB setup
    └── optimization.py # TRPO conjugate gradient
```

## Evaluation Baselines

The evaluation compares MAML against:

| Baseline | Description |
|----------|-------------|
| **Random Init** | Randomly initialized policy adapted on test tasks |
| **Pretrained** | Policy trained without meta-learning, then adapted |
| **Oracle** | Policy trained with task parameters in observation (upper bound) |


## Adding a New Environment

Implement the `MetaEnv` protocol (see `src/maml_rl/envs/base.py`) and register it. Reference `navigation.py` or `ant.py` for examples.

### Required Methods

Your environment class must implement:

| Method | Purpose |
|--------|---------|
| `set_task(task)` | Configure env for a specific task |
| `sample_tasks(num_tasks, low, high)` | Sample task specifications |
| `get_task_obs_dim()` | Dimension of task params for oracle |
| `make_vec_env(tasks, ...)` | Create parallel env |
| `make_oracle_vec_env(tasks, ...)` | Create parallel env with task in obs |
| `get_oracle(tasks, device, checkpoint)` | Load oracle policy |

### Steps

1. **Create env class** in `src/maml_rl/envs/my_env.py` extending `gymnasium.Env`
2. **Create oracle wrapper** that appends task params to observations
3. **Register** in `src/maml_rl/envs/factory.py`:
   ```python
   from maml_rl.envs.my_env import MyEnv
   ENV_REGISTRY["my_env"] = MyEnv
   ```
4. **Create config** `configs/my_env.yaml` with env name and task bounds
5. **Train**: `uv run python run.py --cfg configs/my_env.yaml`

## Oracle Policy

Following the MAML paper, an **oracle** is a policy that receives task parameters (goal position, velocity) directly in its observation. Since the oracle knows the task, it provides an upper-bound baseline for evaluation.

### How Oracle Environments Work

Oracle environments extend the observation space by appending task parameters:

| Environment | Standard Obs | Oracle Obs |
|-------------|--------------|------------|
| `navigation` | `[x, y]` (dim 2) | `[x, y, goal_x, goal_y]` (dim 4) |
| `ant` | state (dim 27) | `[state, goal_velocity]` (dim 28) |

### Training an Oracle

```bash
uv run python run.py --cfg configs/nav.yaml --cfg.oracle true --cfg.inner.num_steps 0
```

This trains a policy on Oracle observations (task params included). Use the checkpoint during evaluation with `--oracle_checkpoint`.

## Configuration

Key config options in `configs/base.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algorithm` | `"trpo"` | Outer loop algorithm (`trpo` or `ppo`) |
| `inner.num_steps` | `1` | Inner loop gradient steps |
| `inner.lr` | `0.1` | Inner loop learning rate |
| `outer.lr` | `0.0003` | Outer loop learning rate (PPO only) |
| `env.num_tasks` | `8` | Number of parallel tasks |
| `env.max_steps` | `200` | Maximum steps per episode |
| `rollout_steps` | `200` | Steps collected per task for adaptation |
| `gamma` | `0.99` | Discount factor |
| `lam` | `0.95` | GAE lambda |

### TRPO-specific options (`trpo.*`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trpo.max_kl` | `0.01` | Maximum KL divergence constraint |
| `trpo.damping` | `0.1` | Conjugate gradient damping factor |
| `trpo.cg_iters` | `10` | Conjugate gradient iterations |

### PPO-specific options (`outer.*`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `outer.clip_eps` | `0.2` | PPO clipping epsilon |
| `outer.ppo_epochs` | `5` | PPO epochs per outer step |
| `outer.entropy_coef` | `0.0` | Entropy bonus coefficient |
