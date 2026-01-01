from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class EnvConfig:
    name: str = "ant"
    num_tasks: int = 8
    max_steps: int = 200
    task_low: float = 0.0
    task_high: float = 3.0
    device: str = "auto"
    norm_obs: bool = True


@dataclass
class ModelConfig:
    hidden_sizes: Tuple[int, ...] = (128, 128)


@dataclass
class InnerConfig:
    lr: float = 0.1
    num_steps: int = 1
    advantage_norm: bool = False


@dataclass
class OuterConfig:
    lr: float = 3e-4
    ppo_epochs: int = 5
    clip_eps: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    clip_value_loss: bool = False
    max_grad_norm: Optional[float] = None
    advantage_norm: bool = False


@dataclass
class TRPOConfig:
    max_kl: float = 0.01
    damping: float = 0.1
    cg_iters: int = 10
    line_search_max_steps: int = 10
    line_search_backtrack_ratio: float = 0.5


@dataclass
class WandbConfig:
    enable: bool = True
    project: str = "maml_rl"
    entity: Optional[str] = None
    group: Optional[str] = None
    mode: str = "online"
    name: Optional[str] = None


@dataclass
class TrainConfig:
    algorithm: str = "trpo"  # "ppo" or "trpo"
    rollout_steps: int = 200
    num_iterations: int = 3
    gamma: float = 0.99
    lam: float = 0.95
    seed: int = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inner: InnerConfig = field(default_factory=InnerConfig)
    outer: OuterConfig = field(default_factory=OuterConfig)
    trpo: TRPOConfig = field(default_factory=TRPOConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


__all__ = [
    "EnvConfig",
    "ModelConfig",
    "InnerConfig",
    "OuterConfig",
    "TRPOConfig",
    "WandbConfig",
    "TrainConfig",
]
