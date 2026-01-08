import torch
from typing import Optional
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded, Bounded
from torchrl.envs import EnvBase


class Navigation2DVectorized(EnvBase):
    """Vectorized 2D Navigation environment.

    Simulates a batch of agents moving in 2D plane to reach task-specific goals.
    All operations are batched tensors.
    """

    def __init__(self, num_tasks: int, device: str = "cpu"):
        super().__init__(device=device, batch_size=torch.Size([num_tasks]))
        self.num_tasks = num_tasks

        # Goal is stateful within the environment logic, but we store it in buffers
        # to persist it. For simplicity in MAML, we can store it as a buffer
        # and update it via set_task or reset(options).
        # We initialize with zeros
        self.register_buffer("goals", torch.zeros(num_tasks, 2, device=device))

        # Specs
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=torch.Size([num_tasks, 2]),
                dtype=torch.float32,
                device=device,
            ),
            goals=Unbounded(
                shape=torch.Size([num_tasks, 2]),
                dtype=torch.float32,
                device=device,
            ),
            state=Unbounded(
                shape=torch.Size([num_tasks, 2]),
                dtype=torch.float32,
                device=device,
            ),
            shape=torch.Size([num_tasks]),
        )
        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=torch.Size([num_tasks, 2]),
            dtype=torch.float32,
            device=device,
        )
        self.reward_spec = Unbounded(
            shape=torch.Size([num_tasks, 1]),
            dtype=torch.float32,
            device=device,
        )

    def _reset(self, t: TensorDict = None, **kwargs) -> TensorDict:
        # If goals are passed in options (during set_task effectively), update them.
        # But standard MAML usually samples tasks outside.
        # Here we just reset the agent position to (0,0)

        batch_size = t.shape if t is not None else self.batch_size
        pos = torch.zeros(*batch_size, 2, device=self.device, dtype=torch.float32)

        out = TensorDict(
            {
                "observation": pos.clone(),
                "state": pos.clone(),
                "goals": self.goals.clone(),  # Pass goals along if needed
            },
            batch_size=batch_size,
            device=self.device,
        )
        return out

    def _step(self, tensordict: TensorDict) -> TensorDict:
        # Use state for physics to avoid issues with transforms modifying observation
        prev_pos = tensordict["state"]
        action = tensordict["action"]

        # Dynamics: pos = pos + clip(action) * 0.1
        # Action is already clipped by spec usually, but let's be safe per original code
        action = action.clamp(-1.0, 1.0) * 0.1
        new_pos = prev_pos + action

        # Reward
        # batch dims auto-broadcast
        dist = torch.linalg.norm(new_pos - self.goals, dim=-1)
        reward = -(dist**2)

        # Done
        # "Terminated" if within 0.01 of goal
        terminated = dist < 0.01

        out = TensorDict(
            {
                "observation": new_pos.clone(),
                "state": new_pos.clone(),
                "goals": self.goals.clone(),
                "reward": reward.unsqueeze(-1),
                "done": terminated.unsqueeze(-1),
                "terminated": terminated.unsqueeze(-1),
            },
            batch_size=tensordict.batch_size,
            device=self.device,
        )
        return out

    def set_task(self, tasks: list[dict]):
        """Update goals from task list."""
        # tasks is list of dicts [{'goal': array([x, y])}, ...]
        import numpy as np

        goals_list = [t["goal"] for t in tasks]
        goals_array = np.stack(goals_list)
        goals_tensor = torch.from_numpy(goals_array).to(
            self.device, dtype=torch.float32
        )

        if goals_tensor.shape[0] != self.batch_size[0]:
            raise ValueError(
                f"Task batch size {goals_tensor.shape[0]} != env batch size {self.batch_size[0]}"
            )

        self.goals.copy_(goals_tensor)

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    @staticmethod
    def sample_tasks(
        num_tasks: int, low: float = -0.5, high: float = 0.5
    ) -> list[dict]:
        import numpy as np

        goals = np.random.uniform(low, high, size=(num_tasks, 2))
        return [{"goal": goal.astype(np.float32)} for goal in goals]
