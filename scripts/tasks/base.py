from abc import ABC, abstractmethod
import torch
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from scripts.track1_env import Track1Env

class BaseTaskHandler(ABC):
    def __init__(self, env: 'Track1Env'):
        self.env = env

    @property
    def device(self):
        return self.env.device

    @abstractmethod
    def evaluate(self) -> Dict[str, torch.Tensor]:
        """Evaluate success/fail conditions for the task."""
        pass

    @abstractmethod
    def compute_dense_reward(self, info: dict, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the dense reward for the task."""
        pass
    
    @abstractmethod
    def initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize task-specific objects and state for a new episode."""
        pass
