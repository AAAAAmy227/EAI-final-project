from abc import ABC, abstractmethod
import torch
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from scripts.envs.track1_env import Track1Env

class BaseTaskHandler(ABC):
    # Default aggregation rules for common metrics
    # "mean": collect values and compute average (for success rate, rewards, etc.)
    # "sum": accumulate total count
    DEFAULT_METRIC_AGGREGATIONS = {
        "success": "mean",       # Success rate (mean of boolean values)
        "fail": "mean",          # Fail rate (mean of boolean values)
        "raw_reward": "mean",    # Average raw reward per episode
        "return": "mean",        # Average episode return
        "episode_len": "mean",   # Average episode length
        "success_once": "mean",  # Whether succeeded at least once
        "fail_once": "mean",     # Whether failed at least once
    }
    
    def __init__(self, env: 'Track1Env'):
        self.env = env
        self.initial_red_cube_pos: Optional[torch.Tensor] = None
        self.initial_green_cube_pos: Optional[torch.Tensor] = None
        self.initial_cube_xy: Optional[torch.Tensor] = None
        self.lift_hold_counter: Optional[torch.Tensor] = None
        self.grasp_hold_counter: Optional[torch.Tensor] = None
        self.prev_action: Optional[torch.Tensor] = None

    @property
    def device(self):
        return self.env.device
    
    @classmethod
    def get_custom_metric_aggregations(cls) -> Dict[str, str]:
        """Return custom metric aggregation rules for this task.
        
        Override this in subclasses to define task-specific metrics.
        These will be merged with DEFAULT_METRIC_AGGREGATIONS.
        
        Returns:
            Dict mapping metric name to aggregation type ("mean" or "sum")
        """
        return {}

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
