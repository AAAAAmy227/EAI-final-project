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
    def get_custom_metric_aggregations(cls, mode: str = "train") -> Dict[str, str]:
        """Return custom metric aggregation rules for this task.
        
        Override _get_train_metrics() and/or _get_eval_metrics() in subclasses
        to define mode-specific metrics.
        
        Args:
            mode: "train" or "eval" to specify which metrics to return
        
        Returns:
            Dict mapping metric name to aggregation type ("mean" or "sum")
        """
        if mode == "train":
            return cls._get_train_metrics()
        elif mode == "eval":
            return cls._get_eval_metrics()
        else:
            # Fallback: merge both
            metrics = cls._get_train_metrics().copy()
            metrics.update(cls._get_eval_metrics())
            return metrics
    
    @classmethod
    def _get_train_metrics(cls) -> Dict[str, str]:
        """Return training-specific metrics.
        
        Override in subclasses to define metrics collected during training.
        By default, returns empty dict (only DEFAULT_METRIC_AGGREGATIONS used).
        
        Returns:
            Dict mapping metric name to aggregation type
        """
        return {}
    
    @classmethod
    def _get_eval_metrics(cls) -> Dict[str, str]:
        """Return evaluation-specific metrics.
        
        Override in subclasses to define metrics collected during evaluation.
        By default, returns the same as training metrics.
        
        Returns:
            Dict mapping metric name to aggregation type
        """
        return cls._get_train_metrics()

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
