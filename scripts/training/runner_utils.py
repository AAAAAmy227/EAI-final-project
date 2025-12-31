"""
Pure utility functions for PPORunner.

These functions are designed to be testable without requiring
a PPORunner instance, following testability design principles.
"""
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch


def compute_reward_logs(episode_metrics: Dict[str, list]) -> Dict[str, float]:
    """Compute reward component logs from episode metrics.
    
    Pure function that transforms episode metrics into wandb log entries
    with appropriate prefixes.
    
    Args:
        episode_metrics: Dict mapping metric names to lists of values
        
    Returns:
        Dict of log entries with appropriate prefixes (rollout/, reward/)
        
    Examples:
        >>> metrics = {"success": [True, False, True], "return": [10.5, 8.2, 12.1]}
        >>> logs = compute_reward_logs(metrics)
        >>> logs["rollout/success_rate"]
        0.666...
    """
    logs = {}
    
    if not episode_metrics:
        return logs
    
    # Build logs for all metrics
    for metric_name, values in episode_metrics.items():
        if not values:
            continue
        
        # Compute mean (works for both float metrics and boolean success/fail)
        mean_value = np.mean(values)
        
        # Add appropriate prefix
        if metric_name in ["success", "fail", "success_once", "fail_once"]:
            # These are rates (mean of boolean values)
            logs[f"rollout/{metric_name}_rate"] = mean_value
        elif metric_name in ["return", "episode_len"]:
            # Episode-level stats
            logs[f"rollout/{metric_name}"] = mean_value
        elif metric_name == "raw_reward":
            logs[f"rollout/raw_reward_mean"] = mean_value
        else:
            # Task-specific reward components
            logs[f"reward/{metric_name}"] = mean_value
    
    return logs


def compute_eval_logs(episode_metrics: Dict[str, list]) -> Dict[str, float]:
    """Compute evaluation logs from episode metrics.
    
    Pure function that transforms episode metrics into wandb log entries
    with 'eval/' prefix.
    
    Args:
        episode_metrics: Dict mapping metric names to lists of values
        
    Returns:
        Dict of evaluation logs with eval/ prefix
        
    Examples:
        >>> metrics = {"success": [True, True, False], "return": [12.5, 11.8, 9.5]}
        >>> logs = compute_eval_logs(metrics)
        >>> logs["eval/success_rate"]
        0.666...
    """
    logs = {}
    
    if not episode_metrics:
        return logs
    
    # Special handling for key metrics
    if "return" in episode_metrics and episode_metrics["return"]:
        logs["eval/return"] = np.mean(episode_metrics["return"])
    
    if "success" in episode_metrics and episode_metrics["success"]:
        logs["eval/success_rate"] = np.mean(episode_metrics["success"])
    
    if "fail" in episode_metrics and episode_metrics["fail"]:
        logs["eval/fail_rate"] = np.mean(episode_metrics["fail"])
    
    # Add all other metrics with eval_reward/ prefix
    for metric_name, values in episode_metrics.items():
        if not values or metric_name in ["return", "success", "fail"]:
            continue
        
        mean_value = np.mean(values)
        
        if metric_name in ["episode_len", "success_once", "fail_once"]:
            logs[f"eval/{metric_name}"] = mean_value
        else:
            # Task-specific reward components
            logs[f"eval_reward/{metric_name}"] = mean_value
    
    return logs


def build_csv_path(base_dir: Path, eval_name: str, env_idx: int) -> Path:
    """Build CSV file path for per-environment step data.
    
    Pure function to construct the path following the convention:
    base_dir/split/eval_name/env{env_idx}/rewards.csv
    
    Args:
        base_dir: Base output directory
        eval_name: Evaluation run name (e.g., "eval0", "eval1")
        env_idx: Environment index
        
    Returns:
        Path to CSV file
        
    Examples:
        >>> build_csv_path(Path("/tmp"), "eval0", 5)
        PosixPath('/tmp/split/eval0/env5/rewards.csv')
    """
    return base_dir / "split" / eval_name / f"env{env_idx}" / "rewards.csv"


def write_csv_file(filepath: Path, data: List[dict]) -> None:
    """Write step data to CSV file.
    
    Pure function to write list of dicts to CSV file.
    Creates parent directories if needed. Handles empty data gracefully.
    
    Args:
        filepath: Path to CSV file
        data: List of dicts with step data (must all have same keys)
        
    Examples:
        >>> write_csv_file(Path("/tmp/test.csv"), [{"step": 0, "reward": 1.0}])
    """
    if not data:
        return
    
    # Create parent directories
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    import csv
    with open(filepath, 'w', newline='') as f:
        fieldnames = list(data[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def extract_metric_from_info(info_dict: dict, metric_name: str) -> Optional[float]:
    """Extract a single metric value from ManiSkill info dict.
    
    Pure function to handle the nested structure of ManiSkill info dicts,
    which may have metrics in either info["episode"][metric] or info[metric].
    
    Args:
        info_dict: Info dict from environment step
        metric_name: Name of metric to extract
        
    Returns:
        Metric value as float, or None if not found
        
    Examples:
        >>> info = {"episode": {"return": 10.5, "success": True}}
        >>> extract_metric_from_info(info, "return")
        10.5
        >>> extract_metric_from_info(info, "unknown")
        None
    """
    value_to_store = None
    
    # Handle "episode" dict from ManiSkill
    if metric_name in ["return", "episode_len", "success_once", "fail_once", "reward"]:
        episode_info = info_dict.get("episode")
        if episode_info is not None and metric_name in episode_info:
            value_to_store = episode_info[metric_name]
    elif metric_name in info_dict:
        value_to_store = info_dict[metric_name]
    
    # Convert to float if found
    if value_to_store is not None:
        if isinstance(value_to_store, torch.Tensor):
            return value_to_store.float().item() if value_to_store.numel() == 1 else value_to_store.float()
        return float(value_to_store)
    
    return None
