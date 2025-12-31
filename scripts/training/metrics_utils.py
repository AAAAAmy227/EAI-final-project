"""Helper functions for metrics collection and aggregation in PPO training."""

import torch
from typing import Dict


def get_metric_specs_from_env(env) -> Dict[str, str]:
    """Get metric aggregation specifications from environment's task handler.
    
    Args:
        env: The environment (assumed to have a task_handler attribute)
        
    Returns:
        Dict mapping metric name to aggregation type ("mean" or "sum")
    """
    # Try to get task handler from the base environment
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    
    # Get task handler
    if not hasattr(base_env, 'task_handler'):
        # Fallback to defaults if no task handler
        from scripts.tasks.base import BaseTaskHandler
        return BaseTaskHandler.DEFAULT_METRIC_AGGREGATIONS.copy()
    
    task_handler = base_env.task_handler
    task_handler_class = type(task_handler)
    
    # Merge default and custom aggregations
    from scripts.tasks.base import BaseTaskHandler
    metric_specs = BaseTaskHandler.DEFAULT_METRIC_AGGREGATIONS.copy()
    metric_specs.update(task_handler_class.get_custom_metric_aggregations())
    
    return metric_specs


def aggregate_metrics(metrics_storage: Dict[str, torch.Tensor], 
                     metric_specs: Dict[str, str],
                     episode_metrics: Dict[str, list]) -> None:
    """Aggregate metrics from a rollout into episode_metrics.
    
    This function processes all metrics collected during a rollout and aggregates
    them based on their type. It extracts values only from completed episodes
    (using the done_mask) and appends them to the episode_metrics lists.
    
    Args:
        metrics_storage: Dict of (num_steps, num_envs) tensors with collected metrics
        metric_specs: Dict mapping metric name to aggregation type
        episode_metrics: Dict to store aggregated metrics (modified in-place)
    """
    done_mask = metrics_storage["done_mask"]  # (num_steps, num_envs)
    
    # Process each metric
    for metric_name, agg_type in metric_specs.items():
        if metric_name not in metrics_storage:
            continue
        
        values = metrics_storage[metric_name]  # (num_steps, num_envs)
        
        # Extract values where episodes completed
        completed_values = values[done_mask]  # 1D tensor of completed episodes
        
        if len(completed_values) == 0:
            continue  # No episodes completed in this rollout
        
        # Initialize list if needed
        if metric_name not in episode_metrics:
            episode_metrics[metric_name] = []
        
        # Aggregate based on type
        if agg_type == "mean":
            # Collect all values for averaging later
            episode_metrics[metric_name].extend(completed_values.cpu().tolist())
        elif agg_type == "sum":
            # Sum all values
            total = completed_values.sum().item()
            if metric_name in episode_metrics:
                episode_metrics[metric_name] += total
            else:
                episode_metrics[metric_name] = total
