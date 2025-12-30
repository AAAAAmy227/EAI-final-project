"""
Info extraction utilities for PPO training.

Provides consistent, early-crash patterns for extracting data from 
ManiSkill/Gymnasium info dictionaries that may have nested structures.
"""
from typing import Any, Optional
import numpy as np
import torch


def get_info_field(info: dict, key: str, required: bool = False) -> Any:
    """Extract field from info, prioritizing final_info for terminó episodes.
    
    Gymnasium/ManiSkill VectorEnvs use auto-reset. When an episode ends, 
    the info for the finished episode is stored in 'final_info', while the 
    top-level info reflects the reset state of the new episode.
    We prioritize 'final_info' to correctly capture episode completion metrics.
    
    Args:
        info: Step info dict from environment
        key: Field name to extract
        required: If True, raise KeyError when missing
        
    Returns:
        The value (or batched values) if found, None otherwise
    """
    # 1. Try to find in final_info (Episode results take priority)
    # final_info can be a single dict (sub-env) or a list/array of dicts (VectorEnv)
    if "final_info" in info and info["final_info"] is not None:
        final_info_bag = info["final_info"]
        
        # Support for VectorEnv (final_info is a list/array of dicts or None)
        if isinstance(final_info_bag, (list, tuple, np.ndarray)):
            # If the key is ALSO at top level, we may need to merge
            base_val = info.get(key)
            
            # If we have a base array, we patch it with final values
            if base_val is not None:
                # Make a copy if it's a tensor/array to avoid mutating original info
                if hasattr(base_val, "clone"):
                    val = base_val.clone()
                elif hasattr(base_val, "copy"):
                    val = base_val.copy()
                else:
                    val = base_val
                
                # Patch indices where final_info is available
                for i, final in enumerate(final_info_bag):
                    if final is not None and isinstance(final, dict) and key in final:
                        # Handle tensor/array assignment
                        try:
                            val[i] = final[key]
                        except Exception:
                            # Fallback for incompatible types or non-indexable val
                            pass
                return val
            
            # If no base_val, try to find the first valid final_info entry as a shortcut
            # (only if we expect a single value, but for VectorEnv we usually want the array)
            # For simplicity, if no top-level, we fall through to top-level check
        
        # Support for single environment dictionary
        elif isinstance(final_info_bag, dict) and key in final_info_bag:
            return final_info_bag[key]

    # 2. Fallback to top-level
    if key in info:
        return info[key]
        
    if required:
        raise KeyError(f"Required info field '{key}' not found in info or final_info")
    return None


def get_reward_components(info: dict) -> Optional[dict]:
    """Extract reward_components from info.
    
    Returns:
        Dict of reward component name -> value, or None if not present
    """
    return get_info_field(info, "reward_components")


def get_reward_components_per_env(info: dict) -> Optional[dict]:
    """Extract reward_components_per_env from info.
    
    This is the per-environment breakdown (each value is a tensor of shape [num_envs]).
    
    Returns:
        Dict of reward component name -> tensor, or None if not present
    """
    return get_info_field(info, "reward_components_per_env")


def extract_scalar(value: Any) -> Optional[float]:
    """Convert tensor/numpy/scalar to Python float.
    
    Handles:
    - None -> None
    - torch.Tensor -> .item()
    - numpy.ndarray -> .item() 
    - Python scalars -> float()
    
    Args:
        value: Value to convert
        
    Returns:
        Python float, or None if input was None
    """
    if value is None:
        return None
    if hasattr(value, 'item'):
        return value.item()
    return float(value)


def extract_bool(value: Any, env_idx: int = 0) -> bool:
    """Extract boolean from info value, handling tensors.
    
    Args:
        value: Value that may be bool, tensor, or None
        env_idx: Environment index if value is batched
        
    Returns:
        Python bool
    """
    if value is None:
        return False
    if hasattr(value, 'item'):
        # Check if it's a 0-dim tensor (scalar) or batched
        if hasattr(value, 'ndim') and value.ndim == 0:
            return bool(value.item())
        elif hasattr(value, '__getitem__'):
            # Batched tensor: index then extract
            return bool(value[env_idx].item())
        else:
            return bool(value.item())
    return bool(value)


def accumulate_reward_components(
    accumulator: dict, 
    reward_comps: dict, 
) -> int:
    """Accumulate reward components into running sums.
    
    Args:
        accumulator: Dict to accumulate into (mutated in place)
        reward_comps: Current step's reward components
        
    Returns:
        1 if accumulated, 0 otherwise
    """
    if reward_comps is None:
        return 0
    
    for k, v in reward_comps.items():
        val = extract_scalar(v)
        if val is not None:
            accumulator[k] = accumulator.get(k, 0.0) + val
    return 1
