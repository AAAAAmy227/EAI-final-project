"""
Info extraction utilities for PPO training.

Provides consistent, early-crash patterns for extracting data from 
ManiSkill/Gymnasium info dictionaries that may have nested structures.
"""
from typing import Any, Optional


def get_info_field(info: dict, key: str, required: bool = False) -> Any:
    """Extract field from info, checking top-level then final_info.
    
    ManiSkill environments may place info fields at either the top level
    or nested under 'final_info' (for terminated episodes). This function
    handles both cases consistently.
    
    Args:
        info: Step info dict from environment
        key: Field name to extract
        required: If True, raise KeyError when missing (early crash pattern)
        
    Returns:
        The value if found, None otherwise (unless required=True)
        
    Raises:
        KeyError: If required=True and field not found anywhere
    """
    if key in info:
        return info[key]
    if "final_info" in info and key in info["final_info"]:
        return info["final_info"][key]
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
    count_ref: list
) -> None:
    """Accumulate reward components into running sums.
    
    Args:
        accumulator: Dict to accumulate into (mutated in place)
        reward_comps: Current step's reward components
        count_ref: Single-element list with count (mutated in place)
    """
    if reward_comps is None:
        return
    
    for k, v in reward_comps.items():
        val = extract_scalar(v)
        if val is not None:
            accumulator[k] = accumulator.get(k, 0.0) + val
    count_ref[0] += 1
