import pytest
import torch
import numpy as np
from scripts.training.info_utils import get_info_field, get_reward_components

def test_maniskill_gpu_autoreset_logic():
    """
    Simulates ManiSkill GPU mode (ManiSkillVectorEnv).
    When an environment resets, final_info is a clone of the pre-reset info (dict of tensors).
    """
    info = {
        "success": torch.tensor([False, False]),
        "final_info": {
            "success": torch.tensor([True, False]),
            "reward_components": {
                "lift": torch.tensor([1.0, 0.0])
            }
        }
    }
    
    success_val = get_info_field(info, "success")
    assert torch.equal(success_val, torch.tensor([True, False]))
    
    reward_comps = get_reward_components(info)
    assert torch.equal(reward_comps["lift"], torch.tensor([1.0, 0.0]))

def test_standard_gym_vector_autoreset_logic():
    """
    Simulates standard Gymnasium VectorEnv (list of dicts).
    """
    info = {
        "success": np.array([False, True]),
        "final_info": [
            {"success": True}, 
            None
        ]
    }
    
    success_val = get_info_field(info, "success")
    assert np.array_equal(success_val, np.array([True, True]))

def test_standard_gym_vector_missing_top_level():
    """
    Test scenario where the key only exists in final_info list, not top-level.
    Currently, the code falls through and returns None.
    """
    info = {
        "final_info": [
            {"custom_metric": 42},
            None
        ]
    }
    result = get_info_field(info, "custom_metric")
    print(f"Extraction result for missing top-level: {result}")
    # This currently expects None based on the source code analysis
    assert result is None

def test_maniskill_gpu_missing_top_level():
    """
    Test scenario where the key only exists in final_info dict, not top-level.
    This SHOULD work in GPU mode because it checks the dict directly.
    """
    info = {
        "final_info": {
            "custom_metric": torch.tensor([42, 0])
        }
    }
    result = get_info_field(info, "custom_metric")
    assert torch.equal(result, torch.tensor([42, 0]))

def test_no_autoreset_logic():
    """Simulates evaluation mode where autoreset=False."""
    info = {
        "success": torch.tensor([True, False]),
        "reward_components": {"lift": torch.tensor([0.8, 0.0])}
    }
    assert torch.equal(get_info_field(info, "success"), torch.tensor([True, False]))
    assert torch.equal(get_reward_components(info)["lift"], torch.tensor([0.8, 0.0]))

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
