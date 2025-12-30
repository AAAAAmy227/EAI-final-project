"""
Unit tests for info_utils module.

Tests the info extraction utilities used in PPORunner for consistent
handling of ManiSkill info dictionaries.
"""
import pytest
import torch
import numpy as np

from scripts.training.info_utils import (
    get_info_field,
    get_reward_components,
    get_reward_components_per_env,
    extract_scalar,
    extract_bool,
    accumulate_reward_components,
)


class TestGetInfoField:
    """Tests for get_info_field function."""
    
    def test_top_level_key(self):
        """Should return value when key is at top level."""
        info = {"success": True, "reward": 1.5}
        assert get_info_field(info, "success") == True
        assert get_info_field(info, "reward") == 1.5
    
    def test_final_info_key(self):
        """Should return value when key is nested in final_info."""
        info = {"final_info": {"success": True, "fail": False}}
        assert get_info_field(info, "success") == True
        assert get_info_field(info, "fail") == False
    
    def test_final_info_priority(self):
        """final_info should take priority over top-level (contains episode results)."""
        info = {
            "success": False,
            "final_info": {"success": True}
        }
        # final_info "success" (True) should be returned
        assert get_info_field(info, "success") == True
    
    def test_missing_key_returns_none(self):
        """Should return None for missing keys by default."""
        info = {"other_key": 123}
        assert get_info_field(info, "success") is None
    
    def test_required_key_raises(self):
        """Should raise KeyError when required=True and key missing."""
        info = {"other_key": 123}
        with pytest.raises(KeyError, match="Required info field"):
            get_info_field(info, "success", required=True)
    
    def test_empty_info(self):
        """Should handle empty info dict."""
        assert get_info_field({}, "success") is None


class TestGetRewardComponents:
    """Tests for reward component extraction."""
    
    def test_top_level_components(self):
        """Should extract reward_components from top level."""
        info = {"reward_components": {"approach": 0.5, "grasp": 0.3}}
        result = get_reward_components(info)
        assert result == {"approach": 0.5, "grasp": 0.3}
    
    def test_final_info_components(self):
        """Should extract reward_components from final_info."""
        info = {
            "final_info": {
                "reward_components": {"lift": 1.0}
            }
        }
        result = get_reward_components(info)
        assert result == {"lift": 1.0}
    
    def test_missing_components(self):
        """Should return None when reward_components not present."""
        info = {"other_data": 123}
        assert get_reward_components(info) is None


class TestExtractScalar:
    """Tests for scalar extraction from various types."""
    
    def test_none_returns_none(self):
        """Should return None for None input."""
        assert extract_scalar(None) is None
    
    def test_python_float(self):
        """Should handle Python float."""
        assert extract_scalar(3.14) == 3.14
    
    def test_python_int(self):
        """Should convert int to float."""
        assert extract_scalar(42) == 42.0
    
    def test_torch_tensor(self):
        """Should extract value from torch tensor."""
        t = torch.tensor(2.718)
        assert abs(extract_scalar(t) - 2.718) < 1e-6
    
    def test_numpy_array(self):
        """Should extract value from numpy scalar."""
        arr = np.array(1.618)
        assert abs(extract_scalar(arr) - 1.618) < 1e-6


class TestExtractBool:
    """Tests for boolean extraction."""
    
    def test_none_returns_false(self):
        """Should return False for None."""
        assert extract_bool(None) == False
    
    def test_python_bool(self):
        """Should handle Python bool."""
        assert extract_bool(True) == True
        assert extract_bool(False) == False
    
    def test_batched_tensor(self):
        """Should extract from batched tensor at given index."""
        t = torch.tensor([True, False, True])
        assert extract_bool(t, 0) == True
        assert extract_bool(t, 1) == False
        assert extract_bool(t, 2) == True
    
    def test_scalar_tensor(self):
        """Should extract from scalar tensor."""
        t = torch.tensor(True)
        assert extract_bool(t) == True


class TestAccumulateRewardComponents:
    """Tests for reward component accumulation."""
    
    def test_accumulate_from_empty(self):
        """Should accumulate into empty dict."""
        acc = {}
        reward_comps = {"approach": 0.5, "grasp": 0.3}
        
        inc = accumulate_reward_components(acc, reward_comps)
        
        assert acc == {"approach": 0.5, "grasp": 0.3}
        assert inc == 1
    
    def test_accumulate_multiple(self):
        """Should sum values across multiple calls."""
        acc = {"approach": 0.5}
        reward_comps = {"approach": 0.5, "lift": 1.0}
        
        inc = accumulate_reward_components(acc, reward_comps)
        
        assert acc["approach"] == 1.0
        assert acc["lift"] == 1.0
        assert inc == 1
    
    def test_none_components_noop(self):
        """Should do nothing when components is None."""
        acc = {"existing": 0.5}
        
        inc = accumulate_reward_components(acc, None)
        
        assert acc == {"existing": 0.5}
        assert inc == 0
    
    def test_tensor_values(self):
        """Should handle tensor values."""
        acc = {}
        reward_comps = {
            "approach": torch.tensor(0.5),
            "grasp": torch.tensor(0.3),
        }
        
        inc = accumulate_reward_components(acc, reward_comps)
        
        assert abs(acc["approach"] - 0.5) < 1e-6
        assert abs(acc["grasp"] - 0.3) < 1e-6
        assert inc == 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
