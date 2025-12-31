"""
Unit tests for runner_utils pure functions.
Tests computation logic without requiring PPORunner instances.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from scripts.training.runner_utils import (
    compute_reward_logs,
    compute_eval_logs,
    build_csv_path,
    write_csv_file,
    extract_metric_from_info,
)


class TestComputeRewardLogs:
    """Test pure function for computing reward logs."""
    
    def test_empty_metrics(self):
        """Test with empty metrics dict."""
        logs = compute_reward_logs({})
        assert logs == {}
    
    def test_basic_metrics(self):
        """Test with basic metrics."""
        metrics = {
            "success": [True, False, True, True],  # 75% success
            "return": [10.5, 8.2, 12.1, 9.8],
            "grasp_reward": [2.0, 1.5, 2.5, 2.2],
        }
        
        logs = compute_reward_logs(metrics)
        
        # Check success rate
        assert "rollout/success_rate" in logs
        assert abs(logs["rollout/success_rate"] - 0.75) < 0.01
        
        # Check return mean
        assert "rollout/return" in logs
        expected_return = np.mean([10.5, 8.2, 12.1, 9.8])
        assert abs(logs["rollout/return"] - expected_return) < 0.01
        
        # Check reward component
        assert "reward/grasp_reward" in logs
        assert abs(logs["reward/grasp_reward"] - 2.05) < 0.01
    
    def test_boolean_to_rate_conversion(self):
        """Test that boolean metrics are converted to rates."""
        metrics = {
            "success": [True, True, False, True, False],
            "fail": [False, False, True, False, True],
        }
        
        logs = compute_reward_logs(metrics)
        
        assert abs(logs["rollout/success_rate"] - 0.6) < 0.01
        assert abs(logs["rollout/fail_rate"] - 0.4) < 0.01
    
    def test_raw_reward_prefix(self):
        """Test raw_reward gets special prefix."""
        metrics = {"raw_reward": [1.0, 2.0, 3.0]}
        logs = compute_reward_logs(metrics)
        assert "rollout/raw_reward_mean" in logs
        assert abs(logs["rollout/raw_reward_mean"] - 2.0) < 0.01
    
    def test_empty_values_skipped(self):
        """Test metrics with empty values are skipped."""
        metrics = {
            "success": [True, False],
            "empty_metric": [],
        }
        logs = compute_reward_logs(metrics)
        assert "rollout/success_rate" in logs
        assert "empty_metric" not in str(logs)


class TestComputeEvalLogs:
    """Test pure function for computing eval logs."""
    
    def test_empty_metrics(self):
        """Test with empty metrics dict."""
        logs = compute_eval_logs({})
        assert logs == {}
    
    def test_basic_eval_metrics(self):
        """Test with basic eval metrics."""
        metrics = {
            "success": [True, True, True, False],
            "return": [12.5, 11.8, 13.2, 9.5],
            "grasp_reward": [2.5, 2.3, 2.7, 1.8],
        }
        
        logs = compute_eval_logs(metrics)
        
        # Check eval prefix
        assert "eval/success_rate" in logs
        assert "eval/return" in logs
        assert "eval_reward/grasp_reward" in logs
        
        # Verify values
        assert abs(logs["eval/success_rate"] - 0.75) < 0.01
        expected_return = np.mean([12.5, 11.8, 13.2, 9.5])
        assert abs(logs["eval/return"] - expected_return) < 0.01
    
    def test_all_success(self):
        """Test with 100% success rate."""
        metrics = {
            "success": [True, True, True],
            "return": [15.0, 14.5, 15.2],
        }
        
        logs = compute_eval_logs(metrics)
        
        assert logs["eval/success_rate"] == 1.0
        assert abs(logs["eval/return"] - np.mean([15.0, 14.5, 15.2])) < 0.01
    
    def test_fail_rate(self):
        """Test fail rate calculation."""
        metrics = {
            "fail": [True, False, True, False],
        }
        
        logs = compute_eval_logs(metrics)
        assert abs(logs["eval/fail_rate"] - 0.5) < 0.01
    
    def test_episode_len_in_eval(self):
        """Test episode_len gets eval/ prefix."""
        metrics = {"episode_len": [100, 120, 110]}
        logs = compute_eval_logs(metrics)
        assert "eval/episode_len" in logs
        assert abs(logs["eval/episode_len"] - 110) < 0.1


class TestBuildCsvPath:
    """Test CSV path building function."""
    
    def test_basic_path(self):
        """Test basic path construction."""
        path = build_csv_path(Path("/tmp"), "eval0", 5)
        assert path == Path("/tmp/split/eval0/env5/rewards.csv")
    
    def test_different_eval_names(self):
        """Test with different eval names."""
        path1 = build_csv_path(Path("/output"), "eval1", 0)
        path2 = build_csv_path(Path("/output"), "eval2", 0)
        
        assert path1 == Path("/output/split/eval1/env0/rewards.csv")
        assert path2 == Path("/output/split/eval2/env0/rewards.csv")
    
    def test_multiple_envs(self):
        """Test with multiple environment indices."""
        paths = [build_csv_path(Path("/tmp"), "eval0", i) for i in range(3)]
        
        assert paths[0] == Path("/tmp/split/eval0/env0/rewards.csv")
        assert paths[1] == Path("/tmp/split/eval0/env1/rewards.csv")
        assert paths[2] == Path("/tmp/split/eval0/env2/rewards.csv")


class TestWriteCsvFile:
    """Test CSV writing function."""
    
    def test_write_basic_csv(self, tmp_path):
        """Test writing basic CSV data."""
        data = [
            {"step": 0, "reward": 1.0},
            {"step": 1, "reward": 2.0},
        ]
        filepath = tmp_path / "test.csv"
        
        write_csv_file(filepath, data)
        
        # Verify file was created
        assert filepath.exists()
        
        # Verify contents
        import csv
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["step"] == "0"
            assert rows[0]["reward"] == "1.0"
            assert rows[1]["step"] == "1"
            assert rows[1]["reward"] == "2.0"
    
    def test_write_empty_data(self, tmp_path):
        """Test that empty data doesn't create file."""
        filepath = tmp_path / "test.csv"
        write_csv_file(filepath, [])
        
        # File should not exist
        assert not filepath.exists()
    
    def test_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created."""
        filepath = tmp_path / "deep" / "nested" / "dir" / "test.csv"
        data = [{"col1": "value1"}]
        
        write_csv_file(filepath, data)
        
        assert filepath.exists()
        assert filepath.parent.exists()
    
    def test_multiple_columns(self, tmp_path):
        """Test writing CSV with multiple columns."""
        data = [
            {"step": 0, "reward": 1.0, "success": True, "metric": 0.5},
            {"step": 1, "reward": 2.0, "success": False, "metric": 0.7},
        ]
        filepath = tmp_path / "test.csv"
        
        write_csv_file(filepath, data)
        
        import csv
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert len(rows[0]) == 4  # 4 columns


class TestExtractMetricFromInfo:
    """Test metric extraction from info dict."""
    
    def test_extract_from_episode_dict(self):
        """Test extracting metrics from nested episode dict."""
        info = {
            "episode": {
                "return": 10.5,
                "success_once": True,
                "episode_len": 100,
            }
        }
        
        assert extract_metric_from_info(info, "return") == 10.5
        assert extract_metric_from_info(info, "success_once") == 1.0
        assert extract_metric_from_info(info, "episode_len") == 100.0
    
    def test_extract_from_top_level(self):
        """Test extracting metrics from top level of info dict."""
        info = {
            "custom_metric": 5.5,
            "another_value": 3.2,
        }
        
        assert extract_metric_from_info(info, "custom_metric") == 5.5
        assert extract_metric_from_info(info, "another_value") == 3.2
    
    def test_metric_not_found(self):
        """Test extracting non-existent metric."""
        info = {"episode": {"return": 10.5}}
        
        result = extract_metric_from_info(info, "nonexistent")
        assert result is None
    
    def test_extract_torch_tensor(self):
        """Test extracting torch tensor values."""
        info = {
            "episode": {
                "return": torch.tensor(15.0),
            },
            "tensor_metric": torch.tensor(7.5),
        }
        
        assert extract_metric_from_info(info, "return") == 15.0
        assert extract_metric_from_info(info, "tensor_metric") == 7.5
    
    def test_episode_dict_priority(self):
        """Test that episode dict has priority for special metrics."""
        info = {
            "return": 5.0,
            "episode": {
                "return": 10.0,
            }
        }
        
        # Should extract from episode dict for special metrics
        assert extract_metric_from_info(info, "return") == 10.0
    
    def test_empty_info_dict(self):
        """Test with empty info dict."""
        result = extract_metric_from_info({}, "return")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
