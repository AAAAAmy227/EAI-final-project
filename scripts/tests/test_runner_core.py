"""
Unit tests for PPORunner core methods.
Tests individual methods with minimal mocking.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch


class TestBuildRewardComponentLogs:
    """Test reward component log building."""
    
    def test_build_reward_component_logs_empty(self):
        """Test building logs when no episodes completed."""
        from scripts.training.runner import PPORunner
        
        with patch.object(PPORunner, '__init__', lambda self, cfg, eval_only: None):
            runner = PPORunner(None, eval_only=True)
            runner.episode_metrics = {}
            runner.avg_returns = []
            
            logs = runner._build_reward_component_logs()
            
            assert logs == {}
    
    def test_build_reward_component_logs_with_metrics(self):
        """Test building logs with collected metrics."""
        from scripts.training.runner import PPORunner
        
        with patch.object(PPORunner, '__init__', lambda self, cfg, eval_only: None):
            runner = PPORunner(None, eval_only=True)
            runner.avg_returns = []
            runner.episode_metrics = {
                "success": [True, False, True, True],  # 75% success
                "return": [10.5, 8.2, 12.1, 9.8],
                "grasp_reward": [2.0, 1.5, 2.5, 2.2],
                "lift_reward": [8.5, 6.7, 9.6, 7.6],
            }
            
            logs = runner._build_reward_component_logs()
            
            # Check success rate
            assert "rollout/success_rate" in logs
            assert abs(logs["rollout/success_rate"] - 0.75) < 0.01
            
            # Check return mean
            assert "rollout/return" in logs
            expected_return = np.mean([10.5, 8.2, 12.1, 9.8])
            assert abs(logs["rollout/return"] - expected_return) < 0.01
            
            # Check reward components
            assert "reward/grasp_reward" in logs
            assert "reward/lift_reward" in logs
            
            # Verify cleared
            assert runner.episode_metrics == {}
    
    def test_build_reward_component_logs_boolean_to_rate(self):
        """Test that boolean metrics are converted to rates."""
        from scripts.training.runner import PPORunner
        
        with patch.object(PPORunner, '__init__', lambda self, cfg, eval_only: None):
            runner = PPORunner(None, eval_only=True)
            runner.avg_returns = []
            runner.episode_metrics = {
                "success": [True, True, False, True, False],
                "fail": [False, False, True, False, True],
            }
            
            logs = runner._build_reward_component_logs()
            
            assert abs(logs["rollout/success_rate"] - 0.6) < 0.01
            assert abs(logs["rollout/fail_rate"] - 0.4) < 0.01


class TestBuildEvalLogs:
    """Test evaluation log building."""
    
    def test_build_eval_logs_empty(self):
        """Test building eval logs when no episodes completed."""
        from scripts.training.runner import PPORunner
        
        with patch.object(PPORunner, '__init__', lambda self, cfg, eval_only: None):
            runner = PPORunner(None, eval_only=True)
            runner.episode_metrics = {}
            
            logs = runner._build_eval_logs()
            
            assert logs == {}
    
    def test_build_eval_logs_with_metrics(self):
        """Test building eval logs with collected metrics."""
        from scripts.training.runner import PPORunner
        
        with patch.object(PPORunner, '__init__', lambda self, cfg, eval_only: None):
            runner = PPORunner(None, eval_only=True)
            runner.episode_metrics = {
                "success": [True, True, True, False],
                "return": [12.5, 11.8, 13.2, 9.5],
                "grasp_reward": [2.5, 2.3, 2.7, 1.8],
            }
            
            logs = runner._build_eval_logs()
            
            # Check eval prefix
            assert "eval/success_rate" in logs
            assert "eval/return" in logs
            assert "eval_reward/grasp_reward" in logs
            
            # Verify values
            assert abs(logs["eval/success_rate"] - 0.75) < 0.01
            expected_return = np.mean([12.5, 11.8, 13.2, 9.5])
            assert abs(logs["eval/return"] - expected_return) < 0.01
            
            # Verify cleared
            assert runner.episode_metrics == {}
    
    def test_build_eval_logs_with_all_success(self):
        """Test eval logs with 100% success rate."""
        from scripts.training.runner import PPORunner
        
        with patch.object(PPORunner, '__init__', lambda self, cfg, eval_only: None):
            runner = PPORunner(None, eval_only=True)
            runner.episode_metrics = {
                "success": [True, True, True],
                "return": [15.0, 14.5, 15.2],
            }
            
            logs = runner._build_eval_logs()
            
            # 100% success
            assert logs["eval/success_rate"] == 1.0
            assert abs(logs["eval/return"] - np.mean([15.0, 14.5, 15.2])) < 0.01


class TestAggregateMetrics:
    """Test metrics aggregation."""
    
    def test_aggregate_metrics_mean(self):
        """Test mean aggregation."""
        from scripts.training.runner import PPORunner
        
        with patch.object(PPORunner, '__init__', lambda self, cfg, eval_only: None):
            runner = PPORunner(None, eval_only=True)
            runner.episode_metrics = {}
            
            metrics_storage = {
                "done_mask": torch.tensor([
                    [False, True, False],
                    [True, False, False],
                ], dtype=torch.bool),
                "success": torch.tensor([
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ], dtype=torch.float32),
                "return": torch.tensor([
                    [5.0, 10.0, 3.0],
                    [8.0, 6.0, 4.0],
                ], dtype=torch.float32),
            }
            
            metric_specs = {
                "success": "mean",
                "return": "mean",
            }
            
            runner._aggregate_metrics(metrics_storage, metric_specs)
            
            # 2 completed episodes
            assert len(runner.episode_metrics["success"]) == 2
            assert len(runner.episode_metrics["return"]) == 2
    
    def test_aggregate_metrics_no_done(self):
        """Test aggregation when no episodes complete."""
        from scripts.training.runner import PPORunner
        
        with patch.object(PPORunner, '__init__', lambda self, cfg, eval_only: None):
            runner = PPORunner(None, eval_only=True)
            runner.episode_metrics = {}
            
            metrics_storage = {
                "done_mask": torch.zeros((5, 3), dtype=torch.bool),  # No done
                "success": torch.ones((5, 3)),
                "return": torch.ones((5, 3)) * 10.0,
            }
            
            metric_specs = {"success": "mean", "return": "mean"}
            
            runner._aggregate_metrics(metrics_storage, metric_specs)
            
            # Should add nothing
            assert runner.episode_metrics == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
