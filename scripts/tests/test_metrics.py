"""
Unit tests for metrics collection and aggregation system.
"""
import pytest
import torch
from typing import Dict

from scripts.training.metrics_utils import get_metric_specs_from_env, aggregate_metrics
from scripts.tasks.base import BaseTaskHandler


class MockTaskHandler(BaseTaskHandler):
    """Mock task handler for testing."""
    
    def __init__(self, env):
        super().__init__(env)
    
    @classmethod
    def _get_train_metrics(cls) -> Dict[str, str]:
        return {
            "train_only_metric": "mean",
            "shared_metric": "mean",
        }
    
    @classmethod
    def _get_eval_metrics(cls) -> Dict[str, str]:
        return {
            "eval_only_metric": "mean",
            "shared_metric": "mean",
        }
    
    def evaluate(self):
        return {"success": torch.tensor([True]), "fail": torch.tensor([False])}
    
    def compute_dense_reward(self, info, action=None):
        return torch.tensor([1.0])
    
    def initialize_episode(self, env_idx, options):
        pass


class MockEnv:
    """Mock environment for testing."""
    
    def __init__(self, task_handler_class=None):
        if task_handler_class:
            self.task_handler = task_handler_class(self)
        self.device = torch.device("cpu")


class TestGetMetricSpecs:
    """Test get_metric_specs_from_env function."""
    
    def test_default_metrics_no_handler(self):
        """Test that default metrics are returned when no task handler exists."""
        env = MockEnv()
        specs = get_metric_specs_from_env(env)
        
        # Should contain default metrics
        assert "success" in specs
        assert "fail" in specs
        assert "return" in specs
        assert specs["success"] == "mean"
    
    def test_train_mode_metrics(self):
        """Test metrics in train mode."""
        env = MockEnv(task_handler_class=MockTaskHandler)
        specs = get_metric_specs_from_env(env, mode="train")
        
        # Should contain default + train-specific
        assert "success" in specs  # Default
        assert "train_only_metric" in specs  # Train-specific
        assert "shared_metric" in specs  # Shared
        assert "eval_only_metric" not in specs  # Not in train
    
    def test_eval_mode_metrics(self):
        """Test metrics in eval mode."""
        env = MockEnv(task_handler_class=MockTaskHandler)
        specs = get_metric_specs_from_env(env, mode="eval")
        
        # Should contain default + eval-specific
        assert "success" in specs  # Default
        assert "eval_only_metric" in specs  # Eval-specific
        assert "shared_metric" in specs  # Shared
        assert "train_only_metric" not in specs  # Not in eval
    
    def test_mode_case_sensitivity(self):
        """Test that mode parameter is case-sensitive."""
        env = MockEnv(task_handler_class=MockTaskHandler)
        
        train_specs = get_metric_specs_from_env(env, mode="train")
        eval_specs = get_metric_specs_from_env(env, mode="eval")
        
        # Different modes should have different specs
        assert "train_only_metric" in train_specs
        assert "train_only_metric" not in eval_specs


class TestAggregateMetrics:
    """Test aggregate_metrics function."""
    
    def test_mean_aggregation(self):
        """Test mean aggregation of metrics."""
        num_steps = 10
        num_envs = 4
        device = torch.device("cpu")
        
        # Create mock metrics storage
        metrics_storage = {
            "done_mask": torch.zeros((num_steps, num_envs), dtype=torch.bool),
            "success": torch.ones((num_steps, num_envs)),
            "reward": torch.arange(num_steps * num_envs, dtype=torch.float32).reshape(num_steps, num_envs),
        }
        
        # Mark some episodes as done (step 5, envs 0 and 2)
        metrics_storage["done_mask"][5, 0] = True
        metrics_storage["done_mask"][5, 2] = True
        
        metric_specs = {
            "success": "mean",
            "reward": "mean",
        }
        
        episode_metrics = {}
        aggregate_metrics(metrics_storage, metric_specs, episode_metrics)
        
        # Should have collected 2 episodes
        assert "success" in episode_metrics
        assert len(episode_metrics["success"]) == 2
        assert episode_metrics["success"] == [1.0, 1.0]
        
        assert "reward" in episode_metrics
        assert len(episode_metrics["reward"]) == 2
    
    def test_sum_aggregation(self):
        """Test sum aggregation of metrics."""
        num_steps = 5
        num_envs = 3
        
        metrics_storage = {
            "done_mask": torch.zeros((num_steps, num_envs), dtype=torch.bool),
            "step_count": torch.ones((num_steps, num_envs)),
        }
        
        # Mark episodes as done
        metrics_storage["done_mask"][2, 0] = True
        metrics_storage["done_mask"][3, 1] = True
        metrics_storage["done_mask"][4, 2] = True
        
        metric_specs = {
            "step_count": "sum",
        }
        
        episode_metrics = {}
        aggregate_metrics(metrics_storage, metric_specs, episode_metrics)
        
        # Should have summed 3 episodes
        assert "step_count" in episode_metrics
        assert episode_metrics["step_count"] == 3.0  # 1 + 1 + 1
    
    def test_no_episodes_completed(self):
        """Test aggregation when no episodes complete."""
        num_steps = 5
        num_envs = 2
        
        metrics_storage = {
            "done_mask": torch.zeros((num_steps, num_envs), dtype=torch.bool),
            "success": torch.ones((num_steps, num_envs)),
        }
        
        metric_specs = {"success": "mean"}
        episode_metrics = {}
        
        aggregate_metrics(metrics_storage, metric_specs, episode_metrics)
        
        # Should not add any metrics (no episodes done)
        assert "success" not in episode_metrics
    
    def test_multiple_done_same_step(self):
        """Test aggregation when multiple envs finish at same step."""
        num_steps = 3
        num_envs = 4
        
        metrics_storage = {
            "done_mask": torch.zeros((num_steps, num_envs), dtype=torch.bool),
            "return": torch.tensor([
                [1.0, 2.0, 3.0, 4.0],
                [1.5, 2.5, 3.5, 4.5],
                [2.0, 3.0, 4.0, 5.0],
            ]),
        }
        
        # All envs finish at step 2
        metrics_storage["done_mask"][2, :] = True
        
        metric_specs = {"return": "mean"}
        episode_metrics = {}
        
        aggregate_metrics(metrics_storage, metric_specs, episode_metrics)
        
        # Should collect all 4 episodes
        assert len(episode_metrics["return"]) == 4
        assert episode_metrics["return"] == [2.0, 3.0, 4.0, 5.0]
    
    def test_gpu_tensor_handling(self):
        """Test that GPU tensors are properly handled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device("cuda")
        num_steps = 3
        num_envs = 2
        
        metrics_storage = {
            "done_mask": torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device),
            "success": torch.ones((num_steps, num_envs), device=device),
        }
        
        metrics_storage["done_mask"][1, 0] = True
        
        metric_specs = {"success": "mean"}
        episode_metrics = {}
        
        aggregate_metrics(metrics_storage, metric_specs, episode_metrics)
        
        # Should transfer to CPU and convert to list
        assert isinstance(episode_metrics["success"], list)
        assert episode_metrics["success"] == [1.0]


class TestDefaultMetricAggregations:
    """Test BaseTaskHandler default metric aggregations."""
    
    def test_default_aggregations_exist(self):
        """Test that default aggregations are defined."""
        assert hasattr(BaseTaskHandler, "DEFAULT_METRIC_AGGREGATIONS")
        assert isinstance(BaseTaskHandler.DEFAULT_METRIC_AGGREGATIONS, dict)
    
    def test_default_aggregations_content(self):
        """Test that default aggregations contain expected metrics."""
        defaults = BaseTaskHandler.DEFAULT_METRIC_AGGREGATIONS
        
        expected = ["success", "fail", "raw_reward", "return", "episode_len", "success_once", "fail_once"]
        for metric in expected:
            assert metric in defaults
            assert defaults[metric] in ["mean", "sum"]


class TestTaskHandlerMetricAPI:
    """Test TaskHandler metric API."""
    
    def test_get_custom_aggregations_train_mode(self):
        """Test getting train metrics."""
        metrics = MockTaskHandler.get_custom_metric_aggregations(mode="train")
        
        assert "train_only_metric" in metrics
        assert "shared_metric" in metrics
        assert "eval_only_metric" not in metrics
    
    def test_get_custom_aggregations_eval_mode(self):
        """Test getting eval metrics."""
        metrics = MockTaskHandler.get_custom_metric_aggregations(mode="eval")
        
        assert "eval_only_metric" in metrics
        assert "shared_metric" in metrics
        assert "train_only_metric" not in metrics
    
    def test_get_custom_aggregations_default_mode(self):
        """Test default mode parameter."""
        # Default should be "train"
        metrics_default = MockTaskHandler.get_custom_metric_aggregations()
        metrics_train = MockTaskHandler.get_custom_metric_aggregations(mode="train")
        
        assert metrics_default == metrics_train
    
    def test_train_metrics_method(self):
        """Test _get_train_metrics method."""
        metrics = MockTaskHandler._get_train_metrics()
        
        assert "train_only_metric" in metrics
        assert "shared_metric" in metrics
    
    def test_eval_metrics_method(self):
        """Test _get_eval_metrics method."""
        metrics = MockTaskHandler._get_eval_metrics()
        
        assert "eval_only_metric" in metrics
        assert "shared_metric" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
