"""
Unit tests for TaskHandler classes.
"""
import pytest
import torch
from typing import Dict

from scripts.tasks.base import BaseTaskHandler
from scripts.tasks.lift import LiftTaskHandler


class MockEnv:
    """Mock environment for testing task handlers."""
    
    def __init__(self):
        self.device = torch.device("cpu")
        self.num_envs = 4
        
        # Mock attributes that task handlers might need
        self.cube_pos = torch.zeros((self.num_envs, 3))
        self.cube_vel = torch.zeros((self.num_envs, 3))
        self.tcp_pose = self._create_mock_pose()
    
    def _create_mock_pose(self):
        """Create mock TCP pose."""
        class MockPose:
            def __init__(self, num_envs):
                self.p = torch.zeros((num_envs, 3))
                self.q = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs)
        
        return MockPose(self.num_envs)


class DummyTaskHandler(BaseTaskHandler):
    """Dummy task handler for testing base class."""
    
    def evaluate(self) -> Dict[str, torch.Tensor]:
        """Dummy evaluation."""
        return {
            "success": torch.tensor([True, False, True, False]),
            "fail": torch.tensor([False, True, False, True]),
        }
    
    def compute_dense_reward(self, info: Dict, action=None) -> torch.Tensor:
        """Dummy reward computation."""
        return torch.ones(self.env.num_envs)
    
    def initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        """Dummy initialization."""
        pass


class TestBaseTaskHandler:
    """Test BaseTaskHandler abstract class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseTaskHandler cannot be instantiated directly."""
        env = MockEnv()
        
        with pytest.raises(TypeError):
            BaseTaskHandler(env)
    
    def test_default_metric_aggregations(self):
        """Test that DEFAULT_METRIC_AGGREGATIONS is properly defined."""
        defaults = BaseTaskHandler.DEFAULT_METRIC_AGGREGATIONS
        
        assert isinstance(defaults, dict)
        assert len(defaults) > 0
        
        # Check expected metrics
        expected_metrics = ["success", "fail", "raw_reward", "return", "episode_len"]
        for metric in expected_metrics:
            assert metric in defaults
            assert defaults[metric] in ["mean", "sum"]
    
    def test_get_custom_metric_aggregations_default(self):
        """Test default implementation of get_custom_metric_aggregations."""
        # Base class should provide empty train/eval metrics
        train_metrics = BaseTaskHandler._get_train_metrics()
        eval_metrics = BaseTaskHandler._get_eval_metrics()
        
        assert isinstance(train_metrics, dict)
        assert isinstance(eval_metrics, dict)
        assert len(train_metrics) == 0  # Default is empty
    
    def test_get_train_metrics_default(self):
        """Test _get_train_metrics default implementation."""
        metrics = BaseTaskHandler._get_train_metrics()
        
        assert isinstance(metrics, dict)
        assert len(metrics) == 0
    
    def test_get_eval_metrics_default(self):
        """Test _get_eval_metrics default implementation."""
        # Default should return same as train
        train_metrics = BaseTaskHandler._get_train_metrics()
        eval_metrics = BaseTaskHandler._get_eval_metrics()
        
        assert eval_metrics == train_metrics
    
    def test_mode_parameter_train(self):
        """Test get_custom_metric_aggregations with mode='train'."""
        metrics = BaseTaskHandler.get_custom_metric_aggregations(mode="train")
        
        assert isinstance(metrics, dict)
    
    def test_mode_parameter_eval(self):
        """Test get_custom_metric_aggregations with mode='eval'."""
        metrics = BaseTaskHandler.get_custom_metric_aggregations(mode="eval")
        
        assert isinstance(metrics, dict)
    
    def test_dummy_handler_instantiation(self):
        """Test that a concrete subclass can be instantiated."""
        env = MockEnv()
        handler = DummyTaskHandler(env)
        
        assert handler.env is env
        assert handler.device == env.device
    
    def test_dummy_handler_evaluate(self):
        """Test evaluate method of dummy handler."""
        env = MockEnv()
        handler = DummyTaskHandler(env)
        
        result = handler.evaluate()
        
        assert "success" in result
        assert "fail" in result
        assert isinstance(result["success"], torch.Tensor)
        assert len(result["success"]) == env.num_envs
    
    def test_dummy_handler_compute_dense_reward(self):
        """Test compute_dense_reward method of dummy handler."""
        env = MockEnv()
        handler = DummyTaskHandler(env)
        
        info = {}
        reward = handler.compute_dense_reward(info)
        
        assert isinstance(reward, torch.Tensor)
        assert len(reward) == env.num_envs


class TestLiftTaskHandler:
    """Test LiftTaskHandler implementation."""
    
    def test_train_metrics_definition(self):
        """Test that LiftTaskHandler defines train metrics."""
        metrics = LiftTaskHandler._get_train_metrics()
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        
        # Expected Lift metrics
        expected = ["grasp_reward", "lift_reward", "moving_distance", "grasp_success", "lift_success"]
        for metric in expected:
            assert metric in metrics
            assert metrics[metric] in ["mean", "sum"]
    
    def test_eval_metrics_same_as_train(self):
        """Test that eval metrics default to train metrics."""
        train_metrics = LiftTaskHandler._get_train_metrics()
        eval_metrics = LiftTaskHandler._get_eval_metrics()
        
        assert eval_metrics == train_metrics
    
    def test_get_custom_aggregations_train_mode(self):
        """Test getting custom aggregations in train mode."""
        metrics = LiftTaskHandler.get_custom_metric_aggregations(mode="train")
        
        assert "grasp_reward" in metrics
        assert "lift_reward" in metrics
    
    def test_get_custom_aggregations_eval_mode(self):
        """Test getting custom aggregations in eval mode."""
        metrics = LiftTaskHandler.get_custom_metric_aggregations(mode="eval")
        
        # Should have same metrics as train (default behavior)
        assert "grasp_reward" in metrics
        assert "lift_reward" in metrics
    
    def test_metric_aggregation_types(self):
        """Test that all Lift metrics use 'mean' aggregation."""
        metrics = LiftTaskHandler._get_train_metrics()
        
        for metric_name, agg_type in metrics.items():
            assert agg_type == "mean", f"{metric_name} should use 'mean' aggregation"
    
    def test_instantiation(self):
        """Test that LiftTaskHandler can be instantiated."""
        env = MockEnv()
        handler = LiftTaskHandler(env)
        
        assert handler.env is env
        assert hasattr(handler, 'device')
    
    def test_has_required_methods(self):
        """Test that LiftTaskHandler implements required abstract methods."""
        assert hasattr(LiftTaskHandler, 'evaluate')
        assert hasattr(LiftTaskHandler, 'compute_dense_reward')
        assert hasattr(LiftTaskHandler, 'initialize_episode')
    
    def test_initialization_attributes(self):
        """Test that LiftTaskHandler initializes expected attributes."""
        env = MockEnv()
        handler = LiftTaskHandler(env)
        
        # Check adaptive state attributes
        assert hasattr(handler, 'grasp_success_rate')
        assert hasattr(handler, 'lift_success_rate')
        assert hasattr(handler, 'task_success_rate')


class TestModeSpecificMetrics:
    """Test mode-specific metrics functionality."""
    
    class TrainHeavyHandler(BaseTaskHandler):
        """Handler with more train metrics than eval."""
        
        @classmethod
        def _get_train_metrics(cls):
            return {
                "debug_metric_1": "mean",
                "debug_metric_2": "mean",
                "shared_metric": "mean",
            }
        
        @classmethod
        def _get_eval_metrics(cls):
            return {
                "shared_metric": "mean",
            }
        
        def evaluate(self):
            return {"success": torch.tensor([True]), "fail": torch.tensor([False])}
        
        def compute_dense_reward(self, info, action=None):
            return torch.tensor([1.0])
        
        def initialize_episode(self, env_idx, options):
            pass
    
    class EvalHeavyHandler(BaseTaskHandler):
        """Handler with more eval metrics than train."""
        
        @classmethod
        def _get_train_metrics(cls):
            return {
                "core_metric": "mean",
            }
        
        @classmethod
        def _get_eval_metrics(cls):
            return {
                "core_metric": "mean",
                "detailed_metric_1": "mean",
                "detailed_metric_2": "mean",
                "detailed_metric_3": "mean",
            }
        
        def evaluate(self):
            return {"success": torch.tensor([True]), "fail": torch.tensor([False])}
        
        def compute_dense_reward(self, info, action=None):
            return torch.tensor([1.0])
        
        def initialize_episode(self, env_idx, options):
            pass
    
    def test_train_heavy_handler_train_mode(self):
        """Test handler with more train metrics in train mode."""
        metrics = self.TrainHeavyHandler.get_custom_metric_aggregations(mode="train")
        
        assert "debug_metric_1" in metrics
        assert "debug_metric_2" in metrics
        assert "shared_metric" in metrics
    
    def test_train_heavy_handler_eval_mode(self):
        """Test handler with more train metrics in eval mode."""
        metrics = self.TrainHeavyHandler.get_custom_metric_aggregations(mode="eval")
        
        assert "debug_metric_1" not in metrics
        assert "debug_metric_2" not in metrics
        assert "shared_metric" in metrics
    
    def test_eval_heavy_handler_train_mode(self):
        """Test handler with more eval metrics in train mode."""
        metrics = self.EvalHeavyHandler.get_custom_metric_aggregations(mode="train")
        
        assert "core_metric" in metrics
        assert "detailed_metric_1" not in metrics
    
    def test_eval_heavy_handler_eval_mode(self):
        """Test handler with more eval metrics in eval mode."""
        metrics = self.EvalHeavyHandler.get_custom_metric_aggregations(mode="eval")
        
        assert "core_metric" in metrics
        assert "detailed_metric_1" in metrics
        assert "detailed_metric_2" in metrics
        assert "detailed_metric_3" in metrics
    
    def test_mode_isolation(self):
        """Test that train and eval metrics are properly isolated."""
        train_metrics = self.EvalHeavyHandler.get_custom_metric_aggregations(mode="train")
        eval_metrics = self.EvalHeavyHandler.get_custom_metric_aggregations(mode="eval")
        
        # Eval should have more metrics
        assert len(eval_metrics) > len(train_metrics)
        
        # Core metric should be in both
        assert "core_metric" in train_metrics
        assert "core_metric" in eval_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
