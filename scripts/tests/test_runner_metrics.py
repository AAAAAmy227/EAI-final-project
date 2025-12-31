import torch
import unittest
from unittest.mock import MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from scripts.training.runner import PPORunner
from scripts.training.info_utils import accumulate_reward_components_gpu

class TestRunnerMetrics(unittest.TestCase):
    def setUp(self):
        # Create a minimal mock of PPORunner
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.runner = MagicMock(spec=PPORunner)
        self.runner.device = self.device
        self.runner.success_count = torch.tensor(0.0, device=self.device)
        self.runner.fail_count = torch.tensor(0.0, device=self.device)
        self.runner.terminated_count = torch.tensor(0, device=self.device, dtype=torch.int64)
        self.runner.truncated_count = torch.tensor(0, device=self.device, dtype=torch.int64)
        self.runner.reward_component_count = 0
        
        # Bind the real _update_metrics method to the mock
        # We need to access the class method directly and bind it to self.runner manually
        # OR just instantiate a dummy runner if init is too complex. 
        # Instantiating PPORunner requires a complex Config, so let's just grab the function code
        # and run it as if it were a method, or monkeypatch.
        
        # Better: use the actual class but mock __init__
        pass

    def test_update_metrics_training(self):
        """Test _update_metrics in training mode."""
        # Setup inputs
        num_envs = 4
        reward = torch.ones(num_envs, device=self.device)
        done = torch.tensor([False, True, False, True], device=self.device) # Envs 1 and 3 done
        terminated = torch.tensor([False, True, False, False], device=self.device) # Env 1 terminated
        # Env 3 is done but not terminated -> truncated
        
        infos = {
            "reward_components": {
                # In real envs (LiftTaskHandler), these are mean scalars
                "comp_a": torch.tensor(0.5, device=self.device),
                "comp_b": torch.tensor(0.2, device=self.device)
            },
            "success_count": torch.tensor(1.0, device=self.device), # sum of successes in batch
            "fail_count": torch.tensor(0.0, device=self.device),
        }
        
        episode_returns = torch.zeros(num_envs, device=self.device)
        # Pre-fill returns to check reset
        episode_returns[1] = 10.0
        episode_returns[3] = 5.0
        
        avg_returns_list = []
        reward_sum_dict = {}
        
        # Call the method (we need to temporarily attach it or just call PPORunner._update_metrics)
        PPORunner._update_metrics(
            self.runner,
            reward, done, terminated, infos,
            episode_returns, avg_returns_list, reward_sum_dict,
            is_training=True
        )
        
        # Verify Metrics - Training specific logic
        # 1. Counts
        self.assertEqual(self.runner.reward_component_count, 1)
        self.assertEqual(self.runner.success_count.item(), 1.0)
        self.assertEqual(self.runner.terminated_count.item(), 1) # Env 1
        self.assertEqual(self.runner.truncated_count.item(), 1) # Env 3
        
        # 2. Reward Components Accumulation (GPU)
        self.assertTrue(torch.is_tensor(reward_sum_dict["comp_a"]))
        self.assertAlmostEqual(reward_sum_dict["comp_a"].item(), 0.5) # Just the scalar value added once 
        # Wait, get_reward_components returns .mean() scalar usually from Track1Env wrappers?
        # Let's check info_utils.py behavior or task handler behavior.
        # In LiftTaskHandler: info["reward_components"]["approach"] = (w * r).mean()
        # So it's a scalar tensor.
        
        # If input was scalar tensor (simulating real env):
        # The mock input above was (num_envs,), but real env (LiftTaskHandler) sends scalar means.
        # Let's adjust test to match reality if validation fails.
        # accumulate_reward_components_gpu expects whatever get_reward_components returns.
        # If it returns a tensor, it sums it.
        
        # 3. Episode returns
        self.assertEqual(len(avg_returns_list), 2)
        # Returns accumulate the current step's reward (1.0) before completion
        self.assertIn(11.0, avg_returns_list) # 10.0 + 1.0
        self.assertIn(6.0, avg_returns_list)  # 5.0 + 1.0
        
        # Returns for done envs should be reset to 0
        self.assertEqual(episode_returns[1].item(), 0.0)
        self.assertEqual(episode_returns[3].item(), 0.0)
        # Returns for active envs should have reward added
        self.assertEqual(episode_returns[0].item(), 1.0) # 0 + 1.0

    def test_update_metrics_eval(self):
        """Test _update_metrics in eval mode."""
        # Setup inputs
        num_envs = 4
        reward = torch.ones(num_envs, device=self.device)
        done = torch.tensor([False, True, False, True], device=self.device) # Envs 1 and 3 done
        terminated = torch.tensor([False, True, False, False], device=self.device)
        
        # Eval mode often has per-env components, but standard reward_components are also present
        infos = {
            "reward_components": {
                "comp_a": torch.tensor(0.5, device=self.device), # Scalar mean
            },
            # Vectorized success/fail fields for list collection
            "success": torch.tensor([False, True, False, False], device=self.device),
            "fail": torch.tensor([False, False, False, True], device=self.device)
        }
        
        episode_returns = torch.zeros(num_envs, device=self.device)
        episode_returns[1] = 10.0
        episode_returns[3] = 5.0
        
        avg_returns_list = []
        reward_sum_dict = {}
        successes_list = []
        fails_list = []
        
        PPORunner._update_metrics(
            self.runner,
            reward, done, terminated, infos,
            episode_returns, avg_returns_list, reward_sum_dict,
            is_training=False,
            successes_list=successes_list,
            fails_list=fails_list
        )
        
        # Verify Metrics - Eval specific
        # 1. NO Training counters updated
        self.assertEqual(self.runner.reward_component_count, 0)
        
        # 2. Reward Components Accumulation (GPU now!)
        self.assertTrue(torch.is_tensor(reward_sum_dict["comp_a"]))
        self.assertAlmostEqual(reward_sum_dict["comp_a"].item(), 0.5)
        
        # 3. Success/Fail List Collection
        self.assertEqual(len(successes_list), 2) # 2 done envs
        self.assertEqual(successes_list[0], True) # Env 1
        self.assertEqual(successes_list[1], False) # Env 3
        
        self.assertEqual(len(fails_list), 2)
        self.assertEqual(fails_list[0], False) # Env 1
        self.assertEqual(fails_list[1], True) # Env 3

if __name__ == "__main__":
    unittest.main()
