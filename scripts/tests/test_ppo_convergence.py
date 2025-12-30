"""
PPO Convergence Tests (Smoke Tests).

End-to-end tests that verify policy improvement on simple tasks.
These are slower but provide high confidence in overall correctness.
"""
import pytest
import torch
import numpy as np

from scripts.training.agent import Agent
from scripts.training.ppo_utils import optimized_gae, make_ppo_update_fn


class MockConfig:
    """Mock config for PPO tests."""
    class PPO:
        clip_coef = 0.2
        clip_vloss = True
        norm_adv = True
        ent_coef = 0.01
        vf_coef = 0.5
        max_grad_norm = 0.5
        logstd_min = -5.0
        logstd_max = 2.0
        gamma = 0.99
        gae_lambda = 0.95
        def get(self, key, default):
            return getattr(self, key, default)
    ppo = PPO()


# ============================================================================
# Simple Environment Simulators
# ============================================================================

class SimpleBandit:
    """
    Simple 1-step bandit: agent must learn to output action=0.5.
    Reward = -|action - 0.5|^2
    """
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.n_obs = 4
        self.n_act = 1
    
    def reset(self):
        return torch.ones(self.num_envs, self.n_obs, device=self.device)
    
    def step(self, action):
        reward = -((action[:, 0] - 0.5) ** 2)
        terminated = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        next_obs = self.reset()
        return next_obs, reward, terminated


class SimpleTarget:
    """
    Multi-step target reaching: agent must output action close to target.
    Observation includes target value.
    Reward = -|action - target|^2
    """
    def __init__(self, num_envs, device, max_steps=10):
        self.num_envs = num_envs
        self.device = device
        self.n_obs = 2  # [current_step/max_steps, target]
        self.n_act = 1
        self.max_steps = max_steps
        self.step_count = None
        self.target = None
    
    def reset(self):
        self.step_count = torch.zeros(self.num_envs, device=self.device)
        self.target = torch.rand(self.num_envs, device=self.device) * 2 - 1  # [-1, 1]
        return self._get_obs()
    
    def _get_obs(self):
        return torch.stack([
            self.step_count / self.max_steps,
            self.target
        ], dim=1)
    
    def step(self, action):
        reward = -((action[:, 0] - self.target) ** 2)
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        
        # Reset terminated envs
        reset_mask = terminated
        if reset_mask.any():
            self.step_count[reset_mask] = 0
            self.target[reset_mask] = torch.rand(reset_mask.sum(), device=self.device) * 2 - 1
        
        return self._get_obs(), reward, terminated


# ============================================================================
# Convergence Tests
# ============================================================================

class TestConvergence:
    """Test that PPO can learn simple tasks."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.mark.slow
    def test_bandit_convergence(self, device):
        """
        PPO should learn to output ~0.5 on simple bandit.
        
        This tests basic policy gradient correctness.
        """
        num_envs = 32
        num_steps = 16
        num_iterations = 150
        
        env = SimpleBandit(num_envs, device)
        agent = Agent(env.n_obs, env.n_act, device=device, logstd_init=-1.0)
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)  # Higher LR for faster convergence
        
        class TinyEntropyConfig(MockConfig):
            class PPO(MockConfig.PPO):
                ent_coef = 0.001
            ppo = PPO()
            
        update_fn = make_ppo_update_fn(agent, optimizer, TinyEntropyConfig())
        
        initial_reward = None
        final_reward = None
        
        for iteration in range(num_iterations):
            # Rollout
            obs = env.reset()
            obs_buffer = []
            action_buffer = []
            logprob_buffer = []
            reward_buffer = []
            value_buffer = []
            
            for step in range(num_steps):
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs)
                
                obs_buffer.append(obs)
                action_buffer.append(action)
                logprob_buffer.append(logprob)
                value_buffer.append(value.flatten())
                
                next_obs, reward, terminated = env.step(action)
                reward_buffer.append(reward)
                obs = next_obs
            
            # Stack
            obs_t = torch.stack(obs_buffer)
            actions_t = torch.stack(action_buffer)
            logprobs_t = torch.stack(logprob_buffer)
            rewards_t = torch.stack(reward_buffer)
            values_t = torch.stack(value_buffer)
            
            # GAE
            with torch.no_grad():
                next_value = agent.get_value(obs).flatten()
            
            advantages, returns = optimized_gae(
                rewards_t, values_t,
                torch.ones_like(rewards_t, dtype=torch.bool),  # All terminal
                next_value,
                gamma=0.99, gae_lambda=0.95
            )
            
            # Flatten and update
            batch_size = num_steps * num_envs
            import tensordict
            container = tensordict.TensorDict({
                "obs": obs_t.view(batch_size, env.n_obs),
                "actions": actions_t.view(batch_size, env.n_act),
                "logprobs": logprobs_t.view(batch_size),
                "advantages": advantages.view(batch_size),
                "returns": returns.view(batch_size),
                "vals": values_t.view(batch_size),
            }, batch_size=[batch_size])
            
            update_fn(container, tensordict_out=tensordict.TensorDict())
            
            # Track rewards
            mean_reward = rewards_t.mean().item()
            if iteration == 0:
                initial_reward = mean_reward
            if iteration == num_iterations - 1:
                final_reward = mean_reward
        
        # Should improve
        assert final_reward > initial_reward, \
            f"Reward should improve: initial={initial_reward:.4f}, final={final_reward:.4f}"
        
        # Should be close to optimal (reward ~ -0 when action ≈ 0.5)
        assert final_reward > -0.15, f"Final reward should be close to 0, got {final_reward:.4f}"
    
    @pytest.mark.slow  
    def test_value_function_learning(self, device):
        """
        Value function should have positive explained variance.
        
        This tests that the critic is learning something useful.
        """
        num_envs = 32
        num_steps = 16
        num_iterations = 100
        
        env = SimpleTarget(num_envs, device, max_steps=10)
        agent = Agent(env.n_obs, env.n_act, device=device, logstd_init=-1.0)
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        update_fn = make_ppo_update_fn(agent, optimizer, MockConfig())
        
        final_explained_var = None
        
        for iteration in range(num_iterations):
            # Rollout
            obs = env.reset()
            obs_buffer = []
            action_buffer = []
            logprob_buffer = []
            reward_buffer = []
            value_buffer = []
            terminated_buffer = []
            
            for step in range(num_steps):
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs)
                
                obs_buffer.append(obs)
                action_buffer.append(action)
                logprob_buffer.append(logprob)
                value_buffer.append(value.flatten())
                
                next_obs, reward, terminated = env.step(action)
                reward_buffer.append(reward)
                terminated_buffer.append(terminated)
                obs = next_obs
            
            # Stack
            values_t = torch.stack(value_buffer)
            rewards_t = torch.stack(reward_buffer)
            terminated_t = torch.stack(terminated_buffer)
            
            # GAE
            with torch.no_grad():
                next_value = agent.get_value(obs).flatten()
            
            advantages, returns = optimized_gae(
                rewards_t, values_t, terminated_t,
                next_value,
                gamma=0.99, gae_lambda=0.95
            )
            
            # Explained variance
            values_flat = values_t.view(-1)
            returns_flat = returns.view(-1)
            var_returns = returns_flat.var()
            explained_var = 1 - (returns_flat - values_flat).var() / (var_returns + 1e-8)
            final_explained_var = explained_var.item()
            
            # Flatten and update
            obs_t = torch.stack(obs_buffer)
            actions_t = torch.stack(action_buffer)
            logprobs_t = torch.stack(logprob_buffer)
            batch_size = num_steps * num_envs
            
            import tensordict
            container = tensordict.TensorDict({
                "obs": obs_t.view(batch_size, env.n_obs),
                "actions": actions_t.view(batch_size, env.n_act),
                "logprobs": logprobs_t.view(batch_size),
                "advantages": advantages.view(batch_size),
                "returns": returns.view(batch_size),
                "vals": values_t.view(batch_size),
            }, batch_size=[batch_size])
            
            update_fn(container, tensordict_out=tensordict.TensorDict())
        
        # Should have positive explained variance (value function is useful)
        assert final_explained_var > 0.0, \
            f"Explained variance should be positive, got {final_explained_var:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
