"""
PPO Integration Tests.

Tests for component interactions:
1. Rollout collection shape validation
2. Checkpoint save/load consistency
3. Observation normalization behavior
"""
import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from scripts.training.agent import Agent
from scripts.training.ppo_utils import optimized_gae, make_ppo_update_fn


# ============================================================================
# Rollout Shape Validation Tests
# ============================================================================

class TestRolloutShapes:
    """Verify rollout collection produces correct shapes."""
    
    @pytest.fixture
    def setup(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_obs, n_act = 49, 6  # Match your env
        num_envs, num_steps = 8, 50
        agent = Agent(n_obs, n_act, device=device)
        return agent, device, n_obs, n_act, num_envs, num_steps
    
    def test_rollout_collection_shapes(self, setup):
        """Simulate rollout collection and verify all tensor shapes match."""
        agent, device, n_obs, n_act, num_envs, num_steps = setup
        
        # Storage tensors (as in PPORunner)
        obs_buffer = torch.zeros(num_steps, num_envs, n_obs, device=device)
        actions_buffer = torch.zeros(num_steps, num_envs, n_act, device=device)
        logprobs_buffer = torch.zeros(num_steps, num_envs, device=device)
        rewards_buffer = torch.zeros(num_steps, num_envs, device=device)
        terminated_buffer = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=device)
        values_buffer = torch.zeros(num_steps, num_envs, device=device)
        
        # Simulate rollout
        obs = torch.randn(num_envs, n_obs, device=device)
        
        for step in range(num_steps):
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs)
            
            # Store
            obs_buffer[step] = obs
            actions_buffer[step] = action
            logprobs_buffer[step] = logprob
            values_buffer[step] = value.flatten()
            
            # Simulate env step
            reward = torch.randn(num_envs, device=device)
            terminated = torch.rand(num_envs, device=device) < 0.01  # 1% termination
            
            rewards_buffer[step] = reward
            terminated_buffer[step] = terminated
            
            # Next obs
            obs = torch.randn(num_envs, n_obs, device=device)
        
        # Verify shapes
        assert obs_buffer.shape == (num_steps, num_envs, n_obs)
        assert actions_buffer.shape == (num_steps, num_envs, n_act)
        assert logprobs_buffer.shape == (num_steps, num_envs)
        assert rewards_buffer.shape == (num_steps, num_envs)
        assert terminated_buffer.shape == (num_steps, num_envs)
        assert values_buffer.shape == (num_steps, num_envs)
    
    def test_gae_input_output_shapes(self, setup):
        """Verify GAE produces correct output shapes."""
        agent, device, n_obs, n_act, num_envs, num_steps = setup
        
        rewards = torch.randn(num_steps, num_envs, device=device)
        vals = torch.randn(num_steps, num_envs, device=device)
        terminated = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=device)
        next_value = torch.randn(num_envs, device=device)
        next_terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        advantages, returns = optimized_gae(
            rewards, vals, terminated, next_value,
            gamma=0.99, gae_lambda=0.95
        )
        
        assert advantages.shape == (num_steps, num_envs)
        assert returns.shape == (num_steps, num_envs)
    
    def test_flatten_for_ppo_update(self, setup):
        """Verify flattening for minibatch update."""
        agent, device, n_obs, n_act, num_envs, num_steps = setup
        
        # Simulated buffers
        obs_buffer = torch.randn(num_steps, num_envs, n_obs, device=device)
        actions_buffer = torch.randn(num_steps, num_envs, n_act, device=device)
        logprobs_buffer = torch.randn(num_steps, num_envs, device=device)
        advantages = torch.randn(num_steps, num_envs, device=device)
        returns = torch.randn(num_steps, num_envs, device=device)
        vals = torch.randn(num_steps, num_envs, device=device)
        
        # Flatten for PPO
        batch_size = num_steps * num_envs
        b_obs = obs_buffer.view(batch_size, n_obs)
        b_actions = actions_buffer.view(batch_size, n_act)
        b_logprobs = logprobs_buffer.view(batch_size)
        b_advantages = advantages.view(batch_size)
        b_returns = returns.view(batch_size)
        b_vals = vals.view(batch_size)
        
        assert b_obs.shape == (batch_size, n_obs)
        assert b_actions.shape == (batch_size, n_act)
        assert b_logprobs.shape == (batch_size,)
        assert b_advantages.shape == (batch_size,)


# ============================================================================
# Checkpoint Consistency Tests
# ============================================================================

class TestCheckpointConsistency:
    """Verify checkpoint save/load produces identical behavior."""
    
    @pytest.fixture
    def agent_setup(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_obs, n_act = 32, 4
        return n_obs, n_act, device
    
    def test_save_load_produces_same_actions(self, agent_setup):
        """Agent should produce identical actions before and after save/load."""
        n_obs, n_act, device = agent_setup
        
        agent1 = Agent(n_obs, n_act, device=device)
        
        # Get deterministic action before save
        obs = torch.randn(16, n_obs, device=device)
        with torch.no_grad():
            action_before = agent1.get_action(obs, deterministic=True)
            value_before = agent1.get_value(obs)
        
        # Save
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        
        torch.save({"agent": agent1.state_dict()}, ckpt_path)
        
        # Create new agent and load
        agent2 = Agent(n_obs, n_act, device=device)
        ckpt = torch.load(ckpt_path, map_location=device)
        agent2.load_state_dict(ckpt["agent"])
        
        # Get action after load
        with torch.no_grad():
            action_after = agent2.get_action(obs, deterministic=True)
            value_after = agent2.get_value(obs)
        
        # Should be identical
        assert torch.allclose(action_before, action_after), "Actions differ after load"
        assert torch.allclose(value_before, value_after), "Values differ after load"
        
        # Cleanup
        Path(ckpt_path).unlink()
    
    def test_logstd_persisted(self, agent_setup):
        """Actor log_std should be saved and loaded correctly."""
        n_obs, n_act, device = agent_setup
        
        agent1 = Agent(n_obs, n_act, device=device, logstd_init=-1.0)
        
        # Modify logstd
        with torch.no_grad():
            agent1.actor_logstd.fill_(-2.5)
        
        # Save
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        
        torch.save({"agent": agent1.state_dict()}, ckpt_path)
        
        # Load into new agent (with different init)
        agent2 = Agent(n_obs, n_act, device=device, logstd_init=0.0)
        ckpt = torch.load(ckpt_path, map_location=device)
        agent2.load_state_dict(ckpt["agent"])
        
        # Should have loaded value
        assert torch.allclose(agent2.actor_logstd, torch.tensor(-2.5, device=device))
        
        # Cleanup
        Path(ckpt_path).unlink()
    
    def test_optimizer_state_persistence(self, agent_setup):
        """Optimizer state should be saveable and loadable."""
        n_obs, n_act, device = agent_setup
        
        agent = Agent(n_obs, n_act, device=device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
        
        # Do some updates
        for _ in range(5):
            obs = torch.randn(8, n_obs, device=device)
            _, logprob, _, value = agent.get_action_and_value(obs)
            loss = -logprob.mean() + value.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Save
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        
        torch.save({
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, ckpt_path)
        
        # Load
        agent2 = Agent(n_obs, n_act, device=device)
        optimizer2 = torch.optim.Adam(agent2.parameters(), lr=1e-4)
        
        ckpt = torch.load(ckpt_path, map_location=device)
        agent2.load_state_dict(ckpt["agent"])
        optimizer2.load_state_dict(ckpt["optimizer"])
        
        # Should not raise
        assert len(optimizer2.state_dict()["state"]) > 0, "Optimizer state should be loaded"
        
        # Cleanup
        Path(ckpt_path).unlink()


# ============================================================================
# Observation Normalization Tests
# ============================================================================

class TestObservationNormalization:
    """Test observation normalization behavior (simulated)."""
    
    def test_running_mean_std_update(self):
        """Test Welford's online algorithm for running mean/std."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Simulate running stats
        n_obs = 32
        mean = torch.zeros(n_obs, device=device)
        var = torch.ones(n_obs, device=device)
        count = torch.tensor(1e-4, device=device)  # Avoid division by zero
        
        # Update with batches (simplified Welford)
        for _ in range(100):
            batch = torch.randn(64, n_obs, device=device) * 3.0 + 2.0  # mean=2, std=3
            batch_mean = batch.mean(dim=0)
            batch_var = batch.var(dim=0)
            batch_count = batch.shape[0]
            
            delta = batch_mean - mean
            tot_count = count + batch_count
            
            mean = mean + delta * batch_count / tot_count
            m_a = var * count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
            var = M2 / tot_count
            count = tot_count
        
        # After many updates, should converge to true mean/std
        assert abs(mean.mean().item() - 2.0) < 0.5, f"Mean should be ~2.0, got {mean.mean().item()}"
        assert abs(var.mean().sqrt().item() - 3.0) < 0.5, f"Std should be ~3.0, got {var.mean().sqrt().item()}"
    
    def test_normalized_obs_range(self):
        """Normalized observations should have ~0 mean and ~1 std."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Simulated obs with specific mean/std
        raw_obs = torch.randn(1000, 32, device=device) * 5.0 + 3.0
        
        # "Learned" statistics (after warmup)
        mean = raw_obs.mean(dim=0)
        std = raw_obs.std(dim=0) + 1e-8
        
        # Normalize
        normalized = (raw_obs - mean) / std
        
        # Check
        assert abs(normalized.mean().item()) < 0.1
        assert abs(normalized.std().item() - 1.0) < 0.1
    
    def test_obs_clipping(self):
        """Clipped normalized obs should be bounded."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_val = 10.0
        
        # Extreme normalized values
        normalized = torch.randn(1000, 32, device=device) * 100  # Very large
        
        clipped = torch.clamp(normalized, -clip_val, clip_val)
        
        assert clipped.min() >= -clip_val
        assert clipped.max() <= clip_val


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
