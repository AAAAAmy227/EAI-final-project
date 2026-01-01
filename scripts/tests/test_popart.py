"""
Unit tests for PopArt (Preserving Outputs Precisely, while Adaptively Rescaling Targets) value normalization.

Tests cover:
1. PopArtValueHead core functionality
2. Output preservation after stats update
3. Normalize/denormalize inverse relationship
4. Agent integration with PopArt
5. Weight preservation invariant
"""
import pytest
import torch
import numpy as np


class TestPopArtValueHead:
    """Test PopArtValueHead core functionality."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def popart_head(self, device):
        from scripts.training.value_normalization import PopArtValueHead
        return PopArtValueHead(in_features=256, device=device, beta=0.001)
    
    def test_forward_shape(self, popart_head):
        """Forward pass should preserve batch dimension."""
        features = torch.randn(32, 256, device=popart_head.mu.device)
        output = popart_head(features)
        assert output.shape == (32, 1)
    
    def test_denormalize_shape(self, popart_head):
        """Denormalize should preserve shape."""
        normalized = torch.randn(32, 1, device=popart_head.mu.device)
        denormalized = popart_head.denormalize(normalized)
        assert denormalized.shape == normalized.shape
    
    def test_normalize_target_shape(self, popart_head):
        """Normalize target should handle various shapes."""
        # 1D input
        targets_1d = torch.randn(32, device=popart_head.mu.device)
        normalized_1d = popart_head.normalize_target(targets_1d)
        assert normalized_1d.shape == targets_1d.shape
        
        # 2D input
        targets_2d = torch.randn(32, 1, device=popart_head.mu.device)
        normalized_2d = popart_head.normalize_target(targets_2d)
        assert normalized_2d.shape == targets_2d.shape
    
    def test_initial_stats(self, popart_head):
        """Initial stats should be mu=0, sigma=1."""
        assert torch.allclose(popart_head.mu, torch.zeros_like(popart_head.mu))
        assert torch.allclose(popart_head.sigma, torch.ones_like(popart_head.sigma))
    
    def test_stats_update(self, device):
        """Stats should update toward batch statistics."""
        from scripts.training.value_normalization import PopArtValueHead
        
        # Use larger beta for faster convergence in test
        head = PopArtValueHead(256, device=device, beta=0.01)
        
        # Create targets with known mean and std
        targets = torch.randn(1000, device=device) * 50 + 100
        
        # Update multiple times to converge
        for _ in range(200):  # More updates needed for convergence
            head.update_stats(targets)
        
        # Should be reasonably close to target statistics (EMA converges slowly)
        assert abs(head.mu.item() - 100) < 15, f"mu={head.mu.item()}, expected ~100"
        assert abs(head.sigma.item() - 50) < 15, f"sigma={head.sigma.item()}, expected ~50"
    
    def test_output_preservation_single_update(self, popart_head, device):
        """Denormalized output should be preserved after single stats update."""
        features = torch.randn(32, 256, device=device)
        
        # Get output before update
        with torch.no_grad():
            output_before = popart_head.denormalize(popart_head(features))
        
        # Update stats with some returns
        returns = torch.randn(64, device=device) * 100 + 50
        popart_head.update_stats(returns)
        
        # Get output after update
        with torch.no_grad():
            output_after = popart_head.denormalize(popart_head(features))
        
        # Outputs should be very close
        diff = (output_before - output_after).abs().mean()
        assert diff < 0.01, f"Output preservation error: {diff.item()}"
    
    def test_output_preservation_many_updates(self, popart_head, device):
        """Denormalized output should remain stable through many updates."""
        features = torch.randn(32, 256, device=device)
        
        # Get baseline output
        with torch.no_grad():
            output_baseline = popart_head.denormalize(popart_head(features))
        
        # Apply many updates with varying returns
        for i in range(50):
            mean = (i - 25) * 10  # Range from -250 to +240
            std = 50 + i
            returns = torch.randn(64, device=device) * std + mean
            popart_head.update_stats(returns)
        
        # Get final output
        with torch.no_grad():
            output_final = popart_head.denormalize(popart_head(features))
        
        # Should still be close to baseline
        diff = (output_baseline - output_final).abs().mean()
        assert diff < 1.0, f"Output drift after many updates: {diff.item()}"
    
    def test_normalize_denormalize_inverse(self, popart_head, device):
        """Normalize and denormalize should be approximate inverses."""
        # First update stats so they're non-trivial
        returns = torch.randn(100, device=device) * 50 + 25
        for _ in range(10):
            popart_head.update_stats(returns)
        
        # Test inverse relationship
        original = torch.randn(32, device=device) * 100
        normalized = popart_head.normalize_target(original)
        denormalized = popart_head.denormalize(normalized.unsqueeze(-1)).squeeze(-1)
        
        # Should recover original (allow for float precision)
        assert torch.allclose(original, denormalized, atol=0.01), \
            f"Inverse error: {(original - denormalized).abs().max().item()}"
    
    def test_get_stats_returns_dict(self, popart_head, device):
        """get_stats should return a dictionary with expected keys."""
        stats = popart_head.get_stats()
        
        assert isinstance(stats, dict)
        assert "popart/mu" in stats
        assert "popart/sigma" in stats
        assert "popart/nu" in stats
        assert "popart/update_count" in stats
    
    def test_update_count_increments(self, popart_head, device):
        """Update count should increment with each update."""
        initial_count = popart_head.update_count.item()
        
        returns = torch.randn(32, device=device)
        popart_head.update_stats(returns)
        
        assert popart_head.update_count.item() == initial_count + 1


class TestAgentWithPopArt:
    """Test Agent integration with PopArt."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_agent_with_popart_creation(self, device):
        """Agent should create successfully with use_popart=True."""
        from scripts.training.agent import Agent
        
        agent = Agent(n_obs=32, n_act=4, device=device, use_popart=True)
        
        assert agent.use_popart is True
        assert agent.get_popart_head() is not None
    
    def test_agent_without_popart_creation(self, device):
        """Agent should create successfully with use_popart=False."""
        from scripts.training.agent import Agent
        
        agent = Agent(n_obs=32, n_act=4, device=device, use_popart=False)
        
        assert agent.use_popart is False
        assert agent.get_popart_head() is None
    
    def test_get_value_returns_normalized(self, device):
        """get_value should return normalized value when PopArt enabled."""
        from scripts.training.agent import Agent
        
        agent = Agent(n_obs=32, n_act=4, device=device, use_popart=True)
        obs = torch.randn(16, 32, device=device)
        
        value = agent.get_value(obs)
        assert value.shape == (16, 1)
    
    def test_get_value_denormalized_returns_original_scale(self, device):
        """get_value_denormalized should return values in original scale."""
        from scripts.training.agent import Agent
        
        agent = Agent(n_obs=32, n_act=4, device=device, use_popart=True)
        obs = torch.randn(16, 32, device=device)
        
        # Update PopArt stats with large returns
        popart_head = agent.get_popart_head()
        returns = torch.randn(100, device=device) * 1000 + 500
        for _ in range(50):
            popart_head.update_stats(returns)
        
        # Now compare normalized vs denormalized
        v_norm = agent.get_value(obs)
        v_denorm = agent.get_value_denormalized(obs)
        
        # Denormalized should have larger scale
        assert v_denorm.abs().mean() > v_norm.abs().mean() * 10, \
            "Denormalized values should have much larger scale after stats update"
    
    def test_get_value_denormalized_equals_get_value_without_popart(self, device):
        """Without PopArt, get_value_denormalized should equal get_value."""
        from scripts.training.agent import Agent
        
        agent = Agent(n_obs=32, n_act=4, device=device, use_popart=False)
        obs = torch.randn(16, 32, device=device)
        
        v1 = agent.get_value(obs)
        v2 = agent.get_value_denormalized(obs)
        
        assert torch.allclose(v1, v2)
    
    def test_critic_property_backward_compatible(self, device):
        """The critic property should work like the old nn.Sequential."""
        from scripts.training.agent import Agent
        
        agent = Agent(n_obs=32, n_act=4, device=device, use_popart=True)
        obs = torch.randn(16, 32, device=device)
        
        # The critic property should be callable and return same as get_value
        value_from_critic = agent.critic(obs)
        value_from_get_value = agent.get_value(obs)
        
        assert torch.allclose(value_from_critic, value_from_get_value)
    
    def test_get_action_and_value_with_popart(self, device):
        """get_action_and_value should work with PopArt enabled."""
        from scripts.training.agent import Agent
        
        agent = Agent(n_obs=32, n_act=4, device=device, use_popart=True)
        obs = torch.randn(16, 32, device=device)
        
        action, logprob, entropy, value = agent.get_action_and_value(obs)
        
        assert action.shape == (16, 4)
        assert logprob.shape == (16,)
        assert entropy.shape == (16,)
        assert value.shape == (16, 1)
    
    def test_state_dict_includes_popart_buffers(self, device):
        """Agent state_dict should include PopArt buffers."""
        from scripts.training.agent import Agent
        
        agent = Agent(n_obs=32, n_act=4, device=device, use_popart=True)
        
        # Update stats
        popart_head = agent.get_popart_head()
        returns = torch.randn(100, device=device) * 50 + 25
        for _ in range(10):
            popart_head.update_stats(returns)
        
        state_dict = agent.state_dict()
        
        # Should contain PopArt buffers
        assert "critic_head.mu" in state_dict
        assert "critic_head.sigma" in state_dict
        assert "critic_head.nu" in state_dict
    
    def test_load_state_dict_restores_popart(self, device):
        """Loading state_dict should restore PopArt statistics."""
        from scripts.training.agent import Agent
        
        agent1 = Agent(n_obs=32, n_act=4, device=device, use_popart=True)
        
        # Update stats significantly
        popart_head = agent1.get_popart_head()
        returns = torch.randn(100, device=device) * 100 + 200
        for _ in range(50):
            popart_head.update_stats(returns)
        
        original_mu = popart_head.mu.clone()
        original_sigma = popart_head.sigma.clone()
        
        # Save state
        state_dict = agent1.state_dict()
        
        # Create new agent and load
        agent2 = Agent(n_obs=32, n_act=4, device=device, use_popart=True)
        agent2.load_state_dict(state_dict)
        
        # Check stats match
        popart_head2 = agent2.get_popart_head()
        assert torch.allclose(popart_head2.mu, original_mu)
        assert torch.allclose(popart_head2.sigma, original_sigma)


class TestPopArtNumericalStability:
    """Test numerical stability of PopArt under edge cases."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_very_large_returns(self, device):
        """PopArt should handle very large returns without NaN/Inf."""
        from scripts.training.value_normalization import PopArtValueHead
        
        head = PopArtValueHead(256, device=device)
        features = torch.randn(32, 256, device=device)
        
        # Very large returns
        returns = torch.randn(64, device=device) * 1e6
        head.update_stats(returns)
        
        output = head(features)
        assert torch.isfinite(output).all(), "Output should be finite for large returns"
        
        denorm = head.denormalize(output)
        assert torch.isfinite(denorm).all(), "Denormalized output should be finite"
    
    def test_very_small_returns(self, device):
        """PopArt should handle very small returns without numerical issues."""
        from scripts.training.value_normalization import PopArtValueHead
        
        head = PopArtValueHead(256, device=device)
        features = torch.randn(32, 256, device=device)
        
        # Very small returns
        returns = torch.randn(64, device=device) * 1e-6
        head.update_stats(returns)
        
        output = head(features)
        assert torch.isfinite(output).all(), "Output should be finite for small returns"
    
    def test_constant_returns(self, device):
        """PopArt should handle constant returns (zero variance)."""
        from scripts.training.value_normalization import PopArtValueHead
        
        head = PopArtValueHead(256, device=device)
        features = torch.randn(32, 256, device=device)
        
        # Constant returns (zero variance)
        returns = torch.ones(64, device=device) * 42.0
        head.update_stats(returns)
        
        output = head(features)
        assert torch.isfinite(output).all(), "Output should be finite for constant returns"
        
        # Sigma should be clamped to epsilon, not zero
        assert head.sigma.item() > 0, "Sigma should be positive even for constant returns"
    
    def test_mixed_sign_returns(self, device):
        """PopArt should handle returns with mixed positive/negative values."""
        from scripts.training.value_normalization import PopArtValueHead
        
        head = PopArtValueHead(256, device=device)
        features = torch.randn(32, 256, device=device)
        
        # Output before
        output_before = head.denormalize(head(features))
        
        # Mixed sign returns centered at 0
        returns = torch.randn(64, device=device) * 100
        head.update_stats(returns)
        
        output_after = head.denormalize(head(features))
        
        # Output preservation should still work
        diff = (output_before - output_after).abs().mean()
        assert diff < 0.1, f"Output preservation failed with mixed returns: {diff.item()}"


class TestPopArtGradients:
    """Test gradient flow through PopArt."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_gradient_flows_through_forward(self, device):
        """Gradients should flow through the forward pass."""
        from scripts.training.value_normalization import PopArtValueHead
        
        head = PopArtValueHead(256, device=device)
        features = torch.randn(32, 256, device=device, requires_grad=True)
        
        output = head(features)
        loss = output.sum()
        loss.backward()
        
        assert features.grad is not None
        assert features.grad.abs().sum() > 0
    
    def test_gradient_flows_to_linear_weights(self, device):
        """Gradients should flow to the Linear layer weights."""
        from scripts.training.value_normalization import PopArtValueHead
        
        head = PopArtValueHead(256, device=device)
        features = torch.randn(32, 256, device=device)
        
        output = head(features)
        loss = output.sum()
        loss.backward()
        
        assert head.linear.weight.grad is not None
        assert head.linear.bias.grad is not None
    
    def test_denormalize_no_grad_on_buffers(self, device):
        """Denormalize should not require gradients on mu/sigma."""
        from scripts.training.value_normalization import PopArtValueHead
        
        head = PopArtValueHead(256, device=device)
        
        # mu and sigma are buffers, not parameters
        assert not head.mu.requires_grad
        assert not head.sigma.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
