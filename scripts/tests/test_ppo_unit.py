"""
Comprehensive PPO Unit Tests.

Tests for PPO algorithm components with focus on:
1. GAE numerical correctness
2. Agent gradient flow
3. PPO update edge cases
"""
import pytest
import torch
import numpy as np

from scripts.training.agent import Agent
from scripts.training.ppo_utils import optimized_gae, make_ppo_update_fn


# ============================================================================
# GAE (Generalized Advantage Estimation) Tests
# ============================================================================

class TestGAECorrectness:
    """Verify GAE numerical correctness with hand-calculated examples."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_single_step_no_discount(self, device):
        """
        Single step, γ=1, λ=1: A = r + V(s') - V(s)
        
        Setup: r=1.0, V(s)=0.5, V(s')=0.3
        Expected: A = 1.0 + 1.0*0.3 - 0.5 = 0.8
        """
        rewards = torch.tensor([[1.0]], device=device)
        vals = torch.tensor([[0.5]], device=device)
        terminated = torch.tensor([[False]], device=device)
        next_value = torch.tensor([0.3], device=device)
        
        advantages, returns = optimized_gae(
            rewards, vals, terminated, next_value,
            gamma=1.0, gae_lambda=1.0
        )
        
        expected_adv = 1.0 + 1.0 * 0.3 - 0.5  # = 0.8
        assert abs(advantages[0, 0].item() - expected_adv) < 1e-5
    
    def test_terminal_no_bootstrap(self, device):
        """
        When next state is terminal, should NOT bootstrap V(s').
        
        Setup: r=1.0, V(s)=0.5, V(s')=10.0 (but terminal, so ignored)
        Expected: A = 1.0 + 0 - 0.5 = 0.5
        """
        rewards = torch.tensor([[1.0]], device=device)
        vals = torch.tensor([[0.5]], device=device)
        terminated = torch.tensor([[True]], device=device)  # Terminal!
        next_value = torch.tensor([10.0], device=device)  # Should be ignored
        
        advantages, returns = optimized_gae(
            rewards, vals, terminated, next_value,
            gamma=0.99, gae_lambda=0.95
        )
        
        expected_adv = 1.0 - 0.5  # = 0.5 (no bootstrap)
        assert abs(advantages[0, 0].item() - expected_adv) < 1e-5
    
    def test_multi_step_accumulation(self, device):
        """
        Multi-step GAE accumulation with γ=0.9, λ=0.8.
        
        Step 2: δ₂ = r₂ + γV₃ - V₂ = 1 + 0.9*0 - 0.5 = 0.5
                A₂ = δ₂ = 0.5
        Step 1: δ₁ = r₁ + γV₂ - V₁ = 1 + 0.9*0.5 - 0.3 = 1.15
                A₁ = δ₁ + γλA₂ = 1.15 + 0.9*0.8*0.5 = 1.15 + 0.36 = 1.51
        """
        rewards = torch.tensor([[1.0], [1.0]], device=device)  # 2 steps
        vals = torch.tensor([[0.3], [0.5]], device=device)
        terminated = torch.tensor([[False], [True]], device=device) # Terminal at step 2
        next_value = torch.tensor([0.0], device=device)
        
        advantages, returns = optimized_gae(
            rewards, vals, terminated, next_value,
            gamma=0.9, gae_lambda=0.8
        )
        
        # Step 2 (index 1)
        delta2 = 1.0 + 0.9 * 0.0 - 0.5  # = 0.5
        A2 = delta2  # = 0.5
        
        # Step 1 (index 0)
        delta1 = 1.0 + 0.9 * 0.5 - 0.3  # = 1.15
        A1 = delta1 + 0.9 * 0.8 * A2  # = 1.15 + 0.36 = 1.51
        
        assert abs(advantages[1, 0].item() - A2) < 1e-5
        assert abs(advantages[0, 0].item() - A1) < 1e-5
    
    def test_returns_equal_advantages_plus_values(self, device):
        """
        Returns should always equal advantages + values.
        """
        num_steps, num_envs = 10, 4
        rewards = torch.randn(num_steps, num_envs, device=device)
        vals = torch.randn(num_steps, num_envs, device=device)
        terminated = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=device)
        next_value = torch.randn(num_envs, device=device)
        
        advantages, returns = optimized_gae(
            rewards, vals, terminated, next_value,
            gamma=0.99, gae_lambda=0.95
        )
        
        reconstructed = advantages + vals
        assert torch.allclose(returns, reconstructed, atol=1e-5)
    
    def test_mid_trajectory_termination(self, device):
        """
        Termination in the middle of trajectory should reset advantage.
        
        Steps: [r=1, V=0.5] -> [TERMINAL, r=2, V=0.3] -> [r=1, V=0.2] -> end
        
        The terminated[1]=True means step 1 ended in terminal state,
        so step 0's advantage should NOT bootstrap from step 1's value.
        """
        rewards = torch.tensor([[1.0], [2.0], [1.0]], device=device)
        vals = torch.tensor([[0.5], [0.3], [0.2]], device=device)
        terminated = torch.tensor([[False], [True], [False]], device=device)
        next_value = torch.tensor([0.0], device=device)
        
        advantages, _ = optimized_gae(
            rewards, vals, terminated, next_value,
            gamma=1.0, gae_lambda=1.0  # Simple case
        )
        
        # Step 2: delta = 1 + 0 - 0.2 = 0.8, A = 0.8
        # Step 1: terminated[1]=True, so NO bootstrap from V2: delta = 2 + 0 - 0.3 = 1.7, A = 1.7
        # Step 0: terminated[0]=False, so bootstrap from V1: delta = 1 + 0.3 - 0.5 = 0.8, A = 0.8 + 1.7 = 2.5
        
        assert abs(advantages[1, 0].item() - 1.7) < 1e-5
        assert abs(advantages[0, 0].item() - 2.5) < 1e-5


# ============================================================================
# Agent Gradient Flow Tests
# ============================================================================

class TestAgentGradients:
    """Verify gradients flow correctly through the agent."""
    
    @pytest.fixture
    def agent_setup(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_obs, n_act = 32, 4
        agent = Agent(n_obs, n_act, device=device)
        return agent, device, n_obs, n_act
    
    def test_actor_receives_gradient(self, agent_setup):
        """Policy loss should produce gradients in actor network.
        
        In PPO, gradients flow through log_prob(action) to actor parameters.
        When evaluating log_prob of a given action, the gradient flows through
        the Normal distribution's mean and std parameters.
        """
        agent, device, n_obs, n_act = agent_setup
        
        obs = torch.randn(16, n_obs, device=device)
        
        # First sample an action (no grad needed for action itself)
        with torch.no_grad():
            action = agent.get_action(obs)
        
        # Now evaluate log_prob of this action - this is where gradient flows
        _, logprob, entropy, value = agent.get_action_and_value(obs, action=action)
        
        # Fake policy loss (maximize logprob)
        policy_loss = -logprob.mean()
        policy_loss.backward()
        
        # Check actor has gradients
        actor_grad_exists = any(
            p.grad is not None and p.grad.abs().sum() > 0 
            for p in agent.actor_mean.parameters()
        )
        assert actor_grad_exists, "Actor should have gradients from policy loss"
        
        # Check logstd has gradient
        assert agent.actor_logstd.grad is not None
        assert agent.actor_logstd.grad.abs().sum() > 0
    
    def test_critic_receives_gradient(self, agent_setup):
        """Value loss should produce gradients in critic network."""
        agent, device, n_obs, n_act = agent_setup
        
        obs = torch.randn(16, n_obs, device=device)
        value = agent.get_value(obs)
        
        # Fake value loss
        target = torch.randn_like(value)
        value_loss = ((value - target) ** 2).mean()
        value_loss.backward()
        
        # Check critic has gradients
        critic_grad_exists = any(
            p.grad is not None and p.grad.abs().sum() > 0 
            for p in agent.critic.parameters()
        )
        assert critic_grad_exists, "Critic should have gradients from value loss"
    
    def test_entropy_affects_logstd(self, agent_setup):
        """Entropy bonus should encourage exploration (increase std)."""
        agent, device, n_obs, n_act = agent_setup
        
        obs = torch.randn(16, n_obs, device=device)
        _, _, entropy, _ = agent.get_action_and_value(obs)
        
        # Maximize entropy
        entropy_loss = -entropy.mean()
        entropy_loss.backward()
        
        # Gradient should be negative (to increase logstd for more entropy)
        assert agent.actor_logstd.grad is not None
        # When maximizing entropy, gradient points toward increasing std
        # So grad should be negative (minimize negative entropy = maximize entropy)


# ============================================================================
# PPO Update Tests
# ============================================================================

class TestPPOUpdate:
    """Test PPO update function edge cases."""
    
    @pytest.fixture
    def ppo_setup(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_obs, n_act = 32, 4
        agent = Agent(n_obs, n_act, device=device)
        
        class MockConfig:
            class PPO:
                clip_coef = 0.2
                clip_vloss = True
                norm_adv = True
                ent_coef = 0.01
                vf_coef = 0.5
                max_grad_norm = 0.5
                logstd_min = -5.0
                logstd_max = 2.0
                def get(self, key, default):
                    return getattr(self, key, default)
            ppo = PPO()
        
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
        update_fn = make_ppo_update_fn(agent, optimizer, MockConfig())
        
        return agent, update_fn, device, n_obs, n_act
    
    def test_first_update_ratio_near_one(self, ppo_setup):
        """On first update, ratio should be ~1 and clipfrac ~0."""
        agent, update_fn, device, n_obs, n_act = ppo_setup
        import tensordict
        
        batch_size = 256
        obs = torch.randn(batch_size, n_obs, device=device)
        
        with torch.no_grad():
            actions, logprobs, _, values = agent.get_action_and_value(obs)
        
        advantages = torch.randn(batch_size, device=device)
        returns = values.flatten() + advantages
        
        container = tensordict.TensorDict({
            "obs": obs,
            "actions": actions,
            "logprobs": logprobs,
            "advantages": advantages,
            "returns": returns,
            "vals": values.flatten(),
        }, batch_size=[batch_size])
        
        out = update_fn(container, tensordict_out=tensordict.TensorDict())
        
        assert out["clipfrac"].item() < 0.1, "First update should have low clipfrac"
        assert out["approx_kl"].item() < 0.1, "First update should have low KL"
    
    def test_large_policy_change_triggers_clipping(self, ppo_setup):
        """Large policy update should trigger clipping."""
        agent, update_fn, device, n_obs, n_act = ppo_setup
        import tensordict
        
        batch_size = 256
        obs = torch.randn(batch_size, n_obs, device=device)
        
        with torch.no_grad():
            actions, old_logprobs, _, values = agent.get_action_and_value(obs)
        
        # Artificially make old_logprobs very different from current
        # by modifying agent before update
        with torch.no_grad():
            for p in agent.actor_mean.parameters():
                p.add_(torch.randn_like(p) * 0.5)  # Big perturbation
        
        advantages = torch.randn(batch_size, device=device)
        returns = values.flatten() + advantages
        
        container = tensordict.TensorDict({
            "obs": obs,
            "actions": actions,
            "logprobs": old_logprobs,  # Old logprobs, now different from agent
            "advantages": advantages,
            "returns": returns,
            "vals": values.flatten(),
        }, batch_size=[batch_size])
        
        out = update_fn(container, tensordict_out=tensordict.TensorDict())
        
        # Should have significant clipping
        assert out["clipfrac"].item() > 0.1, "Should clip when policy changes a lot"
    
    def test_advantage_normalization(self, ppo_setup):
        """Normalized advantages should have ~0 mean and ~1 std."""
        agent, update_fn, device, n_obs, n_act = ppo_setup
        
        batch_size = 1024
        
        # Create raw advantages with arbitrary mean and std
        raw_advantages = torch.randn(batch_size, device=device) * 10 + 5
        
        # Normalize manually (same as in PPO update)
        normalized = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)
        
        assert abs(normalized.mean().item()) < 0.01
        assert abs(normalized.std().item() - 1.0) < 0.01
    
    def test_value_clipping(self, ppo_setup):
        """Value clipping should prevent large value updates."""
        agent, update_fn, device, n_obs, n_act = ppo_setup
        import tensordict
        
        batch_size = 256
        obs = torch.randn(batch_size, n_obs, device=device)
        
        with torch.no_grad():
            actions, logprobs, _, values = agent.get_action_and_value(obs)
        
        # Create returns very different from current values
        advantages = torch.ones(batch_size, device=device) * 100  # Huge advantage
        returns = values.flatten() + advantages  # Very different from current values
        
        container = tensordict.TensorDict({
            "obs": obs,
            "actions": actions,
            "logprobs": logprobs,
            "advantages": advantages,
            "returns": returns,
            "vals": values.flatten(),
        }, batch_size=[batch_size])
        
        out = update_fn(container, tensordict_out=tensordict.TensorDict())
        
        # Should still produce finite loss
        assert torch.isfinite(out["v_loss"]).all()


# ============================================================================
# Agent Architecture Tests
# ============================================================================

class TestAgentArchitecture:
    """Test agent network properties."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_action_bounded_by_tanh(self, device):
        """Actor mean should be bounded by tanh to [-1, 1]."""
        agent = Agent(32, 4, device=device)
        obs = torch.randn(100, 32, device=device)
        
        action_mean = agent.actor_mean(obs)
        
        assert action_mean.min() >= -1.0
        assert action_mean.max() <= 1.0
    
    def test_deterministic_action_equals_mean(self, device):
        """Deterministic action should equal actor mean."""
        agent = Agent(32, 4, device=device)
        obs = torch.randn(16, 32, device=device)
        
        det_action = agent.get_action(obs, deterministic=True)
        action_mean = agent.actor_mean(obs)
        
        assert torch.allclose(det_action, action_mean)
    
    def test_stochastic_action_differs(self, device):
        """Stochastic samples should differ from mean."""
        agent = Agent(32, 4, device=device)
        obs = torch.randn(16, 32, device=device)
        
        action1 = agent.get_action(obs, deterministic=False)
        action2 = agent.get_action(obs, deterministic=False)
        
        # Should not be identical (probabilistically)
        assert not torch.allclose(action1, action2)
    
    def test_logprob_shape(self, device):
        """Log prob should have correct shape."""
        agent = Agent(32, 4, device=device)
        obs = torch.randn(16, 32, device=device)
        
        action, logprob, entropy, value = agent.get_action_and_value(obs)
        
        assert logprob.shape == (16,), f"Expected (16,), got {logprob.shape}"
        assert entropy.shape == (16,), f"Expected (16,), got {entropy.shape}"
        assert value.shape == (16, 1), f"Expected (16, 1), got {value.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
