"""
Unit tests for PPO training components.
Tests to diagnose clipfrac=1, explained_variance=0, approx_kl=4.5 issues.
"""
import torch
import numpy as np
from functools import partial

from scripts.training.agent import Agent
from scripts.training.ppo_utils import optimized_gae, make_ppo_update_fn


def test_agent_consistency():
    """Test that agent produces consistent logprobs for same obs+action pair."""
    print("\n=== Test 1: Agent Log-Prob Consistency ===")
    
    device = torch.device("cuda")
    n_obs, n_act = 49, 6
    batch_size = 32
    
    agent = Agent(n_obs, n_act, device=device)
    
    # Sample observations
    obs = torch.randn(batch_size, n_obs, device=device)
    
    # Get action and logprob from first call
    action1, logprob1, entropy1, value1 = agent.get_action_and_value(obs)
    
    # Now evaluate the SAME action again
    action2, logprob2, entropy2, value2 = agent.get_action_and_value(obs, action=action1)
    
    # Check consistency
    logprob_diff = (logprob1 - logprob2).abs().mean().item()
    value_diff = (value1 - value2).abs().mean().item()
    
    print(f"  logprob1.mean() = {logprob1.mean().item():.4f}")
    print(f"  logprob2.mean() = {logprob2.mean().item():.4f}")
    print(f"  logprob diff (should be ~0): {logprob_diff:.6f}")
    print(f"  value diff (should be 0): {value_diff:.6f}")
    
    if logprob_diff < 0.01:
        print("  ✅ PASS: Agent is consistent")
        return True
    else:
        print("  ❌ FAIL: Agent logprobs are inconsistent!")
        return False


def test_ratio_calculation():
    """Test that PPO ratio calculation is correct."""
    print("\n=== Test 2: Ratio Calculation ===")
    
    device = torch.device("cuda")
    n_obs, n_act = 49, 6
    batch_size = 32
    
    agent = Agent(n_obs, n_act, device=device)
    
    obs = torch.randn(batch_size, n_obs, device=device)
    
    # Step 1: Sample action (simulating rollout)
    with torch.no_grad():
        action, old_logprob, _, value = agent.get_action_and_value(obs)
    
    print(f"  old_logprob.shape = {old_logprob.shape}")
    print(f"  old_logprob.mean() = {old_logprob.mean().item():.4f}")
    
    # Step 2: Evaluate SAME action (simulating PPO update)
    _, new_logprob, _, _ = agent.get_action_and_value(obs, action=action)
    
    print(f"  new_logprob.shape = {new_logprob.shape}")
    print(f"  new_logprob.mean() = {new_logprob.mean().item():.4f}")
    
    # Calculate ratio
    logratio = new_logprob - old_logprob
    ratio = logratio.exp()
    
    print(f"  logratio.mean() = {logratio.mean().item():.4f}")
    print(f"  ratio.mean() = {ratio.mean().item():.4f}")
    print(f"  ratio.min() = {ratio.min().item():.4f}")
    print(f"  ratio.max() = {ratio.max().item():.4f}")
    
    # Before any gradient update, ratio should be ~1.0
    ratio_error = (ratio - 1.0).abs().mean().item()
    print(f"  |ratio - 1| mean (should be ~0): {ratio_error:.6f}")
    
    if ratio_error < 0.01:
        print("  ✅ PASS: Ratio is ~1 before update")
        return True
    else:
        print("  ❌ FAIL: Ratio is NOT 1 before update!")
        return False


def test_ppo_update_basic():
    """Test a basic PPO update step."""
    print("\n=== Test 3: PPO Update Basic ===")
    
    device = torch.device("cuda")
    n_obs, n_act = 49, 6
    batch_size = 3200
    
    agent = Agent(n_obs, n_act, device=device)
    
    # Mock config
    class MockConfig:
        class PPO:
            clip_coef = 0.2
            clip_vloss = True
            norm_adv = True
            ent_coef = 0.0
            vf_coef = 0.5
            max_grad_norm = 0.5
            
            def get(self, key, default=None):
                """Support dict-like .get() access for compatibility with ppo_utils.py"""
                return getattr(self, key, default)
        ppo = PPO()
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
    
    # Generate fake rollout data
    obs = torch.randn(batch_size, n_obs, device=device)
    
    with torch.no_grad():
        actions, logprobs, _, values = agent.get_action_and_value(obs)
    
    # Fake advantages and returns
    advantages = torch.randn(batch_size, device=device)
    returns = values.flatten() + advantages  # returns = advantage + value
    
    print(f"  obs.shape = {obs.shape}")
    print(f"  actions.shape = {actions.shape}")
    print(f"  logprobs.shape = {logprobs.shape}")
    print(f"  values.shape = {values.shape}")
    print(f"  advantages.shape = {advantages.shape}")
    
    # Create update function
    update_fn = make_ppo_update_fn(agent, optimizer, MockConfig())
    
    # Run update (manually, not via TensorDictModule)
    import tensordict
    container = tensordict.TensorDict({
        "obs": obs,
        "actions": actions,
        "logprobs": logprobs,
        "advantages": advantages,
        "returns": returns,
        "vals": values.flatten(),
    }, batch_size=[batch_size])
    
    out = update_fn(container, tensordict_out=tensordict.TensorDict())
    
    print(f"  approx_kl = {out['approx_kl'].item():.4f}")
    print(f"  clipfrac = {out['clipfrac'].item():.4f}")
    print(f"  v_loss = {out['v_loss'].item():.4f}")
    print(f"  pg_loss = {out['pg_loss'].item():.4f}")
    
    # On first step, clipfrac should be 0 (no clipping because ratio~1)
    if out['clipfrac'].item() < 0.1:
        print("  ✅ PASS: clipfrac is low on first update")
        return True
    else:
        print("  ❌ FAIL: clipfrac is high on first update!")
        return False


def test_gae_basic():
    """Test GAE calculation."""
    print("\n=== Test 4: GAE Calculation ===")
    
    device = torch.device("cuda")
    num_steps, num_envs = 50, 16
    gamma, gae_lambda = 0.8, 0.9
    
    # Simple case: constant reward, constant value
    rewards = torch.ones(num_steps, num_envs, device=device) * 0.1
    vals = torch.ones(num_steps, num_envs, device=device) * 1.0
    terminated = torch.zeros(num_steps, num_envs, device=device, dtype=torch.bool)
    next_value = torch.ones(num_envs, device=device) * 1.0
    next_terminated = torch.zeros(num_envs, device=device, dtype=torch.bool)
    
    advantages, returns = optimized_gae(
        rewards, vals, terminated, next_value,
        gamma, gae_lambda
    )
    
    print(f"  advantages.shape = {advantages.shape}")
    print(f"  returns.shape = {returns.shape}")
    print(f"  advantages.mean() = {advantages.mean().item():.4f}")
    print(f"  returns.mean() = {returns.mean().item():.4f}")
    
    # For constant value and reward, advantage should be: r + gamma*V - V = r + V*(gamma - 1)
    # = 0.1 + 1.0 * (0.8 - 1) = 0.1 - 0.2 = -0.1 (approximately, simplified)
    
    if advantages.shape == (num_steps, num_envs):
        print("  ✅ PASS: GAE shapes are correct")
        return True
    else:
        print("  ❌ FAIL: GAE shapes are wrong!")
        return False


def test_inference_vs_update_agent():
    """Test that inference agent and training agent produce same outputs."""
    print("\n=== Test 5: Inference vs Training Agent ===")
    
    device = torch.device("cuda")
    n_obs, n_act = 49, 6
    batch_size = 32
    
    # Create two agents
    agent = Agent(n_obs, n_act, device=device)
    agent_copy = Agent(n_obs, n_act, device=device)
    
    # Copy weights
    from tensordict import from_module
    from_module(agent).data.to_module(agent_copy)
    
    obs = torch.randn(batch_size, n_obs, device=device)
    
    # Get outputs from both
    action1, lp1, ent1, val1 = agent.get_action_and_value(obs)
    
    # Evaluate same action with copy
    action2, lp2, ent2, val2 = agent_copy.get_action_and_value(obs, action=action1)
    
    lp_diff = (lp1 - lp2).abs().mean().item()
    val_diff = (val1 - val2).abs().mean().item()
    
    print(f"  logprob diff = {lp_diff:.6f}")
    print(f"  value diff = {val_diff:.6f}")
    
    if lp_diff < 0.01 and val_diff < 0.01:
        print("  ✅ PASS: Agents are consistent after weight copy")
        return True
    else:
        print("  ❌ FAIL: Agents differ after weight copy!")
        return False


def test_simulated_training_loop():
    """Simulate a mini training loop to see if clipfrac stays reasonable."""
    print("\n=== Test 6: Simulated Training Loop ===")
    
    device = torch.device("cuda")
    n_obs, n_act = 49, 6
    num_envs = 64
    num_steps = 10
    
    agent = Agent(n_obs, n_act, device=device)
    
    class MockConfig:
        class PPO:
            clip_coef = 0.2
            clip_vloss = True
            norm_adv = True
            ent_coef = 0.0
            vf_coef = 0.5
            max_grad_norm = 0.5
            gamma = 0.99
            gae_lambda = 0.95
            
            def get(self, key, default=None):
                """Support dict-like .get() access for compatibility with ppo_utils.py"""
                return getattr(self, key, default)
        ppo = PPO()
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
    update_fn = make_ppo_update_fn(agent, optimizer, MockConfig())
    
    # Simulate 3 iterations
    for iteration in range(3):
        # Collect rollout
        all_obs = []
        all_actions = []
        all_logprobs = []
        all_values = []
        all_rewards = []
        
        obs = torch.randn(num_envs, n_obs, device=device)
        
        for step in range(num_steps):
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs)
            
            all_obs.append(obs)
            all_actions.append(action)
            all_logprobs.append(logprob)
            all_values.append(value.flatten())
            all_rewards.append(torch.randn(num_envs, device=device) * 0.1)
            
            obs = torch.randn(num_envs, n_obs, device=device)  # Fake next obs
        
        # Stack
        obs_batch = torch.stack(all_obs).view(-1, n_obs)
        actions_batch = torch.stack(all_actions).view(-1, n_act)
        logprobs_batch = torch.stack(all_logprobs).view(-1)
        vals_batch = torch.stack(all_values).view(-1)
        
        # Fake advantages
        advantages = torch.randn(num_envs * num_steps, device=device)
        returns = vals_batch + advantages
        
        # Update
        import tensordict
        container = tensordict.TensorDict({
            "obs": obs_batch,
            "actions": actions_batch,
            "logprobs": logprobs_batch,
            "advantages": advantages,
            "returns": returns,
            "vals": vals_batch,
        }, batch_size=[num_envs * num_steps])
        
        out = update_fn(container, tensordict_out=tensordict.TensorDict())
        
        print(f"  Iter {iteration}: approx_kl={out['approx_kl'].item():.4f}, "
              f"clipfrac={out['clipfrac'].item():.4f}")
    
    if out['clipfrac'].item() < 0.5:
        print("  ✅ PASS: clipfrac stays reasonable in simulated loop")
        return True
    else:
        print("  ❌ FAIL: clipfrac too high in simulated loop!")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("PPO Unit Tests")
    print("=" * 60)
    
    results = []
    results.append(("Agent Consistency", test_agent_consistency()))
    results.append(("Ratio Calculation", test_ratio_calculation()))
    results.append(("PPO Update Basic", test_ppo_update_basic()))
    results.append(("GAE Calculation", test_gae_basic()))
    results.append(("Inference vs Training Agent", test_inference_vs_update_agent()))
    results.append(("Simulated Training Loop", test_simulated_training_loop()))
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(p for _, p in results)
    print(f"\nOverall: {'All tests passed!' if all_passed else 'Some tests failed!'}")
