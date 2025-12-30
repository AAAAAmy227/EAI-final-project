import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule

@torch.jit.script
def optimized_gae(
    rewards: torch.Tensor,
    vals: torch.Tensor,
    terminated: torch.Tensor,
    next_value: torch.Tensor,
    next_terminated: torch.Tensor,
    gamma: float,
    gae_lambda: float
):
    """
    GAE calculation with proper terminated vs truncated handling.
    
    Only uses `terminated` (not `done = terminated | truncated`) for bootstrap mask.
    This means truncated episodes still bootstrap V(s_{t+1}), which is theoretically correct.
    
    Args:
        rewards: [num_steps, num_envs] reward tensor
        vals: [num_steps, num_envs] value estimates
        terminated: [num_steps, num_envs] POST-step terminated flags (status of s_{t+1} after step t)
        next_value: [1, num_envs] or [num_envs] value estimate for the state AFTER the last step
        next_terminated: [num_envs] whether the state AFTER the last step is terminal
        gamma: discount factor
        gae_lambda: GAE lambda
    
    Returns:
        advantages, returns
    """
    num_steps: int = rewards.shape[0]
    next_value = next_value.reshape(-1)
    
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(rewards[0])
    
    for t in range(num_steps - 1, -1, -1):
        if t == num_steps - 1:
            # For the last step, use next_terminated and next_value
            nextnonterminal = 1.0 - next_terminated.float()
            nextvalues = next_value
        else:
            # For other steps, use terminated[t] and vals[t+1]
            # terminated[t] corresponds to the end of step t (status of s_{t+1})
            nextnonterminal = 1.0 - terminated[t].float()
            nextvalues = vals[t + 1]
        
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - vals[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
        
    return advantages, advantages + vals


def make_ppo_update_fn(agent, optimizer, cfg):
    """Factory function to create PPO update TensorDictModule.
    
    Args:
        agent: The training agent (with get_action_and_value method)
        optimizer: The optimizer for agent parameters
        cfg: Config with ppo.clip_coef, ppo.clip_vloss, ppo.norm_adv, ppo.ent_coef, ppo.vf_coef, ppo.max_grad_norm
    
    Returns:
        TensorDictModule wrapping the update function
    """
    import tensordict
    
    clip_coef = cfg.ppo.clip_coef
    clip_vloss = cfg.ppo.get("clip_vloss", True)  # Default True (CleanRL default)
    norm_adv = cfg.ppo.get("norm_adv", True)      # Default True (CleanRL default)
    ent_coef = cfg.ppo.ent_coef
    vf_coef = cfg.ppo.vf_coef
    max_grad_norm = cfg.ppo.max_grad_norm
    
    def update(obs, actions, logprobs, advantages, returns, vals):
        optimizer.zero_grad(set_to_none=True)
        
        # Clamp logstd to prevent policy collapse (standard practice in stable RL)
        with torch.no_grad():
            agent.actor_logstd.clamp_(cfg.ppo.get("logstd_min", -5.0), cfg.ppo.get("logstd_max", 2.0))
            
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions)
        logratio = newlogprob - logprobs
        ratio = logratio.exp()
        
        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
        
        # Normalize advantages (configurable, CleanRL default: True)
        if norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        # Value loss (configurable clipping, CleanRL default: True)
        newvalue = newvalue.view(-1)
        if clip_vloss:
            v_loss_unclipped = (newvalue - returns) ** 2
            v_clipped = vals + torch.clamp(newvalue - vals, -clip_coef, clip_coef)
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
        
        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
        
        loss.backward()
        gn = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()
        
        return approx_kl, v_loss, pg_loss, entropy_loss, old_approx_kl, clipfrac, gn
    
    return tensordict.nn.TensorDictModule(
        update,
        in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
        out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac", "gn"],
    )
