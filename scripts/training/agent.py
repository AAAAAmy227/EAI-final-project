"""
LeanRL-style Agent with orthogonal initialization.
Compatible with tensordict and CudaGraphModule.
Supports optional PopArt value normalization.
"""
import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for linear layers."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO Agent with actor-critic architecture (LeanRL-style).
    
    Args:
        n_obs: Observation dimension
        n_act: Action dimension
        device: Device for tensors
        logstd_init: Initial value for log standard deviation
        use_popart: If True, use PopArt value normalization for the critic
        popart_beta: Update rate for PopArt running statistics
    """
    
    def __init__(self, n_obs: int, n_act: int, device=None, logstd_init: float = 0.0,
                 use_popart: bool = False, popart_beta: float = 0.0001):
        super().__init__()
        self.use_popart = use_popart
        
        # Critic backbone (shared layers)
        self.critic_backbone = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
        )
        
        # Critic head - either standard Linear or PopArt
        if use_popart:
            from scripts.training.value_normalization import PopArtValueHead
            self.critic_head = PopArtValueHead(256, device=device, beta=popart_beta)
        else:
            self.critic_head = layer_init(nn.Linear(256, 1, device=device), std=1.0)
        
        # Actor
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, n_act, device=device), std=0.01),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, n_act, device=device) * logstd_init)

    @property
    def critic(self):
        """Compatibility property - returns a callable that acts like the old critic."""
        def _critic_fn(x):
            return self.critic_head(self.critic_backbone(x))
        return _critic_fn

    def get_value(self, x):
        """Get value estimate for observations.
        
        Returns normalized value if PopArt is enabled.
        Use get_value_denormalized() for GAE computation.
        """
        return self.critic_head(self.critic_backbone(x))
    
    def get_value_denormalized(self, x):
        """Get denormalized value estimate (original scale).
        
        For GAE computation, we need values in original reward scale.
        If PopArt is not enabled, this is identical to get_value().
        """
        normalized_value = self.critic_head(self.critic_backbone(x))
        if self.use_popart:
            return self.critic_head.denormalize(normalized_value)
        return normalized_value

    def get_action(self, x, deterministic=False):
        """Get action for observations."""
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, obs, action=None):
        """Get action, log_prob, entropy, and value for observations.
        
        Compatible with tensordict.nn.TensorDictModule wrapping.
        When called with keyword argument 'obs', returns positional outputs.
        
        Returns normalized value (for loss computation with normalized targets).
        """
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            # Sample action using reparameterization for cudagraph compatibility
            action = action_mean + action_std * torch.randn_like(action_mean)
        
        # Get normalized value from critic
        value = self.critic_head(self.critic_backbone(obs))
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value
    
    def get_popart_head(self):
        """Get PopArt head for stats updates. Returns None if PopArt not enabled."""
        if self.use_popart:
            return self.critic_head
        return None

