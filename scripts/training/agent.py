"""
LeanRL-style Agent with orthogonal initialization.
Compatible with tensordict and CudaGraphModule.
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
    """PPO Agent with actor-critic architecture (LeanRL-style)."""
    
    def __init__(self, n_obs: int, n_act: int, device=None):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1, device=device), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, n_act, device=device), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))

    def get_value(self, x):
        """Get value estimate for observations."""
        return self.critic(x)

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
        """
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            # Sample action using reparameterization for cudagraph compatibility
            action = action_mean + action_std * torch.randn_like(action_mean)
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)
