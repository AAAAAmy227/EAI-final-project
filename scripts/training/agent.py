
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class NatureCNN(nn.Module):
    """CNN feature extractor for RGB observations."""
    def __init__(self, sample_obs):
        super().__init__()
        self.out_features = 0
        feature_size = 256
        
        # Check if we have RGB
        if "rgb" in sample_obs:
            in_channels = sample_obs["rgb"].shape[-1]
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )
            
            # Compute flattened size
            with torch.no_grad():
                n_flatten = self.cnn(sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu()).shape[1]
            self.fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
            self.out_features += feature_size
        else:
            self.cnn = None
            self.fc = None

        if "state" in sample_obs:
            state_size = sample_obs["state"].shape[-1]
            self.state_fc = nn.Linear(state_size, 256)
            self.out_features += 256
        else:
            self.state_fc = None

    def forward(self, observations) -> torch.Tensor:
        encoded = []
        # Process RGB
        if self.cnn is not None:
            rgb = observations["rgb"].float().permute(0, 3, 1, 2) / 255.0
            encoded.append(self.fc(self.cnn(rgb)))
        
        # Process state if available
        if self.state_fc is not None and "state" in observations:
            encoded.append(self.state_fc(observations["state"]))
        
        return torch.cat(encoded, dim=1)


class MLP(nn.Module):
    """MLP feature extractor for flat observations."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )
        self.out_features = 256
        
    def forward(self, x):
        return self.net(x)


class Agent(nn.Module):
    """PPO Agent with actor-critic architecture."""
    def __init__(self, envs, sample_obs):
        super().__init__()
        
        # Handle dict vs box observation
        if isinstance(sample_obs, dict):
            self.feature_net = NatureCNN(sample_obs=sample_obs)
        else:
            # Assume flat state vector
            self.feature_net = MLP(input_dim=sample_obs.shape[-1])
            
        latent_size = self.feature_net.out_features
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1), std=1.0),
        )
        
        action_dim = np.prod(envs.unwrapped.single_action_space.shape)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_value(self, x):
        return self.critic(self.feature_net(x))

    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
