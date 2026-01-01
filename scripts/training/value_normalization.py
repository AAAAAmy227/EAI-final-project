"""
PopArt (Preserving Outputs Precisely, while Adaptively Rescaling Targets) Value Normalization.

Based on DeepMind's paper: "Learning values across many orders of magnitude"
https://arxiv.org/abs/1602.07714

This module provides a value head that:
1. Maintains running statistics (mean/std) of value targets
2. Normalizes targets for stable critic training
3. Adjusts network weights to preserve outputs when statistics change
4. Denormalizes critic outputs for accurate GAE computation
"""
import torch
import torch.nn as nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for linear layers."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PopArtValueHead(nn.Module):
    """Value head with PopArt normalization.
    
    PopArt maintains running statistics of value targets and adjusts
    network weights to preserve outputs when statistics change.
    
    The key insight is that when we update the running mean/std, we also
    adjust the final layer's weights and biases so that the denormalized
    output remains the same. This allows the network to learn in a 
    normalized space while producing outputs in the original scale.
    
    Args:
        in_features: Input dimension (from the last hidden layer)
        device: Device to create tensors on
        beta: Update rate for running statistics (like learning rate for stats)
              Smaller = slower adaptation, more stable
        epsilon: Small constant for numerical stability
    """
    
    def __init__(self, in_features: int, device=None, beta: float = 0.0001, epsilon: float = 1e-4):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        
        # Final linear layer (normalized space)
        self.linear = layer_init(nn.Linear(in_features, 1, device=device), std=1.0)
        
        # Running statistics for value targets (buffers = saved in state_dict)
        self.register_buffer("mu", torch.zeros(1, device=device))
        self.register_buffer("nu", torch.ones(1, device=device))  # Second moment (E[x^2])
        self.register_buffer("sigma", torch.ones(1, device=device))  # std = sqrt(nu - mu^2)
        
        # Count for tracking update iterations (optional, for debugging)
        self.register_buffer("update_count", torch.zeros(1, device=device, dtype=torch.long))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns normalized value.
        
        This is what the loss is computed against (with normalized targets).
        """
        return self.linear(x)
    
    def denormalize(self, normalized_value: torch.Tensor) -> torch.Tensor:
        """Convert normalized value back to original scale.
        
        Used for GAE computation where we need V(s) in original reward scale.
        
        V_original = V_normalized * sigma + mu
        """
        return normalized_value * self.sigma + self.mu
    
    def normalize_target(self, target: torch.Tensor) -> torch.Tensor:
        """Normalize target returns for stable loss computation.
        
        target_normalized = (target - mu) / sigma
        """
        return (target - self.mu) / (self.sigma + self.epsilon)
    
    def update_stats(self, targets: torch.Tensor) -> None:
        """Update running statistics and preserve outputs.
        
        This is the core of PopArt:
        1. Compute new statistics from the batch
        2. Update running mean (mu) and second moment (nu)
        3. Compute new sigma from updated statistics
        4. Adjust weights/biases to preserve denormalized output
        
        Args:
            targets: Batch of value targets (returns) in original scale [N] or [N, 1]
        """
        targets = targets.detach().flatten()
        
        # Compute batch statistics
        batch_mean = targets.mean()
        batch_sq_mean = (targets ** 2).mean()
        
        # Store old values for weight adjustment
        old_mu = self.mu.clone()
        old_sigma = self.sigma.clone()
        
        # Update running statistics using exponential moving average
        # mu_new = (1 - beta) * mu_old + beta * batch_mean
        self.mu.copy_((1 - self.beta) * self.mu + self.beta * batch_mean)
        self.nu.copy_((1 - self.beta) * self.nu + self.beta * batch_sq_mean)
        
        # Compute new sigma: sigma = sqrt(E[x^2] - E[x]^2)
        # Clamp to avoid negative variance due to numerical issues
        variance = torch.clamp(self.nu - self.mu ** 2, min=self.epsilon)
        self.sigma.copy_(torch.sqrt(variance))
        
        # Preserve outputs: adjust weights so denormalized output stays the same
        # Old: V_denorm = (w @ x + b) * old_sigma + old_mu
        # New: V_denorm = (w_new @ x + b_new) * new_sigma + new_mu
        # We want V_denorm_old == V_denorm_new
        # 
        # Solution:
        # w_new = w * (old_sigma / new_sigma)
        # b_new = (old_sigma * b + old_mu - new_mu) / new_sigma
        with torch.no_grad():
            scale = old_sigma / (self.sigma + self.epsilon)
            self.linear.weight.data.mul_(scale)
            new_bias = (old_sigma * self.linear.bias.data + old_mu - self.mu) / (self.sigma + self.epsilon)
            self.linear.bias.data.copy_(new_bias)
        
        self.update_count += 1
    
    def get_stats(self) -> dict:
        """Get current statistics for logging."""
        return {
            "popart/mu": self.mu.item(),
            "popart/sigma": self.sigma.item(),
            "popart/nu": self.nu.item(),
            "popart/update_count": self.update_count.item(),
        }
