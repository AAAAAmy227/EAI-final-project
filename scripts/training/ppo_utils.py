import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule

@torch.jit.script
def optimized_gae(
    rewards: torch.Tensor,
    vals: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float
):
    """
    Standard GAE calculation with post-step dones.
    GAE = r_t + gamma * V_{t+1} * (1-d_t) - V_t + gamma * lambda * (1-d_t) * GAE_{t+1}
    """
    num_steps: int = rewards.shape[0]
    next_value = next_value.reshape(-1)
    
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(rewards[0])
    
    # nextvalues starts at V(s_{T+1})
    nextvalues = next_value
    # next_non_terminal is (1 - d_T)
    next_non_terminal = 1.0 - next_done.float()
    
    # Loop backwards from T-1 to 0
    for t in range(num_steps - 1, -1, -1):
        # Advantages are calculated based on the return/value at step t
        # delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)
        # However, our loop usually uses 'nextvalues' which is V(s_{t+1})
        # And 'next_non_terminal' which is (1 - d_t)
        
        # In our storage, dones[t] is the done signal after step t.
        # So it applies to the transition from t to t+1.
        non_terminal = 1.0 - dones[t].float()
        
        delta = rewards[t] + gamma * nextvalues * non_terminal - vals[t]
        lastgaelam = delta + gamma * gae_lambda * non_terminal * lastgaelam
        advantages[t] = lastgaelam
        
        # Set next values for step t-1
        nextvalues = vals[t]
        # Although not strictly used in standard PPO loop for delta calculation inside the loop,
        # we can keep track of it if needed.
        # non_terminal = 1.0 - dones[t].float()
        
    return advantages, advantages + vals
