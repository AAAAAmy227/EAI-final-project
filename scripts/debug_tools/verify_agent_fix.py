
import torch
from scripts.training.agent import Agent

def test_agent():
    n_obs = 36
    n_act = 6
    device = "cpu"
    agent = Agent(n_obs, n_act, device=device)
    
    # Test initial logstd
    print(f"Initial logstd: {agent.actor_logstd.data}")
    
    # Test output range with large inputs
    x = torch.randn(10, n_obs) * 100.0
    action = agent.get_action(x, deterministic=True)
    
    print(f"Action range (deterministic): min={action.min().item():.4f}, max={action.max().item():.4f}")
    
    if action.min() >= -1.0 and action.max() <= 1.0:
        print("SUCCESS: Tanh correctly constrains actions to [-1, 1]!")
    else:
        print("FAILURE: Actions are NOT constrained!")

if __name__ == "__main__":
    test_agent()
