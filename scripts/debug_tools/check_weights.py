
import torch
from scripts.training.agent import Agent

checkpoint_path_1000 = "/home/admin/Desktop/eai-final-project/outputs/2025-12-27/15-48-35/ckpt_1000.pt"
checkpoint_path_100 = "/home/admin/Desktop/eai-final-project/outputs/2025-12-27/15-48-35/ckpt_100.pt"

for cp in [checkpoint_path_100, checkpoint_path_1000]:
    try:
        state_dict = torch.load(cp, map_location="cpu")
        print(f"\n--- {cp} ---")
        logstd = state_dict["actor_logstd"]
        print(f"actor_logstd: mean={logstd.mean().item():.4f}, min={logstd.min().item():.4f}, max={logstd.max().item():.4f}")
        
        actor_w = state_dict["actor_mean.0.weight"]
        print(f"actor_mean.0.weight: std={actor_w.std().item():.4f}, mean={actor_w.mean().item():.4f}")
    except Exception as e:
        print(f"Error loading {cp}: {e}")
