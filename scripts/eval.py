"""
Standalone evaluation script for trained PPO checkpoints.
Reuses PPORunner directly for consistent evaluation logic.

Usage:
    uv run -m scripts.eval checkpoint=/path/to/ckpt.pt
    uv run -m scripts.eval checkpoint=/path/to/ckpt.pt training.num_eval_envs=16
"""
import sys
from pathlib import Path

import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Set float32 matmul precision for speed
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    """Run evaluation on a checkpoint."""
    
    # Validate checkpoint
    if not cfg.checkpoint:
        print("ERROR: Must specify checkpoint path")
        print("Usage: uv run -m scripts.eval checkpoint=/path/to/ckpt.pt")
        sys.exit(1)
    
    checkpoint_path = Path(cfg.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(OmegaConf.to_yaml(cfg))
    
    # Seeding
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    
    # Import here to avoid slow imports when just checking usage
    from scripts.training.runner import PPORunner
    
    # Create PPORunner in eval-only mode (skips training env creation)
    print("\\n" + "="*50)
    print("Creating PPORunner in eval-only mode...")
    print("="*50 + "\\n")
    
    runner = PPORunner(cfg, eval_only=True)
    
    # Load checkpoint into agent
    device = runner.device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if "agent" in checkpoint:
        runner.agent.load_state_dict(checkpoint["agent"])
    else:
        # Assume checkpoint is just the state dict
        runner.agent.load_state_dict(checkpoint)
    
    runner.agent.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Run evaluation using PPORunner's existing _evaluate method
    print("\\n" + "="*50)
    print("Starting evaluation...")
    print("="*50 + "\\n")
    
    runner._evaluate()
    
    # Cleanup
    runner.eval_envs.close()
    print("\\nEvaluation complete!")


if __name__ == "__main__":
    main()
