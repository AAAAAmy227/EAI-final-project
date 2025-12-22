"""
Main training entry point for LeanRL-style PPO.
Uses tensordict, torch.compile, CudaGraphModule, and WandB logging.
"""
import os
import sys
import time
from pathlib import Path

import hydra
import torch
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf

from scripts.training.runner import PPORunner

# Set float32 matmul precision for speed
torch.set_float32_matmul_precision("high")

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Print config
    print(OmegaConf.to_yaml(cfg))
    
    # Seeding
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    
    # Experiment Setup
    exp_name = cfg.exp_name or f"track1_{cfg.env.task}"
    run_name = f"{exp_name}__{cfg.seed}__{int(time.time())}"
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    # WandB Setup
    if cfg.wandb.enabled:
        import subprocess
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                cwd=Path(__file__).parent.parent,
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()[:8]
        except:
            git_commit = "unknown"
            
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            save_code=True,
            tags=[cfg.env.task, cfg.reward.reward_mode if "reward" in cfg else "sparse", cfg.env.obs_mode],
            notes=f"Git commit: {git_commit}"
        )

    # Initialize Runner
    runner = PPORunner(cfg)
    
    # Start Training
    try:
        runner.train()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        runner.envs.close()
        runner.eval_envs.close()
        if cfg.wandb.enabled:
            wandb.finish()
            
    # Cleanup
    if cfg.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    main()
