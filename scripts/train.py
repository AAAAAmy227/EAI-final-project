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

@hydra.main(version_base=None, config_path="./configs", config_name="train")
def main(cfg: DictConfig):
    # Print config
    print(OmegaConf.to_yaml(cfg))
    
    # Seeding
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    
    # Device Setup (Masking for SAPIEN compatibility)
    # SAPIEN/PhysX often requires to be on the "first" visible device (cuda:0)
    # So we mask the visible devices to only include the requested one.
    if "device_id" in cfg:
        target_device = int(cfg.device_id)
        if target_device != 0:
            print(f"Masking visible devices to target GPU {target_device}...")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(target_device)
            # Update config to think it's on device 0 (since it's now the only one visible)
            cfg.device_id = 0
            OmegaConf.update(cfg, "device_id", 0)  # Ensure Hydra config is updated

    
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
        # Define custom x-axis for eval and rollout metrics
        wandb.define_metric("global_step")
        wandb.define_metric("eval/*", step_metric="global_step")
        wandb.define_metric("reward/*", step_metric="global_step")
        wandb.define_metric("rollout/*", step_metric="global_step")
        wandb.define_metric("losses/*", step_metric="global_step")
        wandb.define_metric("charts/*", step_metric="global_step")
        wandb.config.update({"output_dir": str(output_dir)})

    # Initialize Runner
    runner = PPORunner(cfg)
    
    # Start Training
    try:
        runner.train()
        runner.close()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        runner.envs.close()
        runner.eval_envs.close()
    finally:
        if cfg.wandb.enabled:
            wandb.finish()            


if __name__ == "__main__":
    main()
