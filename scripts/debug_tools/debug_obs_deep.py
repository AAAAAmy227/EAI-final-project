
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from scripts.training.env_utils import make_env

def print_obs_structure(obs, prefix=""):
    if isinstance(obs, dict):
        for k in sorted(obs.keys()):
            v = obs[k]
            if isinstance(v, dict):
                print(f"{prefix}{k}: (dict)")
                print_obs_structure(v, prefix + "  ")
            else:
                if isinstance(v, torch.Tensor):
                    print(f"{prefix}{k}: {v.shape} | mean={v.mean().item():.4f} | std={v.std().item():.4f}")
                else:
                    print(f"{prefix}{k}: {type(v)}")
    else:
        print(f"{prefix}tensor: {obs.shape}")

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # We must use the RAW environment WITHOUT FlattenStateWrapper to see the dict
    # But make_env always applies it. So we need a modified version or just look at underlying
    
    # 1. Create eval env
    print("\n--- Deep Inspection: Eval Env (before manual flattening) ---")
    eval_envs = make_env(cfg, num_envs=1, for_eval=True)
    
    # Access the unwrapped env if possible, or just look at the resets
    # Actually, ManiSkillVectorEnv reset returns what's inside.
    # If it's FlattenStateWrapper, it's already a tensor.
    
    # Let's manually create one WITHOUT wrapper
    from scripts.training.env_utils import make_env as original_make_env
    # We'll monkeypatch or just use the logic
    
    import gymnasium as gym
    from scripts.training.env_utils import FlattenStateWrapper
    
    # Create env without wrapper
    from scripts.training.env_utils import OmegaConf
    reward_config = OmegaConf.to_container(cfg.reward, resolve=True) if "reward" in cfg else None
    action_bounds = OmegaConf.to_container(cfg.control.action_bounds, resolve=True) if "control" in cfg else None
    obs_normalization = OmegaConf.to_container(cfg.obs, resolve=True) if "obs" in cfg else None
    
    env_kwargs = dict(
        task=cfg.env.task,
        control_mode=cfg.env.control_mode,
        camera_mode=cfg.env.camera_mode,
        obs_mode=cfg.env.obs_mode,
        domain_randomization=False,
        reward_mode=cfg.reward.reward_mode,
        reward_config=reward_config,
        action_bounds=action_bounds,
        obs_normalization=obs_normalization,
    )
    
    raw_env = gym.make("Track1-v0", num_envs=1, **env_kwargs)
    obs, _ = raw_env.reset(seed=cfg.seed)
    
    print_obs_structure(obs)
    
    # Also print the flattened version to compare with eval.py's DEBUG output
    wrapper = FlattenStateWrapper(raw_env)
    obs_flat = wrapper.observation(obs)
    print(f"\nFlattened obs (36): {obs_flat[0]}")
    
    raw_env.close()

if __name__ == "__main__":
    main()
