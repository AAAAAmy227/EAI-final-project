
import numpy as np
import torch
import gymnasium as gym
import hydra
from omegaconf import DictConfig
from scripts.training.common import make_env

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    cfg.env.task = "lift"
    
    # We must bypass any flattening to see the dict
    from scripts.training.common import OmegaConf
    reward_config = OmegaConf.to_container(cfg.reward, resolve=True) if "reward" in cfg else None
    action_bounds = OmegaConf.to_container(cfg.control.action_bounds, resolve=True) if "control" in cfg else None
    obs_normalization = OmegaConf.to_container(cfg.obs, resolve=True) if "obs" in cfg else None
    
    env_kwargs = dict(
        task=cfg.env.task,
        control_mode=cfg.env.control_mode,
        camera_mode=cfg.env.camera_mode,
        obs_mode=None, # DISABLED flattening by ManiSkill core if possible
        domain_randomization=False,
        reward_mode=cfg.reward.reward_mode,
        reward_config=reward_config,
        action_bounds=action_bounds,
        obs_normalization=obs_normalization,
    )
    
    # Create env and call _get_obs_state_dict manually
    env = gym.make("Track1-v0", num_envs=1, **env_kwargs)
    env.reset(seed=1)
    
    # Access the observations
    obs_dict = env.unwrapped._get_obs_state_dict(info={})
    
    print("\nActual Observation Values (Recursive Sorting):")
    def print_values(d, indent=""):
        if isinstance(d, dict):
            for k in sorted(d.keys()):
                print(f"{indent}{k}:")
                print_values(d[k], indent + "  ")
        elif isinstance(d, torch.Tensor):
            print(f"{indent}{d}")
        else:
            print(f"{indent}{d}")

    print_values(obs_dict)
    
    env.close()

if __name__ == "__main__":
    main()
