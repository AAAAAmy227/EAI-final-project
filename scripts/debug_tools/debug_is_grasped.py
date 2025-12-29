
import numpy as np
import torch
import gymnasium as gym
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    cfg.env.task = "lift"
    
    # Create env
    from scripts.training.env_utils import OmegaConf
    reward_config = OmegaConf.to_container(cfg.reward, resolve=True) if "reward" in cfg else None
    action_bounds = OmegaConf.to_container(cfg.control.action_bounds, resolve=True) if "control" in cfg else None
    obs_normalization = OmegaConf.to_container(cfg.obs, resolve=True) if "obs" in cfg else None
    
    env_kwargs = dict(
        task=cfg.env.task,
        control_mode=cfg.env.control_mode,
        camera_mode=cfg.env.camera_mode,
        obs_mode=None, 
        domain_randomization=False,
        reward_mode=cfg.reward.reward_mode,
        reward_config=reward_config,
        action_bounds=action_bounds,
        obs_normalization=obs_normalization,
    )
    
    env = gym.make("Track1-v0", num_envs=10, **env_kwargs)
    obs, info = env.reset(seed=1)
    
    # Get obs dict manually at step 0
    obs_dict = env.unwrapped._get_obs_state_dict(info)
    is_grasped_0 = obs_dict["extra"]["is_grasped"]
    print(f"\nStep 0 is_grasped (10 envs): {is_grasped_0.cpu().numpy()}")
    
    # Take one empty step
    actions = torch.zeros((10, 6), device=env.unwrapped.device)
    obs, reward, terminated, truncated, info = env.step(actions)
    
    obs_dict_1 = env.unwrapped._get_obs_state_dict(info)
    is_grasped_1 = obs_dict_1["extra"]["is_grasped"]
    print(f"Step 1 is_grasped (10 envs): {is_grasped_1.cpu().numpy()}")

    env.close()

if __name__ == "__main__":
    main()
