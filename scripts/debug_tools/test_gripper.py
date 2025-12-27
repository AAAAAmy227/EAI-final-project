
import numpy as np
import torch
import gymnasium as gym
import hydra
from omegaconf import DictConfig
from scripts.training.common import make_env

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    cfg.env.task = "lift"
    
    # Create env without video/wrappers for speed
    from scripts.training.common import OmegaConf
    reward_config = OmegaConf.to_container(cfg.reward, resolve=True) if "reward" in cfg else None
    action_bounds = OmegaConf.to_container(cfg.control.action_bounds, resolve=True) if "control" in cfg else None
    obs_normalization = OmegaConf.to_container(cfg.obs, resolve=True) if "obs" in cfg else None
    
    env_kwargs = dict(
        task=cfg.env.task,
        control_mode=cfg.env.control_mode,
        camera_mode=cfg.env.camera_mode,
        obs_mode="none",
        domain_randomization=False,
        reward_mode=cfg.reward.reward_mode,
        reward_config=reward_config,
        action_bounds=action_bounds,
        obs_normalization=obs_normalization,
    )
    
    env = gym.make("Track1-v0", num_envs=2, **env_kwargs)
    env.reset(seed=1)
    
    # 1. Close gripper (Action -1.0) on env 0
    # 2. Open gripper (Action 1.0) on env 1
    # Gripper is the 6th action dimension
    actions = torch.zeros((2, 6), device=env.unwrapped.device)
    actions[0, 5] = -1.0
    actions[1, 5] = 1.0
    
    obs, reward, terminated, truncated, info = env.step(actions)
    
    # Check is_grasped in info or obs
    obs_dict = env.unwrapped._get_obs_state_dict(info={})
    is_grasped = obs_dict["extra"]["is_grasped"]
    
    print(f"\nAction -1.0 gripper -> is_grasped: {is_grasped[0].item()}")
    print(f"Action  1.0 gripper -> is_grasped: {is_grasped[1].item()}")
    
    # Also check qpos change
    qpos = obs_dict["agent"]["so101-1"]["qpos"]
    print(f"\nEnv 0 qpos[5] (gripper): {qpos[0, 5].item():.4f}")
    print(f"Env 1 qpos[5] (gripper): {qpos[1, 5].item():.4f}")

    env.close()

if __name__ == "__main__":
    main()
