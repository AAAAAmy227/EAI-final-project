
import numpy as np
import torch
import gymnasium as gym
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    cfg.env.task = "lift"
    
    # Create env
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
    
    env = gym.make("Track1-v0", num_envs=1, **env_kwargs)
    env.reset(seed=1)
    
    robot = env.unwrapped.agent.agents[1].robot
    active_joints = robot.get_active_joints()
    print("\nRobot Active Joints Order:")
    for i, j in enumerate(active_joints):
        print(f"{i}: {j.name}")
    
    env.close()

if __name__ == "__main__":
    main()
