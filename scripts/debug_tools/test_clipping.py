
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
        obs_mode="none",
        domain_randomization=False,
        reward_mode=cfg.reward.reward_mode,
        reward_config=reward_config,
        action_bounds=action_bounds,
        obs_normalization=obs_normalization,
    )
    
    env = gym.make("Track1-v0", num_envs=2, **env_kwargs)
    env.reset(seed=1)
    
    # Init qpos
    qpos0 = env.unwrapped.agent.agents[1].robot.get_qpos()
    
    # Apply action 1.0 (Limit) vs 10.0 (Beyond Limit)
    actions = torch.zeros((2, 6), device=env.unwrapped.device)
    actions[0, 0] = 1.0
    actions[1, 0] = 10.0
    
    env.step(actions)
    
    qpos1 = env.unwrapped.agent.agents[1].robot.get_qpos()
    
    delta0 = (qpos1[0, 0] - qpos0[0, 0]).item()
    delta1 = (qpos1[1, 0] - qpos0[1, 0]).item()
    
    print(f"\nDelta for action 1.0: {delta0:.6f}")
    print(f"Delta for action 10.0: {delta1:.6f}")
    
    if abs(delta0 - delta1) < 1e-4:
        print("SUCCESS: Controller CLIPS actions to [-1, 1]!")
    else:
        print("WARNING: Controller DOES NOT CLIP actions! 10.0 moves more than 1.0!")

    env.close()

if __name__ == "__main__":
    main()
