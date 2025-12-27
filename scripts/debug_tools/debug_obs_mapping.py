
import numpy as np
import torch
import gymnasium as gym
import hydra
from omegaconf import DictConfig
from scripts.training.common import make_env

def get_obs_mapping(space, prefix=""):
    mapping = []
    if isinstance(space, gym.spaces.Dict):
        for k in sorted(space.keys()):
            mapping.extend(get_obs_mapping(space[k], prefix + k + "."))
    elif isinstance(space, gym.spaces.Box):
        size = np.prod(space.shape)
        for i in range(size):
            mapping.append(f"{prefix}[{i}]")
    return mapping

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Use task=lift to match current debug
    cfg.env.task = "lift"
    
    # We need to see the DICT space
    # make_env flattens it, so we need to look at the underlying env
    from scripts.training.common import OmegaConf
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
    
    print("\nObservation Space Mapping (Sorted Keys):")
    mapping = get_obs_mapping(raw_env.observation_space)
    for i, name in enumerate(mapping):
        print(f"{i:2d}: {name}")
    
    print(f"\nTotal Dimensions: {len(mapping)}")
    
    # Also print initial values to compare with mapping
    obs, _ = raw_env.reset(seed=1)
    
    # Flatten manually using sorted keys
    def flatten_recursive(o):
        vals = []
        if isinstance(o, dict):
            for k in sorted(o.keys()):
                vals.extend(flatten_recursive(o[k]))
        else:
            v = o.flatten().cpu().numpy()
            vals.extend(v.tolist())
        return vals

    flat_vals = flatten_recursive(obs)
    print("\nInitial Observation Values (Step 0, Seed 1):")
    for i, v in enumerate(flat_vals):
        print(f"{i:2d}: {v:8.4f}  ({mapping[i]})")

    raw_env.close()

if __name__ == "__main__":
    main()
