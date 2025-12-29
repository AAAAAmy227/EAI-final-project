
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from scripts.training.env_utils import make_env

def print_obs_keys(obs, prefix=""):
    if isinstance(obs, dict):
        for k in sorted(obs.keys()):
            v = obs[k]
            if isinstance(v, dict):
                print(f"{prefix}{k}: (dict)")
                print_obs_keys(v, prefix + "  ")
            else:
                print(f"{prefix}{k}: {v.shape}")
    else:
        print(f"{prefix}tensor: {obs.shape}")

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    print("--- Training Env Obs Keys ---")
    train_envs = make_env(cfg, num_envs=2, for_eval=False)
    obs, _ = train_envs.reset()
    # ManiSkill VectorEnv reset returns dict of tensors [num_envs, ...]
    print_obs_keys(obs)
    train_envs.close()

    print("\n--- Evaluation Env Obs Keys ---")
    eval_envs = make_env(cfg, num_envs=2, for_eval=True)
    obs, _ = eval_envs.reset()
    print_obs_keys(obs)
    eval_envs.close()

if __name__ == "__main__":
    main()
