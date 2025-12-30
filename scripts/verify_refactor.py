import gymnasium as gym
import torch
import numpy as np
import sapien.core as sapien
from scripts.envs.track1_env import Track1Env

def test_env_init():
    print("Testing Track1Env initialization with default config...")
    env = gym.make("Track1-v0", render_mode="rgb_array")
    obs, info = env.reset()
    if isinstance(obs, dict):
        print("Reset successful. Obs keys:", obs.keys())
    else:
        print("Reset successful. Obs shape:", obs.shape)
    
    print("\nTesting Track1Env with explicit cfg object...")
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "env": {
            "task": "lift",
            "domain_randomization": False,
            "camera_mode": "direct_pinhole"
        },
        "reward": {
            "weights": {"lift": 10.0}
        },
        "obs": {
            "enabled": True
        }
    })
    env_cfg = gym.make("Track1-v0", cfg=cfg, render_mode="rgb_array")
    obs, info = env_cfg.reset()
    print("Reset with cfg successful.")
    print("Reward weights:", env_cfg.unwrapped.reward_weights)
    assert env_cfg.unwrapped.reward_weights["lift"] == 10.0
    
    print("\nVerification complete!")

if __name__ == "__main__":
    try:
        test_env_init()
    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()
