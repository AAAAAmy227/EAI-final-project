
import gymnasium as gym
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.training.info_utils import get_info_field, get_reward_components

def investigate_info_flow():
    print("=== Testing Info Flow in ManiSkill VectorEnv ===")
    
    # Use gym.make directly to bypass complex config requirements
    # Track1Env-v1 is our custom env
    env_id = "PickCubeSO101-v1"
    num_envs = 2
    
    print(f"Creating {num_envs} environments of {env_id}...")
    envs = gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
    )
    
    obs, info = envs.reset(seed=42)
    
    print("\n--- Reset Info ---")
    print(f"Top-level keys: {list(info.keys())}")
    if "reward_components" in info:
        print(f"Reset reward_components: {info['reward_components']}")

    # Take steps until one env is done
    max_steps = 100
    for i in range(max_steps):
        actions = envs.action_space.sample()
        obs, reward, terminated, truncated, info = envs.step(actions)
        done = terminated | truncated
        
        if done.any() or i == 0 or i == max_steps - 1:
            print(f"\n--- Step {i} ---")
            print(f"Done: {done}")
            print(f"Top-level keys: {list(info.keys())}")
            
            if done.any():
                print("Detected DONE!")
                if "final_info" in info:
                    # In ManiSkill VectorEnv, final_info is a list/array of dicts
                    final_info_bag = info["final_info"]
                    print(f"final_info available: {type(final_info_bag)}")
                    
                    for idx, fin in enumerate(final_info_bag):
                        if fin is not None:
                            print(f"  Env {idx} final_info keys: {list(fin.keys())}")
                            if "reward_components" in fin:
                                print(f"  Env {idx} final reward_components: {fin['reward_components']}")
                
                # Use our utility and see what it finds
                success = get_info_field(info, "success")
                print(f"Utility get_info_field('success'): {success}")
                
                reward_comps = get_reward_components(info)
                print(f"Utility get_reward_components: {list(reward_comps.keys()) if reward_comps else None}")
                break
        
    envs.close()

if __name__ == "__main__":
    investigate_info_flow()
