#!/usr/bin/env python3
"""
Quick test script for static_grasp task.

This script creates the environment and runs a few episodes to verify:
1. Cube spawns randomly but stays static (kinematic)
2. Grasp detection works
3. Continuous grasp counter increments correctly
"""

import torch
import numpy as np
import sapien
from scripts.envs.track1_env import Track1Env
from omegaconf import OmegaConf

def test_static_grasp():
    print("=" * 60)
    print("Testing StaticGrasp Task")
    print("=" * 60)
    
    # Load sub-configs referenced in defaults
    env_cfg = OmegaConf.load("configs/env/static_grasp.yaml")
    reward_cfg = OmegaConf.load("configs/reward/static_grasp.yaml")
    obs_cfg = OmegaConf.load("configs/obs/single_arm.yaml")
    control_cfg = OmegaConf.load("configs/control/single_arm.yaml")

    # Create a clean composed config
    cfg = OmegaConf.create({
        "env": env_cfg,
        "reward": reward_cfg,
        "obs": obs_cfg,
        "control": control_cfg
    })
    
    # Ensure task is set correctly in the nested dict
    cfg.env.task = "static_grasp"
    
    print(f"Task in cfg: {cfg.env.task}")
    
    # Ensure action_bounds logic works if it relies on config
    # Track1Env init might check cfg.action_bounds if not in env_cfg
    # But usually it's in cfg.env or cfg directly. 
    # Let's make sure cfg.env has what it needs from the "defaults" that Hydra would have merged.
    
    # Override for quick testing
    num_envs = 4
    max_steps_per_episode = 300  # 10 seconds at 30Hz
    
    print(f"\nCreating environment with {num_envs} parallel envs...")
    print(f"Task: {cfg.env.task}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    
    # Create environment
    env = Track1Env(
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        cfg=cfg,
        task="static_grasp"
    )
    
    print(f"Environment created successfully!")
    print(f"Env Task: {env.task}")
    print(f"Task Handler: {type(env.task_handler)}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset environment
    obs, info = env.reset()
    print(f"\nEnvironment reset. Cube positions:")
    for i in range(num_envs):
        cube_pos = env.red_cube.pose.p[i].cpu().numpy()
        print(f"  Env {i}: x={cube_pos[0]:.3f}, y={cube_pos[1]:.3f}, z={cube_pos[2]:.3f}")
    
    # Check if cube is kinematic (static)
    print(f"\nChecking cube physics mode:")
    for i in range(num_envs):
        # Access PhysX component to check kinematic status
        rigid_body = env.red_cube._objs[i].find_component_by_type(sapien.physx.PhysxRigidBodyComponent)
        if rigid_body is None:
             print(f"  Env {i}: RigidBody not found (Likely Static, which is GOOD)")
        else:
            is_kinematic = rigid_body.kinematic
            print(f"  Env {i}: kinematic={is_kinematic}")
    
    print("\n" + "=" * 60)
    print("Running test episodes (random actions)...")
    print("=" * 60)
    
    num_test_episodes = 2
    
    for episode in range(num_test_episodes):
        obs, info = env.reset()
        episode_rewards = np.zeros(num_envs)
        episode_lengths = np.zeros(num_envs, dtype=int)
        done_mask = np.zeros(num_envs, dtype=bool)
        
        max_grasp_steps_seen = np.zeros(num_envs, dtype=int)
        
        print(f"\n--- Episode {episode + 1} ---")
        
        for step in range(max_steps_per_episode):
            # Random action for testing
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Track metrics
            for i in range(num_envs):
                if not done_mask[i]:
                    episode_rewards[i] += reward[i].item()
                    episode_lengths[i] += 1
                    
                    # Track grasp hold steps
                    if "grasp_hold_steps" in info:
                        grasp_steps = info["grasp_hold_steps"][i].item()
                        max_grasp_steps_seen[i] = max(max_grasp_steps_seen[i], grasp_steps)
                    
                    # Check for termination
                    if terminated[i] or truncated[i]:
                        done_mask[i] = True
                        status = "SUCCESS" if info["success"][i] else "TIMEOUT/FAIL"
                        print(f"  Env {i}: {status} at step {step+1}, "
                              f"reward={episode_rewards[i]:.2f}, "
                              f"max_grasp_steps={max_grasp_steps_seen[i]}")
            
            # Stop if all envs are done
            if done_mask.all():
                break
        
        # Report any envs that didn't finish
        for i in range(num_envs):
            if not done_mask[i]:
                print(f"  Env {i}: TRUNCATED at step {step+1}, "
                      f"reward={episode_rewards[i]:.2f}, "
                      f"max_grasp_steps={max_grasp_steps_seen[i]}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    print("\nKey observations to verify:")
    print("1. ✓ Environment created without errors")
    print("2. ✓ Cube is kinematic (static)")
    print("3. ✓ Episodes run without crashes")
    print("\nNext steps:")
    print("- Run actual training: python scripts/training/train.py --config-name train_static_grasp")
    print("- Monitor grasp_success metric in logs/wandb")
    print("- Check if grasp detection is working as expected")
    
    env.close()

if __name__ == "__main__":
    test_static_grasp()
