#!/usr/bin/env python3
"""Generate valid robot poses using Inverse Kinematics.

This script uses the EE controller to move to target positions and records
the resulting joint configurations.

Usage:
    python -m scripts.sample_poses_ik --num-samples 10 --visualize
"""

import argparse
import gymnasium as gym
import torch
import numpy as np
from scripts.track1_env import Track1Env


# Workspace bounds for end-effector (relative to robot base)
WORKSPACE = {
    "x_min": 0.15,
    "x_max": 0.35,
    "y_min": -0.10,
    "y_max": 0.10,
    "z_min": 0.10,
    "z_max": 0.25,
}


def sample_ee_position():
    """Sample a random end-effector position in the workspace."""
    x = np.random.uniform(WORKSPACE["x_min"], WORKSPACE["x_max"])
    y = np.random.uniform(WORKSPACE["y_min"], WORKSPACE["y_max"])
    z = np.random.uniform(WORKSPACE["z_min"], WORKSPACE["z_max"])
    return np.array([x, y, z])


def main():
    parser = argparse.ArgumentParser(description="Generate poses using IK")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of poses to sample")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize found poses")
    parser.add_argument("--steps-per-target", type=int, default=50,
                        help="Steps to take to reach each target")
    args = parser.parse_args()
    
    print("Creating environment...")
    env = gym.make(
        "Track1-v0",
        render_mode="human" if args.visualize else "rgb_array",
        obs_mode="none",
        reward_mode="none",
        task="lift",
        control_mode="pd_ee_delta_pos",
        num_envs=1,
    )
    
    obs, _ = env.reset()
    
    agent = env.unwrapped.agent.agents[0]
    controller = agent.controller.controllers["arm"]
    
    print(f"End link: {controller.ee_link.name}")
    print(f"\nWorkspace bounds:")
    for k, v in WORKSPACE.items():
        print(f"  {k}: {v:.2f}")
    
    valid_poses = []
    
    print(f"\nSampling {args.num_samples} poses...")
    
    for sample_idx in range(args.num_samples):
        # Reset to zero pose first
        env.reset()
        
        # Get current EE position (world frame)
        current_pos = controller.ee_pos.cpu().numpy()[0]
        
        # Sample target position as offset from current (relative movement)
        # Use smaller deltas that are more achievable
        delta = np.array([
            np.random.uniform(0.02, 0.08),   # Forward (smaller)
            np.random.uniform(-0.06, 0.06),  # Left/Right (smaller)
            np.random.uniform(-0.05, 0.05),  # Up/Down (smaller)
        ])
        target_pos = current_pos + delta
        
        # Clamp z to be above table
        target_pos[2] = max(target_pos[2], 0.08)
        
        print(f"\n[{sample_idx+1}/{args.num_samples}] Target: {target_pos}")
        print(f"  Start EE: {current_pos}, Delta: {delta}")
        
        # Step towards target using EE controller
        for step in range(args.steps_per_target):
            current_pos = controller.ee_pos.cpu().numpy()[0]
            diff = target_pos - current_pos
            dist = np.linalg.norm(diff)
            
            if dist < 0.01:  # Reached within 1cm
                break
            
            # Normalize and scale action
            action_arm = np.clip(diff * 5, -1, 1)  # Scale for normalized action
            action_gripper = np.array([0.0])  # Keep gripper fixed
            
            # Combined action for both robots
            action = {
                "so101-0": np.concatenate([action_arm, action_gripper]),
                "so101-1": np.zeros(4),  # Left arm doesn't move
            }
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if args.visualize:
                env.render()
        
        # Check final position
        final_pos = controller.ee_pos.cpu().numpy()[0]
        error = np.linalg.norm(final_pos - target_pos)
        
        # Get joint positions
        qpos = agent.robot.get_qpos().cpu().numpy()[0]
        
        if error < 0.03:  # Accept if within 3cm
            valid_poses.append(qpos)
            print(f"  ✓ Success! Final EE: {final_pos}, Error: {error*100:.1f}cm")
            print(f"  qpos: {qpos}")
            
            if args.visualize:
                input("  Press Enter for next...")
        else:
            print(f"  ✗ Failed to reach target. Error: {error*100:.1f}cm")
    
    print(f"\n=== Results ===")
    print(f"Valid poses: {len(valid_poses)}/{args.num_samples}")
    
    if valid_poses:
        print("\n--- Valid Poses (copy to so101.py) ---")
        for i, qpos in enumerate(valid_poses):
            qpos_str = ", ".join([f"{x:.4f}" for x in qpos])
            print(f"pose_{i+1} = np.array([{qpos_str}])")
    
    env.close()


if __name__ == "__main__":
    main()
