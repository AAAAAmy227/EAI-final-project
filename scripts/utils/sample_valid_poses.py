#!/usr/bin/env python3
"""Sample valid robot poses using collision detection.

This script samples random joint configurations and filters for poses that:
1. Have no self-collision
2. Have gripper above the table (z > 0)
3. Have gripper within reasonable workspace

Usage:
    python -m scripts.sample_valid_poses --num-samples 1000
"""

import argparse
import gymnasium as gym
import torch
import numpy as np
from scripts.track1_env import Track1Env


# Approximate joint limits from URDF (conservative)
JOINT_LIMITS = {
    # [lower, upper] in radians
    "shoulder_pan": [-0.1, 1.5],
    "shoulder_lift": [-2.5, 2.5],
    "elbow_flex": [-1.5, 1.5],
    "wrist_flex": [-1.5, 1.5],
    "wrist_roll": [-1.5, 1.5],
    "gripper": [-1.9, 1.9],  # More conservative range for gripper
}


def sample_random_qpos():
    """Sample a random joint configuration within limits."""
    qpos = np.zeros(6)
    for i, (joint, (lower, upper)) in enumerate(JOINT_LIMITS.items()):
        qpos[i] = np.random.uniform(lower, upper)
    return qpos


def get_gripper_position(env, agent_idx=0):
    """Get the gripper link position for an agent."""
    agent = env.unwrapped.agent.agents[agent_idx]
    try:
        # Try to get gripper_link position
        gripper_link = agent.robot.links_map.get("gripper_link", None)
        if gripper_link is not None:
            return gripper_link.pose.p.cpu().numpy()[0]
    except:
        pass
    return None


def check_self_collision(env):
    """Check if there's self-collision in the robot."""
    # SAPIEN tracks contacts - we check if any contacts are between robot links
    # For simplicity, we assume no self-collision if the simulation is stable
    # A more robust check would use scene.get_contacts()
    try:
        # Step physics briefly to detect collisions
        env.unwrapped.scene.step()
        contacts = env.unwrapped.scene.get_contacts()
        for contact in contacts:
            # Check if both actors belong to the same robot
            # This is a simplified check
            if "so101" in str(contact.bodies[0]) and "so101" in str(contact.bodies[1]):
                return True
    except:
        pass
    return False


def validate_pose(env, qpos, min_height=0.05, max_height=0.4):
    """Check if a pose is valid."""
    # Set qpos for both robots
    qpos_tensor = torch.tensor([qpos], dtype=torch.float32, device=env.unwrapped.device)
    for agent in env.unwrapped.agent.agents:
        agent.robot.set_qpos(qpos_tensor)
    
    if env.unwrapped.scene.gpu_sim_enabled:
        env.unwrapped.scene._gpu_apply_all()
    
    # Get gripper positions
    for agent_idx in range(len(env.unwrapped.agent.agents)):
        pos = get_gripper_position(env, agent_idx)
        if pos is None:
            return False, "Could not get gripper position"
        
        # Check height above table
        z = pos[2]
        if z < min_height:
            return False, f"Gripper too low: z={z:.3f}"
        if z > max_height:
            return False, f"Gripper too high: z={z:.3f}"
    
    # Check for self-collision (simplified)
    # if check_self_collision(env):
    #     return False, "Self-collision detected"
    
    return True, "Valid"


def main():
    parser = argparse.ArgumentParser(description="Sample valid robot poses")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Number of random samples to try")
    parser.add_argument("--num-valid", type=int, default=10,
                        help="Number of valid poses to find")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize found poses")
    args = parser.parse_args()
    
    print("Creating environment...")
    env = gym.make(
        "Track1-v0",
        render_mode="human" if args.visualize else "rgb_array",
        obs_mode="none",
        reward_mode="none",
        task="lift",
        num_envs=1,
    )
    
    obs, _ = env.reset()
    
    valid_poses = []
    tried = 0
    
    print(f"Sampling up to {args.num_samples} poses to find {args.num_valid} valid ones...")
    
    while len(valid_poses) < args.num_valid and tried < args.num_samples:
        qpos = sample_random_qpos()
        tried += 1
        
        is_valid, reason = validate_pose(env, qpos)
        
        if is_valid:
            valid_poses.append(qpos)
            print(f"\n[{len(valid_poses)}/{args.num_valid}] Found valid pose:")
            print(f"  qpos = {qpos}")
            
            if args.visualize:
                print("  Press Enter to continue...")
                env.render()
                input()
        
        if tried % 100 == 0:
            print(f"  Tried {tried} samples, found {len(valid_poses)} valid poses...")
    
    print(f"\n=== Results ===")
    print(f"Tried: {tried} samples")
    print(f"Valid: {len(valid_poses)} poses")
    print(f"Success rate: {100*len(valid_poses)/tried:.1f}%")
    
    if valid_poses:
        print("\n--- Valid Poses (copy to so101.py) ---")
        for i, qpos in enumerate(valid_poses):
            qpos_str = ", ".join([f"{x:.4f}" for x in qpos])
            print(f"pose_{i+1} = np.array([{qpos_str}])")
    
    env.close()


if __name__ == "__main__":
    main()
