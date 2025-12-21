#!/usr/bin/env python3
"""Visualize different keyframes for SO101 robot.

Usage:
    python -m scripts.view_keyframes
    python -m scripts.view_keyframes --keyframe zero
"""

import argparse
import gymnasium as gym
import torch
import numpy as np
from scripts.track1_env import Track1Env


# Define keyframes - sampled valid poses (gripper above table)
KEYFRAMES = {
    "zero": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    # New poses generated via joint control (random walk from zero, gripper z > 0.05)
    "pose_1": np.array([-0.0504, 0.0133, 0.0016, -0.0215, -0.0061, -0.1555]),
    "pose_2": np.array([-0.0461, 0.0496, 0.0305, -0.0267, -0.0008, -0.1685]),
    "pose_3": np.array([0.0146, 0.1061, 0.0112, 0.0220, -0.0252, -0.0255]),
    "pose_4": np.array([-0.0031, 0.1147, -0.0272, 0.0240, -0.1240, -0.1049]),
    "pose_5": np.array([-0.0292, 0.1395, -0.1017, 0.0651, -0.1583, -0.0529]),
}


def set_robot_qpos(env, qpos):
    """Set qpos for all robots in the environment."""
    qpos_tensor = torch.tensor([qpos], dtype=torch.float32, device=env.unwrapped.device)
    for agent in env.unwrapped.agent.agents:
        agent.robot.set_qpos(qpos_tensor)
    # Apply physics state
    if env.unwrapped.scene.gpu_sim_enabled:
        env.unwrapped.scene._gpu_apply_all()


def main():
    parser = argparse.ArgumentParser(description="View SO101 keyframes")
    parser.add_argument("--keyframe", type=str, default="zero",
                        choices=list(KEYFRAMES.keys()),
                        help="Initial keyframe to display")
    args = parser.parse_args()
    
    print("Creating environment...")
    env = gym.make(
        "Track1-v0",
        render_mode="human",
        obs_mode="none",
        reward_mode="none",
        task="lift",
        num_envs=1,
    )
    
    obs, _ = env.reset()
    
    keyframe_list = list(KEYFRAMES.keys())
    current_idx = keyframe_list.index(args.keyframe) if args.keyframe in keyframe_list else 0
    current_keyframe = keyframe_list[current_idx]
    set_robot_qpos(env, KEYFRAMES[current_keyframe])
    
    print(f"\n=== Keyframe Viewer ===")
    print(f"Available: {keyframe_list}")
    print(f"\nControls:")
    print("  - N: Next pose")
    print("  - P: Previous pose")
    print("  - Q: Quit")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom")
    print(f"\n[{current_idx+1}/{len(keyframe_list)}] {current_keyframe}")
    print(f"qpos: {KEYFRAMES[current_keyframe]}")
    
    try:
        while True:
            env.render()
            
            # Simple key input via terminal
            import sys, select
            if select.select([sys.stdin], [], [], 0.01)[0]:
                key = sys.stdin.read(1).lower()
                if key == 'n':
                    current_idx = (current_idx + 1) % len(keyframe_list)
                    current_keyframe = keyframe_list[current_idx]
                    set_robot_qpos(env, KEYFRAMES[current_keyframe])
                    print(f"\n[{current_idx+1}/{len(keyframe_list)}] {current_keyframe}")
                    print(f"qpos: {KEYFRAMES[current_keyframe]}")
                elif key == 'p':
                    current_idx = (current_idx - 1) % len(keyframe_list)
                    current_keyframe = keyframe_list[current_idx]
                    set_robot_qpos(env, KEYFRAMES[current_keyframe])
                    print(f"\n[{current_idx+1}/{len(keyframe_list)}] {current_keyframe}")
                    print(f"qpos: {KEYFRAMES[current_keyframe]}")
                elif key == 'q':
                    break
                    
    except (KeyboardInterrupt, AttributeError) as e:
        print(f"\nExiting... ({e})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
