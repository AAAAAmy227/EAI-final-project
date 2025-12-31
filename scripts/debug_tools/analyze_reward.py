#!/usr/bin/env python3
"""
Analyze trajectory data from eai-dataset to validate reward design.

This script computes reward-relevant metrics from real trajectories:
1. Gripper-to-cube distance over time (for reach_threshold validation)
2. Cube height progression (for lift_target validation)
3. Phase timing (when reach/grasp/lift transitions occur)
4. Reward component distributions

NOTE: The real data doesn't include gripper position or cube position directly.
      We need to estimate from the joint states and forward kinematics,
      OR use the data that IS available in the parquet files.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset root
DATASET_ROOT = Path("/home/admin/Desktop/eai-final-project/eai-dataset")

def load_all_parquets(task_dir: Path) -> pd.DataFrame:
    """Load all parquet files from a task directory."""
    data_dir = task_dir / "data"
    all_dfs = []
    for parquet_file in sorted(data_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_file)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def analyze_joint_trajectories(task_name: str):
    """Analyze joint position trajectories for reward insights."""
    task_dir = DATASET_ROOT / task_name
    
    # Load meta info
    with open(task_dir / "meta" / "info.json") as f:
        info = json.load(f)
    
    fps = info["fps"]
    action_names = info["features"]["action"]["names"]
    
    # Load trajectory data
    df = load_all_parquets(task_dir)
    
    # Convert action and state to numpy arrays
    actions = np.stack(df["action"].values)  # (N, action_dim)
    states = np.stack(df["observation.state"].values)  # (N, state_dim)
    episode_indices = df["episode_index"].values
    frame_indices = df["frame_index"].values
    timestamps = df["timestamp"].values
    
    total_episodes = info["total_episodes"]
    
    print(f"\n{'='*60}")
    print(f"Task: {task_name.upper()}")
    print(f"{'='*60}")
    print(f"Episodes: {total_episodes}, FPS: {fps}")
    print(f"Action/State names: {action_names}")
    
    # Analyze gripper behavior over episode progression
    # Gripper is the last joint (index 5 for single-arm, 5 and 11 for dual-arm)
    gripper_idx = 5 if len(action_names) == 6 else 5  # Right arm gripper for dual
    
    # Collect per-episode statistics
    episode_stats = []
    for ep_idx in range(total_episodes):
        ep_mask = episode_indices == ep_idx
        ep_states = states[ep_mask]
        ep_frames = frame_indices[ep_mask]
        ep_times = timestamps[ep_mask]
        
        if len(ep_states) < 2:
            continue
        
        # Gripper trajectory
        gripper_pos = ep_states[:, gripper_idx]
        
        # Find when gripper closes (assuming lower = closed for grasping)
        # This is a heuristic - actual close position depends on robot calibration
        gripper_min = gripper_pos.min()
        gripper_max = gripper_pos.max()
        gripper_range = gripper_max - gripper_min
        
        # Estimate "grasp time" as when gripper first closes significantly
        if gripper_range > 5:  # Significant gripper motion (degrees)
            close_threshold = gripper_min + 0.3 * gripper_range  # 30% from min
            close_frames = np.where(gripper_pos < close_threshold)[0]
            first_close_frame = close_frames[0] if len(close_frames) > 0 else len(ep_states) - 1
            first_close_time = ep_times[first_close_frame]
        else:
            first_close_frame = len(ep_states) - 1
            first_close_time = ep_times[-1]
        
        episode_stats.append({
            "episode": ep_idx,
            "length": len(ep_states),
            "duration": float(ep_times[-1]),
            "gripper_min": float(gripper_min),
            "gripper_max": float(gripper_max),
            "gripper_range": float(gripper_range),
            "first_close_frame": int(first_close_frame),
            "first_close_time": float(first_close_time),
            "close_fraction": first_close_frame / len(ep_states),  # % of episode when grasp happens
        })
    
    stats_df = pd.DataFrame(episode_stats)
    
    print(f"\n--- Gripper Behavior Analysis ---")
    print(f"Gripper range (degrees): min={stats_df['gripper_range'].min():.1f}, "
          f"max={stats_df['gripper_range'].max():.1f}, mean={stats_df['gripper_range'].mean():.1f}")
    print(f"First grasp timing (fraction of episode):")
    print(f"  Mean: {stats_df['close_fraction'].mean():.2%}")
    print(f"  Std: {stats_df['close_fraction'].std():.2%}")
    print(f"  Min: {stats_df['close_fraction'].min():.2%}, Max: {stats_df['close_fraction'].max():.2%}")
    print(f"First grasp timing (seconds):")
    print(f"  Mean: {stats_df['first_close_time'].mean():.2f}s")
    print(f"  Std: {stats_df['first_close_time'].std():.2f}s")
    
    # Analyze joint motion patterns
    print(f"\n--- Joint Motion Patterns ---")
    
    # Compute velocity profiles for each joint across episodes
    for j_idx, j_name in enumerate(action_names[:6]):  # First 6 for single arm
        # Collect all velocities
        all_velocities = []
        for ep_idx in range(min(total_episodes, 50)):  # Sample 50 episodes
            ep_mask = episode_indices == ep_idx
            ep_states = states[ep_mask]
            if len(ep_states) > 1:
                velocities = np.abs(np.diff(ep_states[:, j_idx])) * fps  # deg/s
                all_velocities.extend(velocities)
        
        all_velocities = np.array(all_velocities)
        print(f"  {j_name}: mean_vel={all_velocities.mean():.1f}°/s, "
              f"max_vel={all_velocities.max():.1f}°/s, p95_vel={np.percentile(all_velocities, 95):.1f}°/s")
    
    return stats_df

def suggest_reward_parameters(stats_dict):
    """Based on trajectory analysis, suggest reward parameters."""
    print(f"\n{'='*60}")
    print("REWARD PARAMETER SUGGESTIONS")
    print(f"{'='*60}")
    
    for task, stats_df in stats_dict.items():
        print(f"\n--- {task.upper()} ---")
        
        # Grasp timing suggests reach phase duration
        avg_grasp_fraction = stats_df['close_fraction'].mean()
        avg_grasp_time = stats_df['first_close_time'].mean()
        
        print(f"Observation: Grasp occurs at ~{avg_grasp_fraction:.0%} of episode ({avg_grasp_time:.1f}s)")
        
        if avg_grasp_fraction > 0.5:
            print("  => Reach phase takes majority of episode")
            print("  => Consider higher reach_reward weight or lower reach_scale")
        elif avg_grasp_fraction < 0.3:
            print("  => Quick reaching, most time spent on manipulation")
            print("  => Current reach_threshold may be appropriate")
        
        # Check gripper range for grasp detection
        avg_gripper_range = stats_df['gripper_range'].mean()
        print(f"Observation: Average gripper motion range = {avg_gripper_range:.1f}°")
        
        if avg_gripper_range < 20:
            print("  => Small gripper motion - grasping may be subtle")
            print("  => Consider using gripper activation rate in reward")

def main():
    print("=" * 80)
    print("Trajectory Analysis for Reward Design Validation")
    print("=" * 80)
    
    tasks = ["lift", "stack"]  # Skip sort for now (dual-arm is more complex)
    stats_dict = {}
    
    for task in tasks:
        stats_df = analyze_joint_trajectories(task)
        stats_dict[task] = stats_df
    
    suggest_reward_parameters(stats_dict)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
