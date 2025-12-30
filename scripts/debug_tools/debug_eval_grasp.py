"""
Debug script to verify grasp detection in eval mode vs training mode.
Simplified version with hardcoded config values.
"""
import sys
sys.path.insert(0, "/home/admin/Desktop/eai-final-project")

import torch
import gymnasium as gym

from scripts.envs.track1_env import Track1Env
from scripts.agents.so101 import SO101

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Testing Grasp Detection: Training vs Eval Environment")
print("=" * 60)

# Hardcoded config values (matching lift.yaml)
reward_config = {
    "reward_mode": "dense",
    "reward_type": "parallel",
    "weights": {"approach": 1.0, "grasp": 1.0, "lift": 1.0, "success": 1.0},
    "grasp_min_force": 1.0,
    "grasp_max_angle": 110,
    "approach_curve": "tanh",
    "approach_tanh_scale": 0.05,
    "gate_lift_with_grasp": True,
    "lift_target": 0.05,
    "stable_hold_time": 3.0,
}

action_bounds = {
    "shoulder_pan": 0.044,
    "shoulder_lift": 0.087,
    "elbow_flex": 0.07,
    "wrist_flex": 0.044,
    "wrist_roll": 0.026,
    "gripper": 0.07,
}

# Create environments
print("\n1. Creating TRAINING environment (eval_mode=False)...")
train_env = gym.make(
    "Track1-v0",
    num_envs=4,
    task="lift",
    control_mode="pd_joint_target_delta_pos",
    camera_mode="direct_pinhole",
    obs_mode="state",
    domain_randomization=False,
    reward_mode="dense",
    reward_config=reward_config,
    action_bounds=action_bounds,
    eval_mode=False,
    render_mode="sensors",
    sim_backend="physx_cuda",
)

print("2. Creating EVAL environment (eval_mode=True)...")
eval_env = gym.make(
    "Track1-v0",
    num_envs=4,
    task="lift",
    control_mode="pd_joint_target_delta_pos",
    camera_mode="direct_pinhole",
    obs_mode="state",
    domain_randomization=False,
    reward_mode="dense",
    reward_config=reward_config,
    action_bounds=action_bounds,
    eval_mode=True,
    render_mode="sensors",
    sim_backend="physx_cuda",
)

# Reset both
train_obs, train_info = train_env.reset()
eval_obs, eval_info = eval_env.reset()

print(f"\n3. Environment eval_mode flags:")
print(f"   Train env.eval_mode: {train_env.unwrapped.eval_mode}")
print(f"   Eval  env.eval_mode: {eval_env.unwrapped.eval_mode}")

grasp_min_force = reward_config["grasp_min_force"]
grasp_max_angle = reward_config["grasp_max_angle"]
print(f"\n4. Grasp detection params: min_force={grasp_min_force}, max_angle={grasp_max_angle}")

print("\n5. Running 20 steps with gripper closing action...")

for step in range(20):
    action = torch.zeros((4, 6), device=device)
    action[:, 5] = -1.0  # Close gripper
    
    train_obs, train_rew, train_term, train_trunc, train_info = train_env.step(action)
    eval_obs, eval_rew, eval_term, eval_trunc, eval_info = eval_env.step(action)
    
    train_rc = train_info.get("reward_components", {})
    eval_rc = eval_info.get("reward_components", {})
    eval_rcpe = eval_info.get("reward_components_per_env", None)
    
    train_base = train_env.unwrapped
    eval_base = eval_env.unwrapped
    
    train_is_grasped = train_base.right_arm.is_grasping(
        train_base.red_cube, min_force=grasp_min_force, max_angle=grasp_max_angle
    )
    eval_is_grasped = eval_base.right_arm.is_grasping(
        eval_base.red_cube, min_force=grasp_min_force, max_angle=grasp_max_angle
    )
    
    train_grasp = train_rc.get('grasp', 'N/A')
    eval_grasp = eval_rc.get('grasp', 'N/A')
    
    # Format tensors for display
    if hasattr(train_grasp, 'item'):
        train_grasp = f"{train_grasp.item():.4f}"
    if hasattr(eval_grasp, 'item'):
        eval_grasp = f"{eval_grasp.item():.4f}"
    
    print(f"\nStep {step}:")
    print(f"  Train is_grasped (direct): {train_is_grasped.tolist()}")
    print(f"  Eval  is_grasped (direct): {eval_is_grasped.tolist()}")
    print(f"  Train reward_components.grasp: {train_grasp}")
    print(f"  Eval  reward_components.grasp: {eval_grasp}")
    print(f"  Eval reward_components_per_env exists: {eval_rcpe is not None}")
    
    if eval_rcpe is not None:
        grasp_per_env = eval_rcpe.get("grasp", None)
        if grasp_per_env is not None:
            print(f"  Eval per-env grasp values: {grasp_per_env.tolist()}")

train_env.close()
eval_env.close()
print("\n" + "=" * 60)
print("Debug complete")
