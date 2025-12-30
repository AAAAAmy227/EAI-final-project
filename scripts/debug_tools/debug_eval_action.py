"""
Debug: Compare action application between Train and Eval environments.
Focus on gripper behavior.
"""
import sys
sys.path.insert(0, "/home/admin/Desktop/eai-final-project")

import torch
import gymnasium as gym

from scripts.envs.track1_env import Track1Env
from scripts.agents.so101 import SO101
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("Testing Action Application: Train vs Eval Environment")
print("=" * 70)

# Config
action_bounds = {
    "shoulder_pan": 0.044,
    "shoulder_lift": 0.087,
    "elbow_flex": 0.07,
    "wrist_flex": 0.044,
    "wrist_roll": 0.026,
    "gripper": 0.2,
}

reward_config = {
    "weights": {"approach": 1.0, "grasp": 1.0, "lift": 1.0, "success": 1.0},
    "grasp_min_force": 1.0,
    "grasp_max_angle": 110,
}

# Create TRAIN environment (exactly like runner.py does)
print("\n1. Creating TRAIN environment...")
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
    reconfiguration_freq=None,  # Train: no reconfiguration
)
train_env = ManiSkillVectorEnv(train_env, num_envs=4, auto_reset=True, ignore_terminations=False)

# Create EVAL environment (exactly like runner.py does)
print("2. Creating EVAL environment...")
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
    reconfiguration_freq=1,  # Eval: reconfigure every reset
)
eval_env = ManiSkillVectorEnv(eval_env, num_envs=4, ignore_terminations=True, record_metrics=True)

# Reset both
train_obs, train_info = train_env.reset()
eval_obs, eval_info = eval_env.reset()

# Get base environments
train_base = train_env.unwrapped
eval_base = eval_env.unwrapped

# Check initial gripper state
print("\n3. Initial gripper qpos after reset:")
train_qpos = train_base.right_arm.robot.qpos
eval_qpos = eval_base.right_arm.robot.qpos
print(f"  Train gripper qpos: {train_qpos[:, 5].tolist()}")
print(f"  Eval  gripper qpos: {eval_qpos[:, 5].tolist()}")

# Check target qpos (controller internal state)
print("\n4. Initial controller target_qpos:")
train_target = train_base.right_arm.controller.target_qpos if hasattr(train_base.right_arm.controller, 'target_qpos') else "N/A"
eval_target = eval_base.right_arm.controller.target_qpos if hasattr(eval_base.right_arm.controller, 'target_qpos') else "N/A"
print(f"  Train target_qpos: {train_target}")
print(f"  Eval  target_qpos: {eval_target}")

# Apply action to OPEN gripper (+1 for normalized action should open)
print("\n5. Applying action to OPEN gripper (action[5] = +1.0)...")
action = torch.zeros((4, 6), device=device)
action[:, 5] = +1.0  # Try to open gripper

for step in range(30):
    train_obs, _, _, _, train_info = train_env.step(action)
    eval_obs, _, _, _, eval_info = eval_env.step(action)

# Check gripper after opening
print("\n6. Gripper qpos after 30 steps of OPEN action:")
train_qpos = train_base.right_arm.robot.qpos
eval_qpos = eval_base.right_arm.robot.qpos
print(f"  Train gripper qpos: {train_qpos[:, 5].tolist()}")
print(f"  Eval  gripper qpos: {eval_qpos[:, 5].tolist()}")

# Now apply action to CLOSE gripper
print("\n7. Applying action to CLOSE gripper (action[5] = -1.0)...")
action[:, 5] = -1.0  # Try to close gripper

for step in range(30):
    train_obs, _, _, _, train_info = train_env.step(action)
    eval_obs, _, _, _, eval_info = eval_env.step(action)

# Check gripper after closing
print("\n8. Gripper qpos after 30 steps of CLOSE action:")
train_qpos = train_base.right_arm.robot.qpos
eval_qpos = eval_base.right_arm.robot.qpos
print(f"  Train gripper qpos: {train_qpos[:, 5].tolist()}")
print(f"  Eval  gripper qpos: {eval_qpos[:, 5].tolist()}")

# Check action bounds
print("\n9. Active action bounds:")
print(f"  Train: {train_base.right_arm._active_action_bounds}")
print(f"  Eval:  {eval_base.right_arm._active_action_bounds}")

# Cleanup
train_env.close()
eval_env.close()
print("\n" + "=" * 70)
print("Debug complete")
