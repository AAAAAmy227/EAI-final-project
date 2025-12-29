"""
Debug: Compare RAW observations between Train and Eval environments.
This helps find any fundamental differences.
"""
import sys
sys.path.insert(0, "/home/admin/Desktop/eai-final-project")

import torch
import gymnasium as gym

from scripts.track1_env import Track1Env
from scripts.so101 import SO101
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("Comparing RAW Observations: Train vs Eval Environment")
print("=" * 70)

# Full config matching training
action_bounds = {
    "shoulder_pan": 0.044,
    "shoulder_lift": 0.087,
    "elbow_flex": 0.07,
    "wrist_flex": 0.044,
    "wrist_roll": 0.026,
    "gripper": 0.2,
}

reward_config = {
    "weights": {"approach": 1.0, "grasp": 1.0, "lift": 1.0, "success": 1.0, "fail": -1.0},
    "grasp_min_force": 1.0,
    "grasp_max_angle": 110,
    "gate_lift_with_grasp": True,
}

obs_normalization = {
    "enabled": False,
    "qpos_scale": 1.5,
    "qvel_clip": [1.0, 2.5, 2.0, 1.0, 0.6, 1.5],
    "relative_pos_clip": 0.5,
    "include_abs_pos": False,
    "include_target_qpos": "relative",
    "action_bounds": action_bounds,
    "include_is_grasped": True,
    "include_tcp_orientation": True,
    "include_cube_displacement": True,
}

def create_env(eval_mode):
    env = gym.make(
        "Track1-v0",
        num_envs=4,
        task="lift",
        control_mode="pd_joint_target_delta_pos",
        camera_mode="direct_pinhole",
        obs_mode="state",
        reward_config=reward_config,
        action_bounds=action_bounds,
        obs_normalization=obs_normalization,
        eval_mode=eval_mode,
        render_mode="sensors",
        sim_backend="physx_cuda",
        reconfiguration_freq=1 if eval_mode else None,
    )
    if eval_mode:
        return ManiSkillVectorEnv(env, num_envs=4, ignore_terminations=True, record_metrics=True)
    else:
        return ManiSkillVectorEnv(env, num_envs=4, auto_reset=True, ignore_terminations=False)

def flatten_obs(obs):
    tensors = []
    def recurse(d):
        if isinstance(d, dict):
            for k in sorted(d.keys()):
                recurse(d[k])
        else:
            if d.ndim > 2:
                tensors.append(d.flatten(start_dim=1))
            else:
                tensors.append(d)
    recurse(obs)
    return torch.cat(tensors, dim=-1)

def compare_obs(train_obs, eval_obs, name=""):
    train_flat = flatten_obs(train_obs)
    eval_flat = flatten_obs(eval_obs)
    
    print(f"\n{name} Observation Comparison (env 0):")
    print(f"  Train obs dim: {train_flat.shape[-1]}")
    print(f"  Eval  obs dim: {eval_flat.shape[-1]}")
    
    if train_flat.shape[-1] == eval_flat.shape[-1]:
        diff = (train_flat[0] - eval_flat[0]).abs()
        print(f"  Max difference: {diff.max().item():.6f}")
        print(f"  Mean difference: {diff.mean().item():.6f}")
        
        # Print element-by-element comparison
        print(f"\n  Element comparison (first 33 dims):")
        for i in range(min(33, train_flat.shape[-1])):
            t_val = train_flat[0, i].item()
            e_val = eval_flat[0, i].item()
            diff_val = abs(t_val - e_val)
            marker = "  *** DIFF!" if diff_val > 0.1 else ""
            print(f"    [{i:2d}] Train: {t_val:8.4f}  Eval: {e_val:8.4f}  Diff: {diff_val:.4f}{marker}")

print("\n1. Creating environments...")
train_env = create_env(eval_mode=False)
eval_env = create_env(eval_mode=True)

print("\n2. Reset both environments...")
train_obs, _ = train_env.reset()
eval_obs, _ = eval_env.reset()

# Compare initial observations
compare_obs(train_obs, eval_obs, "Initial")

# Check base environment states
train_base = train_env.unwrapped
eval_base = eval_env.unwrapped

print("\n\n3. Environment States:")
print(f"  Train eval_mode: {train_base.eval_mode}")
print(f"  Eval  eval_mode: {eval_base.eval_mode}")

print(f"\n  Train qpos: {train_base.right_arm.robot.qpos[0].tolist()}")
print(f"  Eval  qpos: {eval_base.right_arm.robot.qpos[0].tolist()}")

print(f"\n  Train cube pos: {train_base.red_cube.pose.p[0].tolist()}")
print(f"  Eval  cube pos: {eval_base.red_cube.pose.p[0].tolist()}")

# Step both with same action and compare
print("\n\n4. Stepping with zero action...")
action = torch.zeros((4, 6), device=device)
train_obs, _, _, _, _ = train_env.step(action)
eval_obs, _, _, _, _ = eval_env.step(action)

compare_obs(train_obs, eval_obs, "After zero action")

train_env.close()
eval_env.close()
print("\n" + "=" * 70)
print("Debug complete")
