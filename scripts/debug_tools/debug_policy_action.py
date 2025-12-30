"""
Debug: Check policy action output in eval mode with FULL config.
Uses the exact same obs configuration as training.
"""
import sys
sys.path.insert(0, "/home/admin/Desktop/eai-final-project")

import torch
import gymnasium as gym
from omegaconf import OmegaConf

from scripts.envs.track1_env import Track1Env
from scripts.agents.so101 import SO101
from scripts.training.agent import Agent
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config from the training run
config_path = "/home/admin/Desktop/eai-final-project/outputs/2025-12-29/04-27-56/.hydra/config.yaml"
checkpoint_path = "/home/admin/Desktop/eai-final-project/outputs/2025-12-29/04-27-56/latest.pt"

print("=" * 70)
print("Testing Policy Action Output with FULL Config")
print("=" * 70)

# Load config (without resolving hydra interpolations)
print(f"\n1. Loading config: {config_path}")
raw_cfg = OmegaConf.load(config_path)

# Manually extract needed parts without interpolation issues
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

# CRITICAL: Include the full obs config
obs_normalization = {
    "enabled": False,
    "qpos_scale": 1.5,
    "qvel_clip": [1.0, 2.5, 2.0, 1.0, 0.6, 1.5],
    "relative_pos_clip": 0.5,
    "include_abs_pos": False,
    "include_target_qpos": "relative",
    "action_bounds": action_bounds,
    "include_is_grasped": True,  # +1 dim
    "include_tcp_orientation": True,  # +? dim
    "include_cube_displacement": True,  # +? dim
}

# Load checkpoint
print(f"\n2. Loading checkpoint: {checkpoint_path}")
state = torch.load(checkpoint_path, map_location=device)
print(f"   obs_rms_mean shape: {state['obs_rms_mean'].shape}")

# Create environment with FULL config
print("\n3. Creating eval environment with FULL config...")
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
    eval_mode=True,
    render_mode="sensors",
    sim_backend="physx_cuda",
)
env = ManiSkillVectorEnv(env, num_envs=4, ignore_terminations=True, record_metrics=True)

obs, info = env.reset()

# Flatten obs to get n_obs
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

obs_flat = flatten_obs(obs)
n_obs = obs_flat.shape[-1]
n_act = env.single_action_space.shape[0]
print(f"   n_obs={n_obs}, n_act={n_act}")

if n_obs != state['obs_rms_mean'].shape[0]:
    print(f"\n   ⚠️  WARNING: n_obs mismatch!")
    print(f"   Expected (from checkpoint): {state['obs_rms_mean'].shape[0]}")
    print(f"   Got: {n_obs}")
else:
    print(f"   ✓ n_obs matches checkpoint!")

# Create agent and load weights
print("\n4. Loading agent weights...")
agent = Agent(n_obs, n_act, device=device)
try:
    agent.load_state_dict(state["agent"])
    agent.eval()
    print(f"   ✓ Agent loaded successfully")
except Exception as e:
    print(f"   ✗ Failed to load agent: {e}")
    env.close()
    exit(1)

# Load obs normalization stats
obs_rms_mean = state["obs_rms_mean"]
obs_rms_var = state["obs_rms_var"]
obs_clip = 10.0

def normalize_obs(obs_flat):
    normalized = (obs_flat - obs_rms_mean) / torch.sqrt(obs_rms_var + 1e-8)
    return torch.clamp(normalized, -obs_clip, obs_clip)

# Run a few steps and check action output
print("\n5. Checking policy action output...")

obs, info = env.reset()
for step in range(10):
    obs_flat = flatten_obs(obs)
    norm_obs = normalize_obs(obs_flat)
    
    with torch.no_grad():
        action_det = agent.get_action(norm_obs, deterministic=True)
    
    gripper_qpos = env.unwrapped.right_arm.robot.qpos[:, 5]
    
    print(f"\nStep {step}:")
    print(f"  Action: {action_det[0].tolist()}")
    print(f"  Gripper action: {action_det[0, 5].item():.4f}")
    print(f"  Gripper qpos:   {gripper_qpos[0].item():.4f}")
    
    obs, reward, terminated, truncated, info = env.step(action_det)

env.close()
print("\n" + "=" * 70)
print("Debug complete")
