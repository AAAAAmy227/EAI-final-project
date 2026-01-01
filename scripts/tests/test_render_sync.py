"""
Test script to verify if rendered frames are synchronized with simulation state.

Strategy:
1. Create env with known seed
2. Execute a sequence of LARGE actions (to make movement visible)
3. At each step, record:
   - qpos BEFORE render
   - rendered image
   - qpos AFTER render (should be same as before)
4. Analyze: If images lag behind qpos, the render is using stale GPU buffers

Run with: uv run python scripts/tests/test_render_sync.py
"""

import gymnasium as gym
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import json

# Import Track1Env to register it
from scripts.envs.track1_env import Track1Env
from scripts.agents.so101 import SO101


def test_render_sync():
    """Test whether render() returns images synchronized with current state."""
    
    output_dir = Path("outputs/render_sync_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Testing GPU Render Synchronization")
    print("=" * 60)
    
    # 1. Create minimal environment (single env for clarity)
    print("\n1. Creating environment...")
    env = gym.make(
        "Track1-v0",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
        render_mode="sensors",
        sim_backend="physx_cuda:0",
        render_backend="sapien_cuda:0",
        cfg={
            "task": "static_grasp",
            "agents": {"robots_per_side": 1}
        }
    )
    
    print(f"   Env created. GPU sim enabled: {env.unwrapped.gpu_sim_enabled}")
    
    # Get the first robot from agents
    first_agent = env.unwrapped.agent.agents[0]
    
    # 2. Reset with fixed seed
    print("\n2. Resetting environment with seed=42...")
    obs, info = env.reset(seed=42)
    
    # Record initial state
    initial_qpos = first_agent.robot.get_qpos().cpu().numpy().copy()
    print(f"   Initial qpos shape: {initial_qpos.shape}")
    
    # 3. Run several steps with BIG actions
    print("\n3. Running steps with large actions...")
    
    records = []
    num_steps = 20
    
    # Get action space keys
    action_keys = list(env.unwrapped.single_action_space.spaces.keys())
    print(f"   Action keys: {action_keys}")
    action_dim = env.unwrapped.single_action_space[action_keys[0]].shape[0]
    print(f"   Action dim: {action_dim}")
    
    for step in range(num_steps):
        # Create a LARGE oscillating action
        if step % 4 < 2:
            action_value = 0.8  # Move one direction
        else:
            action_value = -0.8  # Move other direction
            
        action = {}
        for key in action_keys:
            action[key] = torch.full((1, action_dim), action_value, device=env.unwrapped.device)
        
        # Record qpos BEFORE step
        qpos_before_step = first_agent.robot.get_qpos().cpu().numpy().copy()
        
        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record qpos AFTER step (this is what the render SHOULD show)
        qpos_after_step = first_agent.robot.get_qpos().cpu().numpy().copy()
        
        # Now render
        img = env.render()
        
        # Record qpos AFTER render (should be same as after_step)
        qpos_after_render = first_agent.robot.get_qpos().cpu().numpy().copy()
        
        # Calculate qpos changes
        qpos_change_from_step = np.abs(qpos_after_step - qpos_before_step).sum()
        qpos_change_from_render = np.abs(qpos_after_render - qpos_after_step).sum()
        
        records.append({
            "step": step,
            "qpos_before_step": qpos_before_step[0].tolist(),  # First env only
            "qpos_after_step": qpos_after_step[0].tolist(),
            "qpos_after_render": qpos_after_render[0].tolist(),
            "qpos_change_from_step": float(qpos_change_from_step),
            "qpos_change_from_render": float(qpos_change_from_render),
        })
        
        # Save frame
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if len(img.shape) == 4:
            img = img[0]  # Remove batch dim
        cv2.imwrite(str(output_dir / f"frame_{step:03d}.jpg"), 
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        print(f"   Step {step}: qpos_change={qpos_change_from_step:.4f}, "
              f"render_lag={qpos_change_from_render:.6f}")
    
    # 4. Additional test: render twice in a row
    print("\n4. Testing double render (same state, should give same image)...")
    img1 = env.render()
    img2 = env.render()
    
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
        img2 = img2.cpu().numpy()
    
    diff = np.abs(img1.astype(float) - img2.astype(float)).mean()
    print(f"   Double render diff: {diff:.4f} (should be ~0)")
    
    # 5. Test: render BEFORE and AFTER step
    print("\n5. Testing render before vs after step...")
    
    action = {}
    for key in action_keys:
        action[key] = torch.full((1, action_dim), 0.9, device=env.unwrapped.device)
    
    # Render before step
    img_before = env.render()
    qpos_before = first_agent.robot.get_qpos().cpu().numpy().copy()
    
    # Step
    env.step(action)
    
    # Render after step  
    img_after = env.render()
    qpos_after = first_agent.robot.get_qpos().cpu().numpy().copy()
    
    if isinstance(img_before, torch.Tensor):
        img_before = img_before.cpu().numpy()
        img_after = img_after.cpu().numpy()
    
    qpos_change = np.abs(qpos_after - qpos_before).sum()
    img_diff = np.abs(img_before.astype(float) - img_after.astype(float)).mean()
    
    print(f"   qpos change: {qpos_change:.4f}")
    print(f"   image diff: {img_diff:.4f}")
    print(f"   Ratio (img_diff / qpos_change): {img_diff / max(qpos_change, 1e-6):.4f}")
    
    # Save comparison
    cv2.imwrite(str(output_dir / "before_step.jpg"), 
                cv2.cvtColor(img_before[0] if len(img_before.shape) == 4 else img_before, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "after_step.jpg"), 
                cv2.cvtColor(img_after[0] if len(img_after.shape) == 4 else img_after, cv2.COLOR_RGB2BGR))
    
    # 6. Save records
    with open(output_dir / "records.json", "w") as f:
        json.dump(records, f, indent=2)
    
    print(f"\n6. Results saved to {output_dir}/")
    
    env.close()
    
    # 7. Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    if diff > 1.0:
        print("⚠️  Double render produces different images!")
        print("   This suggests GPU buffer sync issues.")
    else:
        print("✓  Double render is consistent.")
    
    if img_diff < 5.0 and qpos_change > 0.1:
        print("⚠️  Large qpos change but small image change!")
        print("   This suggests render is LAGGING behind simulation.")
    elif img_diff > 5.0 and qpos_change > 0.1:
        print("✓  Image changes proportionally to qpos change.")
    
    return records


if __name__ == "__main__":
    test_render_sync()
