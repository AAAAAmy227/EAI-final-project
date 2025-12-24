#!/usr/bin/env python3
"""GUI viewer script for Track1 environment inspection.

Usage:
    python scripts/view_env.py --task lift
    python scripts/view_env.py --task lift --camera-mode distorted
    python scripts/view_env.py --task lift --camera-mode distort-twice
    python scripts/view_env.py --task lift --camera-mode direct_pinhole
    python scripts/view_env.py --task lift --domain-randomization
"""

import argparse
import gymnasium as gym
from scripts.track1_env import Track1Env


def main():
    parser = argparse.ArgumentParser(description="View Track1 environment in GUI")
    parser.add_argument("--task", type=str, default="lift", 
                        choices=["lift", "stack", "sort"],
                        help="Task type to visualize")
    parser.add_argument("--domain-randomization", action="store_true",
                        help="Enable domain randomization")
    parser.add_argument("--camera-mode", type=str, default="direct_pinhole",
                        choices=["distorted", "distort-twice", "direct_pinhole"],
                        help="Camera output mode: distorted (raw fisheye), distort-twice (rectified), direct_pinhole (efficient render)")
    parser.add_argument("--control-mode", type=str, default="pd_joint_target_delta_pos",
                        choices=["pd_joint_delta_pos", "pd_joint_target_delta_pos", "pd_ee_delta_pos"],
                        help="Control mode: joint, joint_target, or ee")
    args = parser.parse_args()
    
    print(f"Starting Track1 environment with task={args.task}, camera_mode={args.camera_mode}, control_mode={args.control_mode}")
    print("Controls:")
    print("  - Mouse: Rotate camera view")
    print("  - Scroll: Zoom in/out")
    print("  - R: Reset episode")
    print("  - Q: Quit")
    
    env = gym.make(
        "Track1-v0",
        render_mode="human",
        obs_mode="sensor_data",
        reward_mode="none",  # Disable reward computation
        task=args.task,
        domain_randomization=args.domain_randomization,
        camera_mode=args.camera_mode,
        control_mode=args.control_mode,
        num_envs=1,
    )
    
    obs, _ = env.reset()
    
    # Force gripper wide open for visualization
    unwrapped = env.unwrapped
    right_agent = unwrapped.agent.agents[0]
    # From URDF: gripper joint limits: lower="-0.174533" upper="1.74533"
    # Set to max open (1.7 radians)
    qpos = right_agent.robot.qpos.clone()
    qpos[:, 5] = 1.5  # gripper joint at index 5, set to ~1.5 rad (wide open)
    right_agent.robot.set_qpos(qpos)
    
    print("Environment ready! Use viewer to inspect.")
    print(">>> GRIPPER FORCED WIDE OPEN for visualization <<<")
    print("Close the window or press Ctrl+C to exit.")
    
    # Add debug sphere at gripper midpoint for visualization
    import sapien
    import numpy as np
    
    unwrapped = env.unwrapped
    scene = unwrapped.scene.sub_scenes[0]
    
    # Create visible debug markers
    def make_sphere(scene, name, color, radius=0.01):
        builder = scene.create_actor_builder()
        mat = sapien.render.RenderMaterial(base_color=color)
        builder.add_sphere_visual(radius=radius, material=mat)
        return builder.build_kinematic(name=name)
    
    def make_cube(scene, name, color, half_size=0.015):
        builder = scene.create_actor_builder()
        mat = sapien.render.RenderMaterial(base_color=color)
        builder.add_box_visual(half_size=[half_size, half_size, half_size], material=mat)
        return builder.build_kinematic(name=name)
    
    # Red SPHERE for reference point (3cm diameter = 0.015 radius), blue/green spheres for jaw tips
    ref_cube = make_sphere(scene, "reference", [1, 0, 0, 0.7], radius=0.015)  # 3cm diameter sphere, semi-transparent
    fixed_jaw_sphere = make_sphere(scene, "fixed_jaw", [0, 0, 1, 1], radius=0.008)
    moving_jaw_sphere = make_sphere(scene, "moving_jaw", [0, 1, 0, 1], radius=0.008)
    
    print("\n=== DEBUG MARKERS ===")
    print("  RED SPHERE (3cm): Gripper reference point (where cube center should be)")
    print("  BLUE (small): Fixed jaw tip (gripper_frame_link)")
    print("  GREEN (small): Moving jaw (for reference)")
    
    right_agent = unwrapped.agent.agents[0]
    gripper_link = right_agent.robot.links_map.get("gripper_link")
    gripper_frame = right_agent.robot.links_map.get("gripper_frame_link")
    moving_jaw = right_agent.robot.links_map.get("moving_jaw_so101_v1_link")
    
    # Update scene render
    scene.update_render()
    
    # Config offsets (same as in lift.yaml)
    tip_offset = 0.01  # 1.5cm back from tip
    outward_offset = 0.015  # 1.5cm towards moving jaw
    
    # Keep viewer open, updating debug marker positions each frame
    try:
        while True:
            if gripper_frame and moving_jaw and gripper_link:
                # Fixed jaw tip
                fixed_jaw_tip = gripper_frame.pose.p[0].cpu().numpy()
                gripper_origin = gripper_link.pose.p[0].cpu().numpy()
                moving_jaw_pos = moving_jaw.pose.p[0].cpu().numpy()
                
                # Jaw direction (from gripper_link towards tip)
                jaw_direction = fixed_jaw_tip - gripper_origin
                jaw_length = np.linalg.norm(jaw_direction)
                jaw_unit = jaw_direction / (jaw_length + 1e-6)
                
                # Outward direction (from fixed jaw tip towards moving jaw)
                outward_direction = moving_jaw_pos - fixed_jaw_tip
                outward_length = np.linalg.norm(outward_direction)
                outward_unit = outward_direction / (outward_length + 1e-6)
                
                # Calculate reference point with both offsets
                ref_pos = fixed_jaw_tip - jaw_unit * tip_offset + outward_unit * outward_offset
                
                # Set positions
                fixed_jaw_sphere.set_pose(sapien.Pose(p=fixed_jaw_tip))
                moving_jaw_sphere.set_pose(sapien.Pose(p=moving_jaw_pos))
                ref_cube.set_pose(sapien.Pose(p=ref_pos))
            
            env.render()
    except (KeyboardInterrupt, AttributeError) as e:
        print(f"\nExiting view_env... ({e})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
