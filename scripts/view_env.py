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
from scripts.envs.track1_env import Track1Env


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
    
    # Force gripper to a specific angle for visualization
    unwrapped = env.unwrapped
    right_agent = unwrapped.agent.agents[0]
    # From URDF: gripper joint limits: lower="-0.174533" upper="1.74533"
    # Set to partially closed (0.5 radians) instead of wide open
    qpos = right_agent.robot.qpos.clone()
    qpos[:, 5] = 0.5  # gripper joint at index 5, set to 0.5 rad (partially closed)
    right_agent.robot.set_qpos(qpos)
    
    print("Environment ready! Use viewer to inspect.")
    print(">>> GRIPPER SET TO 0.5 rad (partially closed) <<<")
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
    
    def make_cylinder(scene, name, color, radius=0.003, half_length=0.025):
        """Create a cylinder for axis visualization."""
        builder = scene.create_actor_builder()
        mat = sapien.render.RenderMaterial(base_color=color)
        builder.add_cylinder_visual(radius=radius, half_length=half_length, material=mat)
        return builder.build_kinematic(name=name)
    
    # Red SPHERE for reference point (3cm diameter = 0.015 radius), blue/green spheres for jaw tips
    ref_cube = make_sphere(scene, "reference", [1, 0, 0, 0.7], radius=0.015)  # 3cm diameter sphere, semi-transparent
    fixed_jaw_sphere = make_sphere(scene, "fixed_jaw", [0, 0, 1, 1], radius=0.008)
    moving_jaw_sphere = make_sphere(scene, "moving_jaw", [0, 1, 0, 1], radius=0.008)
    moving_jaw_ref_sphere = make_sphere(scene, "moving_jaw_ref", [1, 1, 0, 0.8], radius=0.012)  # Yellow: moving jaw reference
    
    print("\n=== DEBUG MARKERS ===")
    print("  RED SPHERE (3cm): Fixed jaw reference point (where cube should be)")
    print("  YELLOW SPHERE: Moving jaw reference point (with offsets from config)")
    print("  BLUE (small): Fixed jaw tip (gripper_frame_link)")
    print("  GREEN (small): Moving jaw tip (calculated)")
    print("  === Moving Jaw Coordinate System (5cm from origin) ===")
    print("  WHITE: Origin | RED: +X end | GREEN: +Y end | BLUE: +Z end")
    
    right_agent = unwrapped.agent.agents[0]
    gripper_link = right_agent.robot.links_map.get("gripper_link")
    gripper_frame = right_agent.robot.links_map.get("gripper_frame_link")
    moving_jaw = right_agent.robot.links_map.get("moving_jaw_so101_v1_link")
    
    # Axis visualization: spheres at origin and axis endpoints (5cm away)
    origin_sphere = make_sphere(scene, "origin", [1, 1, 1, 1], radius=0.006)  # White origin
    axis_x_end = make_sphere(scene, "axis_x", [1, 0, 0, 1], radius=0.005)  # Red = +X
    axis_y_end = make_sphere(scene, "axis_y", [0, 1, 0, 1], radius=0.005)  # Green = +Y  
    axis_z_end = make_sphere(scene, "axis_z", [0, 0, 1, 1], radius=0.005)  # Blue = +Z
    
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
                moving_jaw_base = moving_jaw.pose.p[0].cpu().numpy()
                moving_jaw_quat = moving_jaw.pose.q[0].cpu().numpy()  # [w, x, y, z]
                
                # Jaw direction (from gripper_link towards fixed jaw tip)
                jaw_direction = fixed_jaw_tip - gripper_origin
                jaw_length = np.linalg.norm(jaw_direction)
                jaw_unit = jaw_direction / (jaw_length + 1e-6)
                
                # Moving jaw TIP: use moving jaw's own forward direction
                # Get rotation matrix from quaternion to find local axis direction
                from scipy.spatial.transform import Rotation
                rot = Rotation.from_quat([moving_jaw_quat[1], moving_jaw_quat[2], moving_jaw_quat[3], moving_jaw_quat[0]])  # scipy uses [x,y,z,w]
                # The jaw extends roughly along local -Y with +Z and +X components
                # Configurable components to adjust tip direction
                x_component = -0.2   # Adjust this to tune the +X offset
                z_component = 0.23  # Adjust this to tune the +Z offset
                scale = 1.8
                local_forward = np.array([x_component, -1, z_component])
                local_forward = local_forward / np.linalg.norm(local_forward)  # Normalize
                moving_jaw_direction = rot.apply(local_forward)
                
                moving_jaw_tip_offset_base = 0.045  # ~4.5cm from base to tip
                moving_jaw_tip = moving_jaw_base + moving_jaw_direction * moving_jaw_tip_offset_base * scale
                
                # Apply config offsets (from lift.yaml)
                tip_offset_config = 0.015  # moving_jaw_tip_offset from config
                outward_offset_config = 0.015  # moving_jaw_outward_offset from config
                
                # Back along jaw direction
                moving_jaw_ref = moving_jaw_tip - moving_jaw_direction * tip_offset_config
                
                # Outward along local -X
                local_minus_x = np.array([-1.0, 0.0, 0.0])
                outward_dir = rot.apply(local_minus_x)
                moving_jaw_ref = moving_jaw_ref + outward_dir * outward_offset_config
                
                # Visualize coordinate system: spheres at axis endpoints (5cm from origin)
                axis_length = 0.05
                
                # Set origin sphere (white)
                origin_sphere.set_pose(sapien.Pose(p=moving_jaw_base))
                
                # X axis endpoint (RED) - 5cm along local +X
                x_axis_world = rot.apply(np.array([1, 0, 0]))
                axis_x_end.set_pose(sapien.Pose(p=moving_jaw_base + x_axis_world * axis_length))
                
                # Y axis endpoint (GREEN) - 5cm along local +Y
                y_axis_world = rot.apply(np.array([0, 1, 0]))
                axis_y_end.set_pose(sapien.Pose(p=moving_jaw_base + y_axis_world * axis_length))
                
                # Z axis endpoint (BLUE) - 5cm along local +Z
                z_axis_world = rot.apply(np.array([0, 0, 1]))
                axis_z_end.set_pose(sapien.Pose(p=moving_jaw_base + z_axis_world * axis_length))
                
                # Outward direction (from fixed jaw tip towards moving jaw TIP)
                outward_direction = moving_jaw_tip - fixed_jaw_tip
                outward_length = np.linalg.norm(outward_direction)
                outward_unit = outward_direction / (outward_length + 1e-6)
                
                # Calculate reference point with both offsets
                ref_pos = fixed_jaw_tip - jaw_unit * tip_offset + outward_unit * outward_offset
                
                # Set positions
                fixed_jaw_sphere.set_pose(sapien.Pose(p=fixed_jaw_tip))
                moving_jaw_sphere.set_pose(sapien.Pose(p=moving_jaw_tip))  # Green: raw tip position
                moving_jaw_ref_sphere.set_pose(sapien.Pose(p=moving_jaw_ref))  # Yellow: reference with offsets
                ref_cube.set_pose(sapien.Pose(p=ref_pos))
            
            env.render()
    except (KeyboardInterrupt, AttributeError) as e:
        print(f"\nExiting view_env... ({e})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
