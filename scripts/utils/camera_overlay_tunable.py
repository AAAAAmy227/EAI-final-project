#!/usr/bin/env python3
"""
Camera overlay script with manual parameter tuning support.
Allows you to adjust camera position, rotation, and FOV to align simulation with real camera.

Usage:
    # Default parameters
    python -m scripts.camera_overlay_tunable
    
    # Custom parameters
    python -m scripts.camera_overlay_tunable --cam-x 0.316 --cam-y 0.26 --cam-z 0.407 --cam-pitch -90 --fov 73.63
"""
import numpy as np
from PIL import Image
import gymnasium as gym
import argparse
import cv2


def create_overlay(sim_image: np.ndarray, real_image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create an overlay of sim and real images for comparison."""
    if sim_image.shape[:2] != real_image.shape[:2]:
        real_image = cv2.resize(real_image, (sim_image.shape[1], sim_image.shape[0]))
    
    if sim_image.shape[2] == 4:
        sim_image = sim_image[:, :, :3]
    if real_image.shape[2] == 4:
        real_image = real_image[:, :, :3]
    
    overlay = cv2.addWeighted(sim_image, alpha, real_image, 1 - alpha, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser(
        description="Camera overlay with manual tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Camera Parameter Guide:
  --cam-x, --cam-y, --cam-z: Camera position in meters
  --cam-pitch: Pitch angle in degrees (negative = looking down)
  --cam-yaw: Yaw angle in degrees
  --cam-roll: Roll angle in degrees  
  --fov: Vertical field of view in degrees
  
Example workflows:
  1. Start with default, check overlay
  2. Adjust position by small increments (±0.01m)
  3. Adjust angles by small increments (±5°)
  4. Fine-tune FOV if needed (±2°)
        """
    )
    
    # Camera parameters
    parser.add_argument("--cam-x", type=float, default=0.316, help="Camera X position (m)")
    parser.add_argument("--cam-y", type=float, default=0.260, help="Camera Y position (m)")
    parser.add_argument("--cam-z", type=float, default=0.407, help="Camera Z position (m)")
    parser.add_argument("--cam-pitch", type=float, default=-90, help="Pitch angle (degrees, negative=down)")
    parser.add_argument("--cam-yaw", type=float, default=0, help="Yaw angle (degrees)")
    parser.add_argument("--cam-roll", type=float, default=0, help="Roll angle (degrees)")
    parser.add_argument("--fov", type=float, default=73.63, help="Vertical FOV (degrees)")
    
    # Other parameters
    parser.add_argument("--real-image", type=str, 
                        default="eai-2025-fall-final-project-reference-scripts/front_camera.png")
    parser.add_argument("--task", type=str, default="lift", choices=["lift", "stack", "sort"])
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend alpha")
    parser.add_argument("--output", type=str, default="overlay_comparison.png")
    parser.add_argument("--no-distortion", action="store_true", help="Skip distortion simulation")
    
    args = parser.parse_args()
    
    # Print current parameters
    print("=" * 60)
    print("Camera Parameters:")
    print(f"  Position: [{args.cam_x:.3f}, {args.cam_y:.3f}, {args.cam_z:.3f}]")
    print(f"  Rotation: pitch={args.cam_pitch:.1f}°, yaw={args.cam_yaw:.1f}°, roll={args.cam_roll:.1f}°")
    print(f"  FOV: {args.fov:.2f}°")
    print("=" * 60)
    
    # Load real image
    real_img = np.array(Image.open(args.real_image))
    print(f"Loaded real image: {args.real_image}, shape: {real_img.shape}")
    
    # Camera calibration for distortion
    W, H = 640, 480
    MTX = np.array([
        [570.21740069, 0., 327.45975405],
        [0., 570.1797441, 260.83642155],
        [0., 0., 1.]
    ], dtype=np.float64)
    
    DIST = np.array([
        -0.735413911,
        0.949258417,
        0.000189059234,
        -0.00200351391,
        -0.864150312
    ], dtype=np.float64)
    
    # Create a temporary config file to pass camera params to environment
    import tempfile
    import json
    import os
    
    cam_config = {
        "position": [args.cam_x, args.cam_y, args.cam_z],
        "pitch": args.cam_pitch,
        "yaw": args.cam_yaw,
        "roll": args.cam_roll,
        "fov": args.fov
    }
    
    config_path = "/tmp/camera_config.json"
    with open(config_path, 'w') as f:
        json.dump(cam_config, f)
    
    # Set env variable for track1_env to read
    os.environ['CAMERA_CONFIG_PATH'] = config_path
    
    # Create environment
    import scripts.track1_env
    env = gym.make(
        "Track1-v0",
        render_mode="rgb_array",
        obs_mode="sensor_data",
        reward_mode="none",
        task=args.task,
        domain_randomization=False,
        num_envs=1,
    )
    
    obs, _ = env.reset()
    
    # Get sim image
    if 'sensor_data' in obs and 'front_camera' in obs['sensor_data']:
        sim_rgb = obs['sensor_data']['front_camera']['Color']
        if len(sim_rgb.shape) == 4:
            sim_rgb = sim_rgb[0]
        sim_rgb = sim_rgb.cpu().numpy() if hasattr(sim_rgb, 'cpu') else sim_rgb
        sim_rgb = sim_rgb.astype(np.uint8)
        
        print(f"Sim image shape: {sim_rgb.shape}")
        
        # Save pinhole image
        Image.fromarray(sim_rgb).save("sim_camera_view_pinhole.png")
        print("Saved sim_camera_view_pinhole.png")
        
        # Apply distortion if requested
        if not args.no_distortion:
            image_size = (W, H)
            new_mtx_for_fov, _ = cv2.getOptimalNewCameraMatrix(MTX, DIST, image_size, 1.0, image_size)
            new_mtx_for_fov[0, 2] = W / 2
            new_mtx_for_fov[1, 2] = H / 2
            
            xs = np.arange(W)
            ys = np.arange(H)
            xx, yy = np.meshgrid(xs, ys)
            points = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
            points = points.reshape(-1, 1, 2)
            
            undistorted_pts = cv2.undistortPoints(points, cameraMatrix=MTX, distCoeffs=DIST, R=None, P=new_mtx_for_fov)
            map_xy = undistorted_pts.reshape(H, W, 2)
            mapx = map_xy[:, :, 0].astype(np.float32)
            mapy = map_xy[:, :, 1].astype(np.float32)
            
            print("Applying distortion to sim image...")
            sim_distorted = cv2.remap(sim_rgb, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            Image.fromarray(sim_distorted).save("sim_camera_view_distorted.png")
            print("Saved sim_camera_view_distorted.png")
            sim_final = sim_distorted
        else:
            sim_final = sim_rgb
        
        # Create overlay
        overlay = create_overlay(sim_final, real_img, alpha=args.alpha)
        
        # Create comparison
        real_resized = cv2.resize(real_img[:, :, :3] if real_img.shape[2] == 4 else real_img, 
                                   (sim_rgb.shape[1], sim_rgb.shape[0]))
        
        comparison = np.hstack([sim_final[:, :, :3], overlay, real_resized])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        w = sim_rgb.shape[1]
        cv2.putText(comparison, "Sim", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, "Overlay", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, "Real", (2 * w + 10, 30), font, 0.7, (255, 255, 255), 2)
        
        # Save
        Image.fromarray(comparison).save(args.output)
        Image.fromarray(overlay).save("overlay_only.png")
        
        print(f"\nSaved: {args.output}, overlay_only.png")
        print("\nNext steps:")
        print("  1. View overlay: eog overlay_comparison.png")
        print("  2. Adjust parameters and re-run")
        print("  3. When satisfied, update track1_env.py with final values")
        
    else:
        print("Error: Could not get front_camera from sensor_data")
    
    env.close()


if __name__ == "__main__":
    main()
