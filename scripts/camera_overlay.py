"""
Overlay script to compare simulation camera view with real camera view.
Based on lerobot-sim2real/docs/zero_shot_rgb_sim2real.md alignment method.
"""
import numpy as np
from PIL import Image
import gymnasium as gym
import mani_skill.envs
import scripts.track1_env
import argparse
import cv2


def create_overlay(sim_image: np.ndarray, real_image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create an overlay of sim and real images for comparison."""
    # Resize real image to match sim if needed
    if sim_image.shape[:2] != real_image.shape[:2]:
        real_image = cv2.resize(real_image, (sim_image.shape[1], sim_image.shape[0]))
    
    # Convert to same format
    if sim_image.shape[2] == 4:
        sim_image = sim_image[:, :, :3]
    if real_image.shape[2] == 4:
        real_image = real_image[:, :, :3]
    
    # Create overlay
    overlay = cv2.addWeighted(sim_image, alpha, real_image, 1 - alpha, 0)
    
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Compare sim and real camera views")
    parser.add_argument("--real-image", type=str, 
                        default="eai-2025-fall-final-project-reference-scripts/front_camera.png",
                        help="Path to real camera image")
    parser.add_argument("--task", type=str, default="lift", choices=["lift", "stack", "sort"])
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend alpha (0=real, 1=sim)")
    parser.add_argument("--output", type=str, default="overlay_comparison.png", help="Output path")
    args = parser.parse_args()
    
    # Load real image (Distorted)
    real_img = np.array(Image.open(args.real_image))
    print(f"Loaded real image: {args.real_image}, shape: {real_img.shape}")
    
    # Camera Calibration Parameters (from distort.py)
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
    
    # Create sim environment and get camera view
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
    
    # Get front camera image from sim
    if 'sensor_data' in obs and 'front_camera' in obs['sensor_data']:
        sim_rgb = obs['sensor_data']['front_camera']['Color']
        if len(sim_rgb.shape) == 4:
            sim_rgb = sim_rgb[0]
        sim_rgb = sim_rgb.cpu().numpy() if hasattr(sim_rgb, 'cpu') else sim_rgb
        sim_rgb = sim_rgb.astype(np.uint8)
        
        print(f"Sim image shape: {sim_rgb.shape}")
        
        # Save original sim image (Pinhole / Straight)
        Image.fromarray(sim_rgb).save("sim_camera_view_pinhole.png")
        
        # Apply Distortion to Sim Image to match Real
        # Logic from distort.py
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
        
        print("Applying distortion to Sim image...")
        sim_distorted = cv2.remap(sim_rgb, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # Save distorted sim image
        Image.fromarray(sim_distorted).save("sim_camera_view_distorted.png")
        print("Saved sim_camera_view_distorted.png")
        
        # Create overlay
        # sim_distorted vs real_img
        overlay = create_overlay(sim_distorted, real_img, alpha=args.alpha)
        
        # Create side-by-side comparison
        real_resized = cv2.resize(real_img[:, :, :3] if real_img.shape[2] == 4 else real_img, 
                                   (sim_rgb.shape[1], sim_rgb.shape[0]))
        
        # Stack: sim (distorted) | overlay | real
        comparison = np.hstack([sim_distorted[:, :, :3], overlay, real_resized])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        h = sim_rgb.shape[0]
        w = sim_rgb.shape[1]
        cv2.putText(comparison, "Sim (Distorted)", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, "Overlay", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, "Real", (2 * w + 10, 30), font, 0.7, (255, 255, 255), 2)
        
        # Save
        comparison_pil = Image.fromarray(comparison)
        comparison_pil.save(args.output)
        print(f"Saved comparison to {args.output}")
        
        # Also save just the overlay
        overlay_pil = Image.fromarray(overlay)
        overlay_pil.save("overlay_only.png")
        print("Saved overlay_only.png")
        
    else:
        print("Error: Could not get front_camera from sensor_data")
    
    env.close()


if __name__ == "__main__":
    main()
