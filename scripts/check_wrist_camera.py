
import gymnasium as gym
import mani_skill.envs
import cv2
import numpy as np
import torch
from PIL import Image
import scripts.track1_env  # Register env

def main():
    env = gym.make("Track1-v0", camera_mode="direct_pinhole", render_mode="rgb_array")
    
    # Reset specific seed
    obs, _ = env.reset(seed=0)
    
    # Get initial images
    # Wrist cameras are in obs['image']['wrist_camera_0']['rgb'] etc if added
    # But wait, in ManiSkill, extra sensors (added dynamically) might need to be accessed via get_sensor_images or check obs structure
    
    # Obs is likely state, so let's get sensor images directly
    print("Obs type:", type(obs))
    
    sensor_images = env.get_sensor_images()
    print("Sensor images keys:", sensor_images.keys())
    
    # Iterate over available cameras
    for cam_name in ["wrist_camera_0", "wrist_camera_1"]:
        if cam_name in sensor_images:
            # Structure usually: sensor_images[cam_name]['rgb']
            img = sensor_images[cam_name]["rgb"]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            
            # Remove batch dim if present (1, H, W, C) -> (H, W, C)
            if img.ndim == 4:
                img = img[0]
                
            # Save
            filename = f"check_{cam_name}.png"
            Image.fromarray(img).save(filename)
            print(f"Saved {filename}")
        else:
            print(f"{cam_name} not found in sensor_images")

    
    # Clean up
    env.close()

if __name__ == "__main__":
    main()
