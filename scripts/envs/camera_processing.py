"""
Camera processing module for Track1 environment.

This module contains functions for camera setup and image processing
(distortion/undistortion) extracted from Track1Env.
All functions receive the environment instance as the first parameter.
"""
import numpy as np
import torch
import torch.nn.functional as tFunc


def setup_camera_processing_maps(env):
    """Precompute torch grids for camera distortion/undistortion processing via tFunc.grid_sample.
    
    Pipeline:
    - Source: Rendered at (640×scale) × (480×scale) with scaled intrinsic matrix
    - Distortion: Maps source -> 640×480 distorted output
    - Undistortion (alpha=0): Maps 640×480 distorted -> 640×480 clean pinhole
    """
    import cv2
    
    # Camera intrinsic parameters (from real camera calibration)
    env.mtx_intrinsic = np.array([
        [570.21740069, 0., 327.45975405],
        [0., 570.1797441, 260.83642155],
        [0., 0., 1.]
    ], dtype=np.float64)
    
    env.dist_coeffs = np.array([
        -0.735413911, 0.949258417, 0.000189059234, -0.00200351391, -0.864150312
    ], dtype=np.float64)
    
    # Scale factor for high-res rendering
    
    # Source image size (high-res pinhole render)
    OUT_W, OUT_H = 640, 480
    SRC_W = OUT_W * env.render_scale
    SRC_H = OUT_H * env.render_scale
    if env.camera_mode in ["distorted", "distort-twice"]:
        env.front_render_width = SRC_W
        env.front_render_height = SRC_H

        # Get the undistorted intrinsic matrix using getOptimalNewCameraMatrix with alpha=1
        # This gives us the intrinsic for a pinhole camera that covers all distorted pixels
        new_mtx_alpha1, _ = cv2.getOptimalNewCameraMatrix(
            env.mtx_intrinsic, env.dist_coeffs, (OUT_W, OUT_H), 1.0, (SRC_W, SRC_H)
        )
        
        # Scale the new_mtx to render resolution
        env.render_intrinsic = new_mtx_alpha1.copy()
        
        # ============ Distortion Grid (SRC -> OUT distorted) ============
        # For each pixel in the 640×480 distorted output, find where it maps to in the source
        
        # Step 1: Generate grid for distorted output image (640×480)
        xs = np.arange(OUT_W)
        ys = np.arange(OUT_H)
        xx, yy = np.meshgrid(xs, ys)
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32).reshape(-1, 1, 2)
        
        # Step 2: undistortPoints with P=scaled_intrinsic gives coordinates in render space directly
        undistorted_pts = cv2.undistortPoints(
            points, 
            cameraMatrix=env.mtx_intrinsic, 
            distCoeffs=env.dist_coeffs, 
            R=None, 
            P=env.render_intrinsic  # Project to render camera space
        )
        map_xy_render = undistorted_pts.reshape(OUT_H, OUT_W, 2)
        
        # Step 3: Normalize to [-1, 1] for grid_sample
        grid_x = 2.0 * map_xy_render[:, :, 0] / (SRC_W - 1) - 1.0
        grid_y = 2.0 * map_xy_render[:, :, 1] / (SRC_H - 1) - 1.0
        distortion_grid = np.stack((grid_x, grid_y), axis=2).astype(np.float32)
        env.distortion_grid = torch.from_numpy(distortion_grid)  # (OUT_H, OUT_W, 2)
        
        # ============ Undistortion Grid ============
        # This maps 640x480 distorted -> 640x480 clean pinhole
    if env.camera_mode in ["distort-twice", "direct_pinhole"]:
        # Get new camera matrix with configurable alpha
        # alpha=0: crop black borders, alpha=1: keep all pixels (shrinks image)
        # alpha=0.25 is optimal for full work area visibility
        alpha = getattr(env, 'undistort_alpha', 0.25)
        new_mtx_undist, _ = cv2.getOptimalNewCameraMatrix(
            env.mtx_intrinsic, env.dist_coeffs, (OUT_W, OUT_H), alpha, (OUT_W, OUT_H)
        )
        
        if env.camera_mode == "direct_pinhole":
            env.front_render_width = OUT_W
            env.front_render_height = OUT_H
            env.render_intrinsic = new_mtx_undist.copy()
            return
        # initUndistortRectifyMap gives us the mapping from undistorted -> distorted source
        # We need the inverse for grid_sample
        map1, map2 = cv2.initUndistortRectifyMap(
            env.mtx_intrinsic, env.dist_coeffs, None, new_mtx_undist, (OUT_W, OUT_H), cv2.CV_32FC1
        )
        
        # map1, map2 are (OUT_H, OUT_W) containing x, y source coordinates
        # Normalize to [-1, 1]
        undist_grid_x = 2.0 * map1 / (OUT_W - 1) - 1.0
        undist_grid_y = 2.0 * map2 / (OUT_H - 1) - 1.0
        undistortion_grid = np.stack((undist_grid_x, undist_grid_y), axis=2).astype(np.float32)
        env.undistortion_grid = torch.from_numpy(undistortion_grid).to(device=env.device)  # (OUT_H, OUT_W, 2)


def apply_camera_processing(env, obs):
    """Apply camera processing based on camera_mode.
    
    Modes:
    - direct_pinhole: No processing (already rendered with correct params)
    - distorted: Apply distortion to 1920x1440 source -> 640x480 distorted output
    - distort-twice: distorted -> then undistort (alpha=0) -> 640x480 clean
    """
    
    if env.camera_mode == "direct_pinhole":
        return obs  # No processing needed
    
    # Skip if grids not yet initialized (happens during parent __init__ reset)
    if env.distortion_grid is None:
        return obs
    
    # Find the RGB tensor - could be in 'sensor_data' or 'image'
    rgb_tensor = None
    obs_key = None
    
    if isinstance(obs, dict):
        if "sensor_data" in obs and "front_camera" in obs["sensor_data"]:
            if "rgb" in obs["sensor_data"]["front_camera"]:
                rgb_tensor = obs["sensor_data"]["front_camera"]["rgb"]
                obs_key = "sensor_data"
        elif "image" in obs and "front_camera" in obs["image"]:
            if "rgb" in obs["image"]["front_camera"]:
                rgb_tensor = obs["image"]["front_camera"]["rgb"]
                obs_key = "image"
    
    if rgb_tensor is None or not isinstance(rgb_tensor, torch.Tensor):
        return obs

    # Input: (B, SRC_H, SRC_W, C) or (SRC_H, SRC_W, C)
    # For distorted/distort-twice: Source is 1920x1440
    is_batch = len(rgb_tensor.shape) == 4
    if not is_batch:
        img_in = rgb_tensor.unsqueeze(0)
    else:
        img_in = rgb_tensor

    B = img_in.shape[0]
    original_dtype = rgb_tensor.dtype
    
    # Permute to (B, C, H, W) for grid_sample
    img_in = img_in.permute(0, 3, 1, 2).float()
    
    # Ensure grids are on same device as input
    device = img_in.device
    dist_grid = env.distortion_grid.to(device).unsqueeze(0).expand(B, -1, -1, -1)
    
    # Step 1: Apply distortion (1920x1440 -> 640x480)
    distorted = tFunc.grid_sample(img_in, dist_grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    if env.camera_mode == "distorted":
        result = distorted
    elif env.camera_mode == "distort-twice":
        # Step 2: Apply undistortion (640x480 distorted -> 640x480 clean)
        undist_grid = env.undistortion_grid.to(device).unsqueeze(0).expand(B, -1, -1, -1)
        result = tFunc.grid_sample(distorted, undist_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    else:
        result = distorted  # Fallback
    
    # Permute back to (B, H, W, C)
    result = result.permute(0, 2, 3, 1)
    
    # Restore dtype
    if original_dtype == torch.uint8:
        result = result.clamp(0, 255).to(torch.uint8)
    else:
        result = result.to(original_dtype)
        
    if not is_batch:
        obs[obs_key]["front_camera"]["rgb"] = result.squeeze(0)
    else:
        obs[obs_key]["front_camera"]["rgb"] = result

    return obs
