# Camera Extrinsics Calibration Results

This folder contains debug images and results from the camera extrinsics optimization process.
The optimization aligns simulated camera views with real camera images using distance transform based optimization.

## Final Result

- **comparison_final.png** - Final comparison: Sim (distorted mode) | Overlay | Real
  - Uses optimized extrinsic from 315 sample points (3.5 edges)
  - Loss: 12.9 → 0.91

## Optimization Iterations

| File | Description |
|------|-------------|
| `projection_initial.png` | Red dots: initial camera projection on undistorted real image |
| `projection_optimized.png` | Green dots: optimized camera projection |
| `projection_comparison.png` | Red=initial, Green=optimized overlay |
| `comparison_optimized.png` | First optimization attempt (using sapien.Pose directly, incorrect) |
| `comparison_look_at.png` | Second attempt using sapien_utils.look_at conversion |
| `comparison_distorted_optimized.png` | Distorted mode with first working optimization |
| `comparison_improved.png` | With width-sampling improvement (270 points) |
| `comparison_final.png` | Final result with y2 partial edge (315 points) |

## Sim/Real Camera Comparisons

| File | Description |
|------|-------------|
| `overlay_distorted.png` | Original sim (distorted) vs real before optimization |
| `overlay_direct_pinhole.png` | Original sim (direct_pinhole) vs real before optimization |
| `overlay_only.png` | Original overlay image |
| `overlay_optimized.png` | Overlay with first optimized params |
| `compare_distorted.png` | Side-by-side: Sim distorted vs Real raw |
| `compare_undistorted.png` | Side-by-side: Sim direct_pinhole vs Real undistorted |
| `compare_sim_real.png` | Early comparison (wrong: distorted sim vs undistorted real) |

## Simulated Camera Images

| File | Description |
|------|-------------|
| `sim_camera_distorted.png` | Sim render with distortion applied |
| `sim_camera_direct_pinhole.png` | Sim render without distortion |
| `sim_optimized_camera.png` | Sim with first optimized params (incorrect sapien.Pose) |
| `sim_optimized_v2.png` | Sim with look_at optimized params |

## Image Preprocessing Debug

| File | Description |
|------|-------------|
| `debug_undistorted.png` | Real image after undistortion |
| `debug_grayscale.png` | Grayscale of undistorted image |
| `debug_binary_white_bg.png` | Binary threshold (white bg, black tape = 0) |
| `debug_binary_inv.png` | Inverted binary (black bg, tape = 255) |
| `debug_binary.png` | Final binary mask used |
| `debug_distance_field.png` | Distance transform visualization |

## HSV Filtering Debug

| File | Description |
|------|-------------|
| `debug_hsv_raw.png` | Raw HSV black filter result |
| `debug_hsv_black_v80.png` | HSV filter with V≤80 |
| `debug_hsv_black_v100.png` | HSV filter with V≤100 |
| `debug_hsv_closed.png` | After morphological closing |
| `debug_hsv_dilate_erode.png` | After dilate+erode |

## Hough Line Detection Debug

| File | Description |
|------|-------------|
| `debug_hough_lines.png` | Hough line detection mask |
| `debug_lines_overlay.png` | Detected lines on original image (green) |
| `debug_combined.png` | Combined HSV mask + Hough lines |
| `debug_final.png` | Final processed mask |

## Projection Debugging

| File | Description |
|------|-------------|
| `debug_sim_projection.png` | Testing Sapien extrinsic projection |
| `debug_sim_projection_fixed.png` | After Y-flip attempt (still wrong) |
| `debug_sapien_projection.png` | Using model_matrix + projection_matrix |
| `debug_grid_projection.png` | Grid points projected (x0-x1 area, some out of bounds) |
| `debug_grid_x1x2.png` | Grid points projected (x1-x2 area, all in bounds) |

## Optimized Camera Parameters

Final optimized cam2world extrinsic matrix:
```
[[-0.999924,  0.012300,  0.000780,  0.319304],
 [ 0.012322,  0.999045,  0.041926,  0.269718],
 [-0.000264,  0.041932, -0.999120,  0.386226],
 [ 0.0,       0.0,       0.0,       1.0     ]]
```

Equivalent look_at parameters:
- eye: [0.319, 0.270, 0.386]
- target: ≈ [0.319, 0.287, -0.019]
- up: ≈ [-0.012, -0.999, -0.042]

Original parameters:
- eye: [0.316, 0.260, 0.407]
- target: [0.316, 0.260, 0.0]
- up: [0, -1, 0]
