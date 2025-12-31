#!/usr/bin/env python3
"""
Friction Coefficient Verification Experiment

This script verifies how ManiSkill/SAPIEN/PhysX calculates the effective friction coefficient
at the contact interface between two objects with different material properties.

Experiment Design:
1. Create a tilted plane and a box with different friction coefficients
2. Measure the critical angle at which the box starts to slide
3. Use the critical angle to calculate the effective friction coefficient
4. Compare with theoretical predictions (Average, Min, Multiply, Max modes)

Physics principle:
- A box on an inclined plane starts to slide when: tan(θ) = μ_effective
- By measuring the critical angle θ, we can determine the effective friction coefficient
"""

import numpy as np
import sapien
import sapien.physx as physx
import time
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation


@dataclass
class FrictionTestConfig:
    """Configuration for a single friction test"""
    plane_friction: float
    box_friction: float
    box_mass: float = 1.0
    box_half_size: float = 0.05
    sim_timestep: float = 0.001
    max_simulation_time: float = 5.0
    velocity_threshold: float = 0.005  # Consider sliding when velocity > this


def create_scene(config: FrictionTestConfig, angle_degrees: float) -> Tuple[sapien.Scene, sapien.Entity]:
    """Create a SAPIEN scene with a tilted plane and a box"""
    
    # Create scene directly (new SAPIEN API)
    scene = sapien.Scene()
    scene.set_timestep(config.sim_timestep)
    # Gravity is [0, 0, -9.81] by default in SAPIEN
    
    angle_rad = np.deg2rad(angle_degrees)
    
    # Create tilted plane material
    plane_material = physx.PhysxMaterial(
        static_friction=config.plane_friction,
        dynamic_friction=config.plane_friction,
        restitution=0.0
    )
    
    # Create box material  
    box_material = physx.PhysxMaterial(
        static_friction=config.box_friction,
        dynamic_friction=config.box_friction,
        restitution=0.0
    )
    
    # Create tilted plane (using a large thin box as plane)
    plane_builder = scene.create_actor_builder()
    plane_builder.add_box_collision(
        half_size=[2.0, 2.0, 0.01],
        material=plane_material
    )
    plane_builder.add_box_visual(half_size=[2.0, 2.0, 0.01])
    
    # Rotate plane around Y-axis to create incline
    # Use scipy to create rotation quaternion (xyzw format for SAPIEN)
    rot = Rotation.from_euler('y', angle_rad)
    q_scipy = rot.as_quat()  # Returns [x, y, z, w]
    q_sapien = [q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]]  # Convert to [w, x, y, z]
    
    plane_pose = sapien.Pose(p=[0, 0, 0], q=q_sapien)
    plane_builder.initial_pose = plane_pose
    plane = plane_builder.build_static(name="plane")
    
    # Create box on the tilted plane
    box_builder = scene.create_actor_builder()
    box_builder.add_box_collision(
        half_size=[config.box_half_size] * 3,
        material=box_material,
        density=config.box_mass / (8 * config.box_half_size**3)
    )
    box_builder.add_box_visual(half_size=[config.box_half_size] * 3)
    
    # Position box on the tilted plane
    # The box should start at rest on the plane surface
    height_above_plane = config.box_half_size / np.cos(angle_rad) + 0.02
    x_offset = height_above_plane * np.sin(angle_rad)
    z_offset = height_above_plane * np.cos(angle_rad)
    
    box_pose = sapien.Pose(p=[x_offset, 0, z_offset], q=q_sapien)
    box_builder.initial_pose = box_pose
    box = box_builder.build(name="box")
    
    return scene, box


def test_sliding_at_angle(config: FrictionTestConfig, angle_degrees: float) -> Tuple[bool, float]:
    """
    Test if the box slides at a given angle.
    Returns (True, velocity) if box slides, (False, velocity) otherwise.
    """
    scene, box = create_scene(config, angle_degrees)
    
    # Let the box settle
    for _ in range(500):
        scene.step()
    
    # Record initial position
    initial_pos = box.get_pose().p.copy()
    
    # Simulate for a while and track maximum velocity
    simulation_steps = int(config.max_simulation_time / config.sim_timestep)
    max_velocity = 0.0
    total_displacement = 0.0
    
    for step in range(simulation_steps):
        scene.step()
        
        # Get velocity
        body = box.find_component_by_type(physx.PhysxRigidDynamicComponent)
        if body:
            velocity = np.linalg.norm(body.get_linear_velocity())
            max_velocity = max(max_velocity, velocity)
        
        current_pos = box.get_pose().p
        total_displacement = np.linalg.norm(current_pos - initial_pos)
    
    # Consider sliding if significant displacement or velocity
    slides = total_displacement > 0.01 or max_velocity > config.velocity_threshold
    
    return slides, max_velocity


def find_critical_angle(config: FrictionTestConfig, 
                        angle_min: float = 0, 
                        angle_max: float = 75,
                        precision: float = 0.5) -> float:
    """
    Use binary search to find the critical angle at which the box starts to slide.
    """
    print(f"\n  Finding critical angle (plane_μ={config.plane_friction}, box_μ={config.box_friction})...")
    
    iterations = 0
    while angle_max - angle_min > precision and iterations < 20:
        angle_mid = (angle_min + angle_max) / 2
        slides, vel = test_sliding_at_angle(config, angle_mid)
        
        print(f"    Angle {angle_mid:5.1f}°: {'SLIDES' if slides else 'STATIC':6s} (max_vel={vel:.4f})")
        
        if slides:
            angle_max = angle_mid
        else:
            angle_min = angle_mid
        
        iterations += 1
    
    return (angle_min + angle_max) / 2


def calculate_theoretical_friction(mu1: float, mu2: float) -> dict:
    """Calculate theoretical effective friction for each combine mode"""
    return {
        "AVERAGE": (mu1 + mu2) / 2,
        "MIN": min(mu1, mu2),
        "MULTIPLY": mu1 * mu2,
        "MAX": max(mu1, mu2)
    }


def run_friction_experiment():
    """Run the full friction verification experiment"""
    
    print("=" * 70)
    print("FRICTION COEFFICIENT VERIFICATION EXPERIMENT")
    print("=" * 70)
    print("\nThis experiment verifies how PhysX combines friction coefficients")
    print("at the contact interface between two materials.\n")
    print("Method: Measure the critical angle at which a box starts sliding")
    print("        on an inclined plane. tan(θ_critical) = μ_effective")
    
    # Test cases with different friction combinations
    test_cases = [
        {"plane_friction": 0.5, "box_friction": 0.5},   # Same friction
        {"plane_friction": 0.3, "box_friction": 0.7},   # Different friction
        {"plane_friction": 2.0, "box_friction": 0.3},   # Gripper-like vs cube-like
        {"plane_friction": 0.2, "box_friction": 0.8},   # Large difference
    ]
    
    results = []
    
    for i, tc in enumerate(test_cases):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i+1}: plane_μ = {tc['plane_friction']}, box_μ = {tc['box_friction']}")
        print("=" * 70)
        
        config = FrictionTestConfig(**tc)
        
        # Find critical angle through simulation
        critical_angle = find_critical_angle(config)
        measured_friction = np.tan(np.deg2rad(critical_angle))
        
        # Calculate theoretical predictions
        theoretical = calculate_theoretical_friction(tc['plane_friction'], tc['box_friction'])
        
        # Calculate theoretical angles
        theoretical_angles = {mode: np.rad2deg(np.arctan(mu)) for mode, mu in theoretical.items()}
        
        print(f"\n  RESULTS:")
        print(f"  ├─ Critical Angle (measured): {critical_angle:.1f}°")
        print(f"  ├─ Effective μ (measured):    {measured_friction:.3f}")
        print(f"  │")
        print(f"  ├─ Theoretical Predictions:")
        for mode, mu in theoretical.items():
            angle = theoretical_angles[mode]
            error = abs(critical_angle - angle)
            match = "✓ MATCH" if error < 3.0 else ""
            print(f"  │   {mode:8s}: μ = {mu:.3f}, θ = {angle:.1f}° (error: {error:.1f}°) {match}")
        
        # Determine which mode is being used
        errors = {mode: abs(critical_angle - angle) for mode, angle in theoretical_angles.items()}
        best_match = min(errors, key=errors.get)
        
        print(f"  │")
        print(f"  └─ Best Match: {best_match} mode (error: {errors[best_match]:.1f}°)")
        
        results.append({
            "plane_friction": tc['plane_friction'],
            "box_friction": tc['box_friction'],
            "critical_angle": critical_angle,
            "measured_friction": measured_friction,
            "theoretical": theoretical,
            "best_match": best_match,
            "error": errors[best_match]
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\n| Plane μ | Box μ  | Measured μ | Critical θ | Best Match |")
    print("|---------|--------|------------|------------|------------|")
    for r in results:
        print(f"| {r['plane_friction']:7.2f} | {r['box_friction']:6.2f} | {r['measured_friction']:10.3f} | {r['critical_angle']:10.1f}° | {r['best_match']:10s} |")
    
    modes_matched = [r['best_match'] for r in results]
    from collections import Counter
    mode_counts = Counter(modes_matched)
    
    print(f"\nMatched combine modes across all tests:")
    for mode, count in mode_counts.most_common():
        print(f"  {mode}: {count}/{len(results)} tests")
    
    most_common_mode = mode_counts.most_common(1)[0][0]
    print(f"\n⟹ PhysX appears to use '{most_common_mode}' friction combine mode by default")
    
    if most_common_mode == "AVERAGE":
        print("  This matches the expected default behavior documented in PxMaterial.h")
        print("  Formula: μ_effective = (μ_1 + μ_2) / 2")
    
    # Practical example
    print("\n" + "=" * 70)
    print("PRACTICAL EXAMPLE: GRIPPER (μ=2.0) vs CUBE (μ=0.3)")
    print("=" * 70)
    gripper_friction = 2.0
    cube_friction = 0.3
    theoretical = calculate_theoretical_friction(gripper_friction, cube_friction)
    print(f"\n  Theoretical effective friction coefficients:")
    for mode, mu in theoretical.items():
        print(f"    {mode:8s}: μ = {mu:.3f}")
    
    if most_common_mode == "AVERAGE":
        effective_mu = theoretical["AVERAGE"]
        print(f"\n  With AVERAGE mode: μ_effective = {effective_mu:.3f}")
        print(f"  This means for a 100g cube (m=0.1kg, F_gravity=0.98N):")
        print(f"  Friction force available = μ × N = {effective_mu:.3f} × N")
        print(f"  If gripper applies 2N normal force per finger (4N total):")
        print(f"  Maximum friction force = {effective_mu:.3f} × 4N = {effective_mu * 4:.2f}N")
        print(f"  This is {'sufficient' if effective_mu * 4 > 0.98 else 'insufficient'} to hold the cube!")
    
    return results


if __name__ == "__main__":
    results = run_friction_experiment()
