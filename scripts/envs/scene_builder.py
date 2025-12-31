"""
Scene builder module for Track1 environment.

This module contains functions for building the scene (table, tape lines,
grid boundaries, debug markers, cubes) extracted from Track1Env.
All functions receive the environment instance as the first parameter
to access scene, config, and other environment state.
"""
import numpy as np
import sapien
import sapien.render
from mani_skill.utils.structs import Actor
from sapien.physx import PhysxRigidBodyComponent, PhysxRigidDynamicComponent


def build_debug_markers(env):
    """Build debug markers for coordinate system visualization.
    Red at (0,0), Green at (1,0), Blue at (0,1).
    """
    marker_height = 0.005  # Slightly above table/ground
    radius = 0.02
    
    markers = [
        {"pos": [0, 0, marker_height], "color": [1, 0, 0], "name": "debug_origin_red"},
        {"pos": [1, 0, marker_height], "color": [0, 1, 0], "name": "debug_x1_green"},
        {"pos": [0, 1, marker_height], "color": [0, 0, 1], "name": "debug_y1_blue"},
    ]
    
    for marker in markers:
        builder = env.scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius, material=marker["color"])
        builder.initial_pose = sapien.Pose(p=marker["pos"])
        builder.build_static(name=marker["name"])


def build_table(env):
    """Build table with optional visual randomization."""
    if env.domain_randomization:
        tables = []
        for i in range(env.num_envs):
            builder = env.scene.create_actor_builder()
            # Randomize table color slightly
            color = [0.9 + np.random.uniform(-0.05, 0.05)] * 3 + [1]
            builder.add_box_visual(
                half_size=[0.3, 0.3, 0.01], 
                material=sapien.render.RenderMaterial(base_color=color)
            )
            # Use friction material from config for table surface
            table_material = sapien.physx.PhysxMaterial(
                static_friction=env.table_physics["static_friction"],
                dynamic_friction=env.table_physics["dynamic_friction"],
                restitution=env.table_physics["restitution"]
            )
            builder.add_box_collision(
                half_size=[0.3, 0.3, 0.01],
                material=table_material
            )
            builder.initial_pose = sapien.Pose(p=[0.3, 0.3, -0.01])
            builder.set_scene_idxs([i])
            table = builder.build_static(name=f"table_{i}")
            env.scene.remove_from_state_dict_registry(table)
            tables.append(table)
        env.table = Actor.merge(tables, name="table")
        env.scene.add_to_state_dict_registry(env.table)
    else:
        builder = env.scene.create_actor_builder()
        # Use friction material from config for table surface
        table_material = sapien.physx.PhysxMaterial(
            static_friction=env.table_physics["static_friction"],
            dynamic_friction=env.table_physics["dynamic_friction"],
            restitution=env.table_physics["restitution"]
        )
        builder.add_box_visual(half_size=[0.3, 0.3, 0.01], material=[0.9, 0.9, 0.9])
        builder.add_box_collision(
            half_size=[0.3, 0.3, 0.01],
            material=table_material
        )
        builder.initial_pose = sapien.Pose(p=[0.3, 0.3, -0.01])
        env.table = builder.build_static(name="table")


def compute_grids(env):
    """Compute grid coordinates and boundaries with optional randomization."""
    tape_half_width = 0.009
    
    # Base values (Human specified)
    x_1 = 0.204
    x_4 = 0.6
    y_1 = 0.15
    upper_height = 0.164
    
    # Add randomization if enabled
    # if env.domain_randomization:
    #     noise_scale = 0.005 # +/- 5mm
    #     x_1 += np.random.uniform(-noise_scale, noise_scale)
    #     # x_4 (table width) usually fixed or small noise
    #     x_4 += np.random.uniform(-0.002, 0.002) 
    #     y_1 += np.random.uniform(-noise_scale, noise_scale)
    #     upper_height += np.random.uniform(-0.002, 0.002)
    
    # NOTE: User requested independent tape randomization. 
    # We keep the logical grid bounds deterministic (or globally fixed for this episode)
    # so success criteria are consistent, but the Visual Tape will be noisy.
        
    # Calculate derived coordinates
    x = [0.0] * 5
    x[1] = x_1
    x[0] = x[1] - 0.166 - 2 * tape_half_width
    x[4] = x_4
    x[2] = x[4] - 0.204 - 2 * tape_half_width
    x[3] = x[4] - 0.204 + 0.166
    
    y = [0.0] * 3
    y[0] = 0.0
    y[1] = y_1
    y[2] = y_1 + upper_height + 2 * tape_half_width
    
    # Store for build_tape_lines
    env.grid_points = {"x": x, "y": y, "tape_half_width": tape_half_width}
    
    # Calculate logical boundaries for success/placement (Inner areas excluding tape)
    # Left Grid: between col1(x[0]) and col2(x[1]) ?? 
    # Wait, let's map the user's tape logic to logical areas.
    
    # Tape logic from user:
    # row1: y[1] to y[1]+2w (Separates Bottom and Upper?) No, row1 is y[1]. 
    # row2: y[2] to ...
    
    # Based on user code:
    # Row 1 pos y: y[1] + w.  Size y: w. -> Tape is from y[1] to y[1]+2w.
    # Row 2 pos y: y[2] + w.  Size y: w. -> Tape is from y[2] to y[2]+2w.
    
    # Col 1 pos x: x[0] + w.  Size x: w. -> Tape is from x[0] to x[0]+2w.
    # Col 2 pos x: x[1] + w.  Size x: w. -> Tape is from x[1] to x[1]+2w.
    
    # So the grid "Left" is likely between Col 1 and Col 2, and Row 1 and Row 2.
    # Left Grid Bounds:
    # X: (x[0] + 2w) to x[1]
    # Y: (y[1] + 2w) to y[2] 
    
    w = tape_half_width
    
    env.grid_bounds["left"] = {
        "x_min": x[0] + 2*w, "x_max": x[1],
        "y_min": y[1] + 2*w, "y_max": y[2]
    }
    
    env.grid_bounds["mid"] = {
        "x_min": x[1] + 2*w, "x_max": x[2], # Wait, is there a tape between Left and Mid?
        # User code: col1, col4, col2, col3, col5.
        # col1 @ x[0], col2 @ x[1], col3 @ x[2], col4 @ x[3], col5 @ x[4]?
        # Let's re-read user code logic carefully.
        # col1: x[0]. col2: x[1]. col3: x[2]. col4: x[3]. col5: x[4]... 
        # col4 pos: x[3]+w. 
        
        # Left Grid is between x[0] and x[1].
        # Mid Grid is between x[1] and x[2]? Or x[1] and x[2] are edges?
        # x[2] = x[4] - 0.204 - 2w.
        # x[3] = x[4] - 0.204 + 0.166.
        
        # It seems:
        # Left: x[0]...x[1]
        # Gap?
        # Mid: x[1]...x[2] ?? No, x[1]=0.204. x[2] ~ 0.6-0.2-small = 0.38.
        # Right: x[3]...x[4]? x[3] ~ 0.56. x[4]=0.6. width ~4cm? No.
        
        # Let's trust the areas defined by the columns.
        # Left Grid: Inside col1 and col2.
        "y_min": y[1] + 2*w, "y_max": y[2]
    }
    
    # Re-evaluating Mid/Right based on user's manual "draw correctly" code
    # User X array: x[0], x[1], x[2], x[3], x[4]
    # col1 at x[0]
    # col2 at x[1]
    # col3 at x[2]
    # col4 at x[3]
    # col5 at x[4]
    
    # Left Grid: between col1 and col2.
    # Mid Grid: between col2 and col3.
    env.grid_bounds["mid"] = {
        "x_min": x[1] + 2*w, "x_max": x[2],
        "y_min": y[1] + 2*w, "y_max": y[2]
    }
    
    # Right Grid: between col3 and col4 ? 
    # OR col3 and col5?
    # x[3] = x[4] - 0.204 + 0.166.  = 0.562.  x[4]=0.6. Diff = 0.038. Too small for Right grid.
    # x[2] = x[4] - 0.204 - 2w = 0.378.
    # Gap between x[2] and x[3] = 0.562 - 0.378 = 0.184. This looks like the Right Grid!
    
    # So Right Grid is between col3(x[2]) and col4(x[3]).
    env.grid_bounds["right"] = {
        "x_min": x[2] + 2*w, "x_max": x[3],
        "y_min": y[1] + 2*w, "y_max": y[2]
    }
    
    # Bottom Grid (between robot bases)
    # Usually below Mid.
    # User code: col2 and col3 extend down to y[0]?
    # col2 pos y: (y[2]+y[0])/2. Height: (y[2]-y[0])/2. -> Spans y[0] to y[2].
    # col3 pos y: (y[2]+y[0])/2. -> Spans y[0] to y[2].
    # So col2 and col3 go all the way down.
    # Thus Bottom Grid is between col2 and col3, and between row? (no bottom row tape?)
    # row1 is at y[1].
    # So Bottom Grid is y[0] to y[1].
    env.grid_bounds["bottom"] = {
        "x_min": x[1] + 2*w, "x_max": x[2],
        "y_min": y[0], "y_max": y[1]
    }


def build_tape_lines(env):
    """Build black tape lines using computed grid points."""
    tape_material = [0, 0, 0]
    tape_height = 0.001
    
    # Retrieve computed params
    x = env.grid_points["x"]
    y = env.grid_points["y"]
    tape_half_width = env.grid_points["tape_half_width"]
    
    tape_specs = []

    tape_specs.append({
        "half_size": [(x[3]- x[0]) / 2 + tape_half_width, tape_half_width, tape_height],
        "pos": [(x[3] +  x[0]) / 2 + tape_half_width, y[1] + tape_half_width, 0.001],
        "name": "row1"
    })

    tape_specs.append({
        "half_size": [(x[3]- x[0]) / 2 + tape_half_width, tape_half_width, tape_height],
        "pos": [(x[3] +  x[0]) / 2 + tape_half_width, y[2] + tape_half_width, 0.001],
        "name": "row2"
    })


    tape_specs.append({
        "half_size": [tape_half_width, (y[2] - y[1])/2 + tape_half_width , tape_height],
        "pos": [x[0] + tape_half_width, (y[2] + y[1])/2 + tape_half_width, 0.001],
        "name": "col1"
    })

    tape_specs.append({
        "half_size": [tape_half_width, (y[2] - y[1])/2 + tape_half_width , tape_height],
        "pos": [x[3] + tape_half_width, (y[2] + y[1])/2 + tape_half_width, 0.001],
        "name": "col4"
    })

    tape_specs.append({
        "half_size": [tape_half_width, (y[2] - y[0])/2 + tape_half_width , tape_height],
        "pos": [x[1] + tape_half_width, (y[2] + y[0])/2 + tape_half_width, 0.001],
        "name": "col2"
    })
    
    tape_specs.append({
        "half_size": [tape_half_width, (y[2] - y[0])/2 + tape_half_width , tape_height],
        "pos": [x[2] + tape_half_width, (y[2] + y[0])/2 + tape_half_width, 0.001],
        "name": "col3"
    })

    tape_specs.append({
        "half_size": [tape_half_width, 0.6 / 2 , tape_height],
        "pos": [x[4] + tape_half_width, 0.6 / 2, 0.001],
        "name": "col5"
    })
    
    # Build all tape lines
    for spec in tape_specs:
        builder = env.scene.create_actor_builder()
        
        # Apply independent randomization if enabled
        pos = list(spec["pos"])
        half_size = list(spec["half_size"])
        rotation = [1, 0, 0, 0]  # Identity quaternion
        
        if env.domain_randomization:
            # 1. Position Noise (x, y)
            pos_noise = np.random.uniform(-0.005, 0.005, size=2)  # +/- 5mm
            pos[0] += pos_noise[0]
            pos[1] += pos_noise[1]
            
            # 2. Size Noise (length aka half_size[0] mostly, or width)
            size_noise = np.random.uniform(-0.002, 0.002)  # +/- 2mm
            # Don't change thickness (z), maybe slight width/length change
            half_size[0] += size_noise 
            
            # 3. Rotation Noise (Yaw)
            # Small rotation around Z axis
            yaw_noise = np.deg2rad(np.random.uniform(-2, 2))  # +/- 2 degrees
            # simpler:
            q_z = np.sin(yaw_noise / 2)
            q_w = np.cos(yaw_noise / 2)
            rotation = [q_w, 0, 0, q_z]

        builder.add_box_visual(half_size=half_size, material=tape_material)
        builder.initial_pose = sapien.Pose(p=pos, q=rotation)
        builder.build_static(name=spec["name"])



def build_cube(env, name: str, half_size: float, base_color: list, default_pos: list, is_static: bool = False) -> Actor:
    """Build a cube with optional domain randomization.
    
    Args:
        is_static: If True, build as kinematic actor (pose controllable but immune to physics).
    """
    if env.domain_randomization:
        cubes = []
        for i in range(env.num_envs):
            builder = env.scene.create_actor_builder()
            
            # Randomize color slightly
            color = [
                base_color[0] + np.random.uniform(-0.1, 0.1),
                base_color[1] + np.random.uniform(-0.1, 0.1),
                base_color[2] + np.random.uniform(-0.1, 0.1),
                1
            ]
            color = [max(0, min(1, c)) for c in color]
            
            builder.add_box_collision(half_size=[half_size] * 3)
            builder.add_box_visual(
                half_size=[half_size] * 3,
                material=sapien.render.RenderMaterial(base_color=color)
            )
            builder.initial_pose = sapien.Pose(p=default_pos)
            builder.set_scene_idxs([i])
            
            # Always build as standard actor and set kinematic if needed
            cube = builder.build(name=f"{name}_{i}")
            if is_static:
                for obj in cube._objs:
                    dyn = obj.find_component_by_type(PhysxRigidDynamicComponent)
                    if dyn: dyn.kinematic = True
            
            env.scene.remove_from_state_dict_registry(cube)
            cubes.append(cube)
        
        merged = Actor.merge(cubes, name=name)
        env.scene.add_to_state_dict_registry(merged)
        
        apply_cube_physics(env, merged, is_static=is_static)
        return merged
    else:
        builder = env.scene.create_actor_builder()
        builder.add_box_collision(half_size=[half_size] * 3)
        builder.add_box_visual(half_size=[half_size] * 3, material=base_color[:3])
        builder.initial_pose = sapien.Pose(p=default_pos)
        
        cube = builder.build(name=name)
        if is_static:
            for obj in cube._objs:
                dyn = obj.find_component_by_type(PhysxRigidDynamicComponent)
                if dyn: dyn.kinematic = True
                
        apply_cube_physics(env, cube, is_static=is_static)
        return cube


def apply_cube_physics(env, cube: Actor, is_static: bool = False):
    """Apply physics properties to cube (optionally randomized).
    
    Args:
        is_static: If True, skip mass assignment (kinematic objects skip mass).
                  Friction and restitution are always applied.
    """
    p = env.cube_physics
    for obj in cube._objs:
        rigid_body = obj.find_component_by_type(PhysxRigidBodyComponent)
        if rigid_body is not None:
            # 1. Apply Mass (only if not static/kinematic)
            if not is_static:
                # Use randomized or fixed mass
                m = p["mass"] * np.random.uniform(0.5, 1.5) if env.domain_randomization else p["mass"]
                rigid_body.set_mass(m)
            
            # 2. Apply Friction and Restitution to all shapes
            for shape in rigid_body.get_collision_shapes():
                if env.domain_randomization:
                    sf = p["static_friction"] * np.random.uniform(0.8, 1.2)
                    df = p["dynamic_friction"] * np.random.uniform(0.8, 1.2)
                    res = p["restitution"] * np.random.uniform(0.8, 1.2)
                else:
                    sf = p["static_friction"]
                    df = p["dynamic_friction"]
                    res = p["restitution"]
                
                # Create a new material to avoid unintended sharing between parallel environments
                mat = sapien.physx.PhysxMaterial(
                    static_friction=sf,
                    dynamic_friction=df,
                    restitution=res
                )
                shape.physical_material = mat
