import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.pose import Pose
from scripts.so101 import SO101

@register_env("Track1-v0", max_episode_steps=200)
class Track1Env(BaseEnv):
    SUPPORTED_ROBOTS = ["so101", ("so101", "so101")]
    agent: SO101

    def __init__(self, *args, robot_uids=("so101", "so101"), **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        print("DEBUG: _default_sensor_configs called")
        # Front Camera Parameters from Track1_Simulation_Parameters.md
        # Position: [0.316, 0.260, 0.407]
        # Orientation: Top-down view.
        # Identity quaternion [1, 0, 0, 0] in Sapien means:
        # - Camera looks down (-Z world)
        # - Camera Up is Forward (+Y world)
        # - Camera Right is Right (+X world)
        pose = sapien.Pose(p=[0.316, 0.260, 0.407], q=[1, 0, 0, 0])
        
        return [
            CameraConfig(
                "front_camera",
                pose=pose,
                width=640,
                height=480,
                fov=np.deg2rad(73.63), # Vertical FOV from front_camera.py
                near=0.01,
                far=100,
            )
        ]

    def _load_scene(self, options: dict):
        # 1. Ground
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light([0.5, 0, -1], [3.0, 3.0, 3.0], shadow=True)
        
        # Access the underlying sapien scene to add ground
        for scene in self.scene.sub_scenes:
            ground_material = scene.create_physical_material(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
            scene.add_ground(0, material=ground_material)
        
        # Visual ground (Table surface)
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.3, 0.3, 0.01], material=[0.9, 0.9, 0.9]) # White/Light Gray table
        builder.add_box_collision(half_size=[0.3, 0.3, 0.01])
        builder.initial_pose = sapien.Pose(p=[0.3, 0.3, -0.01]) # Center at 0.3, 0.3
        self.table = builder.build_static(name="table")

        # 2. Tape Lines (Visual)
        tape_material = [0, 0, 0] # Black
        tape_half_width = 0.009
        tape_height = 0.001
        
        # Bottom Line
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.27, tape_half_width, tape_height], material=tape_material)
        builder.initial_pose = sapien.Pose(p=[0.316, 0.178, 0.001])
        builder.build_static(name="tape_bottom")
        
        # Top Line
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.27, tape_half_width, tape_height], material=tape_material)
        builder.initial_pose = sapien.Pose(p=[0.316, 0.342, 0.001])
        builder.build_static(name="tape_top")
        
        # Vertical Lines
        # 1. Left of Left Grid
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[tape_half_width, 0.082, tape_height], material=tape_material)
        builder.initial_pose = sapien.Pose(p=[0.051, 0.26, 0.001])
        builder.build_static(name="tape_left_1")
        
        # 2. Right of Left Grid / Left of Mid Grid
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[tape_half_width, 0.082, tape_height], material=tape_material)
        builder.initial_pose = sapien.Pose(p=[0.217, 0.26, 0.001])
        builder.build_static(name="tape_left_2")
        
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[tape_half_width, 0.082, tape_height], material=tape_material)
        builder.initial_pose = sapien.Pose(p=[0.238, 0.26, 0.001])
        builder.build_static(name="tape_mid_1")
        
        # 3. Right of Mid Grid / Left of Right Grid
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[tape_half_width, 0.082, tape_height], material=tape_material)
        builder.initial_pose = sapien.Pose(p=[0.394, 0.26, 0.001])
        builder.build_static(name="tape_mid_2")
        
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[tape_half_width, 0.082, tape_height], material=tape_material)
        builder.initial_pose = sapien.Pose(p=[0.414, 0.26, 0.001])
        builder.build_static(name="tape_right_1")
        
        # 4. Right of Right Grid
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[tape_half_width, 0.082, tape_height], material=tape_material)
        builder.initial_pose = sapien.Pose(p=[0.580, 0.26, 0.001])
        builder.build_static(name="tape_right_2")

        # 3. Robot Bases (Visual)
        base_material = [0.2, 0.2, 0.2] # Dark Grey
        
        # Left Base
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.055, 0.055, 0.005], material=base_material)
        builder.initial_pose = sapien.Pose(p=[0.119, 0.10, 0.005])
        builder.build_static(name="left_base_visual")
        
        # Right Base
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.055, 0.055, 0.005], material=base_material)
        builder.initial_pose = sapien.Pose(p=[0.433, 0.10, 0.005])
        builder.build_static(name="right_base_visual")

    def _load_agent(self, options: dict):
        # Robot Base Positions
        # Left Base: Center X = 0.119m. Y = 0.10m.
        # Right Base: Center X = 0.433m. Y = 0.10m.
        
        # We have two agents now.
        # Poses for the agents.
        # Note: BaseEnv._load_agent handles loading. We just need to pass the poses.
        # If robot_uids is a tuple/list, we should pass a list of poses.
        
        agent_poses = [
            sapien.Pose(p=[0.119, 0.10, 0]), # Left Robot
            sapien.Pose(p=[0.433, 0.10, 0])  # Right Robot
        ]
        
        super()._load_agent(options, agent_poses)

    def _load_objects(self, options: dict):
        # Red Cube
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.015, 0.015, 0.015])
        builder.add_box_visual(half_size=[0.015, 0.015, 0.015], material=[1, 0, 0])
        builder.initial_pose = sapien.Pose(p=[0.497, 0.26, 0.015]) # Default to Right Grid Center
        self.red_cube = builder.build(name="red_cube")

        # Green Cube
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.015, 0.015, 0.015])
        builder.add_box_visual(half_size=[0.015, 0.015, 0.015], material=[0, 1, 0])
        builder.initial_pose = sapien.Pose(p=[0.497, 0.30, 0.015])
        self.green_cube = builder.build(name="green_cube")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Basic initialization
        with torch.device(self.device):
            # Reset robot to rest pose
            # self.agent.reset(self.agent.keyframes["rest"].qpos) # This is done automatically by BaseEnv if keyframes exist?
            # Actually BaseEnv uses the first keyframe or zeros.
            pass

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: dict):
        return {}
