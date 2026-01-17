from typing import Optional
import numpy as np
import sapien
import sapien.render
import torch
import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs import Actor

from scripts.agents.so101 import SO101
from scripts.training.config_utils import Track1Config
from scripts.envs import scene_builder
from scripts.envs import camera_processing




@register_env("Track1-v0", max_episode_steps=800)  # Default high, actual limit from config
class Track1Env(BaseEnv):
    SUPPORTED_ROBOTS = ["so101", ("so101", "so101")]
    agent: SO101

    def __init__(
        self, 
        *args, 
        robot_uids=("so101", "so101"),
        cfg=None,  # Hydra DictConfig - primary way to pass Track1-specific config
        # Legacy explicit params (for backwards compatibility, cfg takes precedence)
        task: str = None,
        domain_randomization: bool = None,
        camera_mode: str = None,
        render_scale: int = 3,
        reward_config: dict = None,
        cube_physics: dict = None,
        table_physics: dict = None,
        action_bounds: dict = None,
        camera_extrinsic: list = None,
        undistort_alpha: float = None,
        obs_normalization: dict = None,
        eval_mode: bool = False,
        **kwargs  # BaseEnv params: obs_mode, reward_mode, control_mode, sim_config, etc.
    ):
        # 1. Initialize Configuration
        if cfg is not None:
            self.track1_cfg = Track1Config.from_hydra(cfg)
            # Priority for explicit task parameter
            if task is not None:
                self.track1_cfg.task = task
        else:
            # Fallback for legacy explicit params
            self.track1_cfg = Track1Config(
                task=task or "lift",
                domain_randomization=domain_randomization if domain_randomization is not None else True,
                camera_mode=camera_mode or "direct_pinhole",
                render_scale=render_scale,
                undistort_alpha=undistort_alpha if undistort_alpha is not None else 0.25,
                action_bounds=action_bounds,
                camera_extrinsic=camera_extrinsic,
            )
            if cube_physics: self.track1_cfg.cube_physics = cube_physics
            if table_physics: self.track1_cfg.table_physics = table_physics
            if obs_normalization: 
                from scripts.training.config_utils import ObsNormalizationConfig
                self.track1_cfg.obs = ObsNormalizationConfig(**obs_normalization)
            if reward_config:
                from scripts.training.config_utils import RewardConfig
                # This is a bit complex for legacy, but from_hydra handles it better
                self.track1_cfg = Track1Config.from_hydra({"env": {"task": self.track1_cfg.task}, "reward": reward_config})

        # 2. Extract configuration to instance attributes (for parity and ease of use)
        cfg = self.track1_cfg
        self.task = cfg.task
        self.domain_randomization = cfg.domain_randomization
        self.eval_mode = eval_mode
        self.camera_extrinsic = cfg.camera_extrinsic
        self.undistort_alpha = cfg.undistort_alpha
        self.camera_mode = cfg.camera_mode
        self.render_scale = cfg.render_scale
        self.cube_physics = {
            "mass": cfg.cube_physics.mass,
            "static_friction": cfg.cube_physics.static_friction,
            "dynamic_friction": cfg.cube_physics.dynamic_friction,
            "restitution": cfg.cube_physics.restitution
        }
        self.table_physics = {
            "static_friction": cfg.table_physics.static_friction,
            "dynamic_friction": cfg.table_physics.dynamic_friction,
            "restitution": cfg.table_physics.restitution
        }
        
        # 3. Observation Normalization
        obs_cfg = cfg.obs
        self.obs_normalize_enabled = obs_cfg.enabled
        self.qpos_scale = obs_cfg.qpos_scale
        self.qvel_clip = obs_cfg.qvel_clip
        self.relative_pos_clip = obs_cfg.relative_pos_clip
        self.include_abs_pos = obs_cfg.include_abs_pos
        self.include_target_qpos = obs_cfg.include_target_qpos
        self.obs_action_bounds = obs_cfg.action_bounds
        self.tcp_pos_norm = obs_cfg.tcp_pos
        self.red_cube_pos_norm = obs_cfg.red_cube_pos
        self.green_cube_pos_norm = obs_cfg.green_cube_pos
        self.include_is_grasped = obs_cfg.include_is_grasped
        self.include_tcp_orientation = obs_cfg.include_tcp_orientation
        self.include_cube_displacement = obs_cfg.include_cube_displacement
        
        # 4. Reward Configuration logic (formerly _setup_reward_config)
        rw_cfg = cfg.reward
        self.reward_type = rw_cfg.reward_type
        self.reward_weights = rw_cfg.weights.copy()
        self.reward_weights["fail"] = rw_cfg.weights.get("fail", 0.0)
        self.reward_weights["approach2"] = rw_cfg.weights.get("approach2", 0.0)
        self.reward_weights["action_rate"] = rw_cfg.weights.get("action_rate", 0.0)
        
        self.grasp_hold_max_steps = rw_cfg.grasp_hold_max_steps
        self.approach_curve = rw_cfg.approach_curve
        self.approach_threshold = rw_cfg.approach_threshold
        self.approach_zero_point = rw_cfg.approach_zero_point
        self.approach_tanh_scale = rw_cfg.approach_tanh_scale
        self.approach_scale = rw_cfg.approach_scale
        self.reach_scale = self.approach_scale
        self.approach_mode = rw_cfg.approach_mode
        self.stage_thresholds = rw_cfg.stage_thresholds
        
        self.lift_target = rw_cfg.lift_target
        self.lift_max_height = rw_cfg.lift_max_height
        self.stable_hold_time = rw_cfg.stable_hold_time
        control_freq = getattr(self, 'control_freq', 30)
        self.stable_hold_steps = int(self.stable_hold_time * control_freq)
        
        self.fail_bounds = rw_cfg.fail_bounds
        self.spawn_bounds = rw_cfg.spawn_bounds
        

        self.horizontal_displacement_threshold = rw_cfg.horizontal_displacement_threshold
        self.grasp_min_force = rw_cfg.grasp_min_force
        self.grasp_max_angle = rw_cfg.grasp_max_angle
        self.stack_height_target = rw_cfg.stack_height_target
        self.stack_height_tolerance = rw_cfg.stack_height_tolerance
        self.stack_xy_tolerance = rw_cfg.stack_xy_tolerance
        self.stack_align_tanh_scale = rw_cfg.stack_align_tanh_scale
        self.green_z_range = rw_cfg.green_z_range
        
        self.adaptive_grasp_enabled = rw_cfg.adaptive_grasp_weight.enabled
        self.adaptive_grasp_alpha = rw_cfg.adaptive_grasp_weight.alpha
        self.adaptive_grasp_eps = rw_cfg.adaptive_grasp_weight.eps
        self.adaptive_grasp_max = rw_cfg.adaptive_grasp_weight.max_weight
        self.adaptive_grasp_tau = rw_cfg.adaptive_grasp_weight.tau
        
        self.gate_lift_with_grasp = rw_cfg.gate_lift_with_grasp
        
        self.adaptive_lift_enabled = rw_cfg.adaptive_lift_weight.enabled
        self.adaptive_lift_alpha = rw_cfg.adaptive_lift_weight.alpha
        self.adaptive_lift_eps = rw_cfg.adaptive_lift_weight.eps
        self.adaptive_lift_max = rw_cfg.adaptive_lift_weight.max_weight
        self.adaptive_lift_tau = rw_cfg.adaptive_lift_weight.tau
        
        self.adaptive_success_enabled = rw_cfg.adaptive_success_weight.enabled
        self.adaptive_success_alpha = rw_cfg.adaptive_success_weight.alpha
        self.adaptive_success_eps = rw_cfg.adaptive_success_weight.eps
        self.adaptive_success_max = rw_cfg.adaptive_success_weight.max_weight
        self.adaptive_success_tau = rw_cfg.adaptive_success_weight.tau

        self.prev_action = None
        self.space_gap = 0.001 
        self.single_arm_mode = (self.task != "sort")
        
        # 4.5 Initialize Camera Grids
        self.distortion_grid: Optional[torch.Tensor] = None
        self.undistortion_grid: Optional[torch.Tensor] = None
        
        # 5. SO101 Agent Setup
        # 5. SO101 Agent Setup (Isolated by Task)
        # Create a configured class with UID = so101_{task}
        ConfiguredSO101 = SO101.create_configured_class(
            task_name=self.task,
            mode="dual" if self.task == "sort" else "single",
            action_bounds=cfg.action_bounds,
            urdf_path=cfg.robot_urdf,
            cfg=cfg.raw_cfg # Pass full config to derive gripper physics
        )
        new_uid = ConfiguredSO101.uid
        
        # 6. Finalize environment setup
        # Ensure super().__init__ uses the configured class UID
        if isinstance(robot_uids, str):
            if robot_uids == "so101": robot_uids = new_uid
        elif isinstance(robot_uids, (list, tuple)):
            robot_uids = tuple(new_uid if uid == "so101" else uid for uid in robot_uids)
        
        self.render_scale = cfg.render_scale
        self.grid_bounds = {}
        self._setup_camera_processing_maps()

        # Update SUPPORTED_ROBOTS dynamically to include our new task-specific UID
        # This silences the ManiSkill warning while maintaining isolation
        self.SUPPORTED_ROBOTS = [new_uid, (new_uid, new_uid)]
        
        self.task_handler = self._create_task_handler(self.task)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self._setup_device()

    def _create_task_handler(self, task):
        if task == "lift":
            from scripts.tasks.lift import LiftTaskHandler
            return LiftTaskHandler(self)
        elif task == "stack":
            from scripts.tasks.stack import StackTaskHandler
            return StackTaskHandler(self)
        elif task == "sort":
            from scripts.tasks.sort import SortTaskHandler
            return SortTaskHandler(self)
        elif task == "static_grasp":
            from scripts.tasks.static_grasp import StaticGraspTaskHandler
            return StaticGraspTaskHandler(self)
        else:
            raise ValueError(f"Unknown task: {task}")


    def _setup_device(self):
        # self.device is guaranteed by BaseEnv
        if self.distortion_grid is not None:
            self.distortion_grid = self.distortion_grid.to(self.device)
        if self.undistortion_grid is not None:
            self.undistortion_grid = self.undistortion_grid.to(self.device)

    def _setup_camera_processing_maps(self):
        """Precompute torch grids for camera distortion/undistortion processing.
        
        Delegates to camera_processing module.
        """
        camera_processing.setup_camera_processing_maps(self)

    def _apply_camera_processing(self, obs):
        """Apply camera processing based on camera_mode.
        
        Delegates to camera_processing module.
        """
        return camera_processing.apply_camera_processing(self, obs)

    @property
    def _default_sensor_configs(self):
        """Front Camera with optional config file override for manual tuning."""
        
        # Use extrinsic matrix from config if provided
        if self.camera_extrinsic is not None:
            extrinsic = np.array(self.camera_extrinsic)
            R = extrinsic[:3, :3]  # Rotation matrix (cam2world)
            eye = extrinsic[:3, 3]  # Camera position
            
            # forward = camera Z axis in world (third column of R)
            forward = R[:, 2]
            
            # up = camera -Y axis in world (images have Y pointing down)
            up = -R[:, 1]
            
            # target = eye + forward * distance (use original distance ~0.407)
            distance = 0.407
            target = eye + forward * distance
            
            pose = sapien_utils.look_at(eye=eye.tolist(), target=target.tolist(), up=up.tolist())
        else:
            # Default look_at parameters
            pose = sapien_utils.look_at(eye=[0.316, 0.260, 0.407], target=[0.316, 0.260, 0.0], up=[0, -1, 0])
        
        if self.domain_randomization and getattr(self, "num_envs", 0) > 1:
            # base_pose = sapien.Pose(p=base_pos, q=q_sapien)
            # pose = Pose.create(base_pose)
            
            # Note: look_at returns a sapien.Pose (cpu). We need to convert if using batch logic?
            # look_at supports batch if inputs are tensors. Here inputs are lists.
            # So 'pose' is a single sapien.Pose.
            
            # Convert to Maniskill Pose to apply randomization
            pose = Pose.create(pose)
            
            pose = pose * Pose.create_from_pq(
                p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
                q=randomization.random_quaternions(
                    n=self.num_envs, device=self.device, bounds=(-np.pi / 24, np.pi / 24)
                ),
            )
        
        # Determine resolution and intrinsic based on camera_mode
        if self.camera_mode == "direct_pinhole":
            return [
                CameraConfig(
                    "front_camera",
                    pose=pose,
                    width=self.front_render_width,
                    height=self.front_render_height,
                    intrinsic=self.render_intrinsic,
                    near=0.01,
                    far=100,
                ),
            ]
        else:
            # High-res source for distortion pipeline using scaled intrinsic
            return [
                CameraConfig(
                    "front_camera",
                    pose=pose,
                    width=self.front_render_width,
                    height=self.front_render_height,
                    intrinsic=self.render_intrinsic,
                    near=0.01,
                    far=100,
                ),
            ]

    def _setup_sensors(self, options: dict):
        """Override to add wrist cameras after agents are loaded."""
        super()._setup_sensors(options)
        
        from mani_skill.sensors.camera import Camera
        wrist_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        
        for i, agent in enumerate(self.agent.agents):
            # Skip left arm wrist camera (index 0, so101-0) for single-arm tasks
            # Right arm is index 1 (so101-1)
            if self.single_arm_mode and i == 0:
                continue
                
            camera_link = agent.robot.links_map.get("camera_link", None)
            if camera_link is not None:
                uid = f"wrist_camera_{i}"
                config = CameraConfig(
                    uid,
                    pose=wrist_pose,
                    width=640,
                    height=480,
                    fov=np.deg2rad(50),
                    near=0.01,
                    far=100,
                    mount=camera_link,
                )
                self._sensors[uid] = Camera(config, self.scene)

    def render_sensors(self):
        """Override to render only RGB images from sensors (no depth/segmentation).
        
        This reduces rendering compute by only requesting RGB texture from the shader,
        unlike the default which requests all textures (rgb + position + segmentation).
        """
        from mani_skill.utils.visualization.misc import tile_images
        from mani_skill.sensors.camera import Camera
        
        # Hide objects that should be hidden for observation
        for obj in self._hidden_objects:
            obj.hide_visual()
        
        # Update render for sensors only
        self.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        self.capture_sensor_data()
        
        images = []
        for name, sensor in self._sensors.items():
            if isinstance(sensor, Camera):
                # Request ONLY RGB - this is the key difference that reduces compute
                obs = sensor.get_obs(
                    rgb=True,
                    depth=False,
                    position=False,
                    segmentation=False,
                    apply_texture_transforms=True
                )
                if 'rgb' in obs:
                    images.append(obs['rgb'])
        
        if len(images) == 0:
            # Fallback to default if no RGB found
            return super().render_sensors()
        
        return tile_images(images)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.camera_mode != "direct_pinhole":
            obs = self._apply_camera_processing(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.camera_mode != "direct_pinhole":
            obs = self._apply_camera_processing(obs)
        return obs, reward, terminated, truncated, info


    def _get_obs_extra(self, info: dict):
        """Return extra observations (state-based).
        
        Includes privileged information like object poses and relative vectors
        to facilitate state-based training.
        """
        obs = dict()
        
        # Helper for position normalization: (pos - mean) / std with optional clipping
        def normalize_pos(pos, norm_config):
            mean = torch.tensor(norm_config["mean"], device=self.device)
            std = torch.tensor(norm_config["std"], device=self.device)
            normalized = (pos - mean) / std
            # Optional clipping to ±clip_with_std
            clip_std = norm_config.get("clip_with_std", None)
            if clip_std is not None:
                normalized = torch.clamp(normalized, -clip_std, clip_std)
            return normalized
        
        # 1. Object State
        red_cube_pos = self.red_cube.pose.p
        obs["red_cube_rot"] = self.red_cube.pose.q
        
        # 1a. Cube Displacement (Relative to initial spawn position)
        initial_red_cube_pos = self.task_handler.initial_red_cube_pos
        if self.include_cube_displacement and initial_red_cube_pos is not None:
            red_disp = red_cube_pos - initial_red_cube_pos
            if self.obs_normalize_enabled:
                # Use relative_pos_clip for displacement as well
                red_disp = torch.clamp(red_disp, -self.relative_pos_clip, self.relative_pos_clip) / self.relative_pos_clip
            obs["red_cube_displacement"] = red_disp
        
        green_cube_pos = None
        if self.green_cube is not None:
            green_cube_pos = self.green_cube.pose.p
            obs["green_cube_rot"] = self.green_cube.pose.q
            
            initial_green_cube_pos = self.task_handler.initial_green_cube_pos
            if self.include_cube_displacement and initial_green_cube_pos is not None:
                green_disp = green_cube_pos - initial_green_cube_pos
                if self.obs_normalize_enabled:
                    green_disp = torch.clamp(green_disp, -self.relative_pos_clip, self.relative_pos_clip) / self.relative_pos_clip
                obs["green_cube_displacement"] = green_disp

        # 2. End-Effector (TCP) State (Right Arm) - using agent.tcp_pos (fingertip midpoint)
        agent = self.right_arm
        
        tcp_pos = agent.tcp_pos  # Uses new fingertip-based calculation
        tcp_pose = agent.tcp_pose
        
        # 2a. is_grasped observation (optional, config controlled)
        if self.include_is_grasped:
            is_grasped = agent.is_grasping(
                self.red_cube, 
                min_force=self.grasp_min_force, 
                max_angle=self.grasp_max_angle
            )
            # Convert bool to -1/1 for better neural network input
            obs["is_grasped"] = is_grasped.float() * 2 - 1
        
        # 2b. TCP orientation (quaternion, optional)
        if self.include_tcp_orientation:
            obs["tcp_orientation"] = tcp_pose.q
        
        # 3. Relative State (Critical for RL efficiency)
        # TCP to Red Cube - apply clip + normalize
        tcp_to_red = red_cube_pos - tcp_pos
        if self.obs_normalize_enabled:
            clip_val = self.relative_pos_clip
            tcp_to_red = torch.clamp(tcp_to_red, -clip_val, clip_val) / clip_val
        obs["tcp_to_red_pos"] = tcp_to_red
        
        # Red to Green (for stack task)
        if self.task == "stack" and green_cube_pos is not None:
            red_to_green = green_cube_pos - red_cube_pos
            if self.obs_normalize_enabled:
                clip_val = self.relative_pos_clip
                red_to_green = torch.clamp(red_to_green, -clip_val, clip_val) / clip_val
            obs["red_to_green_pos"] = red_to_green
        
        # 4. Absolute positions (controlled by include_abs_pos: list, bool, or false)
        # Convert to list for uniform handling
        abs_pos_list = self.include_abs_pos
        if abs_pos_list is True:
            abs_pos_list = ["tcp_pos", "red_cube_pos", "green_cube_pos"]
        elif abs_pos_list is False or abs_pos_list is None:
            abs_pos_list = []
        
        if "tcp_pos" in abs_pos_list:
            if self.obs_normalize_enabled:
                obs["tcp_pos"] = normalize_pos(tcp_pos, self.tcp_pos_norm)
            else:
                obs["tcp_pos"] = tcp_pos
                
        if "red_cube_pos" in abs_pos_list:
            if self.obs_normalize_enabled:
                obs["red_cube_pos"] = normalize_pos(red_cube_pos, self.red_cube_pos_norm)
            else:
                obs["red_cube_pos"] = red_cube_pos
                
        if "green_cube_pos" in abs_pos_list and green_cube_pos is not None:
            if self.obs_normalize_enabled:
                obs["green_cube_pos"] = normalize_pos(green_cube_pos, self.green_cube_pos_norm)
            else:
                obs["green_cube_pos"] = green_cube_pos
            
        return obs


    def _load_lighting(self, options: dict):
        """Load lighting with optional randomization."""
        for i, scene in enumerate(self.scene.sub_scenes):
            if self.domain_randomization:
                # Randomize ambient light intensity
                ambient = np.random.uniform(0.2, 0.5, size=3).tolist()
            else:
                ambient = [0.3, 0.3, 0.3]
            scene.ambient_light = ambient
            scene.add_directional_light([0.5, 0, -1], [3.0, 3.0, 3.0], shadow=True, shadow_scale=5, shadow_map_size=2048)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _load_scene(self, options: dict):
        # Ground
        for scene in self.scene.sub_scenes:
            # Physical ground removed as per user request (objects exceeding table should fail)
            # scene.add_ground(0, material=ground_material, render=False)
            
            # Add visual ground plane (randomized color)
            if self.domain_randomization:
                # Random earth-tone/dark colors
                color = np.random.uniform(0.1, 0.4, size=3).tolist() + [1]
            else:
                color = [0.1, 0.1, 0.1, 1] # Dark gray/blackish
            
            builder = scene.create_actor_builder()
            builder.add_box_visual(half_size=[2.0, 2.0, 0.1], material=sapien.render.RenderMaterial(base_color=color))
            builder.initial_pose = sapien.Pose(p=[0, 0, -0.11], q=[1, 0, 0, 0]) # Below table (-0.01)
            builder.build_static(name="visual_ground")
        
        # Compute grid layout for this reconfiguration
        self._compute_grids()
        
        # Table surface with optional color randomization
        self._build_table()
        self._build_tape_lines()
        self._build_debug_markers()
        
        # Load task objects
        self._load_objects(options)

    def _build_debug_markers(self):
        """Build debug markers for coordinate system visualization.
        Delegates to scene_builder module.
        """
        scene_builder.build_debug_markers(self)

    def _build_table(self):
        """Build table with optional visual randomization.
        Delegates to scene_builder module.
        """
        scene_builder.build_table(self)

    def _compute_grids(self):
        """Compute grid coordinates and boundaries.
        Delegates to scene_builder module.
        """
        scene_builder.compute_grids(self)


    def _build_tape_lines(self):
        """Build black tape lines using computed grid points.
        Delegates to scene_builder module.
        """
        scene_builder.build_tape_lines(self)



    def _load_agent(self, options: dict):
        # Rotate robots to face +Y (90 degrees around Z axis)
        # q = [cos(pi/4), 0, 0, sin(pi/4)]
        rotation = [0.7071068, 0, 0, 0.7071068]
        
        # Base y-position, randomized if domain_randomization is enabled
        if self.domain_randomization:
            left_y = np.random.uniform(0.01, 0.03)
            right_y = np.random.uniform(0.01, 0.03)
        else:
            left_y = 0.02
            right_y = 0.02
        
        agent_poses = [
            sapien.Pose(p=[0.119, left_y, 0], q=rotation),  # Left Robot
            sapien.Pose(p=[0.481, right_y, 0], q=rotation)   # Right Robot
        ]
        
        # Enable per-env building for joint randomization
        if self.domain_randomization:
            super()._load_agent(options, agent_poses, build_separate=True)
            self._randomize_robot_properties()
        else:
            super()._load_agent(options, agent_poses)
        
        # Defensive check: Ensure agent is properly configured as MultiAgent
        # SingleArmWrapper and related code assume dual-arm setup
        from mani_skill.agents.multi_agent import MultiAgent
        if not isinstance(self.agent, MultiAgent):
            raise RuntimeError(
                f"Track1Env requires dual-arm setup but agent is {type(self.agent).__name__}. "
                f"Expected MultiAgent. Check robot_uids configuration: {self.robot_uids}"
            )
        if len(self.agent.agents) != 2:
            raise RuntimeError(
                f"Track1Env requires exactly 2 robots but got {len(self.agent.agents)}. "
                f"Check robot_uids configuration: {self.robot_uids}"
            )
        
        # Create agents dict for easy access by name
        self._agents_dict = {
            "left": self.agent.agents[0],    # so101-0
            "right": self.agent.agents[1],   # so101-1
        }
    
    @property
    def agents_dict(self):
        """Get agents as a dict for intuitive access: agents_dict['left'] or agents_dict['right']."""
        return self._agents_dict
    
    @property
    def right_arm(self):
        """Shortcut to access the right arm agent."""
        return self._agents_dict["right"]
    
    @property
    def left_arm(self):
        """Shortcut to access the left arm agent."""
        return self._agents_dict["left"]

    def _randomize_robot_properties(self):
        """Randomize robot joint friction and damping for domain randomization."""
        for agent in self.agent.agents:
            for joint in agent.robot.joints:
                for i, obj in enumerate(joint._objs):
                    # Randomize joint properties
                    stiffness = np.random.uniform(800, 1200)
                    damping = np.random.uniform(80, 120)
                    obj.set_drive_properties(stiffness=stiffness, damping=damping, force_limit=100)
                    obj.set_friction(friction=np.random.uniform(0.3, 0.7))

    def _load_objects(self, options: dict):
        """Load task-specific objects."""
        # Determine if cubes should be static based on task
        is_static = (self.task == "static_grasp")
        
        # Red cube is always 3cm
        self.red_cube = self._build_cube(
            name="red_cube",
            half_size=0.015,
            base_color=[1, 0, 0, 1],
            default_pos=[0.497, 0.26, 0.015+ self.space_gap],
            is_static=is_static
        )
        
        # Green cube: only load for stack and sort tasks
        if self.task == "lift" or self.task == "static_grasp":
            # Lift and static_grasp tasks: no green cube needed
            self.green_cube = None
        elif self.task == "sort":
            # Sort task: green cube is 1cm
            self.green_cube = self._build_cube(
                name="green_cube",
                half_size=0.005,
                base_color=[0, 1, 0, 1],
                default_pos=[0.497, 0.30, 0.005+ self.space_gap]
            )
        else:  # stack
            # Stack task: green cube is 3cm
            self.green_cube = self._build_cube(
                name="green_cube",
                half_size=0.015,
                base_color=[0, 1, 0, 1],
                default_pos=[0.497, 0.30, 0.015+ self.space_gap]
            )

    def _build_cube(self, name: str, half_size: float, base_color: list, default_pos: list, is_static: bool = False) -> Actor:
        """Build a cube with optional domain randomization.
        Delegates to scene_builder.
        """
        return scene_builder.build_cube(self, name, half_size, base_color, default_pos, is_static)

    def _apply_cube_physics(self, cube: Actor):
        """Apply physics properties to cube.
        Delegates to scene_builder module.
        """
        scene_builder.apply_cube_physics(self, cube)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with randomized object positions and robot poses."""
        super()._initialize_episode(env_idx, options) # Handle robots
        self.task_handler.initialize_episode(env_idx, options)

    def _initialize_robot_poses(self, batch_size: int, env_idx: torch.Tensor):
        """Initialize robot poses with zero + small noise.
        
        Uses zero qpos as base and adds small Gaussian noise to introduce
        variation in initial configurations while keeping the pose valid.
        """
        # Noise standard deviation per joint (radians)
        # Smaller noise for arm joints (0-4), larger for gripper (5)
        qpos_noise_std = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.2], device=self.device)
        
        for agent in self.agent.agents:
            # Zero base pose
            zero_qpos = torch.zeros((batch_size, 6), device=self.device)
            
            # Add Gaussian noise
            noise = torch.randn_like(zero_qpos) * qpos_noise_std
            qpos = zero_qpos + noise
            
            # Clamp to safe joint limits (approximate, from URDF)
            # More conservative than actual limits to ensure valid poses
            qpos_lower = torch.tensor([-0.1, -2.0, -1.5, -1.5, -1.5, -1.0], device=self.device)
            qpos_upper = torch.tensor([1.5, 2.0, 1.5, 1.5, 1.5, 0.5], device=self.device)
            qpos = torch.clamp(qpos, qpos_lower, qpos_upper)
            
            agent.robot.set_qpos(qpos)

    def _random_grid_position(self, batch_size: int, grid: dict, z: float) -> torch.Tensor:
        """Generate random positions within a grid."""
        x = torch.rand(batch_size, device=self.device) * (grid["x_max"] - grid["x_min"]) + grid["x_min"]
        y = torch.rand(batch_size, device=self.device) * (grid["y_max"] - grid["y_min"]) + grid["y_min"]
        z_tensor = torch.full((batch_size,), z, device=self.device)
        return torch.stack([x, y, z_tensor], dim=1), randomization.random_quaternions(batch_size, lock_x=True, lock_y=True, lock_z=False)

    def evaluate(self):
        """Evaluate success/fail based on task (delegated to handler)."""
        return self.task_handler.evaluate()
    # ==================== Dense Reward Functions ====================
    
    def compute_dense_reward(self, obs, action, info):
        """Compute dense reward (delegated to handler)."""
        return self.task_handler.compute_dense_reward(info, action)

    def compute_normalized_dense_reward(self, obs, action, info):
        """Compute normalized dense reward."""
        reward = self.compute_dense_reward(obs, action, info)
        # Apply reward_scale if available in config, else 1.0
        scale = getattr(self.track1_cfg.reward, "reward_scale", 1.0)
        return reward * scale

