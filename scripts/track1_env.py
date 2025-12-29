import numpy as np
import sapien
import sapien.render
import torch
import torch.nn.functional as tFunc
import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs import Actor
from sapien.physx import PhysxRigidBodyComponent
from sapien.render import RenderBodyComponent
from scripts.so101 import SO101
from scripts.training.config_utils import Track1Config
import mani_skill.envs.utils.randomization as randomization




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
        self.gripper_tip_offset = rw_cfg.gripper_tip_offset
        self.gripper_outward_offset = rw_cfg.gripper_outward_offset
        self.stage_thresholds = rw_cfg.stage_thresholds
        
        self.lift_target = rw_cfg.lift_target
        self.lift_max_height = rw_cfg.lift_max_height
        self.stable_hold_time = rw_cfg.stable_hold_time
        control_freq = getattr(self, 'control_freq', 30)
        self.stable_hold_steps = int(self.stable_hold_time * control_freq)
        
        self.fail_bounds = rw_cfg.fail_bounds
        self.spawn_bounds = rw_cfg.spawn_bounds
        
        self.moving_jaw_tip_offset = rw_cfg.moving_jaw_tip_offset
        self.moving_jaw_outward_offset = rw_cfg.moving_jaw_outward_offset
        self.approach2_threshold = rw_cfg.approach2_threshold
        self.approach2_zero_point = rw_cfg.approach2_zero_point
        
        self.horizontal_displacement_threshold = rw_cfg.horizontal_displacement_threshold
        self.grasp_min_force = rw_cfg.grasp_min_force
        self.grasp_max_angle = rw_cfg.grasp_max_angle
        
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
            urdf_path=getattr(cfg.env, 'robot_urdf', None),
            cfg=cfg # Pass full config to derive gripper physics
        )
        new_uid = ConfiguredSO101.uid
        
        # Ensure super().__init__ uses the configured class UID
        if isinstance(robot_uids, str):
            if robot_uids == "so101": robot_uids = new_uid
        elif isinstance(robot_uids, (list, tuple)):
            robot_uids = tuple(new_uid if uid == "so101" else uid for uid in robot_uids)
        
        self.render_scale = cfg.render_scale
        self.grid_bounds = {}
        self._setup_camera_processing_maps()
        
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
        else:
            raise ValueError(f"Unknown task: {task}")

    def get_obs_structure(self):
        """The unbatched, hierarchical observation space for descriptive logging.
        
        Dynamically derived from the actual observation dictionary to ensure parity.
        """
        from mani_skill.utils import gym_utils
        from mani_skill.utils import common as ms_common
        
        # Call _get_obs_state_dict directly to get the hierarchical dictionary.
        # super().get_obs() would return a flattened tensor in state mode.
        obs_dict = self._get_obs_state_dict({})
        
        # Convert to gym space (unbatched)
        obs_numpy = ms_common.to_numpy(obs_dict)
        return gym_utils.convert_observation_to_space(obs_numpy, unbatched=True)
        


    def _setup_device(self):
        # self.device is guaranteed by BaseEnv
        if self.distortion_grid is not None:
            self.distortion_grid = self.distortion_grid.to(self.device)
        if self.undistortion_grid is not None:
            self.undistortion_grid = self.undistortion_grid.to(self.device)

    def _setup_camera_processing_maps(self):
        """Precompute torch grids for camera distortion/undistortion processing via tFunc.grid_sample.
        
        Pipeline:
        - Source: Rendered at (640×scale) × (480×scale) with scaled intrinsic matrix
        - Distortion: Maps source -> 640×480 distorted output
        - Undistortion (alpha=0): Maps 640×480 distorted -> 640×480 clean pinhole
        """
        import cv2
        
        # Camera intrinsic parameters (from real camera calibration)
        self.mtx_intrinsic = np.array([
            [570.21740069, 0., 327.45975405],
            [0., 570.1797441, 260.83642155],
            [0., 0., 1.]
        ], dtype=np.float64)
        
        self.dist_coeffs = np.array([
            -0.735413911, 0.949258417, 0.000189059234, -0.00200351391, -0.864150312
        ], dtype=np.float64)
        
        # Scale factor for high-res rendering
        
        # Source image size (high-res pinhole render)
        OUT_W, OUT_H = 640, 480
        SRC_W = OUT_W * self.render_scale
        SRC_H = OUT_H * self.render_scale
        if self.camera_mode in ["distorted", "distort-twice"]:
            self.front_render_width = SRC_W
            self.front_render_height = SRC_H

            # Get the undistorted intrinsic matrix using getOptimalNewCameraMatrix with alpha=1
            # This gives us the intrinsic for a pinhole camera that covers all distorted pixels
            new_mtx_alpha1, _ = cv2.getOptimalNewCameraMatrix(
                self.mtx_intrinsic, self.dist_coeffs, (OUT_W, OUT_H), 1.0, (SRC_W, SRC_H)
            )
            
            # Scale the new_mtx to render resolution
            self.render_intrinsic = new_mtx_alpha1.copy()
            
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
                cameraMatrix=self.mtx_intrinsic, 
                distCoeffs=self.dist_coeffs, 
                R=None, 
                P=self.render_intrinsic  # Project to render camera space
            )
            map_xy_render = undistorted_pts.reshape(OUT_H, OUT_W, 2)
            
            # Step 3: Normalize to [-1, 1] for grid_sample
            grid_x = 2.0 * map_xy_render[:, :, 0] / (SRC_W - 1) - 1.0
            grid_y = 2.0 * map_xy_render[:, :, 1] / (SRC_H - 1) - 1.0
            distortion_grid = np.stack((grid_x, grid_y), axis=2).astype(np.float32)
            self.distortion_grid = torch.from_numpy(distortion_grid)# .to(device=self.device)  # (OUT_H, OUT_W, 2)
            
            # ============ Undistortion Grid ============
            # This maps 640x480 distorted -> 640x480 clean pinhole
        if self.camera_mode in ["distort-twice", "direct_pinhole"]:
            # Get new camera matrix with configurable alpha
            # alpha=0: crop black borders, alpha=1: keep all pixels (shrinks image)
            # alpha=0.25 is optimal for full work area visibility
            alpha = getattr(self, 'undistort_alpha', 0.25)
            new_mtx_undist, _ = cv2.getOptimalNewCameraMatrix(
                self.mtx_intrinsic, self.dist_coeffs, (OUT_W, OUT_H), alpha, (OUT_W, OUT_H)
            )
            
            if self.camera_mode == "direct_pinhole":
                self.front_render_width = OUT_W
                self.front_render_height = OUT_H
                self.render_intrinsic = new_mtx_undist.copy()
                return
            # initUndistortRectifyMap gives us the mapping from undistorted -> distorted source
            # We need the inverse for grid_sample
            map1, map2 = cv2.initUndistortRectifyMap(
                self.mtx_intrinsic, self.dist_coeffs, None, new_mtx_undist, (OUT_W, OUT_H), cv2.CV_32FC1
            )
            
            # map1, map2 are (OUT_H, OUT_W) containing x, y source coordinates
            # Normalize to [-1, 1]
            undist_grid_x = 2.0 * map1 / (OUT_W - 1) - 1.0
            undist_grid_y = 2.0 * map2 / (OUT_H - 1) - 1.0
            undistortion_grid = np.stack((undist_grid_x, undist_grid_y), axis=2).astype(np.float32)
            self.undistortion_grid = torch.from_numpy(undistortion_grid).to(device=self.device)  # (OUT_H, OUT_W, 2)

    def _apply_camera_processing(self, obs):
        """Apply camera processing based on camera_mode.
        
        Modes:
        - direct_pinhole: No processing (already rendered with correct params)
        - distorted: Apply distortion to 1920x1440 source -> 640x480 distorted output
        - distort-twice: distorted -> then undistort (alpha=0) -> 640x480 clean
        """
        
        if self.camera_mode == "direct_pinhole":
            return obs  # No processing needed
        
        # Skip if grids not yet initialized (happens during parent __init__ reset)
        if self.distortion_grid is None:
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
        dist_grid = self.distortion_grid.to(device).unsqueeze(0).expand(B, -1, -1, -1)
        
        # Step 1: Apply distortion (1920x1440 -> 640x480)
        distorted = tFunc.grid_sample(img_in, dist_grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        if self.camera_mode == "distorted":
            result = distorted
        elif self.camera_mode == "distort-twice":
            # Step 2: Apply undistortion (640x480 distorted -> 640x480 clean)
            undist_grid = self.undistortion_grid.to(device).unsqueeze(0).expand(B, -1, -1, -1)
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

    def _get_obs_state_dict(self, info: dict):
        """Override to filter left arm obs and apply normalization."""
        obs = super()._get_obs_state_dict(info)
        
        # Apply agent state normalization
        if self.obs_normalize_enabled and "agent" in obs:
            qvel_clip = torch.tensor(self.qvel_clip, device=self.device)
            
            for agent_key in obs["agent"]:
                agent_obs = obs["agent"][agent_key]
                
                # Get raw qpos for potential relative calculation
                raw_qpos = agent_obs.get("qpos", None)
                
                # qpos: divide by π → approximately [-1, 1] for typical joint ranges
                if raw_qpos is not None:
                    agent_obs["qpos"] = raw_qpos / self.qpos_scale
                
                # target_qpos handling based on include_target_qpos setting
                if "controller" in agent_obs and "target_qpos" in agent_obs["controller"]:
                    target_qpos = agent_obs["controller"]["target_qpos"]
                    
                    if self.include_target_qpos == "relative" and raw_qpos is not None:
                        # Replace with tracking error: (target_qpos - qpos) / action_bounds
                        tracking_error = target_qpos - raw_qpos
                        
                        # Normalize by action bounds if available, otherwise fall back to qpos_scale
                        if self.obs_action_bounds is not None:
                            # Dynamically get joint order from robot to avoid scrambling
                            active_joints = self.right_arm.robot.get_active_joints()
                            joint_names = [j.name for j in active_joints]
                            bounds_list = [self.obs_action_bounds.get(j, 0.1) for j in joint_names]
                            bounds = torch.tensor(bounds_list, device=self.device)
                            agent_obs["controller"]["target_qpos"] = tracking_error / bounds
                        else:
                            agent_obs["controller"]["target_qpos"] = tracking_error / self.qpos_scale
                    elif self.include_target_qpos:
                        # Include normalized target_qpos
                        agent_obs["controller"]["target_qpos"] = target_qpos / self.qpos_scale
                    else:
                        # Exclude target_qpos entirely
                        del agent_obs["controller"]["target_qpos"]
                        # Remove empty controller dict
                        if not agent_obs["controller"]:
                            del agent_obs["controller"]
                
                # qvel: clip and normalize
                if "qvel" in agent_obs:
                    qvel = agent_obs["qvel"]
                    qvel_clipped = torch.clamp(qvel, -qvel_clip, qvel_clip)
                    agent_obs["qvel"] = qvel_clipped / qvel_clip
        
        return obs

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
        Red at (0,0), Green at (1,0), Blue at (0,1).
        """
        marker_height = 0.005 # Slightly above table/ground
        radius = 0.02
        
        markers = [
            {"pos": [0, 0, marker_height], "color": [1, 0, 0], "name": "debug_origin_red"},
            {"pos": [1, 0, marker_height], "color": [0, 1, 0], "name": "debug_x1_green"},
            {"pos": [0, 1, marker_height], "color": [0, 0, 1], "name": "debug_y1_blue"},
        ]
        
        for marker in markers:
            builder = self.scene.create_actor_builder()
            builder.add_sphere_visual(radius=radius, material=marker["color"])
            builder.initial_pose = sapien.Pose(p=marker["pos"])
            builder.build_static(name=marker["name"])

    def _build_table(self):
        """Build table with optional visual randomization."""
        if self.domain_randomization:
            tables = []
            for i in range(self.num_envs):
                builder = self.scene.create_actor_builder()
                # Randomize table color slightly
                color = [0.9 + np.random.uniform(-0.05, 0.05)] * 3 + [1]
                builder.add_box_visual(
                    half_size=[0.3, 0.3, 0.01], 
                    material=sapien.render.RenderMaterial(base_color=color)
                )
                # Use friction material from config for table surface
                table_material = sapien.physx.PhysxMaterial(
                    static_friction=self.table_physics["static_friction"],
                    dynamic_friction=self.table_physics["dynamic_friction"],
                    restitution=self.table_physics["restitution"]
                )
                builder.add_box_collision(
                    half_size=[0.3, 0.3, 0.01],
                    material=table_material
                )
                builder.initial_pose = sapien.Pose(p=[0.3, 0.3, -0.01])
                builder.set_scene_idxs([i])
                table = builder.build_static(name=f"table_{i}")
                self.scene.remove_from_state_dict_registry(table)
                tables.append(table)
            self.table = Actor.merge(tables, name="table")
            self.scene.add_to_state_dict_registry(self.table)
        else:
            builder = self.scene.create_actor_builder()
            # Use friction material from config for table surface
            table_material = sapien.physx.PhysxMaterial(
                static_friction=self.table_physics["static_friction"],
                dynamic_friction=self.table_physics["dynamic_friction"],
                restitution=self.table_physics["restitution"]
            )
            builder.add_box_visual(half_size=[0.3, 0.3, 0.01], material=[0.9, 0.9, 0.9])
            builder.add_box_collision(
                half_size=[0.3, 0.3, 0.01],
                material=table_material
            )
            builder.initial_pose = sapien.Pose(p=[0.3, 0.3, -0.01])
            self.table = builder.build_static(name="table")

    def _compute_grids(self):
        """Compute grid coordinates and boundaries with optional randomization."""
        tape_half_width = 0.009
        
        # Base values (Human specified)
        x_1 = 0.204
        x_4 = 0.6
        y_1 = 0.15
        upper_height = 0.164
        
        # Add randomization if enabled
        # if self.domain_randomization:
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
        
        # Store for _build_tape_lines
        self.grid_points = {"x": x, "y": y, "tape_half_width": tape_half_width}
        
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
        
        self.grid_bounds["left"] = {
            "x_min": x[0] + 2*w, "x_max": x[1],
            "y_min": y[1] + 2*w, "y_max": y[2]
        }
        
        self.grid_bounds["mid"] = {
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
        self.grid_bounds["mid"] = {
            "x_min": x[1] + 2*w, "x_max": x[2],
            "y_min": y[1] + 2*w, "y_max": y[2]
        }
        
        # Right Grid: between col3 and col4 ? 
        # OR col3 and col5?
        # x[3] = x[4] - 0.204 + 0.166.  = 0.562.  x[4]=0.6. Diff = 0.038. Too small for Right grid.
        # x[2] = x[4] - 0.204 - 2w = 0.378.
        # Gap between x[2] and x[3] = 0.562 - 0.378 = 0.184. This looks like the Right Grid!
        
        # So Right Grid is between col3(x[2]) and col4(x[3]).
        self.grid_bounds["right"] = {
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
        self.grid_bounds["bottom"] = {
            "x_min": x[1] + 2*w, "x_max": x[2],
            "y_min": y[0], "y_max": y[1]
        }


    def _build_tape_lines(self):
        """Build black tape lines using computed grid points."""
        tape_material = [0, 0, 0]
        tape_height = 0.001
        
        # Retrieve computed params
        x = self.grid_points["x"]
        y = self.grid_points["y"]
        tape_half_width = self.grid_points["tape_half_width"]
        
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
            builder = self.scene.create_actor_builder()
            
            # Apply independent randomization if enabled
            pos = list(spec["pos"])
            half_size = list(spec["half_size"])
            rotation = [1, 0, 0, 0] # Identity quaternion
            
            if self.domain_randomization:
                # 1. Position Noise (x, y)
                pos_noise = np.random.uniform(-0.005, 0.005, size=2) # +/- 5mm
                pos[0] += pos_noise[0]
                pos[1] += pos_noise[1]
                
                # 2. Size Noise (length aka half_size[0] mostly, or width)
                size_noise = np.random.uniform(-0.002, 0.002) # +/- 2mm
                # Don't change thickness (z), maybe slight width/length change
                half_size[0] += size_noise 
                
                # 3. Rotation Noise (Yaw)
                # Small rotation around Z axis
                yaw_noise = np.deg2rad(np.random.uniform(-2, 2)) # +/- 2 degrees
                import transforms3d
                rotation = transforms3d.quaternions.axangle2quat([0, 0, 1], yaw_noise)
                # Transforms3d returns [w, x, y, z], Sapien expects [w, x, y, z] match? 
                # Sapien Pose takes q=[w, x, y, z] or [x, y, z, w]?
                # Sapien uses [w, x, y, z] usually. Let's verify or use Sapien's Rotation.
                # Actually sapien.Pose q is [w, x, y, z].
                # Let's use simple randomization without external lib if possible or check imports.
                # simpler:
                q_z = np.sin(yaw_noise / 2)
                q_w = np.cos(yaw_noise / 2)
                rotation = [q_w, 0, 0, q_z]

            builder.add_box_visual(half_size=half_size, material=tape_material)
            builder.initial_pose = sapien.Pose(p=pos, q=rotation)
            builder.build_static(name=spec["name"])



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
        # Red cube is always 3cm
        self.red_cube = self._build_cube(
            name="red_cube",
            half_size=0.015,
            base_color=[1, 0, 0, 1],
            default_pos=[0.497, 0.26, 0.015+ self.space_gap]
        )
        
        # Green cube: only load for stack and sort tasks
        if self.task == "lift":
            # Lift task: no green cube needed
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

    def _build_cube(self, name: str, half_size: float, base_color: list, default_pos: list) -> Actor:
        """Build a cube with optional domain randomization."""
        if self.domain_randomization:
            cubes = []
            for i in range(self.num_envs):
                builder = self.scene.create_actor_builder()
                
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
                cube = builder.build(name=f"{name}_{i}")
                self.scene.remove_from_state_dict_registry(cube)
                cubes.append(cube)
            
            merged = Actor.merge(cubes, name=name)
            self.scene.add_to_state_dict_registry(merged)
            
            # Apply physical properties
            self._apply_cube_physics(merged)
            return merged
        else:
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[half_size] * 3)
            builder.add_box_visual(half_size=[half_size] * 3, material=base_color[:3])
            builder.initial_pose = sapien.Pose(p=default_pos)
            cube = builder.build(name=name)
            
            # Apply physical properties even in non-randomized mode
            self._apply_cube_physics(cube)
            return cube

    def _apply_cube_physics(self, cube: Actor):
        """Apply physics properties to cube (optionally randomized)."""
        p = self.cube_physics
        for i, obj in enumerate(cube._objs):
            rigid_body: PhysxRigidBodyComponent = obj.find_component_by_type(PhysxRigidBodyComponent)
            if rigid_body is not None:
                # Apply mass
                if self.domain_randomization:
                    # Randomize mass around config value (+/- 50%)
                    rigid_body.mass = p["mass"] * np.random.uniform(0.5, 1.5)
                else:
                    rigid_body.mass = p["mass"]
                
                # Apply friction and restitution
                for shape in rigid_body.collision_shapes:
                    if self.domain_randomization:
                        # Randomize friction around config values (+/- 30%)
                        shape.physical_material.static_friction = p["static_friction"] * np.random.uniform(0.7, 1.3)
                        shape.physical_material.dynamic_friction = p["dynamic_friction"] * np.random.uniform(0.7, 1.3)
                    else:
                        shape.physical_material.static_friction = p["static_friction"]
                        shape.physical_material.dynamic_friction = p["dynamic_friction"]
                    
                    shape.physical_material.restitution = p.get("restitution", 0.0)

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


    def _get_gripper_pos(self):
        """Get the gripper reference position for the right arm.
        
        Applies two offsets to gripper_frame_link (fixed jaw tip):
        1. gripper_tip_offset: back along jaw (towards gripper_link)
        2. gripper_outward_offset: perpendicular, towards the moving jaw (where cube should be)
        
        Returns position where cube center should be when properly grasped.
        """
        # Right arm is agents[1] (so101-1, at X=0.481)
        right_agent = self.agent.agents[1]
        gripper_link = right_agent.robot.links_map.get("gripper_link")
        gripper_frame = right_agent.robot.links_map.get("gripper_frame_link")
        moving_jaw = right_agent.robot.links_map.get("moving_jaw_so101_v1_link")
        
        if gripper_frame is None:
            # Fallback
            if gripper_link is not None:
                return gripper_link.pose.p
            return right_agent.robot.links[-1].pose.p
        
        # Start from jaw tip
        ref_pos = gripper_frame.pose.p.clone()
        
        # Apply tip offset (back along jaw direction)
        if self.gripper_tip_offset != 0 and gripper_link is not None:
            jaw_direction = gripper_frame.pose.p - gripper_link.pose.p
            jaw_length = torch.norm(jaw_direction, dim=1, keepdim=True)
            jaw_unit = jaw_direction / (jaw_length + 1e-6)
            ref_pos = ref_pos - jaw_unit * self.gripper_tip_offset
        
        # Apply outward offset (towards moving jaw, perpendicular to fixed jaw)
        if self.gripper_outward_offset != 0 and moving_jaw is not None:
            outward_direction = moving_jaw.pose.p - gripper_frame.pose.p
            outward_length = torch.norm(outward_direction, dim=1, keepdim=True)
            outward_unit = outward_direction / (outward_length + 1e-6)
            ref_pos = ref_pos + outward_unit * self.gripper_outward_offset
        
        return ref_pos

    def _get_moving_jaw_pos(self):
        """Get the moving jaw reference position for the right arm.
        
        Uses calibrated local direction: (-0.2, -1, 0.23) normalized, scaled by 1.8
        Then applies offsets:
        - moving_jaw_tip_offset: back along the jaw direction
        - moving_jaw_outward_offset: along local -X towards cube center
        
        Returns position where cube center should be when properly grasped.
        """
        right_agent = self.agent.agents[1]
        moving_jaw = right_agent.robot.links_map.get("moving_jaw_so101_v1_link")
        
        if moving_jaw is None:
            # Fallback to fixed jaw
            return self._get_gripper_pos()
        
        # Get moving jaw base position and orientation
        moving_jaw_base = moving_jaw.pose.p  # [num_envs, 3]
        moving_jaw_quat = moving_jaw.pose.q  # [num_envs, 4] - SAPIEN uses [w, x, y, z]
        
        # Calibrated local direction to jaw tip: (-0.2, -1, 0.23) normalized
        import torch
        x_comp, y_comp, z_comp = -0.2, -1.0, 0.23
        local_forward = torch.tensor([x_comp, y_comp, z_comp], device=self.device, dtype=torch.float32)
        local_forward = local_forward / torch.norm(local_forward)
        
        # Convert quaternion to rotation matrix and apply to local_forward
        # SAPIEN quaternion: [w, x, y, z]
        w, x, y, z = moving_jaw_quat[:, 0], moving_jaw_quat[:, 1], moving_jaw_quat[:, 2], moving_jaw_quat[:, 3]
        
        # Rotation matrix from quaternion
        R00 = 1 - 2*(y**2 + z**2)
        R01 = 2*(x*y - w*z)
        R02 = 2*(x*z + w*y)
        R10 = 2*(x*y + w*z)
        R11 = 1 - 2*(x**2 + z**2)
        R12 = 2*(y*z - w*x)
        R20 = 2*(x*z - w*y)
        R21 = 2*(y*z + w*x)
        R22 = 1 - 2*(x**2 + y**2)
        
        # Apply rotation to local_forward: world_dir = R @ local_forward
        jaw_direction = torch.stack([
            R00 * local_forward[0] + R01 * local_forward[1] + R02 * local_forward[2],
            R10 * local_forward[0] + R11 * local_forward[1] + R12 * local_forward[2],
            R20 * local_forward[0] + R21 * local_forward[1] + R22 * local_forward[2],
        ], dim=1)  # [num_envs, 3]
        
        # Scale to reach jaw tip (calibrated: 1.8 * 0.045 = 0.081)
        scale = 1.8
        tip_dist = 0.045 * scale
        ref_pos = moving_jaw_base + jaw_direction * tip_dist
        
        # Apply tip offset (back along jaw direction)
        if self.moving_jaw_tip_offset != 0:
            ref_pos = ref_pos - jaw_direction * self.moving_jaw_tip_offset
        
        # Apply outward offset (along local -X, towards cube center)
        if self.moving_jaw_outward_offset != 0:
            local_minus_x = torch.tensor([-1.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
            outward_dir = torch.stack([
                R00 * local_minus_x[0] + R01 * local_minus_x[1] + R02 * local_minus_x[2],
                R10 * local_minus_x[0] + R11 * local_minus_x[1] + R12 * local_minus_x[2],
                R20 * local_minus_x[0] + R21 * local_minus_x[1] + R22 * local_minus_x[2],
            ], dim=1)
            ref_pos = ref_pos + outward_dir * self.moving_jaw_outward_offset
        
        return ref_pos
