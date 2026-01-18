import logging
import numpy as np
import copy
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.sensors.camera import CameraConfig

logger = logging.getLogger(__name__)

# Cache to avoid re-registering the same configuration multiple times
_CONFIGURED_SO101_CLASSES = {}

@register_agent()
class SO101(BaseAgent):
    uid = "so101"
    # Use absolute path to the asset provided in the workspace
    urdf_path = "/home/admin/Desktop/eai-final-project/eai-2025-fall-final-project-reference-scripts/assets/SO101/so101.urdf"

    # Agent instance naming convention (used by ManiSkill)
    # When using robot_uids=("so101", "so101"), instances are named:
    LEFT_AGENT_SUFFIX = "-0"   # First in tuple
    RIGHT_AGENT_SUFFIX = "-1"  # Second in tuple

    @classmethod
    def get_agent_key(cls, side: str) -> str:
        """Get the agent key for left/right arm.
        
        Args:
            side: "left" or "right"
        Returns:
            e.g., "so101-0" for left, "so101-1" for right
        """
        suffix = cls.LEFT_AGENT_SUFFIX if side == "left" else cls.RIGHT_AGENT_SUFFIX
        return f"{cls.uid}{suffix}"


    # Per-joint action bounds (radians) - can be overridden before environment creation
    # Default values based on trajectory analysis (99th percentile + buffer)
    # Keys must match joint names: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    action_bounds_single_arm = {
        "shoulder_pan": 0.044,    # ~2.5 degrees
        "shoulder_lift": 0.087,   # ~5.0 degrees
        "elbow_flex": 0.070,      # ~4.0 degrees
        "wrist_flex": 0.044,      # ~2.5 degrees
        "wrist_roll": 0.026,      # ~1.5 degrees
        "gripper": 0.070,         # ~4.0 degrees
    }
    
    action_bounds_dual_arm = {
        "shoulder_pan": 0.070,    # ~4.0 degrees
        "shoulder_lift": 0.122,   # ~7.0 degrees
        "elbow_flex": 0.122,      # ~7.0 degrees
        "wrist_flex": 0.070,      # ~4.0 degrees
        "wrist_roll": 0.044,      # ~2.5 degrees
        "gripper": 0.113,         # ~6.5 degrees
    }
    
    # Active mode: 'single' or 'dual' - set by environment before agent creation
    active_mode = "single"
    
    # Default urdf_config - can be overridden by environment before agent creation
    # This will be modified by the environment based on config values
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2, dynamic_friction=2, restitution=0.0)
        ),
        link=dict(
            gripper_link=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            moving_jaw_so101_v1_link=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )
    
    @classmethod
    def derive_urdf_config(cls, cfg):
        """Derive urdf_config from config without modifying class state."""
        urdf_config = copy.deepcopy(cls.urdf_config)
        if hasattr(cfg, "env") and "gripper_physics" in cfg.env:
            gripper_cfg = cfg.env.gripper_physics
            urdf_config["_materials"]["gripper"] = dict(
                static_friction=gripper_cfg.get("static_friction", 2.0),
                dynamic_friction=gripper_cfg.get("dynamic_friction", 2.0),
                restitution=gripper_cfg.get("restitution", 0.0),
            )
            urdf_config["link"]["gripper_link"] = dict(
                material="gripper",
                patch_radius=gripper_cfg.get("patch_radius", 0.1),
                min_patch_radius=gripper_cfg.get("min_patch_radius", 0.1),
            )
            urdf_config["link"]["moving_jaw_so101_v1_link"] = dict(
                material="gripper",
                patch_radius=gripper_cfg.get("patch_radius", 0.1),
                min_patch_radius=gripper_cfg.get("min_patch_radius", 0.1),
            )
        return urdf_config



    @classmethod
    def create_configured_class(cls, task_name: str, mode: str, action_bounds: dict = None, urdf_path: str = None, cfg: any = None):
        """Create a new class with specific configuration with a stable UID based on the task name.
        
        Using task-based UIDs (e.g., 'so101_lift') ensures absolute reproducibility and 
        semantic clarity in logs/checkpoints.
        """
        # 1. Generate a stable UID based on the task name
        new_uid = f"{cls.uid}_{task_name}"

        # 2. Check if we already registered this class
        if new_uid in _CONFIGURED_SO101_CLASSES:
            return _CONFIGURED_SO101_CLASSES[new_uid]

        # 3. Derive configurations
        derived_urdf_config = cls.derive_urdf_config(cfg) if cfg else copy.deepcopy(cls.urdf_config)
        
        # 4. Create and register new class
        class ConfiguredSO101(cls):
            uid = new_uid
            active_mode = mode
            if action_bounds:
                if mode == "dual":
                    action_bounds_dual_arm = action_bounds
                else:
                    action_bounds_single_arm = action_bounds
        
        ConfiguredSO101.urdf_path = urdf_path or cls.urdf_path
        ConfiguredSO101.urdf_config = derived_urdf_config
        
        # Register the new class so ManiSkill can find it
        register_agent(new_uid)(ConfiguredSO101)
        _CONFIGURED_SO101_CLASSES[new_uid] = ConfiguredSO101
        
        logger.info(f"Registered stable SO101: {new_uid}")
        return ConfiguredSO101

    arm_joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]
    gripper_joint_names = [
        "gripper",
    ]
    JOINT_NAMES = arm_joint_names + gripper_joint_names
    
    @property
    def _active_action_bounds(self):
        """Get active action bounds based on mode."""
        if self.active_mode == "dual":
            return self.action_bounds_dual_arm
        return self.action_bounds_single_arm

    @property
    def _sensor_configs(self):
        logger.debug("SO101._sensor_configs called")
        return [
            CameraConfig(
                "wrist_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=640,
                height=480,
                fov=np.deg2rad(50),
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"]
            )
        ]

    @property
    def _controller_configs(self):
        # Get joint names
        all_joint_names = [joint.name for joint in self.robot.active_joints]
        arm_joint_names = all_joint_names[:5]  # First 5 joints are arm
        gripper_joint_names = all_joint_names[5:]  # Last joint is gripper
        
        # Get per-joint action bounds based on active mode
        bounds = self._active_action_bounds
        joint_order = self.JOINT_NAMES
        lower = [-bounds[j] for j in joint_order]
        upper = [bounds[j] for j in joint_order]
        
        # Absolute position control (no normalization)
        pd_joint_pos = PDJointPosControllerConfig(
            all_joint_names,
            lower=None,
            upper=None,
            stiffness=[1e3] * 5 + [1e2], # smooth the gripper movement
            damping=[1e2] * 6,
            force_limit=100,
            normalize_action=False,
        )

        # Delta position control from CURRENT position (using configurable bounds)
        pd_joint_delta_pos = PDJointPosControllerConfig(
            all_joint_names,
            lower,
            upper,
            stiffness=[1e3] * 5 + [1e2], # smooth the gripper movement
            damping=[1e2] * 6,
            force_limit=100,
            use_delta=True,
            use_target=False,
            normalize_action=True,
        )

        # Delta position control from TARGET position (recommended for sim2real)
        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        # End effector position control (arm) + joint control (gripper)
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            arm_joint_names,
            pos_lower=-0.02,
            pos_upper=0.02,
            stiffness=[800] * 5,
            damping=[80] * 5,
            force_limit=100,
            ee_link="gripper_frame_link",
            urdf_path=self.urdf_path,
            use_delta=True,
            use_target=False,
            normalize_action=True,
        )
        
        gripper_pd_joint_delta_pos = PDJointPosControllerConfig(
            gripper_joint_names,
            lower=[-0.1],
            upper=[0.1],
            stiffness=[800],
            damping=[80],
            force_limit=100,
            use_delta=True,
            normalize_action=True,
        )
        
        pd_ee_delta_pos = dict(
            arm=arm_pd_ee_delta_pos,
            gripper=gripper_pd_joint_delta_pos,
        )

        controller_configs = dict(
            pd_joint_pos=pd_joint_pos,
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_target_delta_pos=pd_joint_target_delta_pos,
            pd_ee_delta_pos=pd_ee_delta_pos,
        )
        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        # Map gripper links for grasp detection
        self.finger1_link = self.robot.links_map["gripper_link"]  # Fixed jaw
        self.finger2_link = self.robot.links_map["moving_jaw_so101_v1_link"]  # Moving jaw
        
        # Map fingertip auxiliary links for TCP calculation (from so101_new URDF)
        try:
            self.finger1_tip = self.robot.links_map["gripper_link_tip"]
            self.finger2_tip = self.robot.links_map["moving_jaw_so101_v1_link_tip"]
        except KeyError:
            if "gripper_frame_link" in self.robot.links_map:
                # Use gripper_frame_link as valid TCP (it has ~10cm offset from wrist)
                self.finger1_tip = self.robot.links_map["gripper_frame_link"]
                self.finger2_tip = self.robot.links_map["gripper_frame_link"]
            else:
                logger.warning("Fingertip links not found. TCP calculation will fall back to gripper links.")
                self.finger1_tip = self.finger1_link
                self.finger2_tip = self.finger2_link

    @property
    def tcp_pos(self):
        """Compute Tool Center Point as midpoint between the two fingertips."""
        return (self.finger1_tip.pose.p + self.finger2_tip.pose.p) / 2

    @property
    def tcp_pose(self):
        """Get TCP pose (position from fingertips, orientation from fixed jaw)."""
        from mani_skill.utils.structs.pose import Pose
        return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

    def is_grasping(self, object, min_force=0.5, max_angle=110):
        """Check if the robot is grasping an object.
        
        Args:
            object: The object to check if the robot is grasping
            min_force: Minimum contact force in Newtons (default 0.5N)
            max_angle: Maximum angle between contact force and gripper opening direction (default 110°)
        
        Returns:
            Boolean tensor indicating whether each environment is grasping the object.
        """
        import torch
        from mani_skill.utils import common
        
        # Get pairwise contact forces between each finger and the object
        l_contact_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        r_contact_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        
        # Compute force magnitudes
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)
        
        # Gripper opening direction (local Y axis for each finger)
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        
        # Compute angle between contact force and opening direction
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        
        # Grasping if both fingers have sufficient force at correct angle
        lflag = torch.logical_and(lforce >= min_force, torch.rad2deg(langle) <= max_angle)
        rflag = torch.logical_and(rforce >= min_force, torch.rad2deg(rangle) <= max_angle)
        
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold=0.2):
        """Check if the robot arm is static (excluding gripper joint).
        
        Args:
            threshold: Maximum joint velocity to be considered static (default 0.2 rad/s)
        
        Returns:
            Boolean tensor indicating whether each environment's robot is static.
        """
        import torch
        # Exclude gripper joint (last joint)
        qvel = self.robot.get_qvel()[:, :-1]
        return torch.max(torch.abs(qvel), dim=1)[0] <= threshold

