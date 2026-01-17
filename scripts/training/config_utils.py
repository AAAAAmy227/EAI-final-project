import copy
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Dict, Any, Optional, List, Union
import numpy as np
import copy
from omegaconf import DictConfig, OmegaConf


def _to_plain(obj):
    """Recursively convert dataclasses and containers to JSON-serializable plain types."""
    if is_dataclass(obj):
        return _to_plain(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_plain(v) for v in obj)
    return obj

@dataclass
class PhysicsConfig:
    static_friction: float = 0.6
    dynamic_friction: float = 0.6
    restitution: float = 0.0
    mass: float = 0.027

@dataclass
class ObsNormalizationConfig:
    enabled: bool = False
    qpos_scale: float = np.pi
    qvel_clip: List[float] = field(default_factory=lambda: [1.0, 2.5, 2.0, 1.0, 0.6, 1.5])
    relative_pos_clip: float = 0.5
    include_abs_pos: bool = True
    include_target_qpos: Union[bool, str] = True
    action_bounds: Optional[Dict[str, float]] = None
    tcp_pos: Dict[str, List[float]] = field(default_factory=lambda: {"mean": [0.3, 0.3, 0.2], "std": [0.1, 0.1, 0.1]})
    red_cube_pos: Dict[str, List[float]] = field(default_factory=lambda: {"mean": [0.3, 0.3, 0.2], "std": [0.1, 0.1, 0.1]})
    green_cube_pos: Dict[str, List[float]] = field(default_factory=lambda: {"mean": [0.3, 0.3, 0.2], "std": [0.1, 0.1, 0.1]})
    include_is_grasped: bool = False
    include_tcp_orientation: bool = False
    include_cube_displacement: bool = False

@dataclass
class AdaptiveWeightConfig:
    enabled: bool = False
    alpha: float = 0.5
    eps: float = 0.01
    max_weight: float = 1000.0
    tau: float = 0.01

@dataclass
class RewardConfig:
    reward_type: str = "staged"
    weights: Dict[str, float] = field(default_factory=lambda: {
        "approach": 1.0,
        "horizontal_displacement": 0.0,
        "lift": 5.0,
        "hold_progress": 0.0,
        "grasp_hold": 0.0,
        "success": 10.0,
        "fail": 0.0,
        "approach2": 0.0,
        "action_rate": 0.0,
    })
    grasp_hold_max_steps: int = 30
    approach_curve: str = "linear"
    approach_threshold: float = 0.01
    approach_zero_point: float = 0.20
    approach_tanh_scale: float = 0.05
    approach_scale: float = 5.0
    approach_mode: str = "dual_point"
    gripper_tip_offset: float = 0.015
    gripper_outward_offset: float = 0.015
    stage_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "approach": 0.05,
        "reach": 0.05,
        "grasp": 0.03,
        "lift": 0.05,
    })
    lift_target: float = 0.05
    lift_max_height: Optional[float] = None
    stable_hold_time: float = 0.0
    fail_bounds: Optional[Dict[str, float]] = None
    spawn_bounds: Optional[Dict[str, float]] = None
    moving_jaw_tip_offset: float = 0.015
    moving_jaw_outward_offset: float = 0.01
    approach2_threshold: float = 0.01
    approach2_zero_point: float = 0.20
    horizontal_displacement_threshold: float = 0.0
    grasp_min_force: float = 0.5
    grasp_max_angle: float = 110
    stack_height_target: float = 0.03
    stack_height_tolerance: float = 0.005
    stack_xy_tolerance: float = 0.02
    stack_align_tanh_scale: float = 0.03
    green_z_range: List[float] = field(default_factory=lambda: [0.010, 0.020])
    adaptive_grasp_weight: AdaptiveWeightConfig = field(default_factory=AdaptiveWeightConfig)
    gate_lift_with_grasp: bool = False
    adaptive_lift_weight: AdaptiveWeightConfig = field(default_factory=AdaptiveWeightConfig)
    adaptive_success_weight: AdaptiveWeightConfig = field(default_factory=AdaptiveWeightConfig)
    reward_scale: float = 1.0

@dataclass
class Track1Config:
    task: str = "lift"
    domain_randomization: bool = True
    camera_mode: str = "direct_pinhole"
    render_scale: int = 3
    cube_physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    table_physics: PhysicsConfig = field(default_factory=lambda: PhysicsConfig(static_friction=2.0, dynamic_friction=2.0))
    action_bounds: Optional[Dict[str, Any]] = None
    camera_extrinsic: Optional[List[List[float]]] = None
    undistort_alpha: float = 0.25
    obs: ObsNormalizationConfig = field(default_factory=ObsNormalizationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    robot_urdf: Optional[str] = None
    raw_cfg: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_hydra(cls, cfg: Union[DictConfig, Dict[str, Any]]) -> "Track1Config":
        if isinstance(cfg, DictConfig):
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        else:
            cfg_dict = copy.deepcopy(cfg)

        # Work on a deep copy so callers' dictionaries are not mutated (affects env.spec.kwargs)
        cfg_dict = copy.deepcopy(cfg_dict)

        # Initialize with defaults
        config = cls()
        
        env_cfg = cfg_dict.get("env", {})
        # Debug print to trace task identification
        print(f"[Track1Config.from_hydra] Parsing task. env_cfg.get('task')={env_cfg.get('task')}, config.task (default)={config.task}")
        config.task = env_cfg.get("task", config.task)
        print(f"[Track1Config.from_hydra] Resulting config.task={config.task}")
        config.domain_randomization = env_cfg.get("domain_randomization", config.domain_randomization)
        config.camera_mode = env_cfg.get("camera_mode", config.camera_mode)
        config.render_scale = env_cfg.get("render_scale", config.render_scale)
        config.undistort_alpha = env_cfg.get("camera", {}).get("undistort_alpha", config.undistort_alpha)
        
        if "extrinsic" in env_cfg.get("camera", {}):
            config.camera_extrinsic = env_cfg["camera"]["extrinsic"]
        
        if "cube_physics" in env_cfg:
            config.cube_physics = PhysicsConfig(**env_cfg["cube_physics"])
        else:
            # Default mass for 3cm cube if not specified
            config.cube_physics.mass = 0.027
            
        if "table_physics" in env_cfg:
            config.table_physics = PhysicsConfig(**env_cfg["table_physics"])

        if "control" in cfg_dict and "action_bounds" in cfg_dict["control"]:
            config.action_bounds = cfg_dict["control"]["action_bounds"]

        if "obs" in cfg_dict:
            obs_dict = cfg_dict["obs"]
            config.obs = ObsNormalizationConfig(**{k: v for k, v in obs_dict.items() if k in ObsNormalizationConfig.__dataclass_fields__})

        reward_dict = cfg_dict.get("reward", {})
        if reward_dict:
            # Handle weights mapping (approach/reach, etc.)
            weights = reward_dict.get("weights", {}).copy()
            if "reach" in weights and "approach" not in weights:
                weights["approach"] = weights["reach"]
            if "approach" in weights and "reach" not in weights:
                weights["reach"] = weights["approach"]
            
            # Update weights in the dict so RewardConfig can be initialized
            reward_dict["weights"] = {**config.reward.weights, **weights}

            # Handle adaptive weights nested dataclasses
            for key in ["adaptive_grasp_weight", "adaptive_lift_weight", "adaptive_success_weight"]:
                if key in reward_dict:
                    reward_dict[key] = AdaptiveWeightConfig(**reward_dict[key])

            # Handle stage_thresholds
            if "stages" in reward_dict:
                stages = reward_dict["stages"]
                reward_dict["stage_thresholds"] = {
                    "approach": stages.get("approach_threshold", stages.get("reach_threshold", 0.05)),
                    "reach": stages.get("reach_threshold", stages.get("approach_threshold", 0.05)),
                    "grasp": stages.get("grasp_threshold", 0.03),
                    "lift": stages.get("lift_target", 0.05),
                }

        config.reward = RewardConfig(**{k: v for k, v in reward_dict.items() if k in RewardConfig.__dataclass_fields__})

        config.robot_urdf = env_cfg.get("robot_urdf", None)
        config.raw_cfg = _to_plain(cfg_dict)

        return config
