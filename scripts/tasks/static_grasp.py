import torch
from typing import Dict, Optional
from scripts.tasks.base import BaseTaskHandler
from mani_skill.utils.structs.pose import Pose

class StaticGraspTaskHandler(BaseTaskHandler):
    """
    Simplified task for debugging grasp detection.
    
    - Cube spawns at random position but is STATIC (kinematic, won't move)
    - Success: Continuously grasp the cube for N seconds
    - Simplified reward: mainly focuses on approach and grasp
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Grasp hold tracking
        self.continuous_grasp_counter: Optional[torch.Tensor] = None
        self.required_grasp_steps: int = None
    
    @classmethod
    def _get_train_metrics(cls) -> Dict[str, str]:
        """Define StaticGrasp task metrics for training."""
        return {
            "grasp_reward": "mean",
            "grasp_success": "mean",
            "grasp_hold_steps": "mean",
        }

    def initialize_episode(self, env_idx, options):
        b = len(env_idx)
        
        # Random spawn position within configured bounds
        spawn_grid = self.env.spawn_bounds if self.env.spawn_bounds else self.env.grid_bounds["right"]
        red_pos, red_quat = self.env._random_grid_position(b, spawn_grid, z=0.015 + self.env.space_gap)
        
        # Set cube pose
        # Note: Cube is already built as static (kinematic) in _load_objects,
        # so it won't respond to physics forces
        self.env.red_cube.set_pose(Pose.create_from_pq(p=red_pos, q=red_quat))
        
        # Store initial position for observation/debugging
        if self.initial_red_cube_pos is None:
            self.initial_red_cube_pos = torch.zeros((self.env.num_envs, 3), device=self.device)
        self.initial_red_cube_pos[env_idx] = red_pos
        
        # Initialize continuous grasp counter
        if self.continuous_grasp_counter is None:
            self.continuous_grasp_counter = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.int32)
        self.continuous_grasp_counter[env_idx] = 0
        
        # Calculate required grasp steps from stable_hold_time config
        # Default to 3.0 seconds if not specified
        hold_time = getattr(self.env, 'stable_hold_time', 3.0)
        control_freq = getattr(self.env, 'control_freq', 30)
        self.required_grasp_steps = int(hold_time * control_freq)
        
        # Reset prev_action for action rate penalty (if used)
        self.prev_action = None

    def evaluate(self) -> Dict[str, torch.Tensor]:
        """
        Success condition: Continuously grasp the cube for required_grasp_steps.
        
        Since cube is static, we don't check for falling or out-of-bounds.
        """
        # Check if currently grasping the cube
        agent = self.env.right_arm
        is_grasped = agent.is_grasping(
            self.env.red_cube,
            min_force=self.env.grasp_min_force,
            max_angle=self.env.grasp_max_angle
        )
        
        # Update continuous grasp counter
        if self.continuous_grasp_counter is None:
            self.continuous_grasp_counter = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.int32)
        
        # Increment counter when grasping, reset when not
        self.continuous_grasp_counter = torch.where(
            is_grasped,
            self.continuous_grasp_counter + 1,
            torch.zeros_like(self.continuous_grasp_counter)
        )
        
        # Success when counter reaches required steps
        success = self.continuous_grasp_counter >= self.required_grasp_steps
        
        # No failure conditions for this simple task
        # (Could add timeout via max_episode_steps in env registration)
        fail = torch.zeros_like(success, dtype=torch.bool)
        
        return {
            "success": success,
            "fail": fail,
            "is_grasped": is_grasped,
            "grasp_hold_steps": self.continuous_grasp_counter,
        }

    def compute_dense_reward(self, info: dict, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Simplified dense reward for StaticGrasp task.
        
        Components:
        1. Approach reward: Encourage TCP to move close to cube
        2. Grasp reward: Reward for successfully grasping
        3. Hold progress: Progressive reward for maintaining grasp
        4. Success bonus: Large bonus for completing the task
        """
        w = self.env.reward_weights
        
        cube_pos = self.env.red_cube.pose.p
        tcp_pos = self.env.right_arm.tcp_pos
        
        # 1. Approach reward - encourage moving TCP towards cube
        distance = torch.norm(tcp_pos - cube_pos, dim=1)
        
        # Tanh-based approach reward: 1.0 - tanh(distance / scale)
        # Using configured approach_tanh_scale or default 0.05
        tanh_scale = getattr(self.env, 'approach_tanh_scale', 0.05)
        approach_reward = 1.0 - torch.tanh(distance / tanh_scale)
        
        # 2. Grasp reward - binary reward for grasping
        agent = self.env.right_arm
        is_grasped = agent.is_grasping(
            self.env.red_cube,
            min_force=self.env.grasp_min_force,
            max_angle=self.env.grasp_max_angle
        )
        grasp_reward = is_grasped.float()
        
        # 3. Hold progress reward - progressive reward for maintaining grasp
        if self.continuous_grasp_counter is not None and self.required_grasp_steps > 0:
            # Normalized progress: 0 to 1
            hold_progress = torch.clamp(
                self.continuous_grasp_counter.float() / self.required_grasp_steps,
                min=0.0,
                max=1.0
            )
        else:
            hold_progress = torch.zeros(self.env.num_envs, device=self.device)
        
        # 4. Success bonus
        success = info.get("success", torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool))
        success_bonus = success.float()
        
        # 5. Optional: Action rate penalty to encourage smooth motion
        if action is not None and w.get("action_rate", 0.0) != 0:
            if isinstance(action, dict):
                action_tensor = torch.cat([v.flatten(start_dim=1) for v in action.values()], dim=1)
            else:
                action_tensor = action.flatten(start_dim=1) if action.dim() > 1 else action.unsqueeze(0)
            
            if self.prev_action is None:
                action_rate = torch.zeros(self.env.num_envs, device=self.device)
            else:
                action_diff = action_tensor - self.prev_action
                action_rate = torch.sum(action_diff ** 2, dim=1)
            self.prev_action = action_tensor.clone()
        else:
            action_rate = torch.zeros(self.env.num_envs, device=self.device)
        
        # Weighted sum - using config weights or defaults
        reward = (
            w.get("approach", 1.0) * approach_reward +
            w.get("grasp", 5.0) * grasp_reward +
            w.get("hold_progress", 10.0) * hold_progress +
            w.get("success", 50.0) * success_bonus +
            w.get("action_rate", -0.1) * action_rate
        )
        
        # Store reward components for logging
        info["reward_components"] = {
            "approach": (w.get("approach", 1.0) * approach_reward).mean(),
            "grasp": (w.get("grasp", 5.0) * grasp_reward).mean(),
            "hold_progress": (w.get("hold_progress", 10.0) * hold_progress).mean(),
            "success": (w.get("success", 50.0) * success_bonus).mean(),
            "action_rate": (w.get("action_rate", -0.1) * action_rate).mean(),
        }
        
        # Per-env components for detailed eval logging
        if self.env.eval_mode:
            info["reward_components_per_env"] = {
                "approach": w.get("approach", 1.0) * approach_reward,
                "grasp": w.get("grasp", 5.0) * grasp_reward,
                "hold_progress": w.get("hold_progress", 10.0) * hold_progress,
                "success": w.get("success", 50.0) * success_bonus,
                "action_rate": w.get("action_rate", -0.1) * action_rate,
            }
        
        # Logging counters
        info["success_count"] = success.sum()
        info["grasp_count"] = is_grasped.sum()
        
        return reward
