import torch
from typing import Dict, Optional
from scripts.tasks.base import BaseTaskHandler
from mani_skill.envs.utils import randomization
from mani_skill.utils.structs.pose import Pose


class SortTaskHandler(BaseTaskHandler):
    """Sort Task Handler for two-phase sorting (left arm → right arm).
    
    Phase 1 (sort_left): Left arm moves green cube to left grid.
    Phase 2 (sort_right): Right arm moves red cube to right grid.
    
    This handler supports training each phase separately.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Adaptive states (EMA) for potential adaptive rewards
        self.grasp_success_rate: Optional[torch.Tensor] = None
        self.place_success_rate: Optional[torch.Tensor] = None
        self.task_success_rate: Optional[torch.Tensor] = None
    
    @classmethod
    def _get_train_metrics(cls) -> Dict[str, str]:
        """Define Sort task metrics for training."""
        return {
            "reward/approach": "mean",
            "reward/grasp": "mean",
            "reward/grasp_hold": "mean",
            "reward/place": "mean",
            "reward/keep_non_target_in_bounds": "mean",
            "reward/action_rate": "mean",
            "reward/success_bonus": "mean",
            "grasp_success": "mean",
            "target_in_place": "mean",  # Only the active arm's target
        }

    def initialize_episode(self, env_idx, options):
        b = len(env_idx)
        
        # Get spawn parameters from config
        mid_grid = self.env.grid_bounds["mid"]
        min_dist = getattr(self.env, 'min_cube_distance', 0.04)  # Default 4cm
        max_retries = 20
        
        # Red cube z height (3cm half-size cube)
        red_z = 0.015 + self.env.space_gap
        # Green cube z height (3cm half-size cube, same as red)
        green_z = 0.015 + self.env.space_gap
        
        # Generate positions with collision avoidance
        for _ in range(max_retries):
            red_pos, red_quat = self.env._random_grid_position(b, mid_grid, z=red_z)
            green_pos, green_quat = self.env._random_grid_position(b, mid_grid, z=green_z)
            
            # Check distance between cubes (XY plane only)
            xy_dist = torch.norm(red_pos[:, :2] - green_pos[:, :2], dim=1)
            
            # If all environments have sufficient distance, break
            if (xy_dist >= min_dist).all():
                break
            
            # Regenerate only for environments with collision
            collision_mask = xy_dist < min_dist
            if collision_mask.any():
                # Regenerate green cube positions for colliding envs
                new_green_pos, new_green_quat = self.env._random_grid_position(
                    collision_mask.sum().item(), mid_grid, z=green_z
                )
                green_pos[collision_mask] = new_green_pos
                green_quat[collision_mask] = new_green_quat
        
        # Set cube poses
        self.env.red_cube.set_pose(Pose.create_from_pq(p=red_pos, q=red_quat))
        self.env.green_cube.set_pose(Pose.create_from_pq(p=green_pos, q=green_quat))
        
        # Store initial cube positions
        if self.initial_red_cube_pos is None:
            self.initial_red_cube_pos = torch.zeros((self.env.num_envs, 3), device=self.device)
        if self.initial_green_cube_pos is None:
            self.initial_green_cube_pos = torch.zeros((self.env.num_envs, 3), device=self.device)
        
        self.initial_red_cube_pos[env_idx] = red_pos
        self.initial_green_cube_pos[env_idx] = green_pos
        
        # Store initial cube XY for out-of-bounds checking
        if self.initial_cube_xy is None:
            self.initial_cube_xy = torch.zeros((self.env.num_envs, 2), device=self.device)
        self.initial_cube_xy[env_idx] = red_pos[:, :2]
        
        # Reset grasp hold counter
        if self.grasp_hold_counter is None:
            self.grasp_hold_counter = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.int32)
        self.grasp_hold_counter[env_idx] = 0
        
        # Reset was_ever_grasped tracking (must grasp to succeed)
        if self.was_ever_grasped is None:
            self.was_ever_grasped = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
        self.was_ever_grasped[env_idx] = False
        
        # Reset prev_action for action rate penalty
        self.prev_action = None

    def evaluate(self) -> Dict[str, torch.Tensor]:
        """Evaluate sort task success/fail conditions.
        
        Success: green cube in left grid (for left-arm phase).
        Fail: cube fallen OR (if fail_red_out_of_bounds) red leaves mid-grid.
        """
        red_pos = self.env.red_cube.pose.p
        green_pos = self.env.green_cube.pose.p
        
        # Check green in Left Grid
        left_bounds = self.env.grid_bounds["left"]
        green_in_left = (
            (green_pos[:, 0] >= left_bounds["x_min"]) & 
            (green_pos[:, 0] <= left_bounds["x_max"]) &
            (green_pos[:, 1] >= left_bounds["y_min"]) & 
            (green_pos[:, 1] <= left_bounds["y_max"]) &
            (green_pos[:, 2] >= -0.01)  # Not fallen
        )
        
        # Check red in Right Grid (for full task completion reference)
        right_bounds = self.env.grid_bounds["right"]
        red_in_right = (
            (red_pos[:, 0] >= right_bounds["x_min"]) & 
            (red_pos[:, 0] <= right_bounds["x_max"]) &
            (red_pos[:, 1] >= right_bounds["y_min"]) & 
            (red_pos[:, 1] <= right_bounds["y_max"]) &
            (red_pos[:, 2] >= -0.01)  # Not fallen
        )
        
        # Check grasp state and update was_ever_grasped tracking
        agent = self._get_active_agent()
        target_cube = self._get_target_cube()
        is_grasped = agent.is_grasping(
            target_cube,
            min_force=self.env.grasp_min_force,
            max_angle=self.env.grasp_max_angle
        )
        
        # Update grasp hold counter (consecutive grasp steps)
        if self.grasp_hold_counter is None:
            self.grasp_hold_counter = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.int32)
        self.grasp_hold_counter = torch.where(
            is_grasped,
            self.grasp_hold_counter + 1,
            torch.zeros_like(self.grasp_hold_counter)
        )
        
        # Track if cube was ever grasped for sufficient duration (required for success)
        # min_grasp_steps: minimum consecutive grasp steps to count as valid grasp
        min_grasp_steps = getattr(self.env, 'min_grasp_steps', 10)  # Default 10 steps (~0.5s at 20Hz)
        if self.was_ever_grasped is None:
            self.was_ever_grasped = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
        # Once grasp_hold_counter reaches threshold, lock in was_ever_grasped
        self.was_ever_grasped = self.was_ever_grasped | (self.grasp_hold_counter >= min_grasp_steps)
        
        # Success condition: cube in target grid AND was ever grasped (no pushing allowed)
        active_arm = getattr(self.env, 'active_arm', 'left')
        if active_arm == "left":
            in_target = green_in_left
        else:
            in_target = red_in_right
        
        # Must have grasped at some point to count as success
        success = in_target & self.was_ever_grasped
        
        # Failure conditions
        fallen_threshold = -0.05
        red_fallen = red_pos[:, 2] < fallen_threshold
        green_fallen = green_pos[:, 2] < fallen_threshold
        fail = red_fallen | green_fallen
        
        # Optional: fail if non-target cube leaves mid-grid (e.g. red for left-arm phase)
        fail_non_target_out_of_bounds = getattr(self.env, 'fail_non_target_out_of_bounds', False)
        if fail_non_target_out_of_bounds:
            mid_bounds = self.env.grid_bounds["mid"]
            non_target_cube = self._get_non_target_cube()
            non_target_pos = non_target_cube.pose.p
            non_target_out_of_mid = (
                (non_target_pos[:, 0] < mid_bounds["x_min"]) |
                (non_target_pos[:, 0] > mid_bounds["x_max"]) |
                (non_target_pos[:, 1] < mid_bounds["y_min"]) |
                (non_target_pos[:, 1] > mid_bounds["y_max"])
            )
            fail = fail | non_target_out_of_mid
        
        # Ensure success is False if already failed
        success = success & (~fail)
        
        # Only return the active arm's target placement metric
        return {
            "success": success, 
            "fail": fail,
            "target_in_place": in_target,  # green_in_left for left arm, red_in_right for right arm
            "is_grasped": is_grasped,
            "was_ever_grasped": self.was_ever_grasped,
            "grasp_hold_steps": self.grasp_hold_counter,
        }

    def _get_active_agent(self):
        """Get the active arm agent based on config."""
        active_arm = getattr(self.env, 'active_arm', 'left')
        return self.env.left_arm if active_arm == "left" else self.env.right_arm
    
    def _get_target_cube(self):
        """Get the target cube for the active arm."""
        active_arm = getattr(self.env, 'active_arm', 'left')
        # Left arm targets green cube, right arm targets red cube
        return self.env.green_cube if active_arm == "left" else self.env.red_cube
    
    def _get_non_target_cube(self):
        """Get the non-target cube for interference checking."""
        active_arm = getattr(self.env, 'active_arm', 'left')
        return self.env.red_cube if active_arm == "left" else self.env.green_cube

    def compute_dense_reward(self, info: dict, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Dense reward for Sort task (left-arm phase)."""
        w = self.env.reward_weights
        
        agent = self._get_active_agent()
        target_cube = self._get_target_cube()
        non_target_cube = self._get_non_target_cube()
        
        target_pos = target_cube.pose.p
        non_target_pos = non_target_cube.pose.p
        
        # 0. Check grasp state
        is_grasped = agent.is_grasping(
            target_cube,
            min_force=self.env.grasp_min_force,
            max_angle=self.env.grasp_max_angle
        )
        
        # 1. Approach reward (TCP to target cube)
        tcp_pos = agent.tcp_pos
        distance = torch.norm(tcp_pos - target_pos, dim=1)
        
        if self.env.approach_curve == "tanh":
            approach_reward = 1.0 - torch.tanh(distance / self.env.approach_tanh_scale)
        else:
            threshold = self.env.approach_threshold
            zero_point = self.env.approach_zero_point
            approach_reward = torch.where(
                distance < threshold,
                torch.ones_like(distance),
                torch.clamp(1.0 - (distance - threshold) / (zero_point - threshold), min=0.0)
            )
        
        # 2. Grasp reward
        grasp_reward = is_grasped.float()
        
        # 2.5. Grasp hold reward (stability)
        if hasattr(self.env, 'grasp_hold_max_steps') and self.env.grasp_hold_max_steps > 0:
            if self.grasp_hold_counter is None:
                self.grasp_hold_counter = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.int32)
            
            self.grasp_hold_counter = torch.where(
                is_grasped,
                self.grasp_hold_counter + 1,
                torch.zeros_like(self.grasp_hold_counter)
            )
            
            current_count = torch.clamp(self.grasp_hold_counter, max=self.env.grasp_hold_max_steps)
            T = float(self.env.grasp_hold_max_steps)
            grasp_hold_reward = (2.0 * current_count.float()) / (T + 1.0)
        else:
            grasp_hold_reward = torch.zeros(self.env.num_envs, device=self.device)
        
        # 3. Place reward (distance to target grid center)
        active_arm = getattr(self.env, 'active_arm', 'left')
        if active_arm == "left":
            target_grid = self.env.grid_bounds["left"]
        else:
            target_grid = self.env.grid_bounds["right"]
        
        grid_center_x = (target_grid["x_min"] + target_grid["x_max"]) / 2
        grid_center_y = (target_grid["y_min"] + target_grid["y_max"]) / 2
        grid_center = torch.tensor([grid_center_x, grid_center_y], device=self.device)
        
        # Distance from target cube to grid center (XY only)
        place_distance = torch.norm(target_pos[:, :2] - grid_center, dim=1)
        
        # Place reward: only when grasping, based on distance to grid
        place_scale = getattr(self.env, 'place_tanh_scale', 0.1)
        place_reward = (1.0 - torch.tanh(place_distance / place_scale)) * is_grasped.float()
        
        # 4. Keep non-target cube in bounds (penalty if it leaves mid-grid)
        mid_bounds = self.env.grid_bounds["mid"]
        non_target_in_mid = (
            (non_target_pos[:, 0] >= mid_bounds["x_min"]) & 
            (non_target_pos[:, 0] <= mid_bounds["x_max"]) &
            (non_target_pos[:, 1] >= mid_bounds["y_min"]) & 
            (non_target_pos[:, 1] <= mid_bounds["y_max"])
        )
        # Penalty for non-target leaving mid-grid (negative value when out)
        keep_in_bounds_penalty = (~non_target_in_mid).float()
        
        # 5. Action rate penalty
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
        
        # 6. Success bonus
        success = info.get("success", torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool))
        success_bonus = success.float()
        
        # 7. Fail penalty
        fail = info.get("fail", torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool))
        fail_penalty = fail.float()
        
        # Weighted sum
        reward = (
            w.get("approach", 1.0) * approach_reward +
            w.get("grasp", 1.0) * grasp_reward +
            w.get("grasp_hold", 0.0) * grasp_hold_reward +
            w.get("place", 1.0) * place_reward +
            w.get("keep_non_target_in_bounds", 0.0) * (-keep_in_bounds_penalty) +  # Negative weight for penalty
            w.get("action_rate", 0.0) * (-action_rate) +
            w.get("success", 0.0) * success_bonus +
            w.get("fail", 0.0) * fail_penalty
        )
        
        # Store components for logging
        reward_info = {
            "reward/approach": w.get("approach", 1.0) * approach_reward,
            "reward/grasp": w.get("grasp", 1.0) * grasp_reward,
            "reward/grasp_hold": w.get("grasp_hold", 0.0) * grasp_hold_reward,
            "reward/place": w.get("place", 1.0) * place_reward,
            "reward/keep_non_target_in_bounds": w.get("keep_non_target_in_bounds", 0.0) * (-keep_in_bounds_penalty),
            "reward/action_rate": w.get("action_rate", 0.0) * (-action_rate),
            "reward/success_bonus": w.get("success", 0.0) * success_bonus,
        }
        info.update(reward_info)
        
        # Legacy/averaged components
        info["reward_components"] = {
            "approach": reward_info["reward/approach"].mean(),
            "grasp": reward_info["reward/grasp"].mean(),
            "grasp_hold": reward_info["reward/grasp_hold"].mean(),
            "place": reward_info["reward/place"].mean(),
            "keep_non_target_in_bounds": reward_info["reward/keep_non_target_in_bounds"].mean(),
            "action_rate": reward_info["reward/action_rate"].mean(),
        }
        
        info["success_count"] = success.sum()
        info["fail_count"] = fail.sum()
        info["grasp_count"] = is_grasped.sum()
        
        return reward
