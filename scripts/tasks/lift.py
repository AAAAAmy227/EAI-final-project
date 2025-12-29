import torch
from typing import Dict, Optional
from scripts.tasks.base import BaseTaskHandler
from mani_skill.envs.utils import randomization
from mani_skill.utils.structs.pose import Pose

class LiftTaskHandler(BaseTaskHandler):
    def __init__(self, env):
        super().__init__(env)
        
        # Adaptive states (EMA)
        self.grasp_success_rate = None
        self.lift_success_rate = None
        self.task_success_rate = None
        
        # Internal counters
        self.lift_hold_counter = None
        self.grasp_hold_counter = None
        self.prev_action = None
        self.initial_red_cube_pos = None
        self.initial_cube_xy = None

    def initialize_episode(self, env_idx, options):
        b = len(env_idx)
        # Red cube random in configured spawn_bounds (or default grid)
        spawn_grid = self.env.spawn_bounds if self.env.spawn_bounds else self.env.grid_bounds["right"]
        red_pos, red_quat = self.env._random_grid_position(b, spawn_grid, z=0.015 + self.env.space_gap)
        self.env.red_cube.set_pose(Pose.create_from_pq(p=red_pos, q=red_quat))
        
        # Store initial cube full pos for displacement observation
        if self.initial_red_cube_pos is None:
            self.initial_red_cube_pos = torch.zeros((self.env.num_envs, 3), device=self.device)
        self.initial_red_cube_pos[env_idx] = red_pos
        
        # Store initial cube XY for horizontal penalty in reward
        if self.initial_cube_xy is None:
            self.initial_cube_xy = torch.zeros(self.env.num_envs, 2, device=self.device)
        self.initial_cube_xy[env_idx] = red_pos[:, :2]
        
        # Reset stable hold counter for success condition
        if self.lift_hold_counter is None:
            self.lift_hold_counter = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.int32)
        self.lift_hold_counter[env_idx] = 0
        
        # Reset grasp hold counter
        if self.grasp_hold_counter is None:
            self.grasp_hold_counter = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.int32)
        self.grasp_hold_counter[env_idx] = 0
        
        # Reset prev_action for action rate penalty
        self.prev_action = None

    def evaluate(self) -> Dict[str, torch.Tensor]:
        """Lift: red cube >= lift_target AND is_grasped for stable_hold_time seconds."""
        red_z = self.env.red_cube.pose.p[:, 2]
        is_above = red_z >= self.env.lift_target
        
        # Check if currently grasping the cube
        agent = self.env.right_arm
        is_grasped = agent.is_grasping(
            self.env.red_cube,
            min_force=self.env.grasp_min_force,
            max_angle=self.env.grasp_max_angle
        )
        
        # Valid hold: cube is above target AND is being grasped
        is_valid_hold = is_above & is_grasped
        
        if self.env.stable_hold_steps <= 0:
            # Instant success mode (backward compatible)
            success = is_valid_hold
        else:
            # Stable hold mode: increment counter when valid, reset otherwise
            if self.lift_hold_counter is None:
                self.lift_hold_counter = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.int32)
            
            # Increment counter only when both above AND grasped
            self.lift_hold_counter = torch.where(
                is_valid_hold,
                self.lift_hold_counter + 1,
                torch.zeros_like(self.lift_hold_counter)  # Reset if not valid
            )
            
            # Success when counter reaches required hold steps
            success = self.lift_hold_counter >= self.env.stable_hold_steps
            
        # Failure conditions
        fallen_threshold = -0.05
        red_fallen = self.env.red_cube.pose.p[:, 2] < fallen_threshold
        fail = red_fallen
        
        if self.env.fail_bounds is not None:
            red_pos = self.env.red_cube.pose.p
            out_of_bounds = (
                (red_pos[:, 0] < self.env.fail_bounds["x_min"]) |
                (red_pos[:, 0] > self.env.fail_bounds["x_max"]) |
                (red_pos[:, 1] < self.env.fail_bounds["y_min"]) |
                (red_pos[:, 1] > self.env.fail_bounds["y_max"])
            )
            # Only fail on out_of_bounds if NOT grasping (allow moving cube while grasped)
            fail = fail | (out_of_bounds & (~is_grasped))
            
        # Ensure success is False if already failed
        success = success & (~fail)
        
        return {
            "success": success, 
            "fail": fail,
            "red_height": red_z, 
            "hold_steps": self.lift_hold_counter if self.lift_hold_counter is not None else torch.zeros_like(success, dtype=torch.int32)
        }

    def compute_dense_reward(self, info: dict, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Dense reward for Lift task."""
        # Get config values
        w = self.env.reward_weights
        
        cube_pos = self.env.red_cube.pose.p
        cube_height = cube_pos[:, 2]
        
        # 0. Check if currently grasping (used for gating other rewards)
        agent = self.env.right_arm
        is_grasped = agent.is_grasping(
            self.env.red_cube,
            min_force=self.env.grasp_min_force,
            max_angle=self.env.grasp_max_angle
        )
        
        # Approach reward calculation based on approach_mode and approach_curve
        threshold = self.env.approach_threshold
        zero_point = self.env.approach_zero_point
        
        # Helper function to compute approach reward based on curve type
        def compute_approach_reward(distance, th, zp):
            if self.env.approach_curve == "tanh":
                return 1.0 - torch.tanh(distance / self.env.approach_tanh_scale)
            else:
                return torch.where(
                    distance < th,
                    torch.ones_like(distance),
                    torch.clamp(1.0 - (distance - th) / (zp - th), min=0.0)
                )
        
        if self.env.approach_mode == "tcp_midpoint":
            tcp_pos = self.env.right_arm.tcp_pos
            distance = torch.norm(tcp_pos - cube_pos, dim=1)
            approach_reward = compute_approach_reward(distance, threshold, zero_point)
            approach2_reward = torch.zeros_like(approach_reward)
        else:
            gripper_pos = self.env._get_gripper_pos()
            distance = torch.norm(gripper_pos - cube_pos, dim=1)
            approach_reward = compute_approach_reward(distance, threshold, zero_point)
            
            moving_jaw_pos = self.env._get_moving_jaw_pos()
            distance2 = torch.norm(moving_jaw_pos - cube_pos, dim=1)
            threshold2 = self.env.approach2_threshold
            zero_point2 = self.env.approach2_zero_point
            approach2_reward = compute_approach_reward(distance2, threshold2, zero_point2)
        
        # 3. Horizontal displacement
        if self.initial_cube_xy is not None:
            raw_displacement = torch.norm(cube_pos[:, :2] - self.initial_cube_xy, dim=1)
            threshold = self.env.horizontal_displacement_threshold
            horizontal_displacement = torch.clamp(raw_displacement - threshold, min=0.0)
            horizontal_displacement = horizontal_displacement * (~is_grasped).float()
        else:
            horizontal_displacement = torch.zeros(self.env.num_envs, device=self.device)
        
        # 4. Lift reward
        cube_baseline_height = 0.015
        lift_height = cube_height - cube_baseline_height
        if self.env.lift_max_height is not None and self.env.lift_max_height > 0:
            lift_reward = torch.clamp(lift_height, min=0.0, max=self.env.lift_max_height) / self.env.lift_max_height
        else:
            lift_reward = torch.clamp(lift_height, min=0.0)
        
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
        
        # 5.5 Hold progress reward
        if self.env.stable_hold_steps > 0 and self.lift_hold_counter is not None:
            T = float(self.env.stable_hold_steps)
            hold_progress = (2.0 * self.lift_hold_counter.float()) / (T + 1.0)
            hold_progress = torch.clamp(hold_progress, min=0.0, max=2.0)
        else:
            hold_progress = torch.zeros(self.env.num_envs, device=self.device)
        
        # 6. Success bonus
        success = info.get("success", torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool))
        success_bonus = success.float()
        
        # Adaptive success weight
        if self.env.adaptive_success_enabled and not self.env.eval_mode:
            if self.task_success_rate is None:
                self.task_success_rate = torch.tensor(self.env.adaptive_success_eps, device=self.device)
            
            batch_success_rate = success_bonus.mean()
            self.task_success_rate = self.task_success_rate * (1 - self.env.adaptive_success_tau) + batch_success_rate * self.env.adaptive_success_tau
            
            dynamic_success_weight = w.get("success", 0.0) * torch.pow(
                1.0 / (self.task_success_rate + self.env.adaptive_success_eps),
                self.env.adaptive_success_alpha
            )
            dynamic_success_weight = torch.clamp(dynamic_success_weight, max=self.env.adaptive_success_max)
        else:
            dynamic_success_weight = w.get("success", 0.0)
        
        # 7. Fail penalty
        fail = info.get("fail", torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool))
        fail_penalty = fail.float()
        
        # 8. Grasp reward
        grasp_reward = is_grasped.float()
        
        # 8.5 Grasp hold reward
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
        
        # Adaptive grasp weight
        if self.env.adaptive_grasp_enabled and not self.env.eval_mode:
            if self.grasp_success_rate is None:
                self.grasp_success_rate = torch.tensor(self.env.adaptive_grasp_eps, device=self.device)
            
            batch_success_rate = grasp_reward.mean()
            self.grasp_success_rate = self.grasp_success_rate * (1 - self.env.adaptive_grasp_tau) + batch_success_rate * self.env.adaptive_grasp_tau
            
            dynamic_grasp_weight = w.get("grasp", 0.0) * torch.pow(
                1.0 / (self.grasp_success_rate + self.env.adaptive_grasp_eps),
                self.env.adaptive_grasp_alpha
            )
            dynamic_grasp_weight = torch.clamp(dynamic_grasp_weight, max=self.env.adaptive_grasp_max)
        else:
            dynamic_grasp_weight = w.get("grasp", 0.0)
        
        # 9. Gated lift reward
        if self.env.gate_lift_with_grasp:
            effective_lift_reward = lift_reward * is_grasped.float()
        else:
            effective_lift_reward = lift_reward
        
        # 10. Adaptive lift weight
        is_lifting = (effective_lift_reward > 0).float()
        if self.env.adaptive_lift_enabled and not self.env.eval_mode:
            if self.lift_success_rate is None:
                self.lift_success_rate = torch.tensor(self.env.adaptive_lift_eps, device=self.device)
            
            batch_lift_rate = is_lifting.mean()
            self.lift_success_rate = self.lift_success_rate * (1 - self.env.adaptive_lift_tau) + batch_lift_rate * self.env.adaptive_lift_tau
            
            dynamic_lift_weight = w.get("lift", 0.0) * torch.pow(
                1.0 / (self.lift_success_rate + self.env.adaptive_lift_eps),
                self.env.adaptive_lift_alpha
            )
            dynamic_lift_weight = torch.clamp(dynamic_lift_weight, max=self.env.adaptive_lift_max)
        else:
            dynamic_lift_weight = w.get("lift", 0.0)
        
        # 11. Gated hold_progress reward
        if self.env.gate_lift_with_grasp:
            effective_hold_progress = hold_progress * is_grasped.float()
        else:
            effective_hold_progress = hold_progress
        
        # Weighted sum
        reward = (w["approach"] * approach_reward +
                  w["approach"] * approach2_reward +
                  dynamic_grasp_weight * grasp_reward +
                  w.get("grasp_hold", 0.0) * grasp_hold_reward +
                  w["horizontal_displacement"] * horizontal_displacement +
                  dynamic_lift_weight * effective_lift_reward + 
                  w.get("hold_progress", 0.0) * effective_hold_progress +
                  w.get("action_rate", 0.0) * action_rate +
                  dynamic_success_weight * success_bonus +
                  w["fail"] * fail_penalty)
        
        # Store components
        info["reward_components"] = {
            "approach": (w["approach"] * approach_reward).mean(),
            "grasp": (dynamic_grasp_weight * grasp_reward).mean(),
            "grasp_hold": (w.get("grasp_hold", 0.0) * grasp_hold_reward).mean(),
            "horizontal_displacement": (w["horizontal_displacement"] * horizontal_displacement).mean(),
            "lift": (dynamic_lift_weight * effective_lift_reward).mean(),
            "hold_progress": (w.get("hold_progress", 0.0) * effective_hold_progress).mean(),
            "action_rate": (w.get("action_rate", 0.0) * action_rate).mean(),
        }
        
        if self.env.eval_mode:
            info["reward_components_per_env"] = {
                "approach": w["approach"] * approach_reward,
                "grasp": dynamic_grasp_weight * grasp_reward,
                "grasp_hold": w.get("grasp_hold", 0.0) * grasp_hold_reward,
                "horizontal_displacement": w["horizontal_displacement"] * horizontal_displacement,
                "lift": dynamic_lift_weight * effective_lift_reward,
                "hold_progress": w.get("hold_progress", 0.0) * effective_hold_progress,
                "action_rate": w.get("action_rate", 0.0) * action_rate,
            }
            if self.env.approach_mode == "dual_point":
                info["reward_components_per_env"]["approach2"] = w["approach"] * approach2_reward
        
        if self.env.adaptive_grasp_enabled:
            info["reward_components"]["grasp_dynamic_weight"] = dynamic_grasp_weight
            info["reward_components"]["grasp_success_rate"] = self.grasp_success_rate
        if self.env.adaptive_lift_enabled:
            info["reward_components"]["lift_dynamic_weight"] = dynamic_lift_weight
            info["reward_components"]["lift_success_rate"] = self.lift_success_rate
        if self.env.adaptive_success_enabled:
            info["reward_components"]["success_dynamic_weight"] = dynamic_success_weight
            info["reward_components"]["success_rate"] = self.task_success_rate
        if self.env.approach_mode == "dual_point":
            info["reward_components"]["approach2"] = (w["approach"] * approach2_reward).mean()
        
        info["success_count"] = success.sum()
        info["fail_count"] = fail.sum()
        info["grasp_count"] = is_grasped.sum()
        
        return reward
