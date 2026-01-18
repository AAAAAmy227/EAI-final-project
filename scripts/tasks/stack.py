import torch
from typing import Dict, Optional
from scripts.tasks.base import BaseTaskHandler
from mani_skill.envs.utils import randomization
from mani_skill.utils.structs.pose import Pose

class StackTaskHandler(BaseTaskHandler):
    @classmethod
    def _get_train_metrics(cls) -> Dict[str, str]:
        return {
            "reward/reach_red": "mean",
            "reward/grasp_red": "mean",
            "reward/lift_red": "mean",
            "reward/align": "mean",
            "reward/place": "mean",
            "reward/success_bonus": "mean",
            "reward/action_rate": "mean",
            "stack_success": "mean",
        }

    def initialize_episode(self, env_idx, options):
        b = len(env_idx)
        # Both cubes in Right Grid, non-overlapping
        min_dist = 0.043
        red_pos, red_quat = self.env._random_grid_position(b, self.env.grid_bounds["right"], z=0.015)
        green_pos, green_quat = self.env._random_grid_position(b, self.env.grid_bounds["right"], z=0.015)
        
        # Ensure minimum distance
        for _ in range(100):
            dist = torch.norm(red_pos[:, :2] - green_pos[:, :2], dim=1)
            overlap = dist < min_dist
            if not overlap.any():
                break
            new_pos, new_quat = self.env._random_grid_position(overlap.sum().item(), self.env.grid_bounds["right"], z=0.015)
            green_pos[overlap] = new_pos
            green_quat[overlap] = new_quat
        
        self.env.red_cube.set_pose(Pose.create_from_pq(p=red_pos, q=red_quat))
        self.env.green_cube.set_pose(Pose.create_from_pq(p=green_pos, q=green_quat))
        
        # Store initial cube full poses
        if self.initial_red_cube_pos is None:
            self.initial_red_cube_pos = torch.zeros((self.env.num_envs, 3), device=self.device)
        if self.initial_green_cube_pos is None:
            self.initial_green_cube_pos = torch.zeros((self.env.num_envs, 3), device=self.device)
        
        self.initial_red_cube_pos[env_idx] = red_pos
        self.initial_green_cube_pos[env_idx] = green_pos

        # Reset action rate tracking for new episodes
        self.prev_action = None

    def evaluate(self) -> Dict[str, torch.Tensor]:
        """Stack: red cube on top of green cube, stable on table."""
        red_pos = self.env.red_cube.pose.p
        green_pos = self.env.green_cube.pose.p

        # Green cube should stay on table within tolerance
        z_low, z_high = self.env.green_z_range
        green_on_table = (green_pos[:, 2] > z_low) & (green_pos[:, 2] < z_high)
        
        # Check if red is above green (target height with tolerance)
        z_diff = red_pos[:, 2] - green_pos[:, 2]
        target_h = self.env.stack_height_target
        z_tol = self.env.stack_height_tolerance
        z_ok = (z_diff > target_h - z_tol) & (z_diff < target_h + z_tol)

        # Check xy alignment (configurable tolerance)
        xy_dist = torch.norm(red_pos[:, :2] - green_pos[:, :2], dim=1)
        xy_ok = xy_dist < self.env.stack_xy_tolerance
        
        success = green_on_table & z_ok & xy_ok
        # Failure conditions
        fallen_threshold = -0.05
        red_fallen = self.env.red_cube.pose.p[:, 2] < fallen_threshold
        green_fallen = self.env.green_cube.pose.p[:, 2] < fallen_threshold
        fail = red_fallen | green_fallen
        
        success = success & (~fail)
        
        return {
            "success": success, 
            "fail": fail,
            "green_on_table": green_on_table, 
            "z_diff": z_diff, 
            "xy_dist": xy_dist
        }

    def compute_dense_reward(self, info: dict, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Dense reward for Stack task."""
        w = self.env.reward_weights

        red_pos = self.env.red_cube.pose.p
        green_pos = self.env.green_cube.pose.p
        agent = self.env.right_arm
        tcp_pos = agent.tcp_pos

        # 1. Reach red cube
        dist_to_red = torch.norm(tcp_pos - red_pos, dim=1)
        reach_reward = 1.0 - torch.tanh(dist_to_red / self.env.approach_tanh_scale)
        
        # 1.5 Fine Reach: precise guidance when close to object
        fine_reach = 1.0 - torch.tanh(dist_to_red / 0.01)

        # 2. Grasp red cube using contact-based detection
        is_grasped = agent.is_grasping(
            self.env.red_cube,
            min_force=self.env.grasp_min_force,
            max_angle=self.env.grasp_max_angle,
        )
        grasp_reward = is_grasped.float()
        
        # 2.5 Anti-Bow Penalty (Wrist vs Fingertips)
        # Penalize if wrist joint (gripper_link) is lower than fingertips (tcp_pos) when grasping
        # This prevents the "bowed" wrist posture
        
        # We need the actual physical wrist link.
        # In SO101 agent, the physical wrist is 'gripper_link' or 'wrist_link'.
        # Let's use the actual joint positions to detect pitch.
        
        # Simpler approach: Check tcp_pose quaternion pitch
        # Ideally, gripper should be pointing DOWN (-Z) or Level.
        # If pointing UP, that's bad.
        
        # Let's penalize if wrist joint (link index 5) is lower than TCP
        # agent.robot.get_links()[5] is gripper_link (wrist base)
        # agent.tcp_pos is TCP
        wrist_pos = agent.robot.get_links()[5].pose.p
        tcp_pos = agent.tcp_pos
        
        # If wrist is lower than TCP, we have a problem (upside down or excessive pitch)
        # wrist_z < tcp_z is BAD for top-down picking.
        # Limit this penalty to when we are grasping or close to it
        wrist_height_diff = wrist_pos[:, 2] - tcp_pos[:, 2] # Should be positive (Wrist above TCP)
        
        # Penalty grows if wrist is below TCP (diff < 0)
        # Use ReLU to only punish negative values
        # penalty = weight * relu(-diff)
        anti_bow_penalty = torch.relu(-wrist_height_diff) 
        
        # Add to total reward (negative term) - weight 1.0 (strong enough to correct)
        # Only apply when close to object
        close_mask = (dist_to_red < 0.05).float()
        grasp_reward = grasp_reward - (1.0 * anti_bow_penalty * close_mask)
        
        

        # 3. Lift/Transport: Encorage lifting to a safe transport height (stack_target + margin)
        # This helps the robot clear the bottom cube before trying to stack
        z_diff = red_pos[:, 2] - green_pos[:, 2]
        # Dynamically calculated safe height: Target Stack Height + Configured Margin
        transport_margin = getattr(self.env, "transport_height_margin", 0.05)
        target_h = max(self.env.stack_height_target, 1e-4)
        transport_h = target_h + transport_margin
        
        # Reward lifting up to transport height
        # Note: We do NOT penalize being higher than this in the lift term, 
        # because the 'place' term will eventually pull it down.
        # This purely encourages "Get it up off the table/base".
        lift_progress = torch.clamp(z_diff, min=0.0, max=transport_h) / transport_h
        
        if self.env.gate_lift_with_grasp:
            lift_progress = lift_progress * grasp_reward

        # 4. Align horizontally with green cube
        # Gating Idea: "Grasp securely before aligning"
        # We multiply align reward by is_grasped. 
        # Since is_grasped is now based on min_force=0.1 (easy to trigger), this is not too strict.
        # But it prevents the robot from trying to align the "ghost of the red cube" without holding it.
        # To make it even smoother, we can allow a small amount of align reward always (for curiosity),
        # but the bulk comes from holding it.
        # Let's use: 0.1 (always) + 0.9 (if grasped)
        xy_dist = torch.norm(red_pos[:, :2] - green_pos[:, :2], dim=1)
        
        # Base alignment term
        raw_align_reward = 1.0 - torch.tanh(xy_dist / self.env.stack_align_tanh_scale)
        
        # Apply Gating
        # is_grasped is derived from contact sensor (binary-ish float). 
        # Using the raw is_grasped (before anti-bow penalty) to indicate "holding".
        align_reward = raw_align_reward * (0.1 + 0.9 * is_grasped.float())

        # 5. Place reward: Only active when XY is aligned
        # Strong incentive to lower to exact stack height ONLY when safe to do so
        z_dist_to_target = torch.abs(z_diff - target_h)
        place_guidance = 1.0 - torch.tanh(z_dist_to_target / 0.01) # Sharper guidance (1cm scale)
        
        # Condition: Must be within 2x XY tolerance to start caring about precise Z placement
        is_xy_aligned = (xy_dist < (self.env.stack_xy_tolerance * 2.0)).float()
        place_reward = place_guidance * is_xy_aligned

        # 6. Success and fail bonuses/penalties
        success = info.get("success", torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool))
        success_bonus = success.float()
        fail = info.get("fail", torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)).float()

        # 7. Action rate penalty (squared diff between consecutive actions)
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

        reward = (
            w.get("reach_red", 0.0) * reach_reward
            + w.get("fine_reach", 0.0) * fine_reach
            + w.get("grasp_red", 0.0) * grasp_reward
            + w.get("lift_red", 0.0) * lift_progress
            + w.get("align", 0.0) * align_reward
            + w.get("place", 0.0) * place_reward
            + w.get("success", 0.0) * success_bonus
            + w.get("fail", 0.0) * fail
            + w.get("action_rate", 0.0) * action_rate
        )

        reward_info = {
            "reward/reach_red": w.get("reach_red", 0.0) * reach_reward,
            "reward/fine_reach": w.get("fine_reach", 0.0) * fine_reach,
            "reward/grasp_red": w.get("grasp_red", 0.0) * grasp_reward,
            "reward/lift_red": w.get("lift_red", 0.0) * lift_progress,
            "reward/align": w.get("align", 0.0) * align_reward,
            "reward/place": w.get("place", 0.0) * place_reward,
            "reward/success_bonus": w.get("success", 0.0) * success_bonus,
            "reward/action_rate": w.get("action_rate", 0.0) * action_rate,
        }
        info.update(reward_info)

        info["reward_components"] = {k: v.mean() for k, v in reward_info.items()}
        info["stack_success"] = success.float()
        info["success_count"] = success.sum()
        info["fail_count"] = fail.sum()

        if self.env.eval_mode:
            info["reward_components_per_env"] = reward_info

        return reward
