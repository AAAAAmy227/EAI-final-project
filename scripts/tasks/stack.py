import torch
from typing import Dict, Optional
from scripts.tasks.base import BaseTaskHandler
from mani_skill.envs.utils import randomization
from mani_skill.utils.structs.pose import Pose

class StackTaskHandler(BaseTaskHandler):
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

    def evaluate(self) -> Dict[str, torch.Tensor]:
        """Stack: red cube on top of green cube, stable on table."""
        red_pos = self.env.red_cube.pose.p
        green_pos = self.env.green_cube.pose.p
        
        # Check green cube is on table (z ~ 1.5cm for 3cm cube)
        green_on_table = (green_pos[:, 2] > 0.010) & (green_pos[:, 2] < 0.020)
        
        # Check if red is above green (z difference ~ 3cm = cube size, allow ±0.5cm)
        z_diff = red_pos[:, 2] - green_pos[:, 2]
        z_ok = (z_diff > 0.025) & (z_diff < 0.035)
        
        # Check xy alignment (within 1.5cm for stability)
        xy_dist = torch.norm(red_pos[:, :2] - green_pos[:, :2], dim=1)
        xy_ok = xy_dist < 0.015
        
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
        w_reach = 1.0
        w_grasp = 2.0
        w_lift = 2.0
        w_align = 3.0
        w_place = 5.0
        w_success = 10.0
        
        gripper_pos = self.env._get_gripper_pos()
        red_pos = self.env.red_cube.pose.p
        green_pos = self.env.green_cube.pose.p
        
        # 1. Reach reward
        dist_to_red = torch.norm(gripper_pos - red_pos, dim=1)
        reach_reward = 1.0 - torch.tanh(dist_to_red * 5.0)
        
        # 2. Grasp reward
        is_grasping = dist_to_red < 0.03
        grasp_reward = is_grasping.float()
        
        # 3. Lift reward (red above green)
        z_diff = red_pos[:, 2] - green_pos[:, 2]
        lift_reward = torch.clamp(z_diff, min=0.0, max=0.05)  # Cap at 5cm
        
        # 4. Alignment reward (xy distance between red and green)
        xy_dist = torch.norm(red_pos[:, :2] - green_pos[:, :2], dim=1)
        align_reward = 1.0 - torch.tanh(xy_dist * 10.0)
        
        # 5. Place reward (red on top of green)
        is_stacked = (z_diff > 0.025) & (z_diff < 0.04) & (xy_dist < 0.02)
        place_reward = is_stacked.float()
        
        # 6. Success bonus
        success = info.get("success", torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool))
        success_bonus = success.float()
        
        reward = (w_reach * reach_reward +
                  w_grasp * grasp_reward +
                  w_lift * lift_reward +
                  w_align * align_reward +
                  w_place * place_reward +
                  w_success * success_bonus)
        
        return reward
