import torch
from typing import Dict, Optional
from scripts.tasks.base import BaseTaskHandler
from mani_skill.utils import randomization
from mani_skill.utils.structs.pose import Pose

class SortTaskHandler(BaseTaskHandler):
    def initialize_episode(self, env_idx, options):
        b = len(env_idx)
        # Both cubes in Mid Grid
        red_pos, red_quat = self.env._random_grid_position(b, self.env.grid_bounds["mid"], z=0.015)
        green_pos, green_quat = self.env._random_grid_position(b, self.env.grid_bounds["mid"], z=0.005)  # Smaller green cube
        self.env.red_cube.set_pose(Pose.create_from_pq(p=red_pos, q=red_quat))
        self.env.green_cube.set_pose(Pose.create_from_pq(p=green_pos, q=green_quat))
        
        # Store initial cube full poses
        if not hasattr(self, "initial_red_cube_pos"):
            self.initial_red_cube_pos = torch.zeros((self.env.num_envs, 3), device=self.device)
        if not hasattr(self, "initial_green_cube_pos"):
            self.initial_green_cube_pos = torch.zeros((self.env.num_envs, 3), device=self.device)
        
        self.initial_red_cube_pos[env_idx] = red_pos
        self.initial_green_cube_pos[env_idx] = green_pos

    def evaluate(self) -> Dict[str, torch.Tensor]:
        """Sort: green in Left Grid, red in Right Grid."""
        red_pos = self.env.red_cube.pose.p
        green_pos = self.env.green_cube.pose.p
        
        # Check red in Right Grid
        red_in_right = (
            (red_pos[:, 0] >= self.env.grid_bounds["right"]["x_min"]) & 
            (red_pos[:, 0] <= self.env.grid_bounds["right"]["x_max"]) &
            (red_pos[:, 1] >= self.env.grid_bounds["right"]["y_min"]) & 
            (red_pos[:, 1] <= self.env.grid_bounds["right"]["y_max"])
        )
        
        # Check green in Left Grid
        green_in_left = (
            (green_pos[:, 0] >= self.env.grid_bounds["left"]["x_min"]) & 
            (green_pos[:, 0] <= self.env.grid_bounds["left"]["x_max"]) &
            (green_pos[:, 1] >= self.env.grid_bounds["left"]["y_min"]) & 
            (green_pos[:, 1] <= self.env.grid_bounds["left"]["y_max"])
        )
        
        success = red_in_right & green_in_left
        # Failure conditions
        fallen_threshold = -0.05
        red_fallen = self.env.red_cube.pose.p[:, 2] < fallen_threshold
        green_fallen = self.env.green_cube.pose.p[:, 2] < fallen_threshold
        fail = red_fallen | green_fallen
        
        success = success & (~fail)
        
        return {
            "success": success, 
            "fail": fail,
            "red_in_right": red_in_right, 
            "green_in_left": green_in_left
        }

    def compute_dense_reward(self, info: dict, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Dense reward for Sort task (placeholder)."""
        success = info.get("success", torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool))
        return success.float() * 10.0
