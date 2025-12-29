"""
Common utilities for PPO training.
GPU-native normalization wrappers and environment setup.
"""
import gymnasium as gym
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# Import Track1 environment
try:
    from scripts.track1_env import Track1Env
except ImportError:
    import sys
    import os
    sys.path.append(os.getcwd())
    from scripts.track1_env import Track1Env


class RunningMeanStd:
    """GPU-compatible running mean and standard deviation tracker."""
    
    def __init__(self, shape=(), device=None, epsilon=1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon
        self.device = device
    
    def update(self, x: torch.Tensor):
        """Update statistics with a batch of observations."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class NormalizeObservationGPU(gym.Wrapper):
    """GPU-native observation normalization wrapper."""
    
    def __init__(self, env, device=None, epsilon=1e-8, clip=10.0):
        super().__init__(env)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.clip = clip
        
        # Determine observation shape
        obs_shape = env.single_observation_space.shape
        self.rms = RunningMeanStd(shape=obs_shape, device=self.device)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._normalize(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info
    
    def _normalize(self, obs):
        self.rms.update(obs)
        normalized = (obs - self.rms.mean) / torch.sqrt(self.rms.var + self.epsilon)
        return torch.clamp(normalized, -self.clip, self.clip)


class NormalizeRewardGPU(gym.Wrapper):
    """GPU-native reward normalization wrapper using discounted return variance."""
    
    def __init__(self, env, device=None, gamma=0.99, epsilon=1e-8, clip=10.0):
        super().__init__(env)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        self.rms = RunningMeanStd(shape=(), device=self.device)
        self.returns = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Initialize returns tracker
        num_envs = obs.shape[0]
        self.returns = torch.zeros(num_envs, device=self.device)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update discounted returns
        self.returns = self.returns * self.gamma + reward
        self.rms.update(self.returns.unsqueeze(1))
        
        # Normalize reward
        normalized_reward = reward / torch.sqrt(self.rms.var + self.epsilon)
        normalized_reward = torch.clamp(normalized_reward, -self.clip, self.clip)
        
        # Reset returns for done environments
        done = terminated | truncated
        self.returns = self.returns * (~done).float()
        
        return obs, normalized_reward, terminated, truncated, info

class SingleArmWrapper(gym.Wrapper):
    """Filters observation and maps action for single-arm tasks.
    
    Transforms a multi-arm environment into a single-arm one by:
    1. Exposing only the right arm action space (as a Dict).
    2. Filtering out left arm data from agent observations.
    
    Requires obs_mode='state_dict'.
    """
    def __init__(self, env, right_arm_key="so101-1", left_arm_key="so101-0"):
        super().__init__(env)
        self.right_arm_key = right_arm_key
        self.left_arm_key = left_arm_key
        
        from mani_skill.utils.gym_utils import batch_space
        
        # Must use state_dict for structural filtering
        assert self.base_env.obs_mode == "state_dict", f"SingleArmWrapper requires state_dict mode, got {self.base_env.obs_mode}"
        
        # Keep Action Space as a Dict, but only with the right arm
        # This allows FlattenActionWrapper to handle the flattening and naming later
        self.single_action_space = gym.spaces.Dict({
            self.right_arm_key: self.env.single_action_space[self.right_arm_key]
        })
        self.action_space = batch_space(self.single_action_space, n=self.base_env.num_envs)
        
        # Update observation space by informing BaseEnv of the filtered structure
        raw_obs = self.base_env._init_raw_obs
        filtered_raw_obs = self._filter_obs(raw_obs)
        self.base_env.update_obs_space(filtered_raw_obs)

    @property
    def base_env(self):
        return self.env.unwrapped

    def _filter_obs(self, obs):
        """Remove left arm from agent dict."""
        if "agent" in obs and self.left_arm_key in obs["agent"]:
            agent_dict = obs["agent"]
            filtered_agent = {self.right_arm_key: agent_dict[self.right_arm_key]}
            return {**obs, "agent": filtered_agent}
        return obs

    def action(self, action_dict):
        """Map Filtered Dict Action (right arm only) to Full Multi-Agent Dict."""
        # This receives a dict from FlattenActionWrapper
        right_action = action_dict[self.right_arm_key]
        left_zeros = torch.zeros_like(right_action)
        return {
            self.right_arm_key: right_action,
            self.left_arm_key: left_zeros
        }

    def step(self, action):
        # action here might be a Dict coming from FlattenActionWrapper
        obs, reward, terminated, truncated, info = self.env.step(self.action(action))
        return self._filter_obs(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._filter_obs(obs), info


class FlattenActionWrapper(gym.ActionWrapper):
    """Flattens Dict action space into Tensor with Semantic joint name tracking.
    
    Calculates action_names (e.g., 'action/so101-1/shoulder_pan') based on agent metadata.
    Designed for torch.compile optimizations.
    """
    def __init__(self, env):
        super().__init__(env)
        from mani_skill.utils.gym_utils import batch_space
        
        self._action_names = []
        self._leaf_info = [] # List of (key, start, end, original_shape)
        
        start_idx = 0
        # Determine joint names from the agent if possible
        # For ManiSkill Track1, we know the SO101 joint order
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        
        for k, space in self.env.single_action_space.items():
            dim = int(np.prod(space.shape))
            end_idx = start_idx + dim
            
            # Record joint names
            if dim == len(joint_names):
                self._action_names.extend([f"action/{k}/{j}" for j in joint_names])
            else:
                self._action_names.extend([f"action/{k}_{i}" for i in range(dim)])
                
            self._leaf_info.append((k, start_idx, end_idx, space.shape))
            start_idx = end_idx
            
        # Update action space to flattened Box
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(start_idx,), dtype=np.float32
        )
        self.action_space = batch_space(self.single_action_space, n=self.base_env.num_envs)
        
        # Optimize action method
        if hasattr(torch, "compile"):
            self.action = torch.compile(self.action, mode="reduce-overhead")
            
        print(f"[FlattenActionWrapper] Flattened {len(self._leaf_info)} agents into {start_idx} action dims.")

    @property
    def base_env(self):
        return self.env.unwrapped
        
    @property
    def action_names(self):
        return self._action_names

    def action(self, action_tensor):
        """Unflatten Tensor to Dict for underlying environment.
        
        Optimized for torch.compile (static slice and dict creation).
        """
        results = {}
        for key, start, end, shape in self._leaf_info:
            # Slice and reshape if necessary
            val = action_tensor[:, start:end]
            if len(shape) > 1:
                val = val.reshape(-1, *shape)
            results[key] = val
        return results


#NOTE： this is only suited for state-dict obs mode on gpu tensor cases
#NOTE: add support for rgb if we need one day. ask human if agent want to change here.
class FlattenStateWrapper(gym.ObservationWrapper):
    """Flattens dict observations into a single vector, with dimension name tracking.
    
    This wrapper discovers the structure of the observation dictionary during initialization
    and pre-records the paths to each leaf tensor. This ensures that:
    1. Label order (obs_names) and value order are perfectly aligned.
    2. Extraction is high-performance and torch.compile-friendly (no recursion in step).
    
    Attributes:
        obs_names: List of descriptive names for each flattened dimension.
    """

    
    def __init__(self, env) -> None:
        super().__init__(env)
        from mani_skill.envs.sapien_env import BaseEnv
        
        # Get base env for accessing raw obs structure
        base_env: BaseEnv = self.env.unwrapped
        raw_obs = base_env._init_raw_obs
        
        # Discover structure and record paths
        self._leaf_paths = []
        self._obs_names = []
        self._discover_structure(raw_obs)
        
        # Calculate total dimension and update space
        total_dim = len(self._obs_names)
        base_env.update_obs_space(torch.zeros((base_env.num_envs, total_dim), device=base_env.device))
        
        # Compile the observation method for maximum performance
        # PyTorch will unroll the path-following loop during compilation
        if hasattr(torch, "compile"):
            self.observation = torch.compile(self.observation, mode="reduce-overhead")
        
        print(f"[FlattenStateWrapper] Flattened {len(self._leaf_paths)} leaf tensors into {total_dim} dimensions.")
    
    @property
    def base_env(self):
        return self.env.unwrapped
    
    @property
    def obs_names(self) -> list:
        """List of descriptive names for each flattened observation dimension."""
        return self._obs_names
    
    def _discover_structure(self, obs, path=()):
        """Recursively walk the dict to find tensors and record their paths/names."""
        if isinstance(obs, dict):
            # Respect dictionary insertion order (consistent with ManiSkill)
            for key, value in obs.items():
                self._discover_structure(value, path + (key,))
        else:
            # Leaf tensor
            name_prefix = "/".join(path)
            
            # Determine dimensions (B, D...)
            if hasattr(obs, 'shape'):
                # In ManiSkill GPU mode, 1st dimension is always num_envs (batch)
                dim = int(np.prod(obs.shape[1:])) if len(obs.shape) > 1 else 1
            else:
                dim = 1
            
            # Record extraction path
            self._leaf_paths.append(path)
            
            # Generate names for each dimension of this leaf
            if dim == 1:
                self._obs_names.append(name_prefix)
            else:
                for i in range(dim):
                    self._obs_names.append(f"{name_prefix}_{i}")
    
    def observation(self, obs):
        """Perform flattening by following pre-recorded paths.
        
        This loop is easily unrolled/optimized by torch.compile.
        """
        tensors = []
        for path in self._leaf_paths:
            val = obs
            # Manual traversal is fast and compile-friendly
            for key in path:
                val = val[key]
            
            # Standardize to (B, D) tensors
            if val.ndim == 1:
                # [B] -> [B, 1]
                val = val[:, None]
            elif val.ndim > 2:
                # [B, H, W, ...] -> [B, H*W*...]
                val = val.flatten(start_dim=1)
            
            tensors.append(val)
        
        # Final concatenation across features (dim -1)
        return torch.cat(tensors, dim=-1)


def compute_max_episode_steps(cfg: DictConfig, for_eval: bool = False) -> int:
    """Compute max episode steps from config.
    
    Formula: (base * multiplier) + hold_steps, optionally scaled for eval.
    """
    if "episode_steps" in cfg.env:
        base = cfg.env.episode_steps.get("base", 296)
        multiplier = cfg.env.episode_steps.get("multiplier", 1.2)
        max_steps = int(base * multiplier)
        
        # Add stable hold time from reward config (if present)
        if "reward" in cfg and "stable_hold_time" in cfg.reward:
            control_freq = cfg.env.get("control_freq", 30)
            hold_steps = int(cfg.reward.stable_hold_time * control_freq)
            max_steps += hold_steps
        
        # Scale for evaluation if needed
        if for_eval:
            eval_multiplier = cfg.training.get("eval_step_multiplier", 1.0)
            max_steps = int(max_steps * eval_multiplier)
        
        return max_steps
    else:
        return cfg.env.get("max_episode_steps", None)


def make_env(cfg: DictConfig, num_envs: int, for_eval: bool = False, video_dir: str = None):
    """Create Track1 environment with proper wrappers."""
    
    # Build sim_config dict for ManiSkill (BaseEnv needs this, not Track1Env-specific)
    sim_freq = cfg.env.get("sim_freq", 120)
    control_freq = cfg.env.get("control_freq", 30)
    solver_cfg = cfg.env.get("solver", {})
    
    sim_config = {
        "sim_freq": sim_freq,
        "control_freq": control_freq,
        "scene_config": {
            "solver_position_iterations": solver_cfg.get("position_iterations", 20),
            "solver_velocity_iterations": solver_cfg.get("velocity_iterations", 1),
        }
    }
    
    # Configure SO101 class attributes (urdf_path, gripper physics) before environment creation
    from scripts.so101 import SO101
    SO101.configure_from_cfg(cfg)
    
    # Get device ID for GPU selection
    device_id = cfg.get("device_id", 0)
    sim_backend = f"physx_cuda:{device_id}"
    if for_eval or (cfg.env.get("need_render", False)):
        render_backend = f"sapien_cuda:{device_id}" 
        render_mode = "sensors"
    else:
        render_mode = None
        render_backend = None
    
    assert cfg.env.obs_mode == "state_dict", "Only state_dict is supported for now"
    # Build env_kwargs: BaseEnv params + cfg for Track1Env to parse
    env_kwargs = dict(
        # Track1Env extracts its config from cfg
        cfg=cfg,
        eval_mode=for_eval,
        # BaseEnv params
        obs_mode=cfg.env.obs_mode, 
        reward_mode=cfg.reward.reward_mode if "reward" in cfg else "sparse",
        control_mode=cfg.control.control_mode,
        sim_config=sim_config,
        render_mode=render_mode,
        sim_backend=sim_backend,
        render_backend=render_backend,
    )

    # from mani_skill.utils.wrappers import FlattenObservationWrapper
    # from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
    # from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper
    
    reconfiguration_freq = 1 if for_eval else None
    
    max_episode_steps = compute_max_episode_steps(cfg, for_eval)
    
    env = gym.make(
        cfg.env.env_id,
        num_envs=num_envs,
        reconfiguration_freq=reconfiguration_freq,
        max_episode_steps=max_episode_steps,
        **env_kwargs
    )
    
    # Decouple single-arm logic using a wrapper for lift/stack tasks
    if cfg.env.task in ["lift", "stack"]:
        env = SingleArmWrapper(env)
    
    # Flatten Action space and track joint names
    env = FlattenActionWrapper(env)

    # Flatten state_dict observations to tensor (with obs_names tracking)
    env = FlattenStateWrapper(env)

    # Video Recording (only for eval envs with capture_video enabled)
    # Note: RecordEpisode should be AFTER observation wrappers per ManiSkill docs
    if for_eval and video_dir and cfg.capture_video:
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_trajectory=False,
            save_on_reset=False,
            max_steps_per_video=int(1e6),
            video_fps=30,
            info_on_video=True,
            render_substeps=False,
        )

    # Wrap with ManiSkillVectorEnv (different config for training vs eval)
    if for_eval:
        # Eval: ignore terminations to run full episodes, record metrics for final_info
        env = ManiSkillVectorEnv(env, num_envs, auto_reset=False,ignore_terminations=True, record_metrics=True)
    else:
        # Training: auto-reset on terminations (fail/success triggers reset)
        env = ManiSkillVectorEnv(env, num_envs, auto_reset=True, ignore_terminations=False, record_metrics= True)
    
    return env
