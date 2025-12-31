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
    from scripts.envs.track1_env import Track1Env
    from scripts.agents.so101 import SO101
except ImportError as e:
    raise RuntimeError(f"Required modules not found: {e}, Track1 environment not found. Please run `uv run -m scripts.track1_env` to iresolve it.") from e


from typing import TypeVar, Optional, Type

T = TypeVar('T')

def find_wrapper(env, wrapper_type: Type[T]) -> Optional[T]:
    """Traverse wrapper chain to find a specific wrapper type.
    
    Args:
        env: The wrapped environment to search from (outermost wrapper)
        wrapper_type: The wrapper class to find
        
    Returns:
        The wrapper instance if found, None otherwise
        
    Example:
        >>> obs_wrapper = find_wrapper(envs, NormalizeObservationGPU)
        >>> if obs_wrapper is not None:
        ...     print(obs_wrapper.rms.mean)
    """
    curr = env
    while curr is not None:
        if isinstance(curr, wrapper_type):
            return curr
        # Try common wrapper attribute names
        curr = getattr(curr, "env", getattr(curr, "_env", None))
    return None


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

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state_dict):
        self.mean.copy_(state_dict["mean"])
        self.var.copy_(state_dict["var"])
        self.count = state_dict["count"]


class NormalizeObservationGPU(gym.Wrapper):
    """GPU-native observation normalization wrapper."""
    
    def __init__(self, env, device=None, epsilon=1e-8, clip=10.0):
        super().__init__(env)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.clip = clip
        self.update_rms = True
        
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
        if self.update_rms:
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
        self.update_rms = True
        self.rms = RunningMeanStd(shape=(), device=self.device)
        self.returns = torch.zeros(self.base_env.num_envs, device=self.device)

    @property
    def base_env(self):
        return self.env.unwrapped
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Reset returns tracker
        self.returns.zero_()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update discounted returns
        self.returns = self.returns * self.gamma + reward
        
        if self.update_rms:
            self.rms.update(self.returns.unsqueeze(1))
        
        # Normalize reward
        normalized_reward = reward / torch.sqrt(self.rms.var + self.epsilon)
        normalized_reward = torch.clamp(normalized_reward, -self.clip, self.clip)
        
        # Reset returns for done environments
        done = terminated | truncated
        self.returns.mul_( (~done).float() )
        
        # Pass raw reward in info for logging purposes
        info["raw_reward"] = reward
        
        return obs, normalized_reward, terminated, truncated, info

class SingleArmWrapper(gym.Wrapper):
    """Filters observation and maps action for single-arm tasks.
    
    Transforms a multi-arm environment into a single-arm one by:
    1. Exposing only the right arm action space (as a Dict).
    2. Filtering out left arm data from agent observations.
    
    Requires obs_mode='state_dict'.
    """
    def __init__(self, env, right_arm_key=None, left_arm_key=None):
        super().__init__(env)
        
        # Dynamic discovery of agent keys from the environment
        if right_arm_key is None or left_arm_key is None:
            # ManiSkill agent keys are usually sorted in the action space
            # For dual-arm Track1, keys are ['uid-0', 'uid-1']
            all_keys = sorted(self.env.single_action_space.keys())
            
            if len(all_keys) >= 2:
                # Standard dual-arm setup: left is index 0, right is index 1
                left_arm_key = left_arm_key or all_keys[0]
                right_arm_key = right_arm_key or all_keys[1]
            elif len(all_keys) == 1:
                # Single arm setup
                right_arm_key = right_arm_key or all_keys[0]
                left_arm_key = left_arm_key or "none"
            else:
                # Fallback to SO101 defaults if discovery fails
                right_arm_key = right_arm_key or SO101.get_agent_key("right")
                left_arm_key = left_arm_key or SO101.get_agent_key("left")
            
        self.right_arm_key = right_arm_key
        self.left_arm_key = left_arm_key
        
        from gymnasium.vector.utils import batch_space
        
        # Must use state_dict for structural filtering
        assert self.base_env.obs_mode == "state_dict", f"SingleArmWrapper requires state_dict mode, got {self.base_env.obs_mode}"
        
        # Verify the discovery
        if self.right_arm_key not in self.env.single_action_space.spaces:
             raise KeyError(f"SingleArmWrapper: Discovered right_arm_key '{self.right_arm_key}' not found in action space. Available: {list(self.env.single_action_space.keys())}")
        
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
        from gymnasium.vector.utils import batch_space
        
        self._action_names = []
        self._leaf_info = [] # List of (key, start, end, original_shape)
        
        start_idx = 0
        # Use centralized joint names from SO101
        joint_names = SO101.JOINT_NAMES
        
        # Use sorted keys to ensure deterministic flattened ordering
        for k in sorted(self.env.single_action_space.keys()):
            space = self.env.single_action_space[k]
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
            # Use sorted keys for maximum reproducibility across environments/versions
            for key in sorted(obs.keys()):
                value = obs[key]
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


def configure_so101_agent(cfg: DictConfig):
    """Configure SO101 class attributes from config.
    
    Note: Track1Env now uses SO101.create_configured_class() which is more robust,
    but we keep this for any lingering global property needs.
    """
    from scripts.agents.so101 import SO101
    
    task = cfg.env.get("task", "lift")
    SO101.active_mode = "dual" if task == "sort" else "single"
    
    if "control" in cfg and "action_bounds" in cfg.control:
        bounds = OmegaConf.to_container(cfg.control.action_bounds, resolve=True)
        if task == "sort":
            SO101.action_bounds_dual_arm = bounds
        else:
            SO101.action_bounds_single_arm = bounds


def build_sim_config(cfg: DictConfig) -> dict:
    """Build simulation configuration dictionary for ManiSkill."""
    sim_freq = cfg.env.get("sim_freq", 120)
    control_freq = cfg.env.get("control_freq", 30)
    solver_cfg = cfg.env.get("solver", {})
    
    return {
        "sim_freq": sim_freq,
        "control_freq": control_freq,
        "scene_config": {
            "solver_position_iterations": solver_cfg.get("position_iterations", 20),
            "solver_velocity_iterations": solver_cfg.get("velocity_iterations", 1),
        }
    }


def build_env_kwargs(cfg: DictConfig, for_eval: bool, sim_config: dict) -> dict:
    """Build environment keyword arguments."""
    device_id = cfg.get("device_id", 0)
    sim_backend = f"physx_cuda:{device_id}"
    
    if for_eval or (cfg.env.get("need_render", False)):
        render_backend = f"sapien_cuda:{device_id}" 
        render_mode = "sensors"
    else:
        render_mode = None
        render_backend = None
        
    kwargs = dict(
        cfg=cfg,
        eval_mode=for_eval,
        obs_mode=cfg.env.obs_mode, 
        reward_mode=cfg.reward.reward_mode if "reward" in cfg else "sparse",
        control_mode=cfg.control.control_mode,
        sim_config=sim_config,
        sim_backend=sim_backend,
        render_mode=render_mode,
    )
    if render_backend is not None:
        kwargs["render_backend"] = render_backend
        
    return kwargs


def create_base_env(cfg: DictConfig, num_envs: int, for_eval: bool, env_kwargs: dict):
    """Create the base Track1Env instance using gym.make."""
    reconfiguration_freq = 1 if for_eval else None
    max_episode_steps = compute_max_episode_steps(cfg, for_eval)
    
    return gym.make(
        cfg.env.env_id,
        num_envs=num_envs,
        reconfiguration_freq=reconfiguration_freq,
        max_episode_steps=max_episode_steps,
        **env_kwargs
    )


def apply_wrappers(env, cfg: DictConfig, num_envs: int, for_eval: bool, video_dir: str = None):
    """Apply wrappers in correct order."""
    # 1. Single-arm logic (filters actions/obs)
    if cfg.env.task in ["lift", "stack", "static_grasp"]:
        env = SingleArmWrapper(env)
    
    # 2. Spaces flattening (required for training runner)
    env = FlattenActionWrapper(env)
    env = FlattenStateWrapper(env)
    
    # 3. Video/Trajectory recording (eval only, configurable)
    if for_eval and video_dir:
        rec_cfg = cfg.get("recording", {})
        save_video = rec_cfg.get("save_video", True) and cfg.get("capture_video", True)
        save_trajectory = rec_cfg.get("save_trajectory", True)
        
        if save_video or save_trajectory:
            env = RecordEpisode(
                env,
                output_dir=video_dir,
                save_video=save_video,
                save_trajectory=save_trajectory,
                record_env_state=rec_cfg.get("save_env_state", False),
                info_on_video=rec_cfg.get("info_on_video", True),
                video_fps=rec_cfg.get("video_fps", 30),
                save_on_reset=False,
                max_steps_per_video=int(1e6),
                render_substeps=False,
            )
    
    # 4. Vectorization (final layer)
    if for_eval:
        env = ManiSkillVectorEnv(env, num_envs, auto_reset=False, ignore_terminations=True, record_metrics=True)
    else:
        env = ManiSkillVectorEnv(env, num_envs, auto_reset=True, ignore_terminations=False, record_metrics=True)
        
    return env


def make_env(cfg: DictConfig, num_envs: int, for_eval: bool = False, video_dir: str = None):
    """Create Track1 environment with all wrappers."""
    # 1. Configure Agent properties
    configure_so101_agent(cfg)
    
    # 2. Build configurations
    sim_config = build_sim_config(cfg)
    env_kwargs = build_env_kwargs(cfg, for_eval, sim_config)
    
    # 3. Create the base environment
    env = create_base_env(cfg, num_envs, for_eval, env_kwargs)
    
    # 4. Apply wrappers (ordering is critical!)
    env = apply_wrappers(env, cfg, num_envs, for_eval, video_dir)
    
    return env
