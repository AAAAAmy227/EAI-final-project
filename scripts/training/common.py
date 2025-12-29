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


class FlattenStateWrapper(gym.ObservationWrapper):
    """Flattens dict observations into a single vector, with dimension name tracking.
    
    Uses ManiSkill's flatten_state_dict for consistent ordering with their internal logic.
    Records obs_names during init for logging purposes.
    
    Attributes:
        obs_names: List of descriptive names for each flattened dimension.
    """
    
    def __init__(self, env) -> None:
        super().__init__(env)
        from mani_skill.utils import common as ms_common
        from mani_skill.envs.sapien_env import BaseEnv
        
        # Get base env for accessing raw obs structure
        base_env: BaseEnv = self.env.unwrapped
        
        # Get initial raw observation to determine structure and names
        # _init_raw_obs is set by ManiSkill after first reset
        raw_obs = base_env._init_raw_obs
        
        # Flatten to get the structure and update observation space
        flattened = ms_common.flatten_state_dict(raw_obs)
        base_env.update_obs_space(flattened)
        
        # Extract dimension names by walking the raw obs dict
        self._obs_names = self._extract_names(raw_obs)
        
        # Verify count matches
        expected_dim = flattened.shape[-1] if hasattr(flattened, 'shape') else len(flattened)
        if len(self._obs_names) != expected_dim:
            print(f"Warning: obs_names count ({len(self._obs_names)}) != flattened dim ({expected_dim})")
            print(f"Falling back to generic naming")
            self._obs_names = [f"obs_{i}" for i in range(expected_dim)]
    
    @property
    def base_env(self):
        return self.env.unwrapped
    
    @property
    def obs_names(self) -> list:
        """List of descriptive names for each flattened observation dimension."""
        return self._obs_names
    
    def _extract_names(self, obs_dict, prefix="") -> list:
        """Recursively extract names from observation dict structure.
        
        Uses same traversal order as ManiSkill's flatten_state_dict (dict key order).
        """
        from mani_skill.utils import common as ms_common
        names = []
        
        if isinstance(obs_dict, dict):
            for key in obs_dict.keys():  # Same order as flatten_state_dict
                new_prefix = f"{prefix}/{key}" if prefix else key
                names.extend(self._extract_names(obs_dict[key], new_prefix))
        else:
            # Leaf tensor - add one name per element
            if hasattr(obs_dict, 'shape'):
                # Handle batched tensors: shape is (B, *dims)
                # Get the non-batch dimensions
                if len(obs_dict.shape) > 1:
                    dim = int(np.prod(obs_dict.shape[1:]))
                else:
                    dim = 1
            else:
                dim = 1
            
            if dim == 1:
                names.append(prefix)
            else:
                for i in range(dim):
                    names.append(f"{prefix}_{i}")
        
        return names
    
    def observation(self, observation):
        """Flatten observation dict to tensor using ManiSkill's function."""
        from mani_skill.utils import common as ms_common
        return ms_common.flatten_state_dict(observation, use_torch=True)


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

    from mani_skill.utils.wrappers import FlattenObservationWrapper
    from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
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
            info_on_video=False,
        )

    # Wrap with ManiSkillVectorEnv (different config for training vs eval)
    if for_eval:
        # Eval: ignore terminations to run full episodes, record metrics for final_info
        env = ManiSkillVectorEnv(env, num_envs, auto_reset=False,ignore_terminations=True, record_metrics=True)
    else:
        # Training: auto-reset on terminations (fail/success triggers reset)
        env = ManiSkillVectorEnv(env, num_envs, auto_reset=True, ignore_terminations=False, record_metrics= True)
    
    return env
