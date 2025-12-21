
import gymnasium as gym
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# Import Track1 environment (handle running as module)
try:
    from scripts.track1_env import Track1Env
except ImportError:
    import sys
    import os
    sys.path.append(os.getcwd())
    from scripts.track1_env import Track1Env

class DictArray:
    """Helper class for handling dictionary observations in buffer."""
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                             torch.uint8 if v.dtype == np.uint8 else
                             torch.int32 if v.dtype == np.int32 else v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        else:
            for k, v in value.items():
                self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


class FlattenStateWrapper(gym.ObservationWrapper):
    """Flattens the dict observation into a single vector (for State mode).
    Handles GPU tensors efficiently.
    """
    def __init__(self, env):
        super().__init__(env)
        # Calculate flat dimension based on observation space
        self.flat_dim = self._count_dim(env.observation_space)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.flat_dim,), 
            dtype=np.float32
        )
        
    def _count_dim(self, space):
        d = 0
        if isinstance(space, gym.spaces.Dict):
            for v in space.values():
                d += self._count_dim(v)
        elif isinstance(space, gym.spaces.Box):
            d += np.prod(space.shape)
        return d

    def observation(self, observation):
        return self._flatten_recursive(observation)

    def _flatten_recursive(self, obs):
        tensors = []
        if isinstance(obs, dict):
            for k in sorted(obs.keys()):
                v = obs[k]
                tensors.append(self._flatten_recursive(v))
        else:
            # Assume tensor
            if obs.ndim > 2:
                 v = obs.flatten(start_dim=1)
            else:
                 v = obs
            tensors.append(v)
        return torch.cat(tensors, dim=-1)


def make_env(cfg: DictConfig, num_envs: int, for_eval: bool = False, video_dir: str = None):
    """Create Track1 environment with proper wrappers."""
    # Build reward config from Hydra config
    reward_config = OmegaConf.to_container(cfg.reward, resolve=True) if "reward" in cfg else None
    
    env_kwargs = dict(
        task=cfg.env.task,
        control_mode=cfg.env.control_mode,
        camera_mode=cfg.env.camera_mode,
        obs_mode=cfg.env.obs_mode,
        reward_mode=cfg.reward.reward_mode if "reward" in cfg else "sparse",
        reward_config=reward_config,
        render_mode="all",
        sim_backend="physx_cuda",
    )
    
    reconfiguration_freq = 1 if for_eval else None
    
    env = gym.make(
        cfg.env.env_id,
        num_envs=num_envs,
        reconfiguration_freq=reconfiguration_freq,
        **env_kwargs
    )
    
    # Video Recording (only for eval envs with RGB mode or render_mode)
    if for_eval and video_dir and cfg.capture_video:
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_trajectory=False,
            max_steps_per_video=cfg.training.num_eval_steps,
            video_fps=30
        )

    # Flatten observations logic
    if cfg.env.obs_mode == "state":
        # For state mode, we use our custom FlattenStateWrapper
        env = FlattenStateWrapper(env)
    else:
        # For RGB mode, use ManiSkill's wrapper
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=cfg.env.include_state)
    
    # Wrap with ManiSkillVectorEnv (provides auto-reset, GPU metrics)
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)
    
    return env
