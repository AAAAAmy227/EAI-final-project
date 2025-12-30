"""
LeanRL-style PPO Runner with tensordict, torch.compile, and CudaGraphModule.
Based on LeanRL/cleanrl ppo_continuous_action_torchcompile.py
"""
import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import math
import random
import time
from collections import deque
from functools import partial
from pathlib import Path
import sys
import subprocess
import threading
import copy

import gymnasium as gym
import hydra
import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
from omegaconf import OmegaConf
from tensordict import from_module
from tensordict.nn import CudaGraphModule

from scripts.training.agent import Agent
from scripts.training.env_utils import make_env
from scripts.training.ppo_utils import optimized_gae, make_ppo_update_fn

class PPORunner:
    def __init__(self, cfg, eval_only: bool = False):
        """Initialize PPO Runner.
        
        Args:
            cfg: Hydra configuration object
            eval_only: If True, skip training environment creation (for standalone eval)
        """
        self.cfg = cfg
        self.eval_only = eval_only
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Hyperparameters
        self.num_envs = cfg.training.num_envs
        self.num_steps = cfg.training.num_steps
        self.total_timesteps = cfg.training.total_timesteps
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // cfg.ppo.num_minibatches
        self.num_iterations = self.total_timesteps // self.batch_size
        
        # Compile settings
        self.compile = cfg.get("compile", True)
        self.cudagraphs = cfg.get("cudagraphs", True)
        self.anneal_lr = cfg.get("anneal_lr", True)
        
        # Seeding
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = True
        
        # Training env setup (skip for eval_only mode)
        if not eval_only:
            self.envs = make_env(cfg, self.num_envs)
        else:
            self.envs = None
        
        # Eval env with video recording
        self.video_dir = None
        if cfg.capture_video:
            output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            self.video_dir = str(output_dir / "videos")
        self.eval_envs = make_env(cfg, cfg.training.num_eval_envs, for_eval=True, video_dir=self.video_dir)
        self.eval_count = 0  # Counter for eval runs (eval0, eval1, eval2, ...)
        
        # Determine observation/action dimensions (use eval_envs if training envs not created)
        source_env = self.envs if self.envs is not None else self.eval_envs
        obs_space = source_env.single_observation_space
        act_space = source_env.single_action_space
        print(f"Observation space: {obs_space}")
        print(f"Action space: {act_space}")
        
        # Handle different observation modes (Dict vs Box)
        if hasattr(obs_space, "shape") and obs_space.shape is not None:
            self.n_obs = math.prod(obs_space.shape)
        elif isinstance(obs_space, gym.spaces.Dict):
            self.n_obs = sum(math.prod(s.shape) for s in obs_space.values())
        else:
            self.n_obs = sum(math.prod(s.shape) for s in obs_space.spaces.values()) if hasattr(obs_space, "spaces") else 0
            
        if hasattr(act_space, "shape") and act_space.shape is not None:
            self.n_act = math.prod(act_space.shape)
        else:
            self.n_act = sum(math.prod(s.shape) for s in act_space.spaces.values()) if hasattr(act_space, "spaces") else 0
        
        print(f"n_obs: {self.n_obs}, n_act: {self.n_act}")
        
        # Agent setup
        logstd_init = cfg.ppo.get("logstd_init", 0.0)
        self.agent = Agent(self.n_obs, self.n_act, device=self.device, logstd_init=logstd_init)
        
        # Create inference agent only when using CudaGraphModule
        # (CudaGraph captures weights, so we need a separate copy for inference)
        if self.cudagraphs:
            self.agent_inference = Agent(self.n_obs, self.n_act, device=self.device, logstd_init=logstd_init)
            from_module(self.agent).data.to_module(self.agent_inference)
        else:
            self.agent_inference = None  # Not needed without CudaGraph
        
        # Optimizer with fused and capturable for maximum performance
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=torch.tensor(cfg.ppo.learning_rate, device=self.device),
            eps=1e-5,
            fused=True,
            capturable=self.cudagraphs and not self.compile,
        )
        
        if cfg.checkpoint:
            print(f"Loading checkpoint from {cfg.checkpoint}")
            ckpt = torch.load(cfg.checkpoint, map_location=self.device)
            self.agent.load_state_dict(ckpt["agent"])
            if self.normalize_obs and "obs_ema_mean" in ckpt:
                self.obs_ema_mean.copy_(ckpt["obs_ema_mean"])
                self.obs_ema_var.copy_(ckpt["obs_ema_var"])
                print("Loaded observation normalization statistics from checkpoint.")
            if self.cudagraphs:
                from_module(self.agent).data.to_module(self.agent_inference)
        
        # Setup compiled functions
        self._setup_compiled_functions()
        
        # Runtime vars
        self.global_step = 0
        self.avg_returns = deque(maxlen=20)
        self.reward_component_sum = {}  # Accumulated reward components
        self.reward_component_count = 0  # Step count for averaging
        self.success_count = torch.tensor(0, device=self.device, dtype=torch.float32)  # GPU accumulator
        self.fail_count = torch.tensor(0, device=self.device, dtype=torch.float32)  # GPU accumulator
        
        # Termination tracking for logging
        self.terminated_count = 0
        self.truncated_count = 0
        
        # Episode return tracking (per-env accumulator)
        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        self.global_step_burnin = None
        
        # Handle timeout termination (if True, truncated episodes bootstrap)
        self.handle_timeout_termination = cfg.ppo.get("handle_timeout_termination", True)
        
        # Manage Observation Names first (needed for normalization setup)
        self.obs_names = self._get_obs_names_from_wrapper()
        if len(self.obs_names) != self.n_obs:
            print(f"Warning: obs_names count ({len(self.obs_names)}) does not match n_obs ({self.n_obs}).")
            self.obs_names = [f"obs_{i}" for i in range(self.n_obs)]

        # Running Reward Normalization
        self.normalize_reward = self.cfg.get("normalize_reward", False)
        self.reward_clip = self.cfg.get("reward_clip", 10.0)
        if self.normalize_reward:
            from scripts.training.env_utils import NormalizeRewardGPU
            self.envs = NormalizeRewardGPU(
                self.envs, device=self.device, gamma=self.cfg.ppo.gamma, clip=self.reward_clip
            )
            print(f"Running Reward Normalization enabled (gamma={self.cfg.ppo.gamma}, clip={self.reward_clip})")
        
        # Observation statistics logging and normalization
        self.log_obs_stats = self.cfg.get("log_obs_stats", False)
        self.normalize_obs = self.cfg.get("normalize_obs", False)
        self.obs_clip = self.cfg.get("obs_clip", 10.0)
        if self.normalize_obs:
            from scripts.training.env_utils import NormalizeObservationGPU
            self.envs = NormalizeObservationGPU(
                self.envs, device=self.device, clip=self.obs_clip
            )
            print(f"Running Observation Normalization enabled (clip={self.obs_clip})")
            
            # Wrap Eval Envs: shared stats, but don't update them during eval
            self.eval_envs = NormalizeObservationGPU(
                self.eval_envs, device=self.device, clip=self.obs_clip
            )
            self.eval_envs.update_rms = False
            self.eval_envs.rms = self.envs.rms # Shared instance!
        
        # Optional: Initialize stats from config if requested
        if self.cfg.get("init_obs_stats_from_config", False):
            self._initialize_obs_stats_from_config()
        
        print(f"Observation names for logging (count: {len(self.obs_names)})")
        
        # Reward mode for logging
        self.reward_mode = self.cfg.reward.get("reward_mode", "sparse")
        self.staged_reward = self.reward_mode == "staged_dense"
        
        # Manage Action Names for logging
        self.action_names = self._get_action_names_from_wrapper()
        if len(self.action_names) != self.n_act:
            print(f"Warning: action_names count ({len(self.action_names)}) does not match n_act ({self.n_act}).")
            self.action_names = [f"act_{i}" for i in range(self.n_act)]
        print(f"Action names for logging (count: {len(self.action_names)})")
        
        # Async eval infrastructure
        self.async_eval = cfg.get("async_eval", True)  # Enable async eval by default
        self.eval_thread = None
        self.eval_stream = torch.cuda.Stream() if self.async_eval else None
        # Create separate eval agent to avoid race condition with training agent
        if self.async_eval:
            self.eval_agent = Agent(self.n_obs, self.n_act, device=self.device)
        else:
            self.eval_agent = None

    def _setup_compiled_functions(self):
        """Setup torch.compile and CudaGraphModule."""
        cfg = self.cfg
        
        # 1. Policy (Inference)
        # Use agent_inference for CudaGraph (needs separate copy for weight sync)
        # Use agent directly for torch.compile (dynamic, no capture)
        inference_agent = self.agent_inference if self.cudagraphs else self.agent
        self.policy = inference_agent.get_action_and_value
        self.get_value = inference_agent.get_value
        
        # 2. GAE: Use functools.partial to bind gamma/gae_lambda
        self.gae_fn = partial(
            optimized_gae,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda
        )

        # 3. Update: Use factory function from ppo_utils
        self.update_fn = make_ppo_update_fn(self.agent, self.optimizer, cfg)
        
        if self.compile:
            print("Compiling functions...")
            if self.cudagraphs:
                # When using CudaGraphModule, use default compile mode (not reduce-overhead)
                # reduce-overhead internally uses CUDA graphs which conflicts with CudaGraphModule
                self.policy = torch.compile(self.policy)
            else:
                # When not using CudaGraphModule, reduce-overhead is safe
                self.policy = torch.compile(self.policy, mode="reduce-overhead")
            # get_value: compile for consistency (called once per iteration)
            self.get_value = torch.compile(self.get_value, mode="reduce-overhead")
            # Update: Always use reduce-overhead (no CudaGraphModule on update)
            self.update_fn = torch.compile(self.update_fn, mode="reduce-overhead")
        
        if self.cudagraphs:
            print("Applying CudaGraphModule to Policy (Inference Only)...")
            self.policy = CudaGraphModule(self.policy)

    def _get_obs_names_from_wrapper(self) -> list:
        """Get observation names from FlattenStateWrapper.
        
        Traverses the wrapper chain to find FlattenStateWrapper and return its obs_names.
        Falls back to generic naming if not found.
        """
        from scripts.training.env_utils import FlattenStateWrapper, find_wrapper
        
        wrapper = find_wrapper(self.envs, FlattenStateWrapper)
        if wrapper is not None:
            return wrapper.obs_names
        
        # Fallback to generic names
        print("Warning: FlattenStateWrapper not found, using generic obs names")
        return [f"obs_{i}" for i in range(self.n_obs)]

    def _get_action_names_from_wrapper(self) -> list:
        """Get action names from FlattenActionWrapper.
        
        Traverses the wrapper chain to find FlattenActionWrapper and return its action_names.
        Falls back to generic naming if not found.
        """
        from scripts.training.env_utils import FlattenActionWrapper, find_wrapper
        
        wrapper = find_wrapper(self.envs, FlattenActionWrapper)
        if wrapper is not None:
            return wrapper.action_names
        
        # Fallback to generic names
        print("Warning: FlattenActionWrapper not found, using generic action names")
        return [f"act_{i}" for i in range(self.n_act)]

    def _initialize_obs_stats_from_config(self):
        """Initialize obs RMS from environment config."""
        from scripts.training.env_utils import NormalizeObservationGPU, find_wrapper
        
        # Find NormalizeObservationGPU wrapper
        obs_wrapper = find_wrapper(self.envs, NormalizeObservationGPU)
            
        if obs_wrapper is None:
            return

        if "obs" not in self.cfg:
            return
        
        obs_cfg = self.cfg.obs
        print("Initializing observation statistics from config...")
        
        # Helper to find indices by name pattern
        def get_indices(pattern):
            return [i for i, name in enumerate(self.obs_names) if pattern in name]
        
        # Logic for common Track1 features
        for key in obs_cfg.keys():
            if hasattr(obs_cfg[key], "mean") and hasattr(obs_cfg[key], "std"):
                idxs = get_indices(key)
                if len(idxs) > 0:
                    mean_val = torch.tensor(obs_cfg[key].mean, device=self.device)
                    std_val = torch.tensor(obs_cfg[key].std, device=self.device)
                    
                    # Handle broadcasting if size matches
                    if mean_val.numel() == len(idxs):
                        obs_wrapper.rms.mean[idxs] = mean_val
                        obs_wrapper.rms.var[idxs] = std_val ** 2
                        print(f"  Initialized {key} stats at indices {idxs}")
                    elif mean_val.numel() == 1:
                        obs_wrapper.rms.mean[idxs] = mean_val
                        obs_wrapper.rms.var[idxs] = std_val ** 2
                        print(f"  Initialized {key} stats (scalar broadcast) at indices {idxs}")



    def _step_env(self, action):
        """Execute environment step.
        
        Returns:
            next_obs, reward, terminated, truncated, done, info
            where done = terminated | truncated (for episode boundary tracking)
        """
        next_obs, reward, terminations, truncations, info = self.envs.step(action)
        done = terminations | truncations
        return next_obs, reward, terminations, truncations, done, info

    def _rollout(self, obs):
        """Collect trajectories with pre-allocated storage.
        
        Args:
            obs: Initial observations for this rollout
        """
        # 1. Pre-allocate TensorDict (Zero-copy optimization)
        storage = tensordict.TensorDict({
            "obs": torch.empty((self.num_steps, self.num_envs, self.n_obs), device=self.device, dtype=obs.dtype),
            "bootstrap_mask": torch.empty((self.num_steps, self.num_envs), device=self.device, dtype=torch.bool),
            "vals": torch.empty((self.num_steps, self.num_envs), device=self.device),
            "actions": torch.empty((self.num_steps, self.num_envs, self.n_act), device=self.device),
            "logprobs": torch.empty((self.num_steps, self.num_envs), device=self.device),
            "rewards": torch.empty((self.num_steps, self.num_envs), device=self.device),
        }, batch_size=[self.num_steps, self.num_envs], device=self.device)

        for step in range(self.num_steps):
            # 2. Store current state before taking action
            storage["obs"][step] = obs
            
            # Inference (Normalization is already in obs)
            with torch.no_grad():
                action, logprob, _, value = self.policy(obs=obs)
            storage["vals"][step] = value.flatten()
            storage["actions"][step] = action
            storage["logprobs"][step] = logprob
            
            # Environment Step
            next_obs, reward, next_terminated, next_truncated, next_done, infos = self._step_env(action)
            
            # 3. Store result of the step at current index (POST-step storage)
            # This makes storage["bootstrap_mask"][step] the status of next_obs (s_{t+1})
            storage["rewards"][step] = reward
            storage["bootstrap_mask"][step] = next_terminated if self.handle_timeout_termination else next_done
            
            # Accumulate RAW reward for logging
            raw_reward = infos.get("raw_reward", reward)
            self.episode_returns += raw_reward
            
            # Record reward components using info_utils
            from scripts.training.info_utils import get_reward_components, get_info_field, extract_scalar
            reward_comps = get_reward_components(infos)
            if reward_comps is not None:
                for k, v in reward_comps.items():
                    val = extract_scalar(v)
                    if val is not None:
                        self.reward_component_sum[k] = self.reward_component_sum.get(k, 0) + val
            self.reward_component_count += 1

            # Success/Fail counts using info_utils
            s_val = get_info_field(infos, "success_count")
            if s_val is not None:
                self.success_count += s_val
                
            f_val = get_info_field(infos, "fail_count")
            if f_val is not None:
                self.fail_count += f_val

            # Log episodic returns
            if next_done.any():
                for idx in torch.where(next_done)[0]:
                    if next_terminated[idx].item():
                        self.terminated_count += 1
                    else:
                        self.truncated_count += 1
                    
                    self.avg_returns.append(self.episode_returns[idx].item())
                    self.episode_returns[idx] = 0.0

            # Update for next iteration
            obs = next_obs
        
        # After loop, current bootstrap_mask for the START of the next rollout is the last next_done
        last_bootstrap_mask = next_terminated if self.handle_timeout_termination else next_done
        
        storage["rewards"] *= self.cfg.ppo.reward_scale
        return obs, last_bootstrap_mask, storage

    def train(self):
        print(f"\n{'='*60}")
        print(f"Training PPO (LeanRL-style) on Track1 {self.cfg.env.task}")
        print(f"Device: {self.device}, Compile: {self.compile}, CudaGraphs: {self.cudagraphs}")
        print(f"Total Timesteps: {self.total_timesteps}, Batch Size: {self.batch_size}")
        print(f"{'='*60}\n")
        
        # Initial reset
        next_obs, _ = self.envs.reset(seed=self.cfg.seed)
        next_bootstrap_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        pbar = tqdm.tqdm(range(1, self.num_iterations + 1))
        self.global_step_burnin = None
        training_time = 0.0
        measure_burnin = 2
        
        for iteration in pbar:
            if iteration == measure_burnin:
                self.global_step_burnin = self.global_step
                training_time = 0.0
            
            iter_start_time = time.time()
            
            # Learning rate annealing
            self._schedule_learning_rate(iteration)
            
            # Cudagraph marker
            torch.compiler.cudagraph_mark_step_begin()
            
            # Rollout
            next_obs, next_bootstrap_mask, container = self._rollout(next_obs)
            self.global_step += container.numel()
            
            # GAE calculation
            container = self._compute_gae(container, next_obs, next_bootstrap_mask)
            
            # PPO update
            out, clipfracs = self._run_ppo_update(container)
            
            # Sync params to inference agent (CudaGraph only)
            if self.cudagraphs:
                from_module(self.agent).data.to_module(self.agent_inference)
            
            # Logging
            if self.global_step_burnin is not None and training_time > 0:
                self._log_training_metrics(iteration, container, out, clipfracs, training_time)
                avg_return = np.mean(self.avg_returns) if self.avg_returns else 0
                pbar.set_description(f"SPS: {(self.global_step - self.global_step_burnin) / training_time:.0f}, return: {avg_return:.2f}")
            
            # Accumulate training time
            training_time += time.time() - iter_start_time
            
            # Evaluation
            if iteration % self.cfg.training.eval_freq == 0:
                self._handle_evaluation(iteration)
        
        # Cleanup
        if self.async_eval and self.eval_thread is not None and self.eval_thread.is_alive():
            print("Waiting for async eval to complete...")
            self.eval_thread.join()
        
        self.envs.close()
        self.eval_envs.close()
        print("Training complete!")

    def _schedule_learning_rate(self, iteration: int):
        """Anneal learning rate linearly over training."""
        if self.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / self.num_iterations
            lrnow = frac * self.cfg.ppo.learning_rate
            self.optimizer.param_groups[0]["lr"].copy_(lrnow)

    def _compute_gae(self, container, next_obs, next_bootstrap_mask):
        """Compute GAE advantages and returns.
        
        Args:
            container: TensorDict with rollout data (rewards, vals, bootstrap_mask)
            next_obs: Final observations after rollout
            next_bootstrap_mask: Bootstrap mask for final step
            
        Returns:
            Updated container with 'advantages' and 'returns' added
        """
        with torch.no_grad():
            # next_obs is already normalized by wrapper
            next_value = self.get_value(next_obs)
        
        advs, rets = self.gae_fn(
            container["rewards"],
            container["vals"],
            container["bootstrap_mask"],
            next_value,
            next_bootstrap_mask
        )
        container["advantages"] = advs
        container["returns"] = rets
        return container

    def _run_ppo_update(self, container):
        """Run PPO update epochs.
        
        Args:
            container: TensorDict with obs, actions, logprobs, advantages, returns
            
        Returns:
            dict with final update metrics (v_loss, pg_loss, entropy_loss, etc.)
            list of clipfracs from all minibatches
        """
        container_flat = container.view(-1)
        clipfracs = []
        
        for epoch in range(self.cfg.ppo.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=self.device).split(self.minibatch_size)
            for b in b_inds:
                container_local = container_flat[b]
                out = self.update_fn(container_local, tensordict_out=tensordict.TensorDict())
                clipfracs.append(out["clipfrac"].item())
                
                if self.cfg.ppo.target_kl is not None and out["approx_kl"] > self.cfg.ppo.target_kl:
                    break
            else:
                continue
            break
        
        return out, clipfracs

    def _log_training_metrics(self, iteration: int, container, out, clipfracs, training_time: float):
        """Build and log training metrics to console and WandB.
        
        Args:
            iteration: Current training iteration
            container: TensorDict with rollout data
            out: Dict with final PPO update metrics
            clipfracs: List of clip fractions from all minibatches
            training_time: Accumulated training time in seconds
        """
        if training_time <= 0:
            return {}
        
        speed = (self.global_step - self.global_step_burnin) / training_time
        avg_return = np.mean(self.avg_returns) if self.avg_returns else 0
        lr = self.optimizer.param_groups[0]["lr"]
        if isinstance(lr, torch.Tensor):
            lr = lr.item()
        
        # Explained variance
        y_pred = container["vals"].flatten().cpu().numpy()
        y_true = container["returns"].flatten().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # Build base logs
        logs = {
            "charts/SPS": speed,
            "charts/learning_rate": lr,
            "charts/terminated_count": self.terminated_count,
            "charts/truncated_count": self.truncated_count,
            "losses/value_loss": out["v_loss"].item(),
            "losses/policy_loss": out["pg_loss"].item(),
            "losses/entropy": out["entropy_loss"].item(),
            "losses/approx_kl": out["approx_kl"].item(),
            "losses/old_approx_kl": out["old_approx_kl"].item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "losses/grad_norm": out["gn"].item() if isinstance(out["gn"], torch.Tensor) else out["gn"],
            "rollout/ep_return_mean": avg_return,
            "rollout/rewards_mean": container["rewards"].mean().item(),
            "rollout/rewards_max": container["rewards"].max().item(),
        }
        
        # Add reward components
        logs.update(self._build_reward_component_logs())
        
        # Add observation stats if enabled
        if self.log_obs_stats:
            logs.update(self._build_obs_stats_logs(container))
        
        # Log to WandB
        if self.cfg.wandb.enabled:
            wandb.log(logs, step=self.global_step)
        
        return logs

    def _build_reward_component_logs(self) -> dict:
        """Build reward component logs and reset accumulators."""
        logs = {}
        
        if self.reward_component_count > 0:
            for name, total in self.reward_component_sum.items():
                logs[f"reward/{name}"] = total / self.reward_component_count
            logs["reward/success_count"] = self.success_count.item() if hasattr(self.success_count, 'item') else self.success_count
            logs["reward/fail_count"] = self.fail_count.item() if hasattr(self.fail_count, 'item') else self.fail_count
            
            # Reset accumulators
            self.reward_component_sum = {}
            self.reward_component_count = 0
            self.success_count = torch.tensor(0, device=self.device, dtype=torch.float32)
            self.fail_count = torch.tensor(0, device=self.device, dtype=torch.float32)
        
        return logs

    def _build_obs_stats_logs(self, container) -> dict:
        """Build observation statistics logs."""
        from scripts.training.env_utils import NormalizeObservationGPU, NormalizeRewardGPU, find_wrapper
        
        logs = {}
        
        # Raw observation stats
        raw_obs = container["obs"]
        raw_mean = raw_obs.mean(dim=(0, 1))
        raw_std = raw_obs.std(dim=(0, 1))
        logs["obs_raw/mean_avg"] = raw_mean.mean().item()
        logs["obs_raw/std_avg"] = raw_std.mean().item()
        logs["obs_raw/min"] = raw_obs.min().item()
        logs["obs_raw/max"] = raw_obs.max().item()
        
        # Obs wrapper stats
        obs_wrapper = find_wrapper(self.envs, NormalizeObservationGPU)
        if obs_wrapper is not None:
            obs_rms = obs_wrapper.rms
            obs_rms_std = torch.sqrt(obs_rms.var + 1e-8)
            for i, name in enumerate(self.obs_names):
                if i < len(obs_rms.mean):
                    logs[f"obs_rms_mean/{name}"] = obs_rms.mean[i].item()
                    logs[f"obs_rms_std/{name}"] = obs_rms_std[i].item()
        
        # Reward wrapper stats
        reward_wrapper = find_wrapper(self.envs, NormalizeRewardGPU)
        if reward_wrapper is not None:
            r_rms = reward_wrapper.rms
            logs["reward_norm/return_rms_var"] = r_rms.var.item()
            logs["reward_norm/return_rms_std"] = torch.sqrt(r_rms.var + 1e-8).item()
            logs["reward_norm/return_rms_mean"] = r_rms.mean.item()
            logs["reward_norm/return_rms_count"] = r_rms.count if isinstance(r_rms.count, (int, float)) else r_rms.count.item()
            logs["reward_norm/normalized_reward_mean"] = container["rewards"].mean().item()
            logs["reward_norm/normalized_reward_std"] = container["rewards"].std().item()
        
        return logs

    def _handle_evaluation(self, iteration: int):
        """Handle evaluation scheduling (sync or async)."""
        if self.async_eval:
            # Async eval: launch in background
            if self.eval_thread is not None and self.eval_thread.is_alive():
                self.eval_thread.join()
            
            self.eval_agent.load_state_dict(self.agent.state_dict())
            self.eval_thread = threading.Thread(
                target=self._evaluate_async,
                args=(iteration,),
                daemon=True
            )
            self.eval_thread.start()
            print(f"  [Async] Eval launched in background (iteration {iteration})")
        else:
            # Sync eval (blocking)
            eval_start = time.time()
            self._evaluate()
            eval_duration = time.time() - eval_start
            print(f"  Eval took {eval_duration:.2f}s")
            if self.cfg.wandb.enabled:
                wandb.log({"charts/eval_time": eval_duration}, step=self.global_step)
            self._save_checkpoint(iteration)

    def _evaluate(self, agent=None):
        """Run evaluation episodes.
        
        Args:
            agent: Agent to use for evaluation. Defaults to self.agent.
                   For async eval, pass self.eval_agent to avoid race conditions.
        """
        if agent is None:
            agent = self.agent
        print("Running evaluation...")
        
        # Optimization: Instead of recreating env, flush the video wrapper state
        # directly in the existing eval_envs. save=False ignores pre-eval trash.
        self.eval_envs.call("flush_video", save=False)
        
        eval_obs, _ = self.eval_envs.reset()
        eval_returns = []
        eval_successes = []
        eval_fails = []
        episode_rewards = torch.zeros(self.cfg.training.num_eval_envs, device=self.device)
        
        # Track reward components during eval
        eval_reward_components = {}
        eval_component_count = 0
        
        # Structure: {env_idx: [{step, reward, component1, component2, ...}, ...]}
        step_reward_data = {i: [] for i in range(self.cfg.training.num_eval_envs)}
        
        # Compute max_steps consistently: (base * multiplier) + hold_steps
        base = self.cfg.env.episode_steps.get("base", 296)
        multiplier = self.cfg.env.episode_steps.get("multiplier", 1.2)
        hold_steps = 0
        if "reward" in self.cfg and "stable_hold_time" in self.cfg.reward:
            hold_steps = int(self.cfg.reward.stable_hold_time * self.cfg.env.get("control_freq", 30))
        
        training_steps = int(base * multiplier) + hold_steps
        eval_multiplier = self.cfg.training.get("eval_step_multiplier", 1.0)
        max_steps = int(training_steps * eval_multiplier)
        
        # Add a small buffer for safety
        max_steps += 2
        
        for step in range(max_steps):
            # eval_obs is already flat and normalized due to wrappers
            with torch.no_grad():
                eval_action = agent.get_action(eval_obs, deterministic=True)
            eval_obs, reward, terminated, truncated, eval_infos = self.eval_envs.step(eval_action)
            
            episode_rewards += reward
            
            # Use info_utils for cleaner extraction
            from scripts.training.info_utils import (
                get_reward_components, get_reward_components_per_env, 
                get_info_field, extract_scalar, extract_bool
            )
            
            reward_comps = get_reward_components(eval_infos)
            if reward_comps is not None:
                for k, v in reward_comps.items():
                    val = extract_scalar(v)
                    if val is not None:
                        eval_reward_components[k] = eval_reward_components.get(k, 0) + val
                eval_component_count += 1
                
                # Collect per-step data for CSV export (if enabled)
                rec_cfg = self.cfg.get("recording", {})
                if rec_cfg.get("save_step_csv", True):
                    reward_comps_per_env = get_reward_components_per_env(eval_infos)
                    
                    for env_idx in range(self.cfg.training.num_eval_envs):
                        step_data = {
                            "step": step,
                            "reward": reward[env_idx].item(),
                        }
                        
                        # Add reward components (prefer per-env values)
                        if reward_comps_per_env is not None:
                            for k, v in reward_comps_per_env.items():
                                val = v[env_idx].item() if hasattr(v, 'item') else v[env_idx]
                                step_data[k] = val if val is not None else 0.0
                        else:
                            for k, v in reward_comps.items():
                                val = extract_scalar(v)
                                step_data[k] = val if val is not None else 0.0
                        
                        # Add success/fail status using info_utils
                        success_val = get_info_field(eval_infos, "success")
                        step_data["success"] = extract_bool(success_val, env_idx) if success_val is not None else False
                        
                        fail_val = get_info_field(eval_infos, "fail")
                        step_data["fail"] = extract_bool(fail_val, env_idx) if fail_val is not None else False
                        
                        step_reward_data[env_idx].append(step_data)
            
            # Check for episode completion
            done = terminated | truncated
            if done.any():
                for idx in torch.where(done)[0]:
                    eval_returns.append(episode_rewards[idx].item())
                    episode_rewards[idx] = 0.0  # Reset for next episode
                    
                    # Check success/fail using info_utils (final_info is checked automatically)
                    success_val = get_info_field(eval_infos, "success")
                    if success_val is not None:
                        eval_successes.append(extract_bool(success_val, idx))
                    
                    fail_val = get_info_field(eval_infos, "fail")
                    if fail_val is not None:
                        eval_fails.append(extract_bool(fail_val, idx))
            
            # Stop after collecting enough episodes
            if len(eval_returns) >= self.cfg.training.num_eval_envs:
                break
        
        if eval_returns:
            mean_return = np.mean(eval_returns)
            success_rate = np.mean(eval_successes) if eval_successes else 0.0
            fail_rate = np.mean(eval_fails) if eval_fails else 0.0
            print(f"  eval/return = {mean_return:.4f}, success_rate = {success_rate:.2%}, fail_rate = {fail_rate:.2%} (n={len(eval_returns)})")
            
            # Build log dict
            eval_logs = {
                "eval/return": mean_return,
                "eval/success_rate": success_rate,
                "eval/fail_rate": fail_rate,
            }
            
            # Add eval reward components
            if eval_component_count > 0:
                for name, total in eval_reward_components.items():
                    eval_logs[f"eval_reward/{name}"] = total / eval_component_count
            
            if self.cfg.wandb.enabled:
                wandb.log(eval_logs, step=self.global_step)
        
        if self.video_dir is not None:
            # Finalize the evaluation video (saves to videos/*.mp4)
            self.eval_envs.call("flush_video", save=True)
            
            # Split videos and save CSVs to new structure: split/evalN/envM/
            import csv
            from pathlib import Path
            video_dir_path = Path(self.video_dir)
            split_base_dir = video_dir_path.parent / "split"  # outputs/.../split/
            eval_folder = split_base_dir / f"eval{self.eval_count}"  # split/eval0/
            
            # Create eval folder
            eval_folder.mkdir(parents=True, exist_ok=True)
            
            # Save CSVs to split/evalN/envM/rewards.csv
            for env_idx, steps in step_reward_data.items():
                if steps:  # Only save if we have data
                    env_folder = eval_folder / f"env{env_idx}"
                    env_folder.mkdir(exist_ok=True)
                    csv_path = env_folder / "rewards.csv"
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=steps[0].keys())
                        writer.writeheader()
                        writer.writerows(steps)
            
            # Split videos asynchronously to split/evalN/envM/record.mp4
            self._async_split_videos(eval_folder)
            
            self.eval_count += 1  # Increment for next eval

    def _evaluate_async(self, iteration):
        """Run evaluation asynchronously in a background thread.
        
        Uses a separate CUDA stream and a dedicated eval_agent to avoid
        race conditions with the training agent.
        """
        eval_start = time.time()
        
        # Run all CUDA operations on a separate stream
        with torch.cuda.stream(self.eval_stream):
            # Use eval_agent (which has a copy of weights from when eval was triggered)
            # This is completely isolated from self.agent, no race conditions
            self._evaluate(agent=self.eval_agent)
        
        # Sync this stream before logging (ensure eval is complete)
        self.eval_stream.synchronize()
        
        eval_duration = time.time() - eval_start
        print(f"  [Async] Eval completed (iteration {iteration}, took {eval_duration:.2f}s)")
        
        if self.cfg.wandb.enabled:
            # Log from background thread (wandb is thread-safe)
            wandb.log({"charts/eval_time": eval_duration}, step=self.global_step)
        
        # Save checkpoint after eval completes
        self._save_checkpoint(iteration)

    def _save_checkpoint(self, iteration):
        """Save model checkpoint."""
        if self.cfg.save_model:
            from scripts.training.env_utils import NormalizeObservationGPU, NormalizeRewardGPU, find_wrapper
            output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            model_path = output_dir / f"iteration_{iteration}.pt"
            
            # Find wrappers to extract stats
            obs_wrapper = find_wrapper(self.envs, NormalizeObservationGPU)
            reward_wrapper = find_wrapper(self.envs, NormalizeRewardGPU)

            state = {
                "agent": self.agent.state_dict(),
            }
            
            if obs_wrapper is not None:
                state.update({
                    "obs_rms_mean": obs_wrapper.rms.mean,
                    "obs_rms_var": obs_wrapper.rms.var,
                    "obs_rms_count": obs_wrapper.rms.count,
                })
            
            if reward_wrapper is not None:
                state.update({
                    "return_rms_mean": reward_wrapper.rms.mean,
                    "return_rms_var": reward_wrapper.rms.var,
                    "return_rms_count": reward_wrapper.rms.count,
                })

            torch.save(state, model_path)
            torch.save(state, output_dir / "latest.pt")
            print(f"Model and stats saved to {model_path}")

    def _async_split_videos(self, eval_folder):
        """Asynchronously split tiled eval videos into individual env videos.
        
        Args:
            eval_folder: Path to split/evalN/ where env subdirectories will be created
        """
        from scripts.utils.split_video import split_videos_in_dir_custom
        
        if not self.video_dir:
            return
        
        # Run in background thread (non-blocking)
        def split_task():
            split_videos_in_dir_custom(
                self.video_dir,
                self.cfg.training.num_eval_envs,
                eval_folder,  # Custom output directory
                rgb_only=True
            )
        
        thread = threading.Thread(target=split_task, daemon=True)
        thread.start()
