"""
LeanRL-style PPO Runner with tensordict, torch.compile, and CudaGraphModule.
Based on LeanRL/cleanrl ppo_continuous_action_torchcompile.py
"""
import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["EXCLUDE_TD_FROM_PYTREE"] = "1"  # Silence CudaGraph compatibility warning

import math
import random
import time
from collections import deque
from functools import partial
from pathlib import Path
from typing import Dict
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
from scripts.training.metrics_utils import get_metric_specs_from_env, aggregate_metrics

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
        obs_space = source_env.get_wrapper_attr("single_observation_space")
        act_space = source_env.get_wrapper_attr("single_action_space")
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
        
        # Episode metrics storage (populated by _aggregate_metrics)
        # Each key maps to a list of values from completed episodes
        self.episode_metrics = {}
        
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

    def _step_env(self, action, envs):
        """Execute environment step.
        
        Returns:
            next_obs, reward, terminated, truncated, done, info
            where done = terminated | truncated (for episode boundary tracking)
        """
        next_obs, reward, terminations, truncations, info = envs.step(action)
        done = terminations | truncations
        return next_obs, reward, terminations, truncations, done, info

    def _get_metric_specs(self) -> Dict[str, str]:
        """Get metric aggregation specifications from task handler."""
        return get_metric_specs_from_env(self.envs)

    def _aggregate_metrics(self, metrics_storage: Dict[str, torch.Tensor], 
                          metric_specs: Dict[str, str]) -> None:
        """Aggregate rollout metrics into episode_metrics storage."""
        aggregate_metrics(metrics_storage, metric_specs, self.episode_metrics)


    def _rollout(self, obs, num_steps, envs=None, policy_fn=None,
                 collect_for_training=True, record_step_data=False):
        """Unified rollout method for both training and evaluation.
        
        Args:
            obs: Initial observations
            num_steps: Number of steps to rollout
            envs: Environment instance (defaults to self.envs for training)
            policy_fn: Policy function that takes obs and returns (action, logprob, entropy, value).
                      For eval, logprob/entropy/value can be None. Defaults to self.policy.
            collect_for_training: If True, collect training data (obs, actions, logprobs, values, advantages).
                                 If False, only collect metrics.
            record_step_data: If True, record per-env step-by-step data for CSV export (eval only).
            
        Returns:
            next_obs: Final observations after rollout
            storage: TensorDict with rollout data (if collect_for_training, else None)
            step_data_per_env: Dict[env_idx, List[step_data]] (if record_step_data, else None)
        """
        # Defaults
        if envs is None:
            envs = self.envs
        if policy_fn is None:
            policy_fn = lambda obs: self.policy(obs=obs)
        
        num_envs = envs.unwrapped.num_envs
        
        # 1. Pre-allocate TensorDict for training data (only if needed)
        storage = None
        if collect_for_training:
            storage = tensordict.TensorDict({
                "obs": torch.empty((num_steps, num_envs, self.n_obs), device=self.device, dtype=obs.dtype),
                "bootstrap_mask": torch.empty((num_steps, num_envs), device=self.device, dtype=torch.bool),
                "vals": torch.empty((num_steps, num_envs), device=self.device),
                "actions": torch.empty((num_steps, num_envs, self.n_act), device=self.device),
                "logprobs": torch.empty((num_steps, num_envs), device=self.device),
                "entropies": torch.empty((num_steps, num_envs), device=self.device),
                "rewards": torch.empty((num_steps, num_envs), device=self.device),
            }, batch_size=[num_steps, num_envs], device=self.device)

        # 2. Get metric aggregation specs (mode-specific)
        mode = "train" if collect_for_training else "eval"
        metric_specs = get_metric_specs_from_env(envs, mode=mode)
        
        # 3. Pre-allocate storage for metrics (all on GPU!)
        metrics_storage = {
            "done_mask": torch.empty((num_steps, num_envs), dtype=torch.bool, device=self.device),
        }
        for metric_name in metric_specs.keys():
            metrics_storage[metric_name] = torch.empty((num_steps, num_envs), 
                                                       dtype=torch.float32, 
                                                       device=self.device)

        # 4. Optional: per-env step data for eval CSV
        step_data_per_env = None
        if record_step_data:
            step_data_per_env = {i: [] for i in range(num_envs)}

        # 5. Rollout loop
        for step in range(num_steps):
            # Store current observation (training only)
            if collect_for_training:
                storage["obs"][step] = obs
            
            # Policy inference
            with torch.no_grad():
                policy_output = policy_fn(obs)
                # Handle different return formats
                if isinstance(policy_output, tuple) and len(policy_output) == 4:
                    action, logprob, entropy, value = policy_output
                else:
                    # Eval case: might only return action
                    action = policy_output
                    logprob = entropy = value = None
            
            # Store policy outputs (training only)
            if collect_for_training and logprob is not None:
                storage["vals"][step] = value.flatten()
                storage["actions"][step] = action
                storage["logprobs"][step] = logprob
                storage["entropies"][step] = entropy
            
            # Environment step
            next_obs, reward, next_terminated, next_truncated, next_done, infos = self._step_env(action, envs)
            
            # Store rollout data (training only)
            if collect_for_training:
                storage["rewards"][step] = reward
                storage["bootstrap_mask"][step] = next_terminated if self.handle_timeout_termination else next_done
            
            # Extract metrics from final_info if available (episode completion)
            info_to_log = infos.get("final_info", infos)
            
            # Store done mask
            metrics_storage["done_mask"][step] = next_done
            
            # Extract and store each metric (all on GPU)
            for metric_name in metric_specs.keys():
                value_to_store = 0.0  # Default
                
                # Handle "episode" dict from ManiSkill
                if metric_name in ["return", "episode_len", "success_once", "fail_once", "reward"]:
                    episode_info = info_to_log.get("episode")
                    if episode_info is not None and metric_name in episode_info:
                        value_to_store = episode_info[metric_name]
                elif metric_name in info_to_log:
                    value_to_store = info_to_log[metric_name]
                
                # Store the value
                if value_to_store is not None:
                    if isinstance(value_to_store, torch.Tensor):
                        metrics_storage[metric_name][step] = value_to_store.float()
                    else:
                        metrics_storage[metric_name][step] = float(value_to_store)
                else:
                    metrics_storage[metric_name][step] = 0.0
            
            # Record per-env step data (eval CSV only)
            if record_step_data:
                for env_idx in range(num_envs):
                    step_dict = {
                        "step": step,
                        "reward": reward[env_idx].item(),
                    }
                    
                    # Add all metrics for this env
                    for metric_name in metric_specs.keys():
                        metric_value = metrics_storage[metric_name][step, env_idx]
                        if isinstance(metric_value, torch.Tensor):
                            step_dict[metric_name] = metric_value.item()
                        else:
                            step_dict[metric_name] = float(metric_value)
                    
                    step_data_per_env[env_idx].append(step_dict)
            
            # Update for next iteration
            obs = next_obs
        
        # 6. Aggregate all metrics at once (GPU -> CPU transfer happens here)
        aggregate_metrics(metrics_storage, metric_specs, self.episode_metrics)
        
        return next_obs, storage, step_data_per_env


    def train(self):
        print(f"\n{'='*60}")
        print(f"Training PPO (LeanRL-style) on Track1 {self.cfg.env.task}")
        print(f"Device: {self.device}, Compile: {self.compile}, CudaGraphs: {self.cudagraphs}")
        print(f"Total Timesteps: {self.total_timesteps}, Batch Size: {self.batch_size}")
        print(f"{'='*60}\n")
        
        # Initial reset
        next_obs, _ = self.envs.reset(seed=self.cfg.seed)
        
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
            
            # Rollout (training)
            next_obs, container, _ = self._rollout(
                next_obs, 
                self.num_steps,
                envs=self.envs,
                policy_fn=lambda obs: self.policy(obs=obs),
                collect_for_training=True,
                record_step_data=False
            )
            self.global_step += self.num_steps * self.num_envs
            
            # GAE calculation
            container = self._compute_gae(container, next_obs)
            
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

    def _compute_gae(self, container, next_obs):
        """Compute GAE advantages and returns.
        
        Args:
            container: TensorDict with rollout data (rewards, vals, bootstrap_mask)
            next_obs: Final observations after rollout
            
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
            next_value
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
        """Build reward component logs from episode_metrics and reset.
        
        Thin coordination method: uses pure function for computation,
        manages state (avg_returns, episode_metrics clearing).
        """
        from scripts.training.runner_utils import compute_reward_logs
        
        # Pure computation
        logs = compute_reward_logs(self.episode_metrics)
        
        # State management
        if "return" in self.episode_metrics and self.episode_metrics["return"]:
            self.avg_returns.extend(self.episode_metrics["return"])
        
        # Reset for next iteration
        self.episode_metrics = {}
        
        return logs
    
    def _build_eval_logs(self) -> dict:
        """Build evaluation logs from episode_metrics.
        
        Thin coordination method: uses pure function for computation,
        manages state (episode_metrics clearing).
        
        Returns:
            Dict of evaluation logs
        """
        from scripts.training.runner_utils import compute_eval_logs
        
        # Pure computation
        logs = compute_eval_logs(self.episode_metrics)
        
        # State management: clear episode_metrics after logging
        self.episode_metrics = {}
        
        return logs
    
    def _save_step_csvs(self, step_data_per_env: Dict[int, list]) -> None:
        """Save per-environment step-by-step CSV files.
        
        Uses dependency injection for paths and pure functions for I/O.
        
        Args:
            step_data_per_env: Dict mapping environment index to list of step data dicts
        """
        from pathlib import Path
        from scripts.training.runner_utils import build_csv_path, write_csv_file
        
        # Dependency injection: extract config from instance state
        base_dir = Path(self.video_dir).parent
        eval_name = f"eval{self.eval_count}"
        
        # Save CSVs using pure functions
        for env_idx, steps in step_data_per_env.items():
            csv_path = build_csv_path(base_dir, eval_name, env_idx)
            write_csv_file(csv_path, steps)
        
        # Split videos asynchronously
        eval_folder = base_dir / "split" / eval_name
        self._async_split_videos(eval_folder)
        
        # Update state
        self.eval_count += 1

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
            # Capture global_step at eval launch time (not when eval completes)
            eval_global_step = self.global_step
            self.eval_thread = threading.Thread(
                target=self._evaluate_async,
                args=(iteration, eval_global_step),
                daemon=True
            )
            self.eval_thread.start()
            print(f"  [Async] Eval launched in background (iteration {iteration}, step {eval_global_step})")
        else:
            # Sync eval (blocking)
            eval_start = time.time()
            self._evaluate()
            eval_duration = time.time() - eval_start
            print(f"  Eval took {eval_duration:.2f}s")
            if self.cfg.wandb.enabled:
                wandb.log({"charts/eval_time": eval_duration}, step=self.global_step)
            self._save_checkpoint(iteration)

    def _evaluate(self, agent=None, log_step=None):
        """Run evaluation episodes using unified _rollout() method.
        
        Args:
            agent: Agent to use for evaluation. Defaults to self.agent.
                   For async eval, pass self.eval_agent to avoid race conditions.
            log_step: Global step for wandb logging. Defaults to self.global_step.
                     For async eval, should pass the captured step at eval launch time.
        """
        if agent is None:
            agent = self.agent
        if log_step is None:
            log_step = self.global_step
            
        print("Running eval uation...")
        
        # Flush video wrapper state
        self.eval_envs.call("flush_video", save=False)
        
        # Reset eval envs
        eval_obs, _ = self.eval_envs.reset()
        
        # Clear episode_metrics for eval (separate from training metrics)
        self.episode_metrics = {}
        
        # Compute max eval steps
        base = self.cfg.env.episode_steps.get("base", 296)
        multiplier = self.cfg.env.episode_steps.get("multiplier", 1.2)
        hold_steps = 0
        if "reward" in self.cfg and "stable_hold_time" in self.cfg.reward:
            hold_steps = int(self.cfg.reward.stable_hold_time * self.cfg.env.get("control_freq", 30))
        
        training_steps = int(base * multiplier) + hold_steps
        eval_multiplier = self.cfg.training.get("eval_step_multiplier", 1.0)
        max_steps = int(training_steps * eval_multiplier) + 2
        
        # Deterministic policy for eval
        def eval_policy_fn(obs):
            action = agent.get_action(obs, deterministic=True)
            return action, None, None, None  # (action, logprob, entropy, value)
        
        # Check if we need to save CSVs
        rec_cfg = self.cfg.get("recording", {})
        save_csv = rec_cfg.get("save_step_csv", True)
        
        # Run evaluation rollout (reusing unified rollout method)
        _, _, step_data_per_env = self._rollout(
            eval_obs,
            max_steps,
            envs=self.eval_envs,
            policy_fn=eval_policy_fn,
            collect_for_training=False,  # Don't collect training data
            record_step_data=save_csv    # Record per-env data if needed
        )
        
        # Build and log evaluation metrics
        eval_logs = self._build_eval_logs()
        
        if eval_logs:
            print(f"  eval/return = {eval_logs.get('eval/return', 0):.4f}, "
                  f"success_rate = {eval_logs.get('eval/success_rate', 0):.2%}, "
                  f"fail_rate = {eval_logs.get('eval/fail_rate', 0):.2%}")
            
            if self.cfg.wandb.enabled:
                # Use log_step (captured at eval launch for async, or current for sync)
                wandb.log(eval_logs, step=log_step)
        
        # Save videos and CSVs
        
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
            if eval_total_steps > 0:
                for name, total in eval_reward_components.items():
                    eval_logs[f"eval_reward/{name}"] = total / eval_total_steps
            
            if self.cfg.wandb.enabled:
                wandb.log(eval_logs, step=self.global_step)
        
        if self.video_dir is not None:
            # Save videos and CSVs
            if save_csv and step_data_per_env:
                self._save_step_csvs(step_data_per_env)

    def _evaluate_async(self, iteration, eval_global_step):
        """Run evaluation asynchronously in a background thread.
        
        Args:
            iteration: Training iteration number
            eval_global_step: Global step captured at eval launch time (for accurate logging)
        
        Uses a separate CUDA stream and a dedicated eval_agent to avoid
        race conditions with the training agent.
        """
        eval_start = time.time()
        
        # Run all CUDA operations on a separate stream
        with torch.cuda.stream(self.eval_stream):
            # Use eval_agent (which has a copy of weights from when eval was triggered)
            # This is completely isolated from self.agent, no race conditions
            # Pass eval_global_step for accurate wandb logging
            self._evaluate(agent=self.eval_agent, log_step=eval_global_step)
        
        # Sync this stream before logging (ensure eval is complete)
        self.eval_stream.synchronize()
        
        eval_duration = time.time() - eval_start
        print(f"  [Async] Eval completed (iteration {iteration}, took {eval_duration:.2f}s)")
        
        if self.cfg.wandb.enabled:
            # Log from background thread (wandb is thread-safe)
            # Use captured eval_global_step, not self.global_step (which has increased)
            wandb.log({"charts/eval_time": eval_duration}, step=eval_global_step)
        
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
