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
from pathlib import Path

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
from scripts.training.common import make_env

# JIT-compiled GAE function for maximum performance and compiler friendliness
@torch.jit.script
def optimized_gae(
    rewards: torch.Tensor,
    vals: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float
):
    num_steps: int = rewards.shape[0]
    next_value = next_value.reshape(-1)
    
    # Pre-allocate advantages tensor
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(rewards[0])
    
    # In JIT script, we need to be careful with types
    # ~dones is not supported directly in all versions, use 1.0 - x.float()
    nextnonterminal = 1.0 - next_done.float()
    
    # Loop backwards
    for t in range(num_steps - 1, -1, -1):
        if t == num_steps - 1:
            nextnonterminal_t = nextnonterminal
            nextvalues_t = next_value
        else:
            nextnonterminal_t = 1.0 - dones[t + 1].float()
            nextvalues_t = vals[t + 1]
            
        delta = rewards[t] + gamma * nextvalues_t * nextnonterminal_t - vals[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal_t * lastgaelam
        advantages[t] = lastgaelam
        
    return advantages, advantages + vals

class PPORunner:
    def __init__(self, cfg):
        self.cfg = cfg
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
        
        # Env setup
        self.envs = make_env(cfg, self.num_envs)
        self.eval_envs = make_env(cfg, cfg.training.num_eval_envs, for_eval=True)
        
        # Determine observation/action dimensions
        obs_space = self.envs.single_observation_space
        act_space = self.envs.single_action_space
        self.n_obs = math.prod(obs_space.shape)
        self.n_act = math.prod(act_space.shape)
        
        # Agent setup
        self.agent = Agent(self.n_obs, self.n_act, device=self.device)
        # Create inference agent with detached params
        self.agent_inference = Agent(self.n_obs, self.n_act, device=self.device)
        agent_inference_p = from_module(self.agent).data
        agent_inference_p.to_module(self.agent_inference)
        
        # Optimizer with capturable for cudagraphs
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=torch.tensor(cfg.ppo.learning_rate, device=self.device),
            eps=1e-5,
            capturable=self.cudagraphs and not self.compile,
        )
        
        if cfg.checkpoint:
            print(f"Loading checkpoint from {cfg.checkpoint}")
            self.agent.load_state_dict(torch.load(cfg.checkpoint))
            from_module(self.agent).data.to_module(self.agent_inference)
        
        # Setup compiled functions
        self._setup_compiled_functions()
        
        # Runtime vars
        self.global_step = 0
        self.avg_returns = deque(maxlen=20)
        
        # Reward mode for logging
        self.reward_mode = cfg.reward.get("reward_mode", "sparse")
        self.staged_reward = self.reward_mode == "staged_dense"

    def _setup_compiled_functions(self):
        """Setup torch.compile and CudaGraphModule for policy/gae/update."""
        cfg = self.cfg
        
        # Define policy function
        self.policy = self.agent_inference.get_action_and_value
        self.get_value = self.agent_inference.get_value
        
        # GAE function adapter
        # This wrapper prepares data for the JIT-compiled optimized_gae function
        gamma = cfg.ppo.gamma
        gae_lambda = cfg.ppo.gae_lambda
        
        def gae_adapter(next_obs, next_done, container):
            next_value = self.get_value(next_obs)
            
            # Call JIT compiled function
            advantages, returns = optimized_gae(
                container["rewards"],
                container["vals"],
                container["dones"],
                next_value,
                next_done,
                gamma,
                gae_lambda
            )
            
            container["advantages"] = advantages
            container["returns"] = returns
            return container
        
        self.gae_fn = gae_adapter
        
        # Update function
        def update(obs, actions, logprobs, advantages, returns, vals):
            self.optimizer.zero_grad()
            _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(obs, actions)
            logratio = newlogprob - logprobs
            ratio = logratio.exp()
            
            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > cfg.ppo.clip_coef).float().mean()
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Policy loss
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1 - cfg.ppo.clip_coef, 1 + cfg.ppo.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            # Value loss (clipped)
            newvalue = newvalue.view(-1)
            v_loss_unclipped = (newvalue - returns) ** 2
            v_clipped = vals + torch.clamp(newvalue - vals, -cfg.ppo.clip_coef, cfg.ppo.clip_coef)
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            
            entropy_loss = entropy.mean()
            loss = pg_loss - cfg.ppo.ent_coef * entropy_loss + v_loss * cfg.ppo.vf_coef
            
            loss.backward()
            gn = nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.ppo.max_grad_norm)
            self.optimizer.step()
            
            return approx_kl, v_loss.detach(), pg_loss.detach(), entropy_loss.detach(), old_approx_kl, clipfrac, gn
        
        self.update_fn = tensordict.nn.TensorDictModule(
            update,
            in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
            out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac", "gn"],
        )
        
        # Apply torch.compile
        if self.compile:
            self.policy = torch.compile(self.policy)
            # Skip compiling GAE when cudagraphs is enabled:
            # torch.compile's lazy tracing conflicts with CudaGraphModule capture
            # (triggers RuntimeError: Cannot call CUDAGeneratorImpl::current_seed during CUDA graph capture)
            if not self.cudagraphs:
                self.gae_fn = torch.compile(self.gae_fn, fullgraph=True)
            self.update_fn = torch.compile(self.update_fn)
        
        # Apply CudaGraphModule on top of compiled functions
        if self.cudagraphs:
            self.policy = CudaGraphModule(self.policy)
            self.gae_fn = CudaGraphModule(self.gae_fn)
            self.update_fn = CudaGraphModule(self.update_fn)

    def _step_env(self, action):
        """Execute environment step."""
        next_obs, reward, terminations, truncations, info = self.envs.step(action)
        next_done = terminations | truncations
        return next_obs, reward, next_done, info

    def _rollout(self, obs, done):
        """Collect trajectories."""
        ts = []
        for step in range(self.num_steps):
            action, logprob, _, value = self.policy(obs=obs)
            
            next_obs, reward, next_done, infos = self._step_env(action)
            
            # Log episode info
            if "final_info" in infos:
                done_mask = infos["_final_info"]
                for idx in torch.where(done_mask)[0]:
                    ep_info = infos["final_info"]["episode"]
                    # ManiSkill uses 'r' for return, handle both formats
                    if "r" in ep_info:
                        r = float(ep_info["r"][idx])
                    elif "return" in ep_info:
                        r = float(ep_info["return"][idx])
                    else:
                        # Fallback: compute from elapsed steps if needed
                        r = 0.0
                    self.avg_returns.append(r)
            
            ts.append(tensordict.TensorDict._new_unsafe(
                obs=obs,
                dones=done,
                vals=value.flatten(),
                actions=action,
                logprobs=logprob,
                rewards=reward * self.cfg.ppo.reward_scale,
                batch_size=(self.num_envs,),
            ))
            
            obs = next_obs.to(self.device, non_blocking=True)
            done = next_done.to(self.device, non_blocking=True)
        
        container = torch.stack(ts, 0).to(self.device)
        return obs, done, container

    def train(self):
        print(f"\n{'='*60}")
        print(f"Training PPO (LeanRL-style) on Track1 {self.cfg.env.task}")
        print(f"Device: {self.device}, Compile: {self.compile}, CudaGraphs: {self.cudagraphs}")
        print(f"Total Timesteps: {self.total_timesteps}, Batch Size: {self.batch_size}")
        print(f"{'='*60}\n")
        
        # Initial reset
        next_obs, _ = self.envs.reset(seed=self.cfg.seed)
        next_obs = next_obs.to(self.device)
        next_done = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        pbar = tqdm.tqdm(range(1, self.num_iterations + 1))
        global_step_burnin = None
        start_time = None
        measure_burnin = 3
        
        for iteration in pbar:
            if iteration == measure_burnin:
                global_step_burnin = self.global_step
                start_time = time.time()
            
            # LR Annealing
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.cfg.ppo.learning_rate
                self.optimizer.param_groups[0]["lr"].copy_(lrnow)
            
            # Mark step for cudagraph
            torch.compiler.cudagraph_mark_step_begin()
            
            # Rollout
            next_obs, next_done, container = self._rollout(next_obs, next_done)
            self.global_step += container.numel()
            
            # GAE
            container = self.gae_fn(next_obs, next_done, container)
            container_flat = container.view(-1)
            
            # PPO Update
            for epoch in range(self.cfg.ppo.update_epochs):
                b_inds = torch.randperm(container_flat.shape[0], device=self.device).split(self.minibatch_size)
                for b in b_inds:
                    container_local = container_flat[b]
                    out = self.update_fn(container_local, tensordict_out=tensordict.TensorDict())
                    
                    if self.cfg.ppo.target_kl is not None and out["approx_kl"] > self.cfg.ppo.target_kl:
                        break
                else:
                    continue
                break
            
            # Sync params to inference agent
            from_module(self.agent).data.to_module(self.agent_inference)
            
            # Logging (every iteration after burnin)
            if global_step_burnin is not None:
                speed = (self.global_step - global_step_burnin) / (time.time() - start_time)
                avg_return = np.array(self.avg_returns).mean() if self.avg_returns else 0
                lr = self.optimizer.param_groups[0]["lr"]
                if isinstance(lr, torch.Tensor):
                    lr = lr.item()
                
                # Compute explained variance
                y_pred = container["vals"].flatten().cpu().numpy()
                y_true = container["returns"].flatten().cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                
                pbar.set_description(
                    f"SPS: {speed:.0f}, return: {avg_return:.2f}, lr: {lr:.2e}"
                )
                
                logs = {
                    "charts/SPS": speed,
                    "charts/learning_rate": lr,
                    "losses/value_loss": out["v_loss"].item(),
                    "losses/policy_loss": out["pg_loss"].item(),
                    "losses/entropy": out["entropy_loss"].item(),
                    "losses/approx_kl": out["approx_kl"].item(),
                    "losses/old_approx_kl": out["old_approx_kl"].item(),
                    "losses/clipfrac": out["clipfrac"].item(),
                    "losses/explained_variance": explained_var,
                    "losses/grad_norm": out["gn"].item() if isinstance(out["gn"], torch.Tensor) else out["gn"],
                    "rollout/ep_return_mean": avg_return,
                    "rollout/rewards_mean": container["rewards"].mean().item(),
                    "rollout/rewards_max": container["rewards"].max().item(),
                }
                
                if self.cfg.wandb.enabled:
                    wandb.log(logs, step=self.global_step)
                    
            # Evaluation
            if iteration % self.cfg.training.eval_freq == 0:
                self._evaluate()
                self._save_checkpoint(iteration)
        
        self.envs.close()
        self.eval_envs.close()
        print("Training complete!")

    def _evaluate(self):
        """Run evaluation episodes."""
        print("Running evaluation...")
        eval_obs, _ = self.eval_envs.reset()
        eval_returns = []
        
        for _ in range(self.cfg.training.num_eval_steps):
            with torch.no_grad():
                eval_action = self.agent.get_action(eval_obs, deterministic=True)
            eval_obs, _, _, _, eval_infos = self.eval_envs.step(eval_action)
            
            if "final_info" in eval_infos:
                mask = eval_infos["_final_info"]
                for idx in torch.where(mask)[0]:
                    r = float(eval_infos["final_info"]["episode"]["r"][idx])
                    eval_returns.append(r)
        
        if eval_returns:
            mean_return = np.mean(eval_returns)
            print(f"  eval/return = {mean_return:.4f}")
            if self.cfg.wandb.enabled:
                wandb.log({"eval/return": mean_return}, step=self.global_step)

    def _save_checkpoint(self, iteration):
        """Save model checkpoint."""
        if self.cfg.save_model:
            output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            model_path = output_dir / f"ckpt_{iteration}.pt"
            torch.save(self.agent.state_dict(), model_path)
            torch.save(self.agent.state_dict(), output_dir / "latest.pt")
