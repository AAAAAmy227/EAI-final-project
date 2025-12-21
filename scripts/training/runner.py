
import time
import os
from pathlib import Path
from collections import defaultdict

import hydra
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from scripts.training.agent import Agent
from scripts.training.common import make_env, DictArray

class PPORunner:
    def __init__(self, cfg, logger=None):
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
        
        # Env setup
        self.envs = make_env(cfg, self.num_envs)
        self.eval_envs = make_env(cfg, cfg.training.num_eval_envs, for_eval=True, 
                                  video_dir=None) # Video handled separately if needed
        
        # Agent setup
        self.next_obs, _ = self.envs.reset(seed=cfg.seed)
        self.agent = Agent(self.envs, sample_obs=self.next_obs).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=cfg.ppo.learning_rate, eps=1e-5)
        
        if cfg.checkpoint:
            print(f"Loading checkpoint from {cfg.checkpoint}")
            self.agent.load_state_dict(torch.load(cfg.checkpoint))
            
        # Storage setup
        if isinstance(self.envs.single_observation_space, gym.spaces.Dict):
            self.obs_buffer = DictArray((self.num_steps, self.num_envs), self.envs.single_observation_space, device=self.device)
        else:
            # Assume Box (flat state)
            obs_shape = self.envs.single_observation_space.shape
            self.obs_buffer = torch.zeros((self.num_steps, self.num_envs) + obs_shape, device=self.device)
            
        self.actions = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        
        # Runtime vars
        self.global_step = 0
        self.start_time = time.time()
        self.next_done = torch.zeros(self.num_envs, device=self.device)
        
        # Logging
        self.logger = logger
        if self.logger is None:
             output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
             self.logger = SummaryWriter(output_dir / "tensorboard")

    def train(self):
        print(f"\n{'='*60}")
        print(f"Training PPO on Track1 {self.cfg.env.task}")
        print(f"Device: {self.device}")
        print(f"Total Timesteps: {self.total_timesteps}")
        print(f"{'='*60}\n")
        
        for iteration in range(1, self.num_iterations + 1):
            self._train_iteration(iteration)
            
        self.envs.close()
        self.eval_envs.close()
        self.logger.close()
        print("Training complete!")

    def _train_iteration(self, iteration):
        # 1. Anneal learning rate (optional, not implemented)
        
        # 2. Evaluation
        if iteration % self.cfg.training.eval_freq == 1:
            self._evaluate()
            self._save_checkpoint(iteration)

        # 3. Policy Rollout
        self.agent.eval()
        for step in range(self.num_steps):
            self.global_step += self.num_envs
            self.obs_buffer[step] = self.next_obs
            self.dones[step] = self.next_done

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(self.next_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            self.next_obs, reward, terminations, truncations, infos = self.envs.step(action)
            self.next_done = (terminations | truncations).float()
            self.rewards[step] = reward * self.cfg.ppo.reward_scale

            # Log episode info
            if "final_info" in infos:
                done_mask = infos["_final_info"]
                for k, v in infos["final_info"]["episode"].items():
                    self.logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), self.global_step)

        # 4. Bootstrap value
        with torch.no_grad():
            next_value = self.agent.get_value(self.next_obs).flatten()
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.cfg.ppo.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.cfg.ppo.gamma * self.cfg.ppo.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        # 5. Flatten batch
        if isinstance(self.obs_buffer, DictArray):
            b_obs = self.obs_buffer.reshape((-1,))
        else:
            b_obs = self.obs_buffer.reshape((-1,) + self.envs.single_observation_space.shape)
            
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # 6. PPO Update
        self.agent.train()
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        
        for epoch in range(self.cfg.ppo.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > self.cfg.ppo.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                # Normalize advantage
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.ppo.clip_coef, 1 + self.cfg.ppo.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                newvalue = newvalue.flatten()
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -self.cfg.ppo.clip_coef,
                    self.cfg.ppo.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                # v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean() # Unclipped version

                entropy_loss = entropy.mean()
                loss = pg_loss - self.cfg.ppo.ent_coef * entropy_loss + v_loss * self.cfg.ppo.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.cfg.ppo.max_grad_norm)
                self.optimizer.step()

            if self.cfg.ppo.target_kl is not None and approx_kl > self.cfg.ppo.target_kl:
                break

        # 7. Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        self.logger.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
        self.logger.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
        self.logger.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
        self.logger.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
        self.logger.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
        self.logger.add_scalar("losses/explained_variance", explained_var, self.global_step)
        
        sps = int(self.global_step / (time.time() - self.start_time))
        self.logger.add_scalar("charts/SPS", sps, self.global_step)
        print(f"Iter {iteration}: SPS={sps}, v_loss={v_loss.item():.3f}, p_loss={pg_loss.item():.3f}")

    def _evaluate(self):
        print("Running evaluation...")
        eval_obs, _ = self.eval_envs.reset()
        eval_metrics = defaultdict(list)
        num_episodes = 0
        
        # Simple loop for now
        for _ in range(self.cfg.training.num_eval_steps):
            with torch.no_grad():
                eval_action = self.agent.get_action(eval_obs, deterministic=True)
            eval_obs, _, _, _, eval_infos = self.eval_envs.step(eval_action)
            
            if "final_info" in eval_infos:
                mask = eval_infos["_final_info"]
                num_episodes += mask.sum()
                for k, v in eval_infos["final_info"]["episode"].items():
                    eval_metrics[k].append(v)
        
        # Save video if available (handled by Wrapper automatically, but we can verify)
        
        for k, v in eval_metrics.items():
            if len(v) > 0:
                mean_val = torch.stack(v).float().mean().item()
                self.logger.add_scalar(f"eval/{k}", mean_val, self.global_step)
                print(f"  eval/{k} = {mean_val:.4f}")

    def _save_checkpoint(self, iteration):
        if self.cfg.save_model:
             output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
             model_path = output_dir / f"ckpt_{iteration}.pt"
             torch.save(self.agent.state_dict(), model_path)
             
             # Also save as 'latest.pt'
             torch.save(self.agent.state_dict(), output_dir / "latest.pt")
