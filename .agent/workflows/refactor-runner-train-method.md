---
description: 重构 PPORunner.train() 方法 - 提取子方法降低复杂度
---

# 任务：PPORunner.train() 子方法提取

## 📋 任务概述

`scripts/training/runner.py` 中的 `train()` 方法约 250 行，职责过多。需要提取子方法来降低复杂度，提高可读性。

## 🎯 目标

1. 将 `train()` 方法从 ~250 行减少到 ~80 行
2. 提取 4-5 个职责单一的私有方法
3. 保持功能完全不变

---

## 📍 当前 train() 结构分析

```python
def train(self):
    # 第 436-447 行: 初始化和首次 reset
    # 第 448-451 行: 训练循环设置
    
    for iteration in pbar:
        # 第 455-457 行: burnin 逻辑
        # 第 460-466 行: 学习率调度
        # 第 468-469 行: cudagraph 标记
        
        # 第 471-473 行: Rollout
        # 第 475-491 行: GAE 计算
        # 第 493-518 行: PPO Update
        
        # 第 520-639 行: 日志记录 (120+ 行!)
        
        # 第 641-672 行: 评估调度
    
    # 第 674-681 行: 清理
```

---

## 📝 重构方案

### 提取的子方法

| 新方法名 | 原位置 | 职责 |
|---------|-------|------|
| `_log_training_metrics()` | 520-639 | 构建并记录日志 |
| `_run_ppo_update()` | 493-518 | PPO 更新循环 |
| `_compute_gae()` | 475-491 | GAE 计算 |
| `_schedule_learning_rate()` | 462-466 | 学习率调度 |
| `_handle_evaluation()` | 644-672 | 评估调度 |

---

## 📍 需要修改的代码

### 文件: `scripts/training/runner.py`

#### 新增方法 1: `_schedule_learning_rate()` 

**在 `_rollout()` 方法之后添加:**

```python
def _schedule_learning_rate(self, iteration: int):
    """Anneal learning rate linearly over training."""
    if self.anneal_lr:
        frac = 1.0 - (iteration - 1.0) / self.num_iterations
        lrnow = frac * self.cfg.ppo.learning_rate
        self.optimizer.param_groups[0]["lr"].copy_(lrnow)
```

---

#### 新增方法 2: `_compute_gae()`

```python
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
```

---

#### 新增方法 3: `_run_ppo_update()`

```python
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
```

---

#### 新增方法 4: `_log_training_metrics()` 

这是最大的提取，约 120 行。

```python
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
```

---

#### 新增辅助方法 5: `_build_reward_component_logs()` 

```python
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
```

---

#### 新增辅助方法 6: `_build_obs_stats_logs()`

```python
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
```

---

#### 新增方法 7: `_handle_evaluation()`

```python
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
```

---

#### 重构后的 `train()` 方法

**原来 ~250 行，重构后 ~80 行:**

```python
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
        next_obs, next_bootstrap_mask, container = self._rollout(next_obs, next_bootstrap_mask)
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
```

---

## ⚠️ 注意事项

1. **实例变量**: 新方法需要访问 `self.global_step_burnin`，需要在 `__init__` 中初始化为 `None`
2. **Import 位置**: `find_wrapper` import 应放在需要的方法内部 (延迟导入)
3. **方法顺序**: 新方法应放在 `train()` 之后，`_evaluate()` 之前
4. **Progress bar**: 保持 pbar.set_description 更新

---

## ✅ 验收标准

1. **语法正确**:
   ```bash
   uv run python -m py_compile scripts/training/runner.py
   ```

2. **行数减少**: `train()` 方法从 ~250 行减少到 ~80 行

3. **功能不变**: 训练/评估/日志行为与重构前一致

---

## 📁 相关文件路径

- `/home/admin/Desktop/eai-final-project/scripts/training/runner.py`

## 🔗 前置依赖

- `/refactor-wrapper-traversal` 必须已完成 (提供 `find_wrapper`)
- `/fix-runner-missing-methods` 必须已完成 (修复未定义方法)

---

// turbo-all
