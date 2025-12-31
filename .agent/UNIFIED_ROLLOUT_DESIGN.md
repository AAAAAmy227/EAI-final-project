# 统一 Train 和 Eval Rollout 的设计方案

## 核心思路

将 `_rollout()` 改为通用方法，通过参数区分 train 和 eval 行为。

## 设计方案

### 1. 统一的 `_rollout()` 签名

```python
def _rollout(self, obs, num_steps, envs, policy_fn, 
             collect_for_training=True, 
             record_step_data=False):
    """
    Args:
        obs: 初始观测
        num_steps: rollout 步数
        envs: 环境实例 (self.envs 或 self.eval_envs)
        policy_fn: policy 函数 (lambda obs: self.policy(obs) for training, 
                                 lambda obs: self.agent.get_action(obs, deterministic=True) for eval)
        collect_for_training: 是否收集 training data (obs, actions, logprobs, values)
        record_step_data: 是否记录每个环境的 per-step data (用于 eval CSV)
    
    Returns:
        next_obs: 最后的观测
        storage: TensorDict with rollout data (if collect_for_training)
        step_data_per_env: Dict[int, List[dict]] (if record_step_data)
    """
```

### 2. train() 调用方式

```python
def train(self):
    # ...
    for iteration in pbar:
        # Rollout for training
        next_obs, storage = self._rollout(
            next_obs, 
            self.num_steps, 
            envs=self.envs,
            policy_fn=lambda obs: self.policy(obs=obs),
            collect_for_training=True,
            record_step_data=False
        )
        
        # GAE, PPO update, etc.
```

### 3. evaluate() 调用方式

```python
def _evaluate(self, agent=None):
    if agent is None:
        agent = self.agent
    
    eval_obs, _ = self.eval_envs.reset()
    
    # Deterministic policy for eval
    policy_fn = lambda obs: (agent.get_action(obs, deterministic=True), None, None, None)
    
    # Rollout for evaluation
    _, _, step_data_per_env = self._rollout(
        eval_obs,
        max_steps,
        envs=self.eval_envs,
        policy_fn=policy_fn,
        collect_for_training=False,  # 不需要收集 actions/logprobs/values
        record_step_data=self.cfg.get("recording", {}).get("save_step_csv", True)
    )
    
    # episode_metrics 已经被 _aggregate_metrics 填充
    # 直接用于 logging
    eval_logs = self._build_eval_logs()
    
    # 保存 per-env CSV
    if step_data_per_env:
        self._save_step_csvs(step_data_per_env)
```

### 4. 统一的 metrics 收集

无论 train 还是 eval，都使用相同的 metrics 收集逻辑：

```python
def _rollout(...):
    # ...
    
    # Get metric specs (same for train and eval)
    metric_specs = self._get_metric_specs(envs)
    
    # Pre-allocate metrics storage
    metrics_storage = {...}
    
    # Optional: per-env step data for eval CSV
    step_data_per_env = {i: [] for i in range(num_envs)} if record_step_data else None
    
    for step in range(num_steps):
        # Policy inference
        action, logprob, entropy, value = policy_fn(obs)
        
        # Store training data (only if collect_for_training)
        if collect_for_training:
            storage["actions"][step] = action
            storage["logprobs"][step] = logprob
            # ...
        
        # Environment step
        next_obs, reward, terminated, truncated, done, infos = self._step_env(action, envs)
        
        # Extract metrics (same logic for train and eval)
        info_to_log = infos.get("final_info", infos)
        for metric_name in metric_specs.keys():
            # ... (same as before)
        
        # Record per-env step data (only if record_step_data)
        if record_step_data:
            for env_idx in range(envs.num_envs):
                step_data_per_env[env_idx].append({
                    "step": step,
                    "reward": reward[env_idx].item(),
                    # Extract per-env metrics from info_to_log[metric_name][env_idx]
                })
        
        obs = next_obs
    
    # Aggregate metrics (populates self.episode_metrics)
    self._aggregate_metrics(metrics_storage, metric_specs)
    
    return next_obs, storage if collect_for_training else None, step_data_per_env
```

## 优势

1. ✅ **代码统一**：train 和 eval 使用相同的 metrics 收集逻辑
2. ✅ **减少重复**：不需要在 `_evaluate()` 中重复写 metrics 提取
3. ✅ **灵活性**：通过参数控制行为，易于扩展
4. ✅ **一致性**：train 和 eval 的 metrics 保证是一致的

## 关键差异处理

| 维度 | Train | Eval |
|------|-------|------|
| Policy | `self.policy(obs)` (stochastic) | `agent.get_action(obs, deterministic=True)` |
| 收集数据 | actions, logprobs, values, advantages | 只收集 metrics |
| Per-step 记录 | 不需要 | 可选（用于 CSV） |
| Env | `self.envs` | `self.eval_envs` |
| Episode metrics | 累加到 `self.episode_metrics` | 累加到 `self.episode_metrics`（分开 log） |

## 注意事项

1. **episode_metrics 的管理**：
   - Train: 每次 iteration 后清空（在 `_build_reward_component_logs()` 中）
   - Eval: 在 evaluate 开始时清空，结束后 log

2. **Per-env step data**：
   - 只在 eval 且 `save_step_csv=True` 时记录
   - 记录每个环境每一步的详细信息（reward, components, success/fail）

3. **Policy function**：
   - Train: 返回 (action, logprob, entropy, value)
   - Eval: 返回 (action, None, None, None) 或只返回 action，需要统一接口
