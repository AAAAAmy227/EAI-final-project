# 统一 Rollout 实现总结

## ✅ 完成状态

已成功实现统一的 `_rollout()` 方法，同时支持 training 和 evaluation。

## 关键修改

### 1. **统一的 `_rollout()` 方法** (`scripts/training/runner.py`)

**新签名**：
```python
def _rollout(self, obs, num_steps, envs=None, policy_fn=None,
             collect_for_training=True, record_step_data=False):
```

**参数说明**：
- `envs`: 环境实例（默认 `self.envs` for training，可传 `self.eval_envs` for eval）
- `policy_fn`: Policy 函数（training 用 `self.policy`，eval 用 deterministic action）
- `collect_for_training`: 是否收集训练数据（actions, logprobs, values, advantages）
- `record_step_data`: 是否记录 per-env step-by-step 数据（用于 eval CSV）

**返回值**：
```python
return next_obs, storage, step_data_per_env
```
- `storage`: TensorDict（仅 training 时非 None）
- `step_data_per_env`: 每环境的step数据（仅 `record_step_data=True` 时非 None）

### 2. **Training 调用**

```python
# In train()
next_obs, container, _ = self._rollout(
    next_obs, 
    self.num_steps,
    envs=self.envs,
    policy_fn=lambda obs: self.policy(obs=obs),
    collect_for_training=True,
    record_step_data=False
)
```

### 3. **Evaluation 调用**

```python
# In _evaluate()
def eval_policy_fn(obs):
    action = agent.get_action(obs, deterministic=True)
    return action, None, None, None

_, _, step_data_per_env = self._rollout(
    eval_obs,
    max_steps,
    envs=self.eval_envs,
    policy_fn=eval_policy_fn,
    collect_for_training=False,
    record_step_data=save_csv
)
```

### 4. **新增辅助方法**

#### `_build_eval_logs()` - 处理 Eval 指标
```python
def _build_eval_logs(self) -> dict:
    """从 episode_metrics 构建 eval 日志"""
    logs = {}
    
    # 特殊处理关键指标
    logs["eval/return"] = mean(episode_metrics["return"])
    logs["eval/success_rate"] = mean(episode_metrics["success"])
    logs["eval/fail_rate"] = mean(episode_metrics["fail"])
    
    # 其他指标用 eval_reward/ 前缀
    logs[f"eval_reward/{metric_name}"] = mean(values)
    
    return logs
```

#### `_save_step_csvs()` - 保存 Per-Env CSV
```python
def _save_step_csvs(self, step_data_per_env: Dict[int, list]):
    """保存每个环境的 step-by-step CSV 文件"""
    # 保存到: split/evalN/envM/rewards.csv
    # 包含所有 metrics（success, fail, reward components, etc.）
```

### 5. **Metrics 收集统一**

无论 train 还是 eval，都使用相同的逻辑：

```python
# 1. 获取 metric specs（从 task handler）
metric_specs = get_metric_specs_from_env(envs)

# 2. 预分配 GPU tensors
metrics_storage = {
    "done_mask": torch.empty((num_steps, num_envs), ...),
    "success": torch.empty((num_steps, num_envs), ...),
    # ...
}

# 3. 每一步收集
info_to_log = infos.get("final_info", infos)
for metric_name in metric_specs.keys():
    metrics_storage[metric_name][step] = info_to_log.get(metric_name, 0.0)

# 4. Rollout 结束后聚合
aggregate_metrics(metrics_storage, metric_specs, self.episode_metrics)
```

## 主要优势

### ✅ 代码统一
- Train 和 Eval 使用相同的 rollout 逻辑
- 消除了 ~100 行重复代码
- Metrics 收集保证一致性

### ✅ 灵活性
- 通过参数控制行为，易于扩展
- 支持不同的 policy函数（stochastic vs deterministic）
- 可选的 per-env step data 记录

### ✅ 性能优化
- 所有 metrics 在 GPU 上收集
- 延迟传输（rollout 结束后一次性 GPU→CPU）
- 向量化操作

### ✅ 可维护性
- 清晰的职责分离
- 易于添加新指标（只需在 TaskHandler 中声明）
- 统一的 logging 逻辑

## Metrics 处理流程

### Training
1. `_rollout()` 收集 metrics → `self.episode_metrics`
2. `_build_reward_component_logs()` 处理为 wandb logs
3. 使用 `rollout/` 前缀（如 `rollout/success_rate`）
4. 清空 `self.episode_metrics`

### Evaluation
1. 清空 `self.episode_metrics`（与 training 分离）
2. `_rollout()` 收集 metrics → `self.episode_metrics`
3. `_build_eval_logs()` 处理为 wandb logs
4. 使用 `eval/` 前缀（如 `eval/success_rate`）
5. 清空 `self.episode_metrics`

### Per-Env CSV（仅 Eval）
- 可选功能（`save_step_csv = True`）
- 记录每个环境每一步的所有 metrics
- 保存到 `split/evalN/envM/rewards.csv`
- 包含：step, reward, success, fail, grasp_reward, lift_reward, 等等

## 测试建议

1. **Training**: 运行几个 iterations，检查 `rollout/*` logs
2. **Evaluation**: 触发 eval，检查 `eval/*` logs 和 CSV 文件
3. **Metrics 一致性**: 对比 train 和 eval 的 metrics 定义
4. **CSV 内容**: 验证 CSV 包含所有预期的 metrics

## 回答你的问题

### 问题 1: episode_info 字段

- **`return`**: Episode累计总回报
- **`reward`**: 平均每步 reward = `return / episode_len`
- **`episode_len`**: Episode 已运行步数
- **`success_once`**: ✅ 支持 `ignore_terminations=True`（记录是否曾经成功）

### 问题 2: Per-Env 记录设计

**新设计**：
- 使用统一的 metrics 收集（与 training 一致）
- 通过 `record_step_data=True` 开启
- 每环境每步记录所有 metrics
- 一次性保存到 CSV

### 问题 3: 重用 Rollout

✅ **已实现！**
- Train 和 Eval 完全重用同一个 `_rollout()` 方法
- 通过参数控制差异行为
- 大幅减少代码重复，提高可维护性
