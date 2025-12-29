---
description: 修复 PPORunner 中未定义的方法调用
---

# 任务：修复 PPORunner 未定义方法

## 📋 任务概述

`scripts/training/runner.py` 中调用了 `self._flatten_obs()` 和 `self._normalize_obs()` 方法，但这些方法**未在类中定义**。需要分析这些调用是否必要，并进行修复。

## 🎯 目标

1. 分析 wrapper 链是否已处理 flatten 和 normalize
2. 如果 wrapper 已处理：删除冗余调用
3. 如果需要保留：实现这些方法

---

## 📍 问题定位

### 问题调用点

| 行号 | 代码 | 上下文 |
|------|------|--------|
| 445 | `next_obs = self._flatten_obs(next_obs).to(self.device)` | `train()` 初始 reset 后 |
| 480 | `norm_next_obs = self._normalize_obs(next_obs)` | GAE 计算前 |
| 496 | `container["obs"] = self._normalize_obs(container["obs"])` | PPO update 前 |

---

## 🔍 分析背景

### Wrapper 链结构 (从外到内)

```
ManiSkillVectorEnv
  └── RecordEpisode (eval only)
        └── NormalizeObservationGPU (if normalize_obs=true)
              └── FlattenStateWrapper  ← 已处理 flatten
                    └── FlattenActionWrapper
                          └── SingleArmWrapper (lift/stack)
                                └── Track1Env (BaseEnv)
```

### 关键发现

1. **`FlattenStateWrapper`** (env_utils.py:266-363): 
   - 在 `observation()` 方法中已将 dict obs 展平为 tensor
   - `reset()` 和 `step()` 自动调用 `self.observation()`

2. **`NormalizeObservationGPU`** (env_utils.py:60-86):
   - 在 `_normalize()` 方法中已标准化观测
   - `reset()` 和 `step()` 自动调用 `self._normalize()`

### 结论

- `self._flatten_obs()`: **冗余** - FlattenStateWrapper 已处理
- `self._normalize_obs()`: **需要保留但应改用 wrapper** - 在 rollout 期间 wrapper 自动处理，但 GAE 和 PPO update 需要手动调用

---

## 📝 修改方案

### 方案 A: 删除冗余调用 (推荐)

由于 wrapper 已处理 flatten 和 normalize，rollout 阶段返回的 obs 已经是展平且标准化的。问题在于:

1. **第 445 行**: `reset()` 返回的 obs 已经过 wrapper 处理，无需再 flatten
2. **第 480, 496 行**: 需要确认 rollout 阶段存储的 obs 是否已标准化

### 决定因素

查看 `_rollout()` 方法:
- 第 372 行: `storage["obs"][step] = obs` - 存储当前 obs
- 第 430 行: `obs = next_obs` - next_obs 来自 `_step_env()`

如果 `NormalizeObservationGPU` wrapper 在 step 时返回标准化的 obs，则 storage 中的 obs 已经是标准化的，第 496 行是**重复标准化**！

---

## 📍 需要修改的代码

### 文件: `scripts/training/runner.py`

#### 修改点 1: 删除 `_flatten_obs` 调用 (第 445 行)

**当前代码:**
```python
# Initial reset
next_obs, _ = self.envs.reset(seed=self.cfg.seed)
next_obs = self._flatten_obs(next_obs).to(self.device)
next_bootstrap_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
```

**修改后:**
```python
# Initial reset
# Note: FlattenStateWrapper already flattens obs, NormalizeObservationGPU already normalizes
next_obs, _ = self.envs.reset(seed=self.cfg.seed)
next_obs = next_obs.to(self.device) if not next_obs.device == self.device else next_obs
next_bootstrap_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
```

或者更简洁:
```python
# Initial reset (obs already flattened and normalized by wrappers)
next_obs, _ = self.envs.reset(seed=self.cfg.seed)
# next_obs is already on GPU from ManiSkillVectorEnv
next_bootstrap_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
```

---

#### 修改点 2: GAE 计算前的 normalize (第 478-481 行)

**当前代码:**
```python
# GAE Calculation
with torch.no_grad():
    norm_next_obs = self._normalize_obs(next_obs)
    next_value = self.get_value(norm_next_obs)
```

**分析:**
- `next_obs` 来自最后一次 `_step_env()`，已经过 `NormalizeObservationGPU` 处理
- 因此 `next_obs` 已经是标准化的！

**修改后:**
```python
# GAE Calculation
# Note: next_obs is already normalized by NormalizeObservationGPU wrapper
with torch.no_grad():
    next_value = self.get_value(next_obs)
```

---

#### 修改点 3: PPO Update 前的 normalize (第 493-496 行)

**当前代码:**
```python
# CRITICAL FIX: Normalize observations BEFORE flattening for PPO update
# This ensures the update phase sees the same distribution as the rollout phase
if self.normalize_obs:
    container["obs"] = self._normalize_obs(container["obs"])
```

**分析:**
- `container["obs"]` 在 `_rollout()` 第 372 行被填充
- 填充的 `obs` 来自第 370 行的循环，初始值来自 `train()` 传入的 `obs`
- 这些 obs 已经过 wrapper 处理，所以已经是标准化的！

**修改后:**
```python
# Note: container["obs"] was populated during rollout with already-normalized obs
# (NormalizeObservationGPU wrapper processes observations in step/reset)
# No additional normalization needed here
```

即：**删除这整个 if 块**

---

## ⚠️ 注意事项

### 1. 关于 `normalize_obs` 配置

当前代码中 `self.normalize_obs` 控制是否应用标准化。修改后:
- `normalize_obs=True`: wrapper 处理，无需手动调用
- `normalize_obs=False`: wrapper 不添加，obs 未标准化

### 2. 确认 Wrapper 顺序

修改前确认 `make_env()` 中 wrapper 添加顺序:
```python
# env_utils.py make_env()
env = FlattenStateWrapper(env)  # First: flatten
# ...
if normalize_obs:
    env = NormalizeObservationGPU(env)  # After flatten
```

### 3. 测试验证

修改后必须验证训练是否正常:
```bash
# 快速测试 (10 iterations)
uv run python scripts/train.py training.total_timesteps=100000 training.num_envs=64
```

---

## ✅ 验收标准

1. **语法正确**:
   ```bash
   uv run python -m py_compile scripts/training/runner.py
   ```

2. **运行测试**:
   ```bash
   uv run python -c "from scripts.training.runner import PPORunner; print('Import OK')"
   ```

3. **无 AttributeError**: 训练时不再出现 `_flatten_obs` 或 `_normalize_obs` 未定义错误

---

## 📁 相关文件路径

- `/home/admin/Desktop/eai-final-project/scripts/training/runner.py`
- `/home/admin/Desktop/eai-final-project/scripts/training/env_utils.py` (参考)

---

## 🔗 前置依赖

- `/refactor-wrapper-traversal` 应已完成 (提供 `find_wrapper` 工具函数)

---

// turbo-all
