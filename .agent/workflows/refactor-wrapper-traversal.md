---
description: 重构 Wrapper 遍历代码 - 提取公共工具函数消除重复
---

# 任务：重构 Wrapper 遍历代码

## 📋 任务概述

在 `scripts/training/runner.py` 中存在 **6 处重复的 Wrapper 遍历模式**。需要提取一个通用工具函数来消除重复代码，提高可维护性。

## 🎯 目标

1. 在 `scripts/training/env_utils.py` 中添加 `find_wrapper()` 工具函数
2. 重构 `runner.py` 中所有重复的遍历代码，使用新的工具函数
3. 确保所有功能保持不变（向后兼容）

---

## 📍 需要修改的文件

### 文件 1: `scripts/training/env_utils.py`

**新增函数** (建议添加在 `RunningMeanStd` 类之前，约第 20 行):

```python
from typing import TypeVar, Optional, Type

T = TypeVar('T')

def find_wrapper(env, wrapper_type: Type[T]) -> Optional[T]:
    """Traverse wrapper chain to find a specific wrapper type.
    
    Args:
        env: The wrapped environment to search from (outermost wrapper)
        wrapper_type: The wrapper class to find
        
    Returns:
        The wrapper instance if found, None otherwise
        
    Example:
        >>> obs_wrapper = find_wrapper(envs, NormalizeObservationGPU)
        >>> if obs_wrapper is not None:
        ...     print(obs_wrapper.rms.mean)
    """
    curr = env
    while curr is not None:
        if isinstance(curr, wrapper_type):
            return curr
        # Try common wrapper attribute names
        curr = getattr(curr, "env", getattr(curr, "_env", None))
    return None
```

---

### 文件 2: `scripts/training/runner.py`

需要修改 **6 处** wrapper 遍历代码：

#### 修改点 1: `_get_obs_names_from_wrapper()` (第 246-269 行)

**当前代码:**
```python
def _get_obs_names_from_wrapper(self) -> list:
    from scripts.training.env_utils import FlattenStateWrapper
    
    curr_env = self.envs
    while curr_env is not None:
        if isinstance(curr_env, FlattenStateWrapper):
            return curr_env.obs_names
        if hasattr(curr_env, "_env"):
            curr_env = curr_env._env
        elif hasattr(curr_env, "env"):
            curr_env = curr_env.env
        else:
            break
    
    print("Warning: FlattenStateWrapper not found, using generic obs names")
    return [f"obs_{i}" for i in range(self.n_obs)]
```

**重构后:**
```python
def _get_obs_names_from_wrapper(self) -> list:
    from scripts.training.env_utils import FlattenStateWrapper, find_wrapper
    
    wrapper = find_wrapper(self.envs, FlattenStateWrapper)
    if wrapper is not None:
        return wrapper.obs_names
    
    print("Warning: FlattenStateWrapper not found, using generic obs names")
    return [f"obs_{i}" for i in range(self.n_obs)]
```

---

#### 修改点 2: `_get_action_names_from_wrapper()` (第 271-294 行)

**当前代码:**
```python
def _get_action_names_from_wrapper(self) -> list:
    from scripts.training.env_utils import FlattenActionWrapper
    
    curr_env = self.envs
    while curr_env is not None:
        if isinstance(curr_env, FlattenActionWrapper):
            return curr_env.action_names
        if hasattr(curr_env, "_env"):
            curr_env = curr_env._env
        elif hasattr(curr_env, "env"):
            curr_env = curr_env.env
        else:
            break
    
    print("Warning: FlattenActionWrapper not found, using generic action names")
    return [f"act_{i}" for i in range(self.n_act)]
```

**重构后:**
```python
def _get_action_names_from_wrapper(self) -> list:
    from scripts.training.env_utils import FlattenActionWrapper, find_wrapper
    
    wrapper = find_wrapper(self.envs, FlattenActionWrapper)
    if wrapper is not None:
        return wrapper.action_names
    
    print("Warning: FlattenActionWrapper not found, using generic action names")
    return [f"act_{i}" for i in range(self.n_act)]
```

---

#### 修改点 3: `_initialize_obs_stats_from_config()` (第 296-310 行)

**当前代码:**
```python
def _initialize_obs_stats_from_config(self):
    from scripts.training.env_utils import NormalizeObservationGPU
    
    obs_wrapper = None
    curr_env = self.envs
    while curr_env is not None:
        if isinstance(curr_env, NormalizeObservationGPU):
            obs_wrapper = curr_env
            break
        curr_env = getattr(curr_env, "env", None)
        
    if obs_wrapper is None:
        return
    # ... rest of function
```

**重构后:**
```python
def _initialize_obs_stats_from_config(self):
    from scripts.training.env_utils import NormalizeObservationGPU, find_wrapper
    
    obs_wrapper = find_wrapper(self.envs, NormalizeObservationGPU)
    if obs_wrapper is None:
        return
    # ... rest of function (保持不变)
```

---

#### 修改点 4: `train()` 方法内日志部分 - 查找 obs wrapper (第 603-610 行)

**当前代码:** (在 `if self.log_obs_stats:` 块内)
```python
from scripts.training.env_utils import NormalizeObservationGPU, NormalizeRewardGPU

# Log Observation statistics from wrapper
obs_wrapper = None
curr_env = self.envs
while curr_env is not None:
    if isinstance(curr_env, NormalizeObservationGPU):
        obs_wrapper = curr_env
        break
    curr_env = getattr(curr_env, "env", None)
```

**重构后:**
```python
from scripts.training.env_utils import NormalizeObservationGPU, NormalizeRewardGPU, find_wrapper

# Log Observation statistics from wrapper
obs_wrapper = find_wrapper(self.envs, NormalizeObservationGPU)
```

---

#### 修改点 5: `train()` 方法内日志部分 - 查找 reward wrapper (第 620-627 行)

**当前代码:**
```python
# Log Reward statistics from wrapper
reward_wrapper = None
curr_env = self.envs
while curr_env is not None:
    if isinstance(curr_env, NormalizeRewardGPU):
        reward_wrapper = curr_env
        break
    curr_env = getattr(curr_env, "env", None)
```

**重构后:**
```python
# Log Reward statistics from wrapper
reward_wrapper = find_wrapper(self.envs, NormalizeRewardGPU)
```

---

#### 修改点 6: `_save_checkpoint()` 方法 (第 901-910 行)

**当前代码:**
```python
def _save_checkpoint(self, iteration):
    if self.cfg.save_model:
        from scripts.training.env_utils import NormalizeObservationGPU, NormalizeRewardGPU
        # ...
        
        # Find wrappers to extract stats
        obs_wrapper = None
        reward_wrapper = None
        curr_env = self.envs
        while curr_env is not None:
            if isinstance(curr_env, NormalizeObservationGPU):
                obs_wrapper = curr_env
            elif isinstance(curr_env, NormalizeRewardGPU):
                reward_wrapper = curr_env
            curr_env = getattr(curr_env, "env", None)
```

**重构后:**
```python
def _save_checkpoint(self, iteration):
    if self.cfg.save_model:
        from scripts.training.env_utils import NormalizeObservationGPU, NormalizeRewardGPU, find_wrapper
        # ...
        
        # Find wrappers to extract stats
        obs_wrapper = find_wrapper(self.envs, NormalizeObservationGPU)
        reward_wrapper = find_wrapper(self.envs, NormalizeRewardGPU)
```

---

## ⚠️ 注意事项

### 1. Wrapper 链遍历顺序
- ManiSkill 的 wrapper 链结构是 `VectorEnv -> RecordEpisode -> NormalizeObs -> ... -> BaseEnv`
- 必须支持两种属性名: `.env` (gymnasium 标准) 和 `._env` (某些 wrapper 使用)

### 2. 类型安全
- `find_wrapper` 返回 `Optional[T]`，调用方必须检查 `None`
- 原代码中的 fallback 逻辑必须保留

### 3. import 语句位置
- 原代码将 import 放在函数内部 (延迟导入以避免循环依赖)
- 重构时也应保持这一模式，将 `find_wrapper` 加入现有 import 语句

### 4. 不要修改的逻辑
- `_save_checkpoint` 中同时查找两种 wrapper 的逻辑 (非 early-return 模式) 
- 各函数的 fallback 行为 (如打印 warning, 返回默认值)

---

## ✅ 验收标准

1. **功能不变:** 训练流程运行正常，日志输出与重构前一致
2. **代码简化:** `runner.py` 减少约 30-40 行重复代码
3. **类型正确:** `find_wrapper` 有正确的类型注解
4. **测试通过:** 运行以下命令验证:
   ```bash
   cd /home/admin/Desktop/eai-final-project
   # 语法检查
   uv run python -m py_compile scripts/training/env_utils.py
   uv run python -m py_compile scripts/training/runner.py
   
   # 导入测试
   uv run python -c "from scripts.training.env_utils import find_wrapper; print('OK')"
   ```

---

## 📁 相关文件路径

- `/home/admin/Desktop/eai-final-project/scripts/training/env_utils.py` (新增函数)
- `/home/admin/Desktop/eai-final-project/scripts/training/runner.py` (重构 6 处)

---

## 🔧 执行建议

1. 先在 `env_utils.py` 添加 `find_wrapper` 函数
2. 从最简单的修改点开始 (`_get_obs_names_from_wrapper`)
3. 逐个修改并测试
4. 最后运行完整语法检查

// turbo-all
