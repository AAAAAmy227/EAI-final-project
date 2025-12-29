---
description: 重构 Track1Env 奖励函数 - 拆分至独立模块
---

# 任务：重构 Track1Env 奖励函数

## 📋 任务概述

`scripts/track1_env.py` 中的奖励计算逻辑（约 1698-2053 行）非常庞大且难以维护。需要将其拆分到独立的任务奖励类中，由 `Track1Env` 进行调度。

## 🎯 目标

1. 创建 `scripts/tasks/` 目录和任务基类
2. 为 `lift`, `stack`, `sort` 任务创建独立的处理器类
3. 重构 `Track1Env` 使其使用任务处理器
4. 确保自适应权重的 EMA 状态得到正确保留

---

## 📍 架构设计

创建以下文件结构：
```
scripts/tasks/
├── __init__.py
├── base.py       # 任务基类
├── lift.py       # Lift 任务逻辑
├── stack.py      # Stack 任务逻辑
└── sort.py       # Sort 任务逻辑
```

### 1. 基类定义 (`scripts/tasks/base.py`)

```python
from abc import ABC, abstractmethod
import torch

class BaseTaskHandler(ABC):
    def __init__(self, env):
        self.env = env
        self.device = env.device

    @abstractmethod
    def evaluate(self) -> dict:
        pass

    @abstractmethod
    def compute_dense_reward(self, info, action=None) -> torch.Tensor:
        pass
    
    @abstractmethod
    def initialize_episode(self, env_idx, options):
        pass
```

### 2. 任务分发重构 (`scripts/track1_env.py`)

**在 `__init__` 中初始化处理器:**
```python
def __init__(self, ...):
    # ...
    self.task_handler = self._create_task_handler(self.task)
    # ...

def _create_task_handler(self, task):
    if task == "lift":
        from scripts.tasks.lift import LiftTaskHandler
        return LiftTaskHandler(self)
    elif task == "stack":
        from scripts.tasks.stack import StackTaskHandler
        return StackTaskHandler(self)
    # ...
```

**重写入口方法:**
```python
def evaluate(self):
    return self.task_handler.evaluate()

def compute_dense_reward(self, obs, action, info):
    return self.task_handler.compute_dense_reward(info, action)

def _initialize_episode(self, env_idx, options):
    super()._initialize_episode(env_idx, options) # Handle robots
    self.task_handler.initialize_episode(env_idx, options)
```

---

## 📝 详细迁移指导

### 迁移项 A: 自适应权重状态
`self.grasp_success_rate`, `self.lift_success_rate`, `self.task_success_rate` 等 EMA 状态应迁移到 `LiftTaskHandler` 内部。

### 迁移项 B: 辅助方法
`_get_gripper_pos()`, `_get_moving_jaw_pos()` 等几何计算方法建议保留在 `Track1Env` 中作为工具方法，或者移动到 `scripts/utils/geometry.py`。

### 迁移项 C: 奖励组件日志
确保 `info["reward_components"]` 的填充逻辑在新的 Handler 中完整保留。

---

## ⚠️ 注意事项

1. **循环引用**: 确保 `tasks/*.py` 文件中只在方法内导入 `Track1Env`（如果需要类型注解使用 `TYPE_CHECKING`）。
2. **性能**: 奖励函数每步都会调用，确保使用的 Tensor 操作是高效的并保持在 GPU 上。
3. **兼容性**: 确保 `info` 字典中返回的键名与 `PPORunner` 期望的完全一致。

---

## ✅ 验收标准

1. **代码精简**: `track1_env.py` 减少约 400-500 行代码。
2. **功能一致**: 
   - 运行测试确保 `reward` 值与重构前完全匹配。
   - 验证 `success` 和 `fail` 触发逻辑正常。
3. **可扩展性**: 增加新任务现在只需添加一个文件，而无需修改主环境类。

---

// turbo-all
