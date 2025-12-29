---
description: 重构 Track1Env 配置管理 - 提取配置解析逻辑
---

# 任务：重构 Track1Env 配置管理

## 📋 任务概述

`scripts/track1_env.py` 的构造函数和 `_setup_reward_config` 包含了大量繁琐的配置提取逻辑（约 270 行）。需要将其提取到结构化的 Dataclass 中，以提高可读性和类型安全。

## 🎯 目标

1. 定义 `Track1Config` 及其嵌套的配置类（Physics, Reward, Obs）
2. 实现从 Hydra `DictConfig` 到 `Track1Config` 的解析器
3. 简化 `Track1Env.__init__` 的初始化逻辑

---

## 📍 设计方案

### 1. 配置模型 (`scripts/training/config_utils.py`)

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class PhysicsConfig:
    static_friction: float = 0.6
    dynamic_friction: float = 0.6
    restitution: float = 0.0
    mass: Optional[float] = None

@dataclass
class RewardConfig:
    weights: Dict[str, float] = field(default_factory=dict)
    approach_mode: str = "dual_point"
    lift_target: float = 0.05
    stable_hold_time: float = 0.0
    # ... 其他 20+ 个参数

@dataclass
class Track1Config:
    task: str = "lift"
    domain_randomization: bool = True
    reward: RewardConfig = field(default_factory=RewardConfig)
    cube_physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    # ...
```

### 2. 重构 `Track1Env`

**初始化简化:**

```python
def __init__(self, *args, cfg=None, **kwargs):
    # 解析配置
    self.track1_cfg = Track1Config.from_hydra(cfg) if cfg else Track1Config()
    
    # 赋值
    self.task = self.track1_cfg.task
    self.reward_weights = self.track1_cfg.reward.weights
    # ...
```

---

## 📝 详细工作项

1. **识别所有配置项**: 仔细扫描 `track1_env.py` 第 48-155 行和 195-335 行的所有参数。
2. **处理默认值**: 确保 Dataclass 中的默认值与原代码逻辑完全一致。
3. **支持 Legacy 模式**: `Track1Env` 仍需支持直接传入参数（如 `task="lift"`），可以在 `from_hydra` 之后手动覆盖对象属性。
4. **清理方法**: 移除庞大的 `_setup_reward_config` 方法，改为直接从 `self.track1_cfg.reward` 读取。

---

## ⚠️ 注意事项

1. **类型匹配**: 注意 Hydra 配置中有些值可能是 `DictConfig` 或者是 `None`，解析时需调用 `OmegaConf.to_container`。
2. **层级结构**: 保持配置的层级结构（env, reward, obs, control）与 YAML 配置文件一致，方便理解。

---

## ✅ 验收标准

1. **代码可读性**: `__init__` 方法应从原来的 130 行缩减至 40 行以内。
2. **零功能变动**: 运行环境并打印 `self` 中的配置属性，确保与重构前完全相同。
3. **IDE 支持**: 使用属性访问取代 `get("key", default)`，获得良好的代码补全和类型检查。

---

// turbo-all
