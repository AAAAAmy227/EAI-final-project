---
description: 重构 SO101 类属性管理 - 消除全局状态修改
---

# 任务：重构 SO101 类属性管理

## 📋 任务概述

`scripts/so101.py` 和 `scripts/training/env_utils.py` 中存在多处**修改类属性**的全局状态操作，这在多进程/多配置环境中可能导致问题。需要将这些配置改为实例级别或使用工厂模式。

---

## 🎯 目标

1. 消除 `SO101.active_mode = "dual"` 等类属性直接修改
2. 提供更安全的配置传递机制
3. 将硬编码的 Agent 名称（如 `"so101-1"`）集中管理

---

## 📍 问题定位

### 问题 1: 类属性全局修改

**文件**: `scripts/track1_env.py` (约 150-165 行)
```python
# 当前代码 - 修改类属性
if self.task == "sort":
    SO101.active_mode = "dual"
else:
    SO101.active_mode = "single"

if cfg.action_bounds is not None:
    if self.task == "sort":
        SO101.action_bounds_dual_arm = cfg.action_bounds
    else:
        SO101.action_bounds_single_arm = cfg.action_bounds
```

**问题**: 如果同一进程中创建多个不同配置的环境，类属性会被覆盖。

---

### 问题 2: 硬编码的 Agent 名称

**文件**: `scripts/training/env_utils.py` (第 171 行)
```python
def __init__(self, env, right_arm_key="so101-1", left_arm_key="so101-0"):
```

**问题**: 
- `"so101-0"` 和 `"so101-1"` 是 ManiSkill 根据 `robot_uids` 自动生成的
- 如果 `uid` 改变或使用不同机器人，这里会失效

---

## 📝 重构方案

### 方案 A: SO101 常量定义

在 `so101.py` 中添加常量：

```python
class SO101(BaseAgent):
    uid = "so101"
    
    # Agent instance naming convention (used by ManiSkill)
    # When using robot_uids=("so101", "so101"), instances are named:
    LEFT_AGENT_SUFFIX = "-0"   # First in tuple
    RIGHT_AGENT_SUFFIX = "-1"  # Second in tuple
    
    @classmethod
    def get_agent_key(cls, side: str) -> str:
        """Get the agent key for left/right arm.
        
        Args:
            side: "left" or "right"
        Returns:
            e.g., "so101-0" for left, "so101-1" for right
        """
        suffix = cls.LEFT_AGENT_SUFFIX if side == "left" else cls.RIGHT_AGENT_SUFFIX
        return f"{cls.uid}{suffix}"
```

### 方案 B: 更新 SingleArmWrapper

```python
class SingleArmWrapper(gym.ActionWrapper):
    def __init__(self, env, right_arm_key=None, left_arm_key=None):
        super().__init__(env)
        
        # Auto-detect from SO101 if not provided
        if right_arm_key is None or left_arm_key is None:
            from scripts.so101 import SO101
            right_arm_key = right_arm_key or SO101.get_agent_key("right")
            left_arm_key = left_arm_key or SO101.get_agent_key("left")
        
        self.right_arm_key = right_arm_key
        self.left_arm_key = left_arm_key
        # ...
```

### 方案 C: 解决类属性修改问题

**选项 C1**: 使用 `configure_from_cfg` 返回配置后的类（当前已有，但仍修改全局）

**选项 C2**: 使用动态类创建（推荐）

```python
# so101.py
@classmethod
def create_configured_class(cls, mode: str, action_bounds: dict = None):
    """Create a new class with specific configuration.
    
    This avoids modifying global class state.
    """
    class ConfiguredSO101(cls):
        active_mode = mode
        
        if mode == "dual" and action_bounds:
            action_bounds_dual_arm = action_bounds
        elif mode == "single" and action_bounds:
            action_bounds_single_arm = action_bounds
    
    return ConfiguredSO101
```

然后在 `track1_env.py` 中：
```python
ConfiguredSO101 = SO101.create_configured_class(
    mode="single" if self.task != "sort" else "dual",
    action_bounds=cfg.action_bounds
)
# 使用 ConfiguredSO101 或注册新的 agent
```

**注意**: ManiSkill 的 agent 注册机制可能不支持动态类。需要评估可行性。

---

## ⚠️ 注意事项

1. **ManiSkill 兼容性**: 确保修改不破坏 `register_agent()` 装饰器的行为。
2. **向后兼容**: 保留 `configure_from_cfg` 方法作为 fallback。
3. **测试**: 必须在单进程中创建多个不同配置的环境来验证隔离性。

---

## ✅ 验收标准

1. **常量定义**: `SO101.get_agent_key("left")` 返回 `"so101-0"`
2. **消除硬编码**: `SingleArmWrapper` 不再包含字符串 `"so101-0"` 或 `"so101-1"`
3. **功能正常**: 现有训练流程无回归

---

// turbo-all
