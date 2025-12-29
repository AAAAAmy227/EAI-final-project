---
description: 重构 Track1Env 属性初始化 - 消除 hasattr 检查
---

# 任务：重构 Track1Env 属性初始化

## 📋 任务概述

`scripts/track1_env.py` 中散布着大量 `hasattr(self, '...')` 检查（约 20 处）。这种模式会隐藏潜在的初始化问题，且不利于静态检查。需要将所有动态属性集中到 `__init__` 中初始化。

## 🎯 目标

1. 识别所有延迟初始化的属性
2. 在 `__init__` 或专用的 `_init_state_tensors` 方法中预先定义它们
3. 用显式的空值检查（`if self.prop is not None`）替换 `hasattr`

---

## 📍 审计列表

需要处理的典型属性包括：

- **状态 Tensor**: `initial_red_cube_pos`, `initial_cube_xy`, `lift_hold_counter`, `grasp_hold_counter`, `prev_action`
- **配置参数**: `gripper_tip_offset`, `moving_jaw_tip_offset`, `lift_max_height`
- **内部组件**: `distortion_grid`, `undistortion_grid`

---

## 📝 修改方案

### 1. 显式初始化 (`__init__`)

```python
def __init__(self, ...):
    # ...
    # Initialize all dynamic attributes to None or default
    self.initial_red_cube_pos: Optional[torch.Tensor] = None
    self.initial_cube_xy: Optional[torch.Tensor] = None
    self.lift_hold_counter: Optional[torch.Tensor] = None
    
    # Offsets (with defaults from config)
    self.gripper_tip_offset = 0.0
    # ...
```

### 2. 状态重置中的初始化

对于需要在环境重置时初始化的 Tensor，可以在 `_initialize_episode` 中确保它们已存在：

```python
def _initialize_episode(self, env_idx, options):
    if self.initial_red_cube_pos is None:
        self.initial_red_cube_pos = torch.zeros((self.num_envs, 3), device=self.device)
    # ...
```

### 3. 代码清理

将如下模式：
```python
if hasattr(self, 'lift_hold_counter'):
    self.lift_hold_counter[env_idx] = 0
```
替换为：
```python
if self.lift_hold_counter is not None:
    self.lift_hold_counter[env_idx] = 0
```

---

## ⚠️ 注意事项

1. **类型安全性**: 加上类型提示（如 `: torch.Tensor`）有助于 IDE 更好地支持开发。
2. **Device 兼容性**: 确保在 `__init__` 中初始化的 Tensor 最终都能被正确地移动到正确的设备（通常在 `_setup_device` 阶段）。
3. **ManiSkill 属性**: 注意不要覆盖父类 `BaseEnv` 中同名的关键属性。

---

## ✅ 验收标准

1. **代码一致性**: 搜索文件确保 `hasattr(self, ...)` 的数量大幅减少（除了极少数真正需要动态探测的情况）。
2. **逻辑验证**: 确保原本依赖 `hasattr` 触发的逻辑（如懒加载）现在通过明确的初始化或重构的任务 Handler 正常工作。

---

// turbo-all
