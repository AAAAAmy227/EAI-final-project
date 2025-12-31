# Static Grasp Task - Simplified Debugging Task

## 概述

`static_grasp` 任务是一个简化的调试任务，用于隔离和测试抓取检测功能，消除物理模拟的干扰因素。

### 关键特性

1. **固定位置的Cube（静态物理）**
   - Cube在每个episode开始时**随机生成**在配置的spawn区域
   - Cube是**kinematic**（运动学）模式 - 不受重力、碰撞等物理力影响
   - Cube保持在初始位置，像"固定在桌面上"一样

2. **简单的成功标准**
   - 持续抓取cube **N秒**（默认3秒）
   - 不需要lift、不需要考虑cube掉落

3. **简化的奖励函数**
   - **Approach**: 鼓励机械臂接近cube
   - **Grasp**: 成功抓取的奖励
   - **Hold Progress**: 持续抓取的渐进奖励
   - **Success**: 完成任务的大额奖励

### 与Lift任务的对比

| 特性 | Lift Task | Static Grasp Task |
|------|-----------|-------------------|
| Cube物理 | Dynamic（动态） | Kinematic（静态） |
| Cube位置 | 随机 + 物理模拟 | 随机但固定 |
| 成功条件 | Lift到高度 + 保持 | 持续抓取N秒 |
| 失败条件 | 掉落、出界 | 仅超时 |
| 奖励复杂度 | 8+ 组件 | 4 组件 |

## 使用方法

### 1. 快速测试

运行测试脚本验证任务是否正常工作：

```bash
python test_static_grasp.py
```

这将：
- 创建4个并行环境
- 运行2个测试episode
- 验证cube是kinematic模式
- 显示grasp检测和计数器是否工作

### 2. 训练

使用专用配置文件启动训练：

```bash
python scripts/training/train.py --config-name train_static_grasp
```

或者指定实验名称：

```bash
python scripts/training/train.py --config-name train_static_grasp exp_name=debug_grasp_v1
```

### 3. 评估

使用训练好的checkpoint进行评估：

```bash
python scripts/training/train.py --config-name train_static_grasp \
    checkpoint=runs/static_grasp_debug/model.pt \
    training.num_envs=0  # 仅评估模式
```

## 配置文件

### 主配置：`configs/train_static_grasp.yaml`

关键修改：
- `env.task: static_grasp`
- 更少的并行环境（512 vs 2048）用于更清晰的监控
- 更频繁的评估（每10次更新 vs 25次）

### 奖励配置：`configs/reward/static_grasp.yaml`

简化的奖励权重：
```yaml
weights:
  approach: 1.0
  grasp: 5.0
  hold_progress: 10.0
  success: 50.0
  action_rate: -0.1
```

## 预期输出和调试

### 关键指标

在训练日志或WandB中查看：

1. **grasp_success** - 抓取成功率
2. **grasp_hold_steps** - 平均持续抓取步数
3. **success** - 任务成功率（持续抓取>=N秒）

### 调试场景

#### 场景1: Grasp检测不工作
**症状**: `grasp_success` 始终为0或非常低
**可能原因**:
- `grasp_min_force` 阈值过高
- `grasp_max_angle` 阈值过严格
- 接触力向量计算有误

**调试**:
- 降低 `grasp_min_force` (例如 0.5)
- 增加 `grasp_max_angle` (例如 120)
- 添加日志打印接触力和角度值

#### 场景2: Grasp成功但hold_steps不增加
**症状**: `grasp_success > 0` 但 `grasp_hold_steps` 很低
**可能原因**:
- 抓取不稳定，频繁失去接触
- Counter重置逻辑有误

**调试**:
- 查看评估视频，观察机械臂行为
- 添加日志记录每一步的is_grasped状态

#### 场景3: 训练无进展
**症状**: 奖励不增长，成功率为0
**可能原因**:
- Approach奖励不足以引导
- 探索不够

**调试**:
- 增加approach权重
- 增加action entropy coefficient
- 检查observation normalization是否合理

## 代码位置

- **任务Handler**: `scripts/tasks/static_grasp.py`
- **环境注册**: `scripts/envs/track1_env.py:205`
- **配置文件**: 
  - `configs/train_static_grasp.yaml`
  - `configs/reward/static_grasp.yaml`
- **测试脚本**: `test_static_grasp.py`

## 实现细节

### Kinematic模式设置

在 `initialize_episode()` 中：

```python
# Make cube STATIC (kinematic) - won't be affected by physics
for i, obj in enumerate(self.env.red_cube._objs):
    obj.set_kinematic(True)
```

### 持续抓取计数

```python
# Increment counter when grasping, reset when not
self.continuous_grasp_counter = torch.where(
    is_grasped,
    self.continuous_grasp_counter + 1,
    torch.zeros_like(self.continuous_grasp_counter)
)
```

### 成功条件

```python
success = self.continuous_grasp_counter >= self.required_grasp_steps
```

## 下一步

1. **验证Grasp检测**: 如果static_grasp任务能正常工作，说明grasp检测功能是正常的
2. **对比Lift任务**: 如果static_grasp成功但lift失败，问题可能在于：
   - Cube物理参数（摩擦力、质量等）
   - Lift高度控制
   - 水平位移约束
3. **渐进式调试**: 可以创建中间任务，例如"static_lift"（cube固定，只测试lift动作）

## 常见问题

**Q: Cube真的不会动吗？**
A: 是的，kinematic模式下cube完全不受物理力影响。即使机械臂碰撞它，它也不会移动。

**Q: 如何知道抓取检测是否工作？**
A: 查看`is_grasped`观测值或在reward components中查看`grasp_reward`。如果机械臂接触cube时这些值变为1，说明检测正常。

**Q: 这个任务太简单了，会不会训练出的策略无法迁移？**
A: 这个任务的目的是**调试**，不是训练最终策略。一旦确认grasp检测和基本接近行为正常，应该切换回lift任务。

**Q: 可以修改持续抓取时间吗？**
A: 可以，修改`configs/reward/static_grasp.yaml`中的`stable_hold_time`（单位：秒）。
