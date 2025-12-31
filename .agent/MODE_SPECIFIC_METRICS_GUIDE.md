# Mode-Specific Metrics 使用示例

## 概述

现在 TaskHandler 支持为 training 和 evaluation 定义不同的 metrics。

## 实现方式

### 1. BaseTaskHandler 提供的方法

```python
class BaseTaskHandler(ABC):
    @classmethod
    def _get_train_metrics(cls) -> Dict[str, str]:
        """定义 training 期间收集的 metrics"""
        return {}
    
    @classmethod
    def _get_eval_metrics(cls) -> Dict[str, str]:
        """定义 evaluation 期间收集的 metrics
        
        默认返回 _get_train_metrics()（即 train 和 eval 相同）
        """
        return cls._get_train_metrics()
```

### 2. 在 TaskHandler 中使用

#### 示例 1: Train 和 Eval 使用相同的 metrics（默认行为）

```python
class LiftTaskHandler(BaseTaskHandler):
    @classmethod
    def _get_train_metrics(cls) -> Dict[str, str]:
        """Train 和 Eval 都使用这些 metrics"""
        return {
            "grasp_reward": "mean",
            "lift_reward": "mean",
            "moving_distance": "mean",
            "grasp_success": "mean",
            "lift_success": "mean",
        }
    # _get_eval_metrics() 不需要覆盖，默认返回 _get_train_metrics()
```

#### 示例 2: Train 和 Eval 使用不同的 metrics

```python
class StackTaskHandler(BaseTaskHandler):
    @classmethod
    def _get_train_metrics(cls) -> Dict[str, str]:
        """Training 只收集关键 metrics（减少开销）"""
        return {
            "grasp_reward": "mean",
            "stack_reward": "mean",
        }
    
    @classmethod
    def _get_eval_metrics(cls) -> Dict[str, str]:
        """Evaluation 收集更详细的 metrics"""
        return {
            # 包含所有 train metrics
            "grasp_reward": "mean",
            "stack_reward": "mean",
            # 额外的 eval-only metrics
            "cube_alignment": "mean",
            "stack_stability": "mean",
            "gripper_distance": "mean",
            "cube_velocity": "mean",
            "detailed_success_breakdown": "mean",
        }
```

#### 示例 3: Train 收集更多，Eval 只收集关键指标

```python
class ComplexTaskHandler(BaseTaskHandler):
    @classmethod
    def _get_train_metrics(cls) -> Dict[str, str]:
        """Training 收集所有 metrics 用于调试"""
        return {
            "step_reward": "mean",
            "grasp_reward": "mean",
            "approach_reward": "mean",
            "velocity_penalty": "mean",
            "smoothness_reward": "mean",
            # ... 更多调试用的 metrics
        }
    
    @classmethod
    def _get_eval_metrics(cls) -> Dict[str, str]:
        """Evaluation 只关心最终结果"""
        return {
            "final_success": "mean",
            "time_to_completion": "mean",
        }
```

## 自动模式切换

在 `_rollout()` 中自动根据 `collect_for_training` 参数选择 mode：

```python
# Training
next_obs, container, _ = self._rollout(
    next_obs, self.num_steps,
    collect_for_training=True,  # → mode="train"
    # ...
)

# Evaluation  
_, _, step_data = self._rollout(
    eval_obs, max_steps,
    collect_for_training=False,  # → mode="eval"
    # ...
)
```

## 实际效果

### Training Logs
```python
wandb.log({
    "rollout/success_rate": 0.75,
    "rollout/return": 10.5,
    "reward/grasp_reward": 2.3,
    "reward/stack_reward": 8.2,
}, step=10240)
```

### Evaluation Logs
```python
wandb.log({
    "eval/success_rate": 0.82,
    "eval/return": 12.1,
    "eval_reward/grasp_reward": 2.5,
    "eval_reward/stack_reward": 9.6,
    # 额外的 eval-only metrics
    "eval_reward/cube_alignment": 0.95,
    "eval_reward/stack_stability": 0.87,
    "eval_reward/gripper_distance": 0.12,
}, step=10240)
```

## 默认 Metrics

无论 train 还是 eval，以下 metrics 总是可用（来自 `DEFAULT_METRIC_AGGREGATIONS`）：

- `success`: 成功率
- `fail`: 失败率
- `raw_reward`: 原始奖励
- `return`: Episode 总回报
- `episode_len`: Episode 长度
- `success_once`: 是否成功过
- `fail_once`: 是否失败过

## 使用建议

1. **默认情况**: 不需要覆盖任何方法，train 和 eval 使用相同的 metrics
2. **轻量 Training**: 在 `_get_train_metrics()` 中只返回必要的 metrics，减少计算开销
3. **详细 Evaluation**: 在 `_get_eval_metrics()` 中返回更多 metrics，用于详细分析
4. **性能考虑**: 每个 metric 都需要 GPU 内存和计算，只收集需要的

## 迁移现有代码

如果你有旧的 `get_custom_metric_aggregations()` 方法：

```python
# 旧代码
@classmethod
def get_custom_metric_aggregations(cls) -> Dict[str, str]:
    return {"grasp_reward": "mean"}

# 新代码（重命名即可）
@classmethod
def _get_train_metrics(cls) -> Dict[str, str]:
    return {"grasp_reward": "mean"}
```

Eval 会自动使用相同的 metrics（通过 base class 的 `_get_eval_metrics()` 默认实现）。
