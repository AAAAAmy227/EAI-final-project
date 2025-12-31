# Metrics Collection Refactoring Summary

## 概述

成功实现了混合方案的 metrics 收集系统，解决了你提出的 ManiSkill autoreset 处理问题。

## 关键设计决策

### 1. **ManiSkill Autoreset 行为理解**
- ✅ 只要任意一个环境 done，`final_info` 就会包含**所有环境**的旧状态  
- ✅ `infos["_final_info"]` 是布尔 mask，标记哪些环境实际 done 了
- ✅ 不需要复杂的合并逻辑 - 直接使用 `final_info` 即可获取 episode 完成时的信息

### 2. **混合方案设计**
采用约定优于配置的方式：
- **默认规则**: `BaseTaskHandler.DEFAULT_METRIC_AGGREGATIONS` 定义常见指标（success/fail/return/episode_len）
- **自定义扩展**: `TaskHandler.get_custom_metric_aggregations()` 定义任务特定指标
- **自动聚合**: 根据类型自动决定聚合方式（mean 或 sum）

### 3. **GPU 优化的 Rollout 收集**
- **预分配**: 为所有指标预分配 `(num_steps, num_envs)` GPU tensors
- **延迟传输**: 整个 rollout 完成后才进行一次 GPU → CPU 传输
- **向量化**: 使用 done_mask 一次性提取所有完成 episode 的数据

## 代码结构

### 文件修改

1. **`scripts/tasks/base.py`**
   - 添加 `DEFAULT_METRIC_AGGREGATIONS` 定义默认指标
   - 添加 `get_custom_metric_aggregations()` classmethod

2. **`scripts/tasks/lift.py`**
   - 实现 `get_custom_metric_aggregations()` 定义 Lift 任务的自定义指标
   - 包括: grasp_reward, lift_reward, moving_distance, grasp_success, lift_success

3. **`scripts/training/metrics_utils.py`** (新文件)
   - `get_metric_specs_from_env()`: 从环境的 task_handler 获取 metric specs
   - `aggregate_metrics()`: 批量聚合 metrics，使用 done_mask 提取完成 episodes

4. **`scripts/training/runner.py`**
   - 简化了 `__init__`: 移除复杂的累加器，使用简单的 `episode_metrics` dict
   - 重构了 `_rollout()`: 预分配 metrics storage，延迟聚合
   - 添加了 `_get_metric_specs()` 和 `_aggregate_metrics()` helper 方法
   - 重构了 `_build_reward_component_logs()`: 使用新的 episode_metrics 结构

## 默认处理的指标

### 1. **从 NormalizeRewardGPU 引入**
- `raw_reward`: 原始奖励（未归一化）

### 2. **从 ManiSkill VectorEnv 引入** (record_metrics=True)
- `return`: episode 总回报
- `episode_len`: episode 长度  
- `success_once`: 是否成功过（布尔）
- `fail_once`: 是否失败过（布尔）

### 3. **从 TaskHandler 引入**
- `success`: 最终是否成功
- `fail`: 最终是否失败
- Lift 任务额外指标: grasp_reward, lift_reward, moving_distance, grasp_success, lift_success

## 聚合类型

- **"mean"**: 收集所有值并计算平均（适用于 rewards、success rate 等）
- **"sum"**: 累加所有值（适用于 counts）

## 工作流程

1. **Rollout 开始**: 根据 metric_specs 预分配 GPU tensors
2. **每个 step**: 
   - 从 `infos.get("final_info", infos)` 提取指标
   - 存储到 pre-allocated tensors（全在 GPU）
3. **Rollout 结束**: 
   - 调用 `_aggregate_metrics()`
   - 使用 done_mask 提取完成 episodes
   - 一次性传输到 CPU 并添加到 episode_metrics
4. **Logging**: 
   - 从 episode_metrics 计算 mean
   - 清空 episode_metrics

## 优势

✅ **简洁**: 不需要为每个 episode 手动判断 done 和提取指标  
✅ **高效**: GPU 批量操作，减少 CPU-GPU 传输  
✅ **灵活**: 容易添加新的 task-specific 指标  
✅ **正确**: 正确处理 ManiSkill 的 autoreset 和 final_info 结构  

## 注意事项

- `final_info` 是一个完整的字典（不是列表），结构与普通 `infos` 相同
- 所有 episode-level 指标都来自 `infos["episode"]` 子字典（ManiSkill 提供）
- 不需要复杂的合并逻辑 - 只要有 done，就用 final_info
