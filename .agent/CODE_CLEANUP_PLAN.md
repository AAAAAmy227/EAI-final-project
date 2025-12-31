# 代码清理建议 - info_utils 和 _update_metrics

## 当前状态分析

### _update_metrics 方法
- **位置**: `scripts/training/runner.py` L493-557
- **状态**: ❌ 无任何调用
- **替代**: 新的 `_rollout()` + `aggregate_metrics()` 系统

### info_utils.py 使用情况

#### 在 runner.py 中
所有使用都在 `_update_metrics` 方法内：
- `get_info_field` (L505, L517, L520, L543, L552)
- `get_reward_components` (L509)
- `accumulate_reward_components_gpu` (L511)

由于 `_update_metrics` 不再被调用，这些import实际上也不需要了。

#### 在测试文件中
- `scripts/tests/test_info_utils.py` - 专门测试 info_utils
- `scripts/tests/test_autoreset_logic.py` - 使用 get_info_field
- `scripts/tests/test_runner_metrics.py` - 使用 accumulate_reward_components_gpu
- `scripts/tests/investigate_info.py` - 调查脚本

## 推荐的清理方案

### 方案 A: 完全删除（激进）

**删除**:
1. `scripts/training/runner.py` 中的 `_update_metrics` 方法
2. `scripts/training/runner.py` 中的 info_utils imports
3. `scripts/training/info_utils.py` 整个文件
4. `scripts/tests/test_info_utils.py` 测试文件
5. `scripts/tests/test_runner_metrics.py` 中相关测试
6. `scripts/tests/test_autoreset_logic.py` 更新为使用新系统
7. `scripts/tests/investigate_info.py` 调查脚本（如果不需要）

**优点**:
- 代码库更清洁
- 消除技术债
- 没有死代码

**缺点**:
- 需要更新/删除多个测试文件
- 如果将来需要参考旧实现会很麻烦

### 方案 B: 保守删除（推荐）

**第一阶段**（安全）:
1. ✅ 删除 `scripts/training/runner.py` 中的 `_update_metrics` 方法
2. ✅ 删除 `scripts/training/runner.py` 中未使用的 info_utils imports

**第二阶段**（可选，观察一段时间后）:
3. ⚠️ 保留 `info_utils.py` 文件（标记为 deprecated）
4. ⚠️ 保留相关测试文件（作为文档/参考）

**优点**:
- 立即清理主代码
- 保留测试作为文档
- 可以将来再决定是否完全删除

**缺点**:
- info_utils.py 成为死代码

### 方案 C: 仅删除 runner.py 中的死代码

**删除**:
1. ✅ `scripts/training/runner.py` 中的 `_update_metrics` 方法  
2. ✅ `scripts/training/runner.py` 中的 info_utils imports

**保留**:
- `scripts/training/info_utils.py` （标记为 Legacy/Deprecated）
- 所有测试文件

**优点**:
- 最小改动
- 保留所有历史代码和测试

**缺点**:
- info_utils.py 没人用但还在

## 我的建议

**立即执行 - 方案 B 第一阶段**:
```python
# 1. 删除 runner.py 中的 _update_metrics 方法 (L493-557)
# 2. 删除未使用的 imports
```

**添加 deprecation 标记**:
```python
# scripts/training/info_utils.py 顶部添加
"""
DEPRECATED: This module is no longer used in the main training loop.
Replaced by scripts/training/metrics_utils.py.

Kept for reference and legacy tests only.
"""
import warnings
warnings.warn(
    "info_utils is deprecated. Use metrics_utils instead.",
    DeprecationWarning,
    stacklevel=2
)
```

**在 README 中说明**:
```markdown
### Deprecated Modules

- `scripts/training/info_utils.py` - 旧的 metrics 系统，已被 `metrics_utils.py` 替代
```

## 执行命令

### 删除 runner.py 中的死代码
```bash
# 手动编辑或使用以下步骤：
# 1. 删除 _update_metrics 方法 (L493-557)
# 2. 更新 imports (L36-40) - 移除未使用的
```

### 添加 deprecation 警告
```bash
# 在 info_utils.py 顶部添加 deprecation 说明
```

### 更新文档
```bash
# 在 README.md 中添加 deprecated modules 部分
```

## 总结

**推荐**: 方案 B
- ✅ 立即删除 `_update_metrics` 和未使用的 imports
- ✅ 标记 `info_utils.py` 为 deprecated
- ⚠️ 保留测试文件作为历史参考
- 📅 6个月后再评估是否完全删除

**这样做的好处**:
1. 主代码立即变清洁
2. 不破坏现有测试基础设施
3. 有明确的迁移路径
4. 保留历史参考
