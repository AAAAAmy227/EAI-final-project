# 测试文档

## 测试概述

本项目包含全面的单元测试，覆盖 Metrics 系统和 TaskHandler 功能。

## 运行测试

### 运行所有测试
```bash
uv run pytest tests/ -v
```

### 运行特定测试文件
```bash
# Metrics 测试
uv run pytest tests/test_metrics.py -v

# TaskHandler 测试
uv run pytest tests/test_task_handlers.py -v
```

### 运行特定测试类
```bash
uv run pytest tests/test_metrics.py::TestGetMetricSpecs -v
uv run pytest tests/test_task_handlers.py::TestLiftTaskHandler -v
```

### 运行特定测试方法
```bash
uv run pytest tests/test_metrics.py::TestGetMetricSpecs::test_train_mode_metrics -v
```

## 测试覆盖

### test_metrics.py (16 个测试)

#### TestGetMetricSpecs (4 tests)
测试 `get_metric_specs_from_env()` 函数：
- ✅ 无 task handler 时返回默认 metrics
- ✅ Train mode 正确返回 train-specific metrics
- ✅ Eval mode 正确返回 eval-specific metrics
- ✅ Mode 参数大小写敏感

####TestAggregateMetrics (5 tests)
测试 `aggregate_metrics()` 函数：
- ✅ Mean 聚合正确计算
- ✅ Sum 聚合正确累加
- ✅ 无完成 episode 时不报错
- ✅ 多环境同时完成时正确处理
- ✅ GPU tensor 正确处理并传输到 CPU

#### TestDefaultMetricAggregations (2 tests)
测试 `BaseTaskHandler.DEFAULT_METRIC_AGGREGATIONS`：
- ✅ 默认 aggregations 存在
- ✅ 包含预期的 metrics（success, fail, return, etc.）

#### TestTaskHandlerMetricAPI (5 tests)
测试 TaskHandler 的 metric API：
- ✅ Train mode 调用正确
- ✅ Eval mode 调用正确
- ✅ 默认 mode 参数为 "train"
- ✅ `_get_train_metrics()` 方法工作
- ✅ `_get_eval_metrics()` 方法工作

### test_task_handlers.py (23 个测试)

#### TestBaseTaskHandler (10 tests)
测试 `BaseTaskHandler` 抽象类：
- ✅ 不能直接实例化抽象类
- ✅ DEFAULT_METRIC_AGGREGATIONS 正确定义
- ✅ `get_custom_metric_aggregations()` 默认实现
- ✅ `_get_train_metrics()` 默认返回空字典
- ✅ `_get_eval_metrics()` 默认与 train 相同
- ✅ Mode 参数 "train" 和 "eval" 工作
- ✅ 具体子类可以正常实例化
- ✅ Dummy handler 的 evaluate() 方法工作
- ✅ Dummy handler 的 compute_dense_reward() 方法工作

#### TestLiftTaskHandler (8 tests)
测试 `LiftTaskHandler` 实现：
- ✅ Train metrics 正确定义
- ✅ Eval metrics 默认与 train 相同
- ✅ Train mode 获取正确的 custom aggregations
- ✅ Eval mode 获取正确的 custom aggregations
- ✅ 所有 Lift metrics 使用 "mean" 聚合
- ✅ 可以正常实例化
- ✅ 实现了所有必需的抽象方法
- ✅ 初始化了预期的属性（adaptive states）

#### TestModeSpecificMetrics (5 tests)
测试 mode-specific metrics 功能：
- ✅ Train-heavy handler 在 train mode 返回正确 metrics
- ✅ Train-heavy handler 在 eval mode 过滤掉 train-only metrics
- ✅ Eval-heavy handler 在 train mode 只返回核心 metrics
- ✅ Eval-heavy handler 在 eval mode 返回详细 metrics
- ✅ Train 和 eval metrics 正确隔离

## 测试统计

- **总测试数**: 39
- **通过**: 39 ✅
- **失败**: 0
- **跳过**: 0
- **警告**: 1 (pkg_resources deprecation - 来自 sapien)

## Mock 对象

### MockEnv
模拟环境，用于测试 TaskHandler：
```python
class MockEnv:
    def __init__(self, task_handler_class=None):
        self.device = torch.device("cpu")
        self.num_envs = 4
        self.task_handler = task_handler_class(self) if task_handler_class else None
```

### MockTaskHandler
模拟 TaskHandler，用于测试 metrics 系统：
```python
class MockTaskHandler(BaseTaskHandler):
    @classmethod
    def _get_train_metrics(cls):
        return {"train_only_metric": "mean", "shared_metric": "mean"}
    
    @classmethod
    def _get_eval_metrics(cls):
        return {"eval_only_metric": "mean", "shared_metric": "mean"}
```

## 边界情况测试

### Metrics Aggregation
- ✅ 空完成 episodes（done_mask 全 False）
- ✅ 多环境同时完成
- ✅ GPU vs CPU tensor
- ✅ Sum vs Mean aggregation
- ✅ 不存在的 metric（gracefully skip）

### Mode-Specific Metrics
- ✅ Train-only metrics
- ✅ Eval-only metrics
- ✅ Shared metrics
- ✅ Mode 参数大小写
- ✅ 默认 mode 行为

## 添加新测试

### 测试 Metrics 功能
在 `tests/test_metrics.py` 中添加：
```python
class TestYourFeature:
    def test_something(self):
        # Setup
        ...
        
        # Execute
        ...
        
        # Assert
        assert ...
```

### 测试 TaskHandler
在 `tests/test_task_handlers.py` 中添加：
```python
class TestYourTaskHandler:
    def test_custom_metrics(self):
        metrics = YourTaskHandler._get_train_metrics()
        assert "your_metric" in metrics
```

## CI/CD 集成

可以在 GitHub Actions 中添加：
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: astral-sh/setup-uv@v1
      - run: uv sync
      - run: uv run pytest tests/ -v
```

## 测试最佳实践

1. **命名**: 使用描述性的测试名称（`test_<feature>_<scenario>_<expected>`）
2. **隔离**: 每个测试应该独立运行
3. **清晰**: 使用 AAA 模式（Arrange, Act, Assert）
4. **覆盖**: 包括正常情况、边界情况和错误情况
5. **Mock**: 使用 mock 对象避免依赖外部状态

## 下一步

- [ ] 添加 integration tests（完整 rollout 测试）
- [ ] 添加 performance benchmarks
- [ ] 测试覆盖率报告（pytest-cov）
- [ ] 测试 runner.py 的关键方法
- [ ] 测试环境 wrappers

---

最后更新: 2025-12-31
测试通过率: 100% (39/39)
