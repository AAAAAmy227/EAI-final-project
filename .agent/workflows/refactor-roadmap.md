---
description: 代码重构完整路线图 - 执行顺序和依赖关系
---

# 🗺️ 代码重构路线图

## 📋 执行顺序概览

按以下顺序串行执行，每个阶段的 workflow 在前一阶段完成后才能开始：

```
阶段 1: 基础设施 (无依赖) ✅ 已完成
├── 1.1 /refactor-wrapper-traversal     ✅ 已完成
└── 1.2 /refactor-so101-logging         ✅ 已完成

阶段 2: PPORunner 清理 ✅ 已完成
├── 2.1 /fix-runner-missing-methods     ✅ 已完成
└── 2.2 /refactor-runner-train-method   ✅ 已完成

阶段 3: Track1Env 重构 ✅ 已完成
├── 3.1 /refactor-track1-rewards        ✅ 已完成 (-691行, 34%精简)
├── 3.2 /refactor-track1-config         ✅ 已完成 (config_utils.py)
└── 3.3 /refactor-track1-init-attrs     ✅ 已完成 (hasattr -> Optional)

阶段 4: 高级重构 (当前阶段)
├── 4.1 /refactor-so101-class-attrs     📝 待执行 (类属性+硬编码)
└── 4.2 /refactor-make-env              📝 待执行 (函数拆分)

阶段 5: 收尾 (可选)
└── 5.1 /refactor-extract-constants     📝 待规划
```

## ⚙️ 依赖关系

| Workflow | 前置依赖 | 说明 |
|----------|---------|------|
| 1.1 refactor-wrapper-traversal | 无 | 创建基础工具函数 |
| 1.2 refactor-so101-logging | 无 | 独立修改 |
| 2.1 fix-runner-missing-methods | 1.1 | 使用 find_wrapper |
| 2.2 refactor-runner-train-method | 2.1 | 依赖无错误的 runner |
| 3.1 refactor-track1-rewards | 无 | 独立重构 |
| 3.2 refactor-track1-config | 3.1 | 奖励配置先拆分 |
| 3.3 refactor-track1-init-attrs | 3.2 | 依赖配置结构 |
| 4.1 refactor-so101-class-attrs | 1.2 | 依赖 logging 基础 |
| 4.2 refactor-make-env | 4.1 | 依赖 SO101 新接口 |
| 5.1 refactor-extract-constants | 全部 | 收尾工作 |

## 📊 工作量估计

| Workflow | 复杂度 | 预计代码变更 | 风险 |
|----------|-------|-------------|------|
| 1.1 wrapper-traversal | ⭐ | +20, -60 行 | 低 |
| 1.2 so101-logging | ⭐ | +5, -2 行 | 极低 |
| 2.1 missing-methods | ⭐⭐ | +0, -10 行 | 中 |
| 2.2 train-method | ⭐⭐⭐ | +50, -0 行 | 中 |
| 3.1 track1-rewards | ⭐⭐⭐⭐ | +200, -150 行 | 高 |
| 3.2 track1-config | ⭐⭐⭐ | +80, -50 行 | 中 |
| 3.3 init-attrs | ⭐⭐ | +20, -20 行 | 低 |
| 4.1 so101-attrs | ⭐⭐⭐⭐ | +100, -50 行 | 高 |
| 4.2 make-env | ⭐⭐⭐ | +60, -40 行 | 中 |
| 5.1 constants | ⭐ | +30, -0 行 | 极低 |

## 🚀 快速开始命令

```bash
# 查看某个 workflow
cat .agent/workflows/refactor-wrapper-traversal.md

# 执行整个阶段
# 在 agent 中使用 /workflow-name 触发
```

## ⚠️ 重要提醒

1. **顺序执行**: 每完成一个 workflow，验证无误后再执行下一个
2. **增量验证**: 每步执行后运行 `uv run python -m py_compile` 验证
3. **上下文独立**: 每个 workflow 由独立 agent 执行，包含完整上下文
4. **代码变化**: 后续 workflow 中的行号可能与当前不同，需要根据内容定位
