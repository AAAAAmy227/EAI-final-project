# EAI Final Project - Robotic Manipulation with PPO

强化学习训练机器人操作任务（基于 ManiSkill 和 PPO 算法）

## 项目概述

本项目使用 PPO (Proximal Policy Optimization) 算法训练双臂机器人（SO-101）完成各种操作任务。核心特性包括：

- 🤖 **双臂机器人**: SO-101 双臂协作控制
- 🎯 **多任务支持**: Lift, Stack, Sort 等任务
- ⚡ **GPU 加速**: 并行环境模拟和训练
- 📊 **统一 Metrics 系统**: 训练和评估的指标收集
- 🔄 **异步评估**: 后台评估不影响训练速度

## 环境要求

- **Python**: 3.10+
- **包管理**: uv
- **GPU**: CUDA-capable GPU (推荐)
- **ManiSkill**: 已安装在 `.venv`

## 快速开始

### 安装依赖

```bash
# 使用 uv 创建虚拟环境并安装依赖
uv sync
```

### 训练

```bash
# 基础训练（Lift 任务）
uv run python scripts/train.py

# 指定任务
uv run python scripts/train.py env.task=stack

# 自定义配置
uv run python scripts/train.py \
    training.num_envs=512 \
    ppo.learning_rate=3e-4 \
    wandb.enabled=true
```

### 评估

```bash
# 评估已训练的模型
uv run python scripts/train.py \
    checkpoint=path/to/checkpoint.pth \
    training.num_eval_envs=16 \
    capture_video=true
```

## 项目结构

```
eai-final-project/
├── scripts/
│   ├── training/              # 训练相关模块
│   │   ├── runner.py          # PPO Runner（核心训练循环）
│   │   ├── agent.py           # Actor-Critic 网络
│   │   ├── ppo_utils.py       # PPO 算法实现（GAE, 更新）
│   │   ├── env_utils.py       # 环境工具（wrappers, make_env）
│   │   └── metrics_utils.py   # Metrics 收集和聚合
│   │
│   ├── tasks/                 # 任务定义
│   │   ├── base.py            # BaseTaskHandler（基类）
│   │   ├── lift.py            # LiftTaskHandler
│   │   ├── stack.py           # StackTaskHandler
│   │   └── sort.py            # SortTaskHandler
│   │
│   ├── envs/                  # 环境定义
│   │   └── track1_env.py      # Track1Env（主环境）
│   │
│   ├── tests/                 # 单元测试
│   │   ├── test_metrics.py        # Metrics 系统测试
│   │   ├── test_task_handlers.py  # TaskHandler 测试
│   │   ├── test_ppo_unit.py       # PPO 单元测试
│   │   ├── test_ppo_integration.py # PPO 集成测试
│   │   └── README.md              # 测试文档
│   │
│   ├── so101.py               # SO-101 双臂机器人定义
│   ├── train.py               # 训练入口
│   ├── view_env.py            # 环境可视化
│   │
│   ├── benchmarks/            # 性能测试
│   │   ├── benchmark_full_loop.py
│   │   ├── benchmark_gae.py
│   │   └── benchmark_ppo.py
│   │
│   └── utils/                 # 工具脚本
│       ├── camera_overlay.py
│       ├── check_wrist_camera.py
│       ├── sample_poses_ik.py
│       └── ...
│
├── configs/                   # Hydra 配置文件
│   └── train.yaml            # 默认训练配置
│
├── outputs/                   # 训练输出（自动生成）
│   └── YYYY-MM-DD/
│       └── HH-MM-SS/
│           ├── checkpoints/   # 模型检查点
│           ├── videos/        # 评估视频
│           ├── split/         # 分环境视频和CSV
│           └── .hydra/        # Hydra 配置快照
│
├── assets/                    # 资源文件
│   └── screenshots/
│
└── README.md                  # 本文件
```

## 核心模块说明

### 1. Training Pipeline (`scripts/training/`)

#### `runner.py` - PPO Runner
核心训练循环，负责：
- 环境交互（rollout）
- GAE 计算
- PPO 更新
- 评估调度
- 指标记录

**关键方法**：
- `_rollout()`: 统一的 rollout 方法（支持 train 和 eval）
- `_compute_gae()`: GAE 优势估计
- `_run_ppo_update()`: PPO 参数更新
- `_evaluate()`: 评估循环

#### `metrics_utils.py` - Metrics 系统
统一的指标收集和聚合系统：
- `get_metric_specs_from_env()`: 从 TaskHandler 获取 metric specs
- `aggregate_metrics()`: 批量聚合 rollout 的 metrics

**特性**：
- ✅ GPU 批量操作
- ✅ 延迟 CPU 传输
- ✅ Mode-specific metrics（train vs eval）
- ✅ 自动聚合（mean / sum）

### 2. Task System (`scripts/tasks/`)

#### `base.py` - BaseTaskHandler
任务处理器基类，定义接口：
- `evaluate()`: 评估成功/失败条件
- `compute_dense_reward()`: 计算密集奖励
- `initialize_episode()`: 初始化 episode

**Metrics 定义**：
```python
class BaseTaskHandler:
    # 默认 metrics（所有任务共享）
    DEFAULT_METRIC_AGGREGATIONS = {
        "success": "mean",
        "fail": "mean",
        "raw_reward": "mean",
        "return": "mean",
        "episode_len": "mean",
    }
    
    @classmethod
    def _get_train_metrics(cls) -> Dict[str, str]:
        """定义 training 专用 metrics"""
        return {}
    
    @classmethod
    def _get_eval_metrics(cls) -> Dict[str, str]:
        """定义 evaluation 专用 metrics（默认与 train 相同）"""
        return cls._get_train_metrics()
```

#### `lift.py` - LiftTaskHandler
Lift 任务实现（抓取并举起物体）

**自定义 Metrics**：
```python
class LiftTaskHandler(BaseTaskHandler):
    @classmethod
    def _get_train_metrics(cls):
        return {
            "grasp_reward": "mean",
            "lift_reward": "mean",
            "moving_distance": "mean",
            "grasp_success": "mean",
            "lift_success": "mean",
        }
```

### 3. Environment (`scripts/envs/`)

#### `track1_env.py` - Track1Env
主环境类，继承自 ManiSkill 的 `BaseEnv`：
- 场景设置（robot, objects, cameras）
- 观测空间定义
- 动作空间定义
- 奖励计算（委托给 TaskHandler）

### 4. Robot (`scripts/so101.py`)

SO-101 双臂机器人定义：
- URDF 加载
- 控制器配置
- 运动学参数

## Metrics 系统详解

### Mode-Specific Metrics

支持为 training 和 evaluation 定义不同的 metrics：

```python
class DetailedTaskHandler(BaseTaskHandler):
    @classmethod
    def _get_train_metrics(cls):
        """Training: 只收集关键 metrics（性能优先）"""
        return {
            "grasp_reward": "mean",
            "lift_reward": "mean",
        }
    
    @classmethod
    def _get_eval_metrics(cls):
        """Evaluation: 收集详细 metrics（分析优先）"""
        return {
            "grasp_reward": "mean",
            "lift_reward": "mean",
            "cube_velocity": "mean",      # Eval-only
            "gripper_distance": "mean",   # Eval-only
            "stability_score": "mean",    # Eval-only
        }
```

### 自动模式切换

在 `_rollout()` 中自动根据 `collect_for_training` 参数选择：
- `collect_for_training=True` → `mode="train"`
- `collect_for_training=False` → `mode="eval"`

### 聚合类型

- **`"mean"`**: 计算平均值（适用于 rewards, success rate）
- **`"sum"`**: 累加总和（适用于 counts）

### Logging

**Training Logs**:
```python
wandb.log({
    "rollout/success_rate": 0.75,
    "rollout/return": 10.5,
    "reward/grasp_reward": 2.3,
    "reward/lift_reward": 8.2,
}, step=10240)
```

**Evaluation Logs**:
```python
wandb.log({
    "eval/success_rate": 0.82,
    "eval/return": 12.1,
    "eval_reward/grasp_reward": 2.5,
    "eval_reward/lift_reward": 9.6,
    "eval_reward/cube_velocity": 0.15,  # Eval-only
}, step=10240)
```

## 配置系统

使用 Hydra 进行配置管理（`configs/train.yaml`）：

```yaml
# 环境配置
env:
  task: lift
  num_envs: 256
  robot_urdf: assets/so101.urdf

# PPO 配置
ppo:
  learning_rate: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  
# Training 配置
training:
  total_timesteps: 10_000_000
  num_steps: 2048
  num_eval_envs: 16
  eval_freq: 10

# WandB 配置
wandb:
  enabled: true
  project: eai-final-project
```

## 性能优化

### GPU 优化
- ✅ 所有 metrics 在 GPU 上收集
- ✅ 延迟 CPU 传输（rollout 结束后批量）
- ✅ 向量化操作

### 异步评估
- ✅ 后台线程运行 evaluation
- ✅ 独立的 CUDA stream
- ✅ 不影响 training 速度
- ✅ 准确的 step logging

### 编译和 CUDA Graphs
- ✅ `torch.compile` 加速
- ✅ CudaGraphModule 用于 policy inference
- ✅ Reduce-overhead mode 用于 update

## 测试

```bash
# 运行所有测试
uv run pytest scripts/tests/ -v

# 运行特定测试
uv run pytest scripts/tests/test_metrics.py
uv run pytest scripts/tests/test_task_handlers.py
```

## 开发指南

### 添加新任务

1. 创建 TaskHandler:
```python
# scripts/tasks/my_task.py
from scripts.tasks.base import BaseTaskHandler

class MyTaskHandler(BaseTaskHandler):
    @classmethod
    def _get_train_metrics(cls):
        return {"my_reward": "mean"}
    
    def evaluate(self):
        # 实现评估逻辑
        return {"success": ..., "fail": ...}
    
    def compute_dense_reward(self, info, action):
        # 实现奖励计算
        return reward
```

2. 在 `track1_env.py` 中注册:
```python
if self.task == "my_task":
    from scripts.tasks.my_task import MyTaskHandler
    return MyTaskHandler(self)
```

### 添加新 Metrics

只需在 TaskHandler 的 `_get_train_metrics()` 或 `_get_eval_metrics()` 中声明：
```python
@classmethod
def _get_train_metrics(cls):
    return {
        "new_metric": "mean",  # 或 "sum"
    }
```

然后在 `compute_dense_reward()` 中填充到 `info`:
```python
def compute_dense_reward(self, info, action):
    info["new_metric"] = ...  # 计算值
    return reward
```

## 常见问题

### Q: 如何查看训练进度？
A: 使用 WandB:
```bash
uv run python scripts/train.py wandb.enabled=true
```

### Q: 如何调整并行环境数？
A: 修改 `training.num_envs`:
```bash
uv run python scripts/train.py training.num_envs=512
```

### Q: 如何保存/加载检查点？
A: 检查点自动保存到 `outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/`，使用 `checkpoint` 参数加载：
```bash
uv run python scripts/train.py checkpoint=path/to/checkpoint.pth
```

### Q: 评估视频在哪里？
A: `outputs/.../videos/` (完整视频) 和 `outputs/.../split/evalN/envM/` (分环境视频)

## 相关资源

- **ManiSkill**: https://github.com/haosulab/ManiSkill
- **PPO 论文**: https://arxiv.org/abs/1707.06347
- **项目文档**: `.agent/` 目录下的 Markdown 文件

## License

[Your License Here]

## 贡献者

[Your Name]

---

最后更新: 2025-12-31