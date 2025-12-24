# LeRobot 模仿学习训练指南

本文档记录了使用 LeRobot 框架在 eai-dataset 上训练模仿学习策略的完整流程。

## 环境说明

项目中有三个 Python 环境：

| 环境 | 位置 | 用途 |
|------|------|------|
| `lerobot/.venv` | LeRobot 虚拟环境 | 模仿学习训练 |
| `.venv` | 项目主虚拟环境 | PPO 训练、模拟器 |
| `conda maniskill` | Conda 环境 | ManiSkill 相关 |

## 数据集预处理

### 相机去畸变

前置相机 (front camera) 存在径向畸变，需要去畸变后才能用于训练。

**相机标定参数**：
```python
MTX = np.array([
    [570.217, 0., 327.460],
    [0., 570.180, 260.836],
    [0., 0., 1.]
], dtype=np.float64)
DIST = np.array([-0.735, 0.949, 0.000189, -0.00200, -0.864], dtype=np.float64)
```

**Alpha 参数选择**：
- `alpha=0`: 裁剪所有黑边，保持原始 FOV (fx=570)
- `alpha=0.25`: **推荐值**，可看到全部工作区域 (fx≈477)
- `alpha=1`: 保留所有像素，但图像会缩小变形

### 预处理命令

```bash
source lerobot/.venv/bin/activate

# 单个任务
python scripts/preprocess_undistort.py --task lift --alpha 0.25

# 所有任务
python scripts/preprocess_undistort.py --all --alpha 0.25 --workers 2
```

预处理后的数据集保存在 `eai-dataset-undistorted/` 目录。

### 任务数据集特性

| 任务 | Episodes | Frames | Action DoF | 图像 |
|------|----------|--------|------------|------|
| **lift** | 50 | 8,934 | 6 (单臂) | front + wrist |
| **stack** | 100 | 24,883 | 6 (单臂) | front + wrist |
| **sort** | 100 | 33,526 | 12 (双臂) | front + left_wrist + right_wrist |

## LeRobot 本地数据集支持

LeRobot 默认会尝试连接 HuggingFace Hub 验证数据集版本。使用本地数据集时需要修改代码跳过网络调用。

**已修改的文件**：
- `lerobot/src/lerobot/datasets/utils.py` - `get_safe_version()` 函数
- `lerobot/src/lerobot/datasets/lerobot_dataset.py` - `LeRobotDatasetMetadata.__init__` 和 `LeRobotDataset.__init__`

关键修改：当 `repo_id == "local"` 或路径以 `/` 开头时，跳过 HuggingFace API 调用。

## 训练命令

### 基本训练

```bash
source lerobot/.venv/bin/activate
cd lerobot

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id local \
  --dataset.root /path/to/eai-dataset-undistorted/lift \
  --policy.type diffusion \
  --policy.push_to_hub false \
  --batch_size 32 \
  --steps 50000 \
  --log_freq 100 \
  --save_freq 5000 \
  --eval_freq 0 \
  --output_dir outputs/lift_diffusion
```

### 带 WandB 日志

```bash
# 首次使用需要登录
wandb login

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id local \
  --dataset.root /path/to/eai-dataset-undistorted/lift \
  --policy.type diffusion \
  --policy.push_to_hub false \
  --wandb.enable true \
  --wandb.project eai-lerobot \
  --batch_size 32 \
  --steps 50000 \
  --output_dir outputs/lift_diffusion
```

### WandB 参数

| 参数 | 说明 |
|------|------|
| `--wandb.enable true` | 启用 WandB |
| `--wandb.project NAME` | 项目名 |
| `--wandb.entity USERNAME` | 用户名/团队名 |
| `--wandb.notes "描述"` | 实验备注 |
| `--wandb.mode offline` | 离线模式 |

## 可用策略

LeRobot 支持多种策略：

| 策略 | 类型 | 参数量 |
|------|------|--------|
| `diffusion` | Diffusion Policy | ~267M |
| `act` | Action Chunking Transformer | 较小 |
| `vqbet` | VQ-BeT | 较小 |
| `pi0` | Pi0 | - |
| `tdmpc` | TD-MPC | - |

使用 `--policy.type` 参数指定策略类型。

## Diffusion Policy 配置

默认配置：
- `horizon`: 16
- `n_obs_steps`: 2
- `n_action_steps`: 8
- `vision_backbone`: resnet18
- `num_train_timesteps`: 100

## 常见问题

### 1. SSL/网络错误
使用 `--dataset.repo_id local` 并确保已应用本地数据集补丁。

### 2. 输出目录已存在
删除旧目录或使用 `--resume true` 继续训练。

### 3. 视频解码错误
确保预处理后的视频使用 H.264 编码（非 AV1）。

### 4. 双臂任务 wrist camera 缺失
确保预处理脚本复制了 `left_wrist` 和 `right_wrist` 目录。
