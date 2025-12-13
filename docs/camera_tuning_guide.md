# 相机参数手动微调指南

## 工具说明

`scripts/camera_overlay_tunable.py` 是一个支持命令行参数调整的相机对齐工具。你可以通过调整参数，实时查看模拟图像与真实图像的叠加效果。

## 基本使用

```bash
# 默认参数（初始设置）
python -m scripts.camera_overlay_tunable

# 查看生成的叠加图
eog overlay_comparison.png
```

## 参数说明

### 相机位置（单位：米）
- `--cam-x`: X 坐标（默认 0.316）
- `--cam-y`: Y 坐标（默认 0.260）
- `--cam-z`: Z 坐标（默认 0.407）

### 相机旋转（单位：度）
- `--cam-pitch`: 俯仰角（默认 -90，负值表示向下看）
- `--cam-yaw`: 偏航角（默认 0）
- `--cam-roll`: 滚转角（默认 0）

### 其他参数
- `--fov`: 垂直视场角（默认 73.63度）
- `--no-distortion`: 跳过畸变模拟（用于调试）

## 调参工作流

根据你的空间感知能力，我建议这个调参顺序：

### 1. 初步评估
```bash
python -m scripts.camera_overlay_tunable
eog overlay_comparison.png
```
观察：
- 胶带线的对齐程度
- 方块的位置是否一致
- 整体画面的旋转和缩放

### 2. 位置微调
如果画面整体偏移，调整位置参数（每次 ±0.01米）：

```bash
# 示例：向右移动 1cm
python -m scripts.camera_overlay_tunable --cam-x 0.326

# 示例：向前移动 1cm
python -m scripts.camera_overlay_tunable --cam-y 0.270

# 示例：向上移动 1cm
python -m scripts.camera_overlay_tunable --cam-z 0.417
```

### 3. 旋转微调
如果画面有旋转不对齐，调整角度参数（每次 ±5度）：

```bash
# 示例：调整俯仰角
python -m scripts.camera_overlay_tunable --cam-pitch -85

# 示例：调整偏航角
python -m scripts.camera_overlay_tunable --cam-yaw 5

# 示例：调整滚转角
python -m scripts.camera_overlay_tunable --cam-roll 3
```

### 4. 视场角微调
如果画面缩放不对，调整 FOV（每次 ±2度）：

```bash
python -m scripts.camera_overlay_tunable --fov 75.63
```

### 5. 组合调整
一旦找到大致方向，可以组合多个参数：

```bash
python -m scripts.camera_overlay_tunable --cam-x 0.326 --cam-y 0.270 --cam-pitch -85 --fov 75
```

## 快速实验技巧

### 创建快捷脚本
```bash
# 保存当前最佳参数到脚本
cat > test_camera.sh << 'EOF'
#!/bin/bash
python -m scripts.camera_overlay_tunable \
  --cam-x 0.316 \
  --cam-y 0.260 \
  --cam-z 0.407 \
  --cam-pitch -90 \
  --cam-yaw 0 \
  --cam-roll 0 \
  --fov 73.63
eog overlay_comparison.png
EOF

chmod +x test_camera.sh
./test_camera.sh
```

### 批量测试
创建测试脚本尝试多个参数组合：

```bash
#!/bin/bash
for pitch in -95 -90 -85; do
  for yaw in -5 0 5; do
    echo "Testing pitch=$pitch, yaw=$yaw"
    python -m scripts.camera_overlay_tunable \
      --cam-pitch $pitch \
      --cam-yaw $yaw \
      --output "test_p${pitch}_y${yaw}.png"
  done
done
```

## 参数理解

### 坐标系
- **X轴**：向右为正
- **Y轴**：向前为正（进入场景）
- **Z轴**：向上为正

### 欧拉角（旋转顺序：XYZ）
- **Pitch（俯仰）**：绕 X 轴旋转，负值向下看
- **Yaw（偏航）**：绕 Z 轴旋转，正值向左转
- **Roll（滚转）**：绕 Y 轴旋转，正值右倾

## 最终步骤

当你找到最佳参数后：

1. **记录最终参数**：
```bash
# 示例输出
Camera Parameters:
  Position: [0.320, 0.265, 0.410]
  Rotation: pitch=-88.0°, yaw=2.0°, roll=1.0°
  FOV: 74.50°
```

2. **更新环境文件**：
告诉我最终参数，我会更新 `track1_env.py` 中的默认值。

3. **验证**：
使用新参数重新生成叠加图，确认对齐效果。

## 故障排查

### 问题：看不到方块
- 检查 `sim_camera_view_pinhole.png`确认方块是否在视野内
-尝试调整 Z 位置（相机高度）

### 问题：画面是侧着的
- 检查 `roll` 参数（可能需要 ±90度）

### 问题：视野太窄/太宽
- 调整 `fov` 参数

### 问题：畸变不匹配
- 使用 `--no-distortion` 检查原始对齐
- 畸变参数已硬编码，来自 `distort.py`
