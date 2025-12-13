# Track 1 项目状态总结（会话切换文档）

**日期**: 2025-12-13  
**目的**: 记录当前项目状态，供新对话引用

---

## 项目概况

**项目**: EAI Final Project - Track 1 双臂机器人模拟环境  
**主要目标**: 创建 ManiSkill 模拟环境，支持 Lift/Stack/Sort 任务，实现 sim2real 对齐

---

## 当前状态

### ✅ 已完成

1. **环境基础架构**
   - ✅ `scripts/so101.py` - SO101 机器人定义
   - ✅ `scripts/track1_env.py` - Track1 环境主文件
   - ✅ 双臂机器人配置（左右各一个 SO101）
   - ✅ 三类任务逻辑：Lift, Stack, Sort
   - ✅ 域随机化（Domain Randomization）
     - 视觉：物体/桌面颜色
     - 物理：质量、摩擦力
     - 机器人：关节属性
     - 相机：位置/旋转随机化
     - 光照随机化
   - ✅ 相机配置：1个前置 + 2个腕部相机

2. **胶带线布局**
   - ✅ **品字形布局**（最新修改）
   - 4个正方形：3个上排（LEFT/MID/RIGHT Grid），1个下排（BOTTOM Grid，在两个机械臂底座之间）
   - 共享边只绘制一次
   - 位置：
     ```
     [Left]  [Mid]  [Right]  <- 上排
        [Bottom]             <- 下排（两个机器人之间）
     ```

3. **相机对齐工具**
   - ✅ `scripts/camera_overlay_tunable.py` - **可调参数的相机对齐工具**
   - ✅ `docs/camera_tuning_guide.md` - 详细使用指南
   - ✅ `track1_env.py` 支持通过配置文件读取相机参数
   - 支持调整参数：
     - 位置：`--cam-x`, `--cam-y`, `--cam-z`
     - 旋转：`--cam-pitch`, `--cam-yaw`, `--cam-roll`
     - 视场角：`--fov`
     - 畸变模拟：默认启用，可用 `--no-distortion` 禁用

### 🔄 进行中

1. **相机参数微调**（需要用户手动操作）
   - 当前默认参数：
     - 位置：`[0.316, 0.260, 0.407]`
     - 旋转：`pitch=-90°, yaw=0°, roll=0°`
     - FOV：`73.63°`
   - **待办**：用户使用 `camera_overlay_tunable.py` 微调参数，直到对齐满意

### ❌ 已知问题

1. **`scripts/view_env.py` GUI查看器有bug**
   - 错误：`AttributeError: 'NoneType' object has no attribute 'should_close'`
   - 原因：ManiSkill viewer 在静态渲染模式下的问题
   - **影响**：无法使用GUI查看器检查环境
   - **临时方案**：使用 `camera_overlay_tunable.py` 生成图片查看

2. **坐标系混淆（已解决）**
   - 之前错误地尝试统一 `front_camera.py` 和 `Track1_Simulation_Parameters.md` 的坐标系
   - **解决方案**：使用 `git checkout` 恢复环境到 Track1_Simulation_Parameters.md 坐标系
   - 相机对齐在 `camera_overlay_tunable.py` 中处理，不修改环境本身

---

## 关键文件说明

### 核心环境文件

1. **`scripts/track1_env.py`** (418行)
   - Track1 环境主类
   - **重要修改**：
     - `_default_sensor_configs()` - 支持从 `/tmp/camera_config.json` 读取相机参数
     - `_build_tape_lines()` - 品字形胶带布局
     - Grid 定义：LEFT/MID/RIGHT (上排) + BOTTOM (下排，新增但未完全启用)
   
2. **`scripts/so101.py`**
   - SO101 机器人定义
   - 包含腕部相机挂载点

### 工具脚本

1. **`scripts/camera_overlay_tunable.py`** ⭐
   - **核心功能**：命令行调整相机参数，生成对齐叠加图
   - **使用方法**：
     ```bash
     # 默认参数
     python -m scripts.camera_overlay_tunable
     
     # 调整位置
     python -m scripts.camera_overlay_tunable --cam-x 0.320
     
     # 调整旋转
     python -m scripts.camera_overlay_tunable --cam-pitch -85
     
     # 组合调整
     python -m scripts.camera_overlay_tunable --cam-x 0.320 --cam-pitch -88 --fov 74
     
     # 查看结果
     eog overlay_comparison.png
     ```
   - **输出文件**：
     - `sim_camera_view_pinhole.png` - 模拟器原始图像（针孔，无畸变）
     - `sim_camera_view_distorted.png` - 应用畸变后的模拟图像
     - `overlay_comparison.png` - 三图对比（Sim | Overlay | Real）
     - `overlay_only.png` - 仅叠加图

2. **`scripts/camera_overlay.py`**
   - 旧版overlay脚本（无可调参数）
   - 可能已过时，建议使用 `camera_overlay_tunable.py`

3. **`scripts/view_env.py`**
   - GUI查看器（**有bug，暂不可用**）

### 文档

1. **`docs/camera_tuning_guide.md`** ⭐
   - 详细的相机微调指南
   - 包含参数说明、工作流、快速实验技巧

2. **Artifact 文件**（在 `/home/admin/.gemini/antigravity/brain/.../`）
   - `task.md` - 任务清单
   - `implementation_plan.md` - 实现计划
   - `walkthrough.md` - 完成报告

---

## 坐标系统

### Track1_Simulation_Parameters.md 坐标系（当前环境使用）

```
原点：桌面左下角
X 轴：向右（+X）
Y 轴：向前/进入场景（+Y）
Z 轴：向上（+Z）
```

### Grid 定义

```python
# 上排（Y = 0.178 ~ 0.342）
LEFT_GRID = {"x_min": 0.051, "x_max": 0.217, "y_min": 0.178, "y_max": 0.342}
MID_GRID = {"x_min": 0.238, "x_max": 0.394, "y_min": 0.178, "y_max": 0.342}
RIGHT_GRID = {"x_min": 0.414, "x_max": 0.580, "y_min": 0.178, "y_max": 0.342}

# 下排（新增，Y = 0.01 ~ 0.174）
# Bottom Grid: X=[0.238, 0.394], Y=[0.01, 0.174]
```

### 机器人底座位置

```python
左机器人: [0.119, 0.10, 0]
右机器人: [0.433, 0.10, 0]
```

---

## 相机参数

### 当前默认值

```python
位置: [0.316, 0.260, 0.407]  # [X, Y, Z] 米
旋转: pitch=-90°, yaw=0°, roll=0°  # 欧拉角（XYZ顺序）
FOV: 73.63°  # 垂直视场角
```

### 畸变参数（硬编码，来自 distort.py）

```python
MTX = [[570.217, 0, 327.460],
       [0, 570.180, 260.836],
       [0, 0, 1]]

DIST = [-0.7354, 0.9493, 0.0002, -0.0020, -0.8642]
```

---

## 下一步行动

### 立即需要完成

1. **相机参数微调**（用户操作）
   ```bash
   cd /home/admin/Desktop/eai-final-project
   
   # 运行默认参数
   python -m scripts.camera_overlay_tunable
   eog overlay_comparison.png
   
   # 根据叠加效果微调参数
   # 示例：
   python -m scripts.camera_overlay_tunable --cam-x 0.320 --cam-pitch -88
   
   # 找到最佳参数后，记录下来
   ```

2. **更新环境默认相机参数**
   - 当用户找到最佳参数后
   - 更新 `scripts/track1_env.py` 中 `_default_sensor_configs()` 的默认值

### 后续任务

3. **验证品字形胶带布局**
   - 运行 `camera_overlay_tunable.py` 查看胶带线是否正确
   - 检查是否有4个正方形，共享边是否正确

4. **测试多环境GPU模式**
   ```bash
   python -m scripts.test_env --num-envs 4 --task lift
   ```

5. **完成域随机化验证**
   - 确认所有随机化功能正常工作

---

## 参考资料

### 源文件

- **Track1参数文档**: `Track1_Simulation_Parameters.md`
- **参考脚本**: `eai-2025-fall-final-project-reference-scripts/`
  - `front_camera.py` - 相机参考设置
  - `front_camera.png` - 真实相机图像
  - `distort.py` / `undistort.py` - 畸变处理

### 相关文档

- **Sim2Real指南**: `lerobot-sim2real/docs/zero_shot_rgb_sim2real.md`

---

## 环境配置

### Conda 环境

```bash
conda activate maniskill
```

### 代理设置（多环境GPU测试需要）

```bash
export https_proxy="http://127.0.0.1:7890"
export http_proxy="http://127.0.0.1:7890"
export all_proxy="socks5://127.0.0.1:7890"
```

### 工作流文件

- `.agent/workflows/run-maniskill.md` - ManiSkill 运行命令（包含代理设置）

---

## 常用命令

```bash
# 进入项目目录
cd /home/admin/Desktop/eai-final-project

# 相机参数微调
python -m scripts.camera_overlay_tunable
python -m scripts.camera_overlay_tunable --cam-x 0.320 --cam-pitch -88 --fov 74

# 查看结果
eog overlay_comparison.png

# 测试环境（单环境）
python -m scripts.test_env --task lift

# 测试环境（多环境，GPU）
# turbo-all
export https_proxy="http://127.0.0.1:7890"
export http_proxy="http://127.0.0.1:7890"
export all_proxy="socks5://127.0.0.1:7890"
python -m scripts.test_env --num-envs 4 --task lift

# Git 操作（如需恢复）
git status
git checkout scripts/track1_env.py  # 恢复文件
```

---

## 重要提醒

1. **不要再尝试修改环境坐标系**
   - 环境使用 Track1_Simulation_Parameters.md 坐标系
   - 相机对齐通过 `camera_overlay_tunable.py` 工具完成

2. **用户空间感知能力强**
   - 相机微调应由用户主导
   - AI 提供工具和指导，不要代替用户做调参决策

3. **品字形胶带布局**
   - 已实现，但需要视觉验证
   - 4个正方形应该清晰可见，共享边没有重复

4. **view_env.py 暂不可用**
   - GUI查看器有bug
   - 使用 overlay 图片替代

---

## 下次对话建议

### 首先询问用户

1. 是否已完成相机参数微调？
   - 如果是，请提供最终参数
   - 如果否，是否需要帮助？

2. 胶带线布局是否满意？
   - 检查 overlay 图片中的黑色胶带
   - 应该看到4个正方形

### 然后继续

根据用户反馈决定下一步：
- 更新环境相机参数
- 调整胶带线
- 进行其他测试

---

**文档结束**
