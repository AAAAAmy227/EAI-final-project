---
description: 重构 make_env 函数 - 拆分环境创建逻辑
---

# 任务：重构 make_env 函数

## 📋 任务概述

`scripts/training/env_utils.py` 中的 `make_env` 函数（约 100 行）承担了太多职责：配置解析、环境创建、Wrapper 应用、SO101 配置等。需要拆分为更小的、职责单一的函数。

---

## 🎯 目标

1. 将 `make_env` 拆分为 3-4 个专注的辅助函数
2. 提高代码可读性和可测试性
3. 消除函数内的硬编码值

---

## 📍 当前结构分析

```python
def make_env(cfg, num_envs, for_eval=False, video_dir=None):
    # 1. 配置 SO101 类属性 (20 行)
    # 2. 构建 sim_config (15 行)
    # 3. 构建 env_kwargs (10 行)
    # 4. 创建 Track1Env (5 行)
    # 5. 应用 Wrappers (50 行)
    #    - SingleArmWrapper
    #    - FlattenActionWrapper
    #    - FlattenStateWrapper
    #    - RecordEpisode
    #    - ManiSkillVectorEnv
    return env
```

---

## 📝 重构方案

### 新函数结构

```python
def make_env(cfg, num_envs, for_eval=False, video_dir=None):
    """Create Track1 environment with all wrappers."""
    # 1. 配置 Agent
    configure_so101_agent(cfg)
    
    # 2. 构建配置
    sim_config = build_sim_config(cfg)
    env_kwargs = build_env_kwargs(cfg, num_envs, for_eval, sim_config)
    
    # 3. 创建基础环境
    env = create_base_env(cfg, env_kwargs)
    
    # 4. 应用 Wrappers
    env = apply_wrappers(env, cfg, num_envs, for_eval, video_dir)
    
    return env


def configure_so101_agent(cfg: DictConfig):
    """Configure SO101 class attributes from config.
    
    Warning: This modifies global state. Consider using SO101.create_configured_class()
    after refactor-so101-class-attrs is complete.
    """
    from scripts.so101 import SO101
    
    task = cfg.env.get("task", "lift")
    SO101.active_mode = "dual" if task == "sort" else "single"
    
    if "control" in cfg and "action_bounds" in cfg.control:
        bounds = OmegaConf.to_container(cfg.control.action_bounds, resolve=True)
        if task == "sort":
            SO101.action_bounds_dual_arm = bounds
        else:
            SO101.action_bounds_single_arm = bounds
    
    # Gripper physics
    if "gripper_physics" in cfg.env:
        # ...


def build_sim_config(cfg: DictConfig) -> dict:
    """Build simulation configuration dictionary."""
    sim_config = dict(
        sim_freq=120,
        control_freq=cfg.env.control_freq,
        gpu_memory_config=dict(
            max_rigid_contact_count=2**21,
            max_rigid_patch_count=2**19,
        ),
    )
    
    if "solver_position_iterations" in cfg.env:
        sim_config["solver_position_iterations"] = cfg.env.solver_position_iterations
    if "solver_velocity_iterations" in cfg.env:
        sim_config["solver_velocity_iterations"] = cfg.env.solver_velocity_iterations
    
    return sim_config


def build_env_kwargs(cfg: DictConfig, num_envs: int, for_eval: bool, sim_config: dict) -> dict:
    """Build environment keyword arguments."""
    return dict(
        num_envs=num_envs,
        obs_mode="state_dict",
        reward_mode="dense",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        cfg=cfg,
        sim_config=sim_config,
        eval_mode=for_eval,
    )


def create_base_env(cfg: DictConfig, env_kwargs: dict):
    """Create the base Track1Env instance."""
    from scripts.track1_env import Track1Env
    return Track1Env(**env_kwargs)


def apply_wrappers(env, cfg: DictConfig, num_envs: int, for_eval: bool, video_dir: str = None):
    """Apply wrappers in correct order."""
    task = cfg.env.get("task", "lift")
    
    # 1. Single-arm filtering (for non-sort tasks)
    if task in ["lift", "stack"]:
        env = SingleArmWrapper(env)
    
    # 2. Flatten action/state spaces
    env = FlattenActionWrapper(env)
    env = FlattenStateWrapper(env)
    
    # 3. Video recording (eval only)
    if for_eval and video_dir and cfg.capture_video:
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_trajectory=False,
            info_on_video=True,
            max_steps_per_video=compute_max_episode_steps(cfg) + 100,
        )
    
    # 4. Vectorization
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=not for_eval)
    
    return env
```

---

## ⚠️ 注意事项

1. **Wrapper 顺序**: 顺序至关重要！必须是:
   `SingleArmWrapper -> FlattenActionWrapper -> FlattenStateWrapper -> RecordEpisode -> ManiSkillVectorEnv`
   
2. **延迟导入**: 保持 `from scripts.track1_env import Track1Env` 在函数内部（避免循环导入）。

3. **配置传递**: 确保 `cfg` 对象在所有辅助函数间正确传递。

---

## ✅ 验收标准

1. **功能不变**: 训练/评估流程正常运行
2. **代码简化**: `make_env` 主函数减少到 ~20 行
3. **可测试性**: 每个辅助函数可以单独测试

---

## 🔗 依赖

- **推荐先完成**: `/refactor-so101-class-attrs` (SO101 类属性问题)
- 完成后 `configure_so101_agent` 可以简化

---

// turbo-all
