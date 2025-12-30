```
scripts/
├── benchmarks/           # 性能测试脚本
│   ├── benchmark_full_loop.py
│   ├── benchmark_gae.py
│   └── benchmark_ppo.py
├── utils/                # 工具脚本
│   ├── camera_overlay.py
│   ├── camera_overlay_tunable.py
│   ├── check_wrist_camera.py
│   ├── sample_poses_ik.py
│   ├── sample_valid_poses.py
│   ├── test_env.py
│   ├── view_camera.py
│   └── view_keyframes.py
├── training/             # 训练模块
├── so101.py              # 机器人定义
├── track1_env.py         # 环境
├── train.py              # 训练入口
└── view_env.py           # 保留原位

assets/
└── screenshots/          # PNG 文件
    ├── check_wrist_camera_0.png
    ├── overlay_comparison.png
    └── ... (14 files)
```


我现在正在研究如何整理与重构代码。我的环境是uv, 代码入口是scripts/train.py。我核心想要研究的是scripts/track1_env.py与scripts/so101.py。我们核心要打交道的是maniskill env,位置在.venv/lib/python3.10/site-packages/mani_skill/envs, 特别是BaseEnv .venv/lib/python3.10/site-packages/mani_skill/envs/sapien_env.py。 