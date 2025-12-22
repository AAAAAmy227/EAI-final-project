# Project Context Handoff

**Last Updated:** December 22, 2025
**Project:** EAI Final Project (Track 1 - ManiSkill Robotics)

## 1. Project Overview & Objectives
The goal is to solve robotic manipulation tasks (`lift`, `stack`, `sort`) in the `Track1` environment using Reinforcement Learning (PPO). The project emphasizes Sim-to-Real transfer potential, currently focusing on pure simulation training with plans for RMA (Rapid Motor Adaptation) if needed.

**Core Requirements:**
- **Environment**: Custom `Track1Env` based on ManiSkill 3 (Gymnasium-compliant).
- **Algorithm**: PPO (Proximal Policy Optimization).
- **Backend**: Pure GPU simulation (PhysX + CUDA) for high-performance training.
- **Config Management**: Hydra.
- **Logging**: WandB + Tensorboard.

---

## 2. Key Accomplishments (What We Have Done)

### A. Environment (`scripts/track1_env.py`)
- **Initialization**: Fixed initialization order for `SingleArmEnv` and Action Space.
- **Observations**:
  - Implemented `obs_mode="state"` support: Flattened vector of Proprioception + Object Poses + Relative Vectors (TCP-to-Obj).
  - Implemented `obs_mode="rgb"`: Standard ManiSkill RGBD support.
- **Rewards**:
  - **Staged Dense Reward**: Curriculum-based reward (Reach -> Grasp -> Lift -> Success).
  - **Parallel Dense Reward**: Traditional weighted sum.
  - Configurable via Hydra (`configs/reward/lift.yaml`).
- **Domain Randomization**: Configurable in `Track1Env`.

### B. Training Infrastructure (`scripts/training/`)
Refactored from a monolithic script and SB3 dependency to a **Modular CleanRL-style Pure GPU** implementation.
- **`scripts/train.py`**: Clean entry point with Hydra + WandB integration.
- **`scripts/training/runner.py`**: High-performance PPO training loop (all tensors on GPU). Auto-handles `Dict` vs `Box` observation spaces.
- **`scripts/training/agent.py`**: 
  - `Agent` class supporting both MLP (for State) and NatureCNN (for RGB).
  - Automatic network selection based on input type.
- **`scripts/training/common.py`**: Utilities like `make_env`, `DictArray`, and `FlattenStateWrapper` (crucial for State-based training).

### C. Configuration (`configs/`)
- `train.yaml`: Main PPO/Training config.
- `env/track1.yaml`: Environment specific settings.
- `reward/lift.yaml`: Staged reward weights and thresholds.

---

## 3. Current Code Structure

```
eai-final-project/
├── configs/
│   ├── train.yaml              # Main training config
│   ├── env/track1.yaml         # Env config
│   └── reward/lift.yaml        # Reward config
├── scripts/
│   ├── track1_env.py           # The Environment
│   ├── train.py                # Main Entry Point
│   └── training/               # PPO Module
│       ├── agent.py            # Neural Networks
│       ├── common.py           # Wrappers & Utils
│       └── runner.py           # Training Loop
└── outputs/                    # Hydra logs & checkpoints
```

---

## 4. Work in Progress & Future Plans

### Immediate Next Steps
1.  **Run Large-Scale Training (State-Based)**: 
    - Verify that the `state` based policy can master the `lift` task using Staged Rewards.
    - Command: `python -m scripts.train env.obs_mode=state env.task=lift training.num_envs=512 training.total_timesteps=10000000`
2.  **Visual Training (RGB)**:
    - Once State-Based works (Teacher), try training directly with RGB (`env.obs_mode=rgb`).
    - If RGB fails to learn, consider **RMA (Rapid Motor Adaptation)**: Distill State-Based Teacher into RGB Student.

### Future Features
- **Stack Task**: Implement Staged Reward for `stack` task (currently placeholder).
- **Sort Task**: Implement logic for sorting.
- **Sim-to-Real**:
  - Verify Domain Randomization effectiveness.
  - Potential Real-World deployment using the Student policy.

---

## 5. Technical Notes for Next Session
- **Dependencies**: Removed Stable-Baselines3 to avoid numpy<2.0 conflicts. Using `gymnasium==0.29.1` and `numpy<2.0` to match ManiSkill requirements.
- **GPU Handling**: All PPO operations are on GPU. Do not use `.cpu()` or `.numpy()` inside the inner loop unless necessary for logging.
- **WandB**: Auto-logs Config + Git Commit Hash. Make sure to commit changes before long runs to track experiments accurately.
