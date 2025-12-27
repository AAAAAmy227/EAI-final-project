# Training Stabilization and Bug Fix Report

This document summarizes the changes made to resolve the train/eval discrepancy and stabilize the PPO training process for the SO101 robot arm.

## 1. Core Issues Identified

### Policy Collapse (LogStd Explosion)
Training logs revealed that `actor_logstd` (which controls exploration noise) was growing uncontrollably (e.g., from -0.5 to +1.4). This resulted in extremely noisy actions that prevented the model from converging on a stable policy.

### Unbounded Action Outputs
The agent's actor network lacked a squashing activation function. Consequently, the mean action values could grow far beyond the valid `[-1, 1]` range (observed values as high as 14.5), leading to saturated gradients and inconsistent behavior.

### Observation Scrambling (Joint Order Mismatch)
A critical bug was discovered in `track1_env.py` where a hardcoded joint order was used for `target_qpos` normalization. This frequently mismatched the actual kinematic order provided by ManiSkill, causing the agent to receive "scrambled" input features.

### Eval/Train Discrepancy
The evaluation loop initially lacked the internal observation flattening logic used during training, meaning the evaluation agent was receiving raw dictionaries while expecting flattened tensors.

## 2. Implemented Solutions

### Agent Architecture Improvements (`agent.py`)
- **Action Squashing**: Added `nn.Tanh()` as the final layer of the `actor_mean` network.
- **Stable Initialization**: Set the default `actor_logstd` initialization to a configurable value (default `-0.5`, std ≈ 0.6) to reduce initial jitter.

### Training Stability Enhancements (`ppo_utils.py`)
- **LogStd Clamping**: Added a hard clamp to `actor_logstd` (default range `[-5, 2]`) within the PPO update loop to prevent policy collapse.

### Robust Environment Logic (`track1_env.py`)
- **Dynamic Joint Normalization**: Replaced hardcoded joint lists with dynamic lookups from the robot's `active_joints`, ensuring normalization always matches the correct physical joint.
- **Reward Normalization**: Updated the `lift` reward to be normalized by `lift_max_height`, mapping the base reward to a clear `[0, 1]` range.

### Runner & Infrastructure Fixes (`runner.py`)
- **Async Eval Isolation**: Implemented a separate `eval_agent` and CUDA stream for evaluation to prevent race conditions with training weights.
- **Logging Fixes**: Resolved variable name errors (`obs_flat`, `obs_dim`) that crashed the runner when `log_obs_stats` was enabled.
- **Reward Component Tracking**: Unified reward component logging between training and evaluation steps.

### Configuration Flexibility (`train.yaml`)
- Exposed `logstd_init`, `logstd_max`, and `logstd_min` as hyperparameters.
- Enabled `log_obs_stats` option for real-time monitoring of input normalization quality.

## 3. Verification & Cleanup

- **Benchmark Run**: Successfully executed a 1M step training run confirming stability and positive reward trends.
- **Debug Tools**: Archived all investigation scripts (joint checks, weight stats, observation mapping) into `scripts/debug_tools/` for future reference.
- **Git Housekeeping**: Updated `.gitignore` to exclude log files.

---
*Report generated on Dec 27, 2025*
