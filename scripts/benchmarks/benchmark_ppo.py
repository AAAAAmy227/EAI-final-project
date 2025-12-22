"""
Benchmark script to test PPO components (policy inference, update) 
with and without torch.compile + CudaGraphModule.
Excludes simulator to isolate RL algorithm performance.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from functools import partial

# Import from project
import sys
sys.path.insert(0, "/home/admin/Desktop/eai-final-project")
from scripts.training.agent import Agent
from scripts.training.ppo_utils import optimized_gae, make_ppo_update_fn
from tensordict import from_module
from tensordict.nn import CudaGraphModule
import tensordict


def benchmark_policy_inference(agent, obs, warmup=10, iters=100, use_compile=False, use_cudagraph=False):
    """Benchmark policy inference."""
    policy = agent.get_action_and_value
    
    if use_compile:
        if use_cudagraph:
            policy = torch.compile(policy)  # default mode for cudagraph compatibility
        else:
            policy = torch.compile(policy, mode="reduce-overhead")
    
    if use_cudagraph:
        policy = CudaGraphModule(policy)
    
    # First call (compilation)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = policy(obs=obs)
    torch.cuda.synchronize()
    compile_time = time.perf_counter() - t0
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = policy(obs=obs)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = policy(obs=obs)
    torch.cuda.synchronize()
    avg_time = (time.perf_counter() - t0) / iters * 1000
    
    return compile_time * 1000, avg_time


def benchmark_update(agent, optimizer, cfg, batch, warmup=10, iters=100, use_compile=False):
    """Benchmark PPO update step."""
    update_fn = make_ppo_update_fn(agent, optimizer, cfg)
    
    if use_compile:
        update_fn = torch.compile(update_fn, mode="reduce-overhead")
    
    # First call (compilation)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = update_fn(batch.clone(), tensordict_out=tensordict.TensorDict())
    torch.cuda.synchronize()
    compile_time = time.perf_counter() - t0
    
    # Warmup
    for _ in range(warmup):
        _ = update_fn(batch.clone(), tensordict_out=tensordict.TensorDict())
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = update_fn(batch.clone(), tensordict_out=tensordict.TensorDict())
    torch.cuda.synchronize()
    avg_time = (time.perf_counter() - t0) / iters * 1000
    
    return compile_time * 1000, avg_time


class MockConfig:
    """Mock config for PPO."""
    class PPO:
        clip_coef = 0.2
        ent_coef = 0.0
        vf_coef = 0.5
        max_grad_norm = 0.5
        learning_rate = 3e-4
    ppo = PPO()


def main():
    device = torch.device("cuda")
    print(f"Device: {device}")
    
    # Config
    num_envs = 2048
    num_steps = 50
    n_obs = 49
    n_act = 6
    minibatch_size = num_envs * num_steps // 32  # 3200
    
    warmup_iters = 10
    bench_iters = 100
    
    print(f"\nConfig: num_envs={num_envs}, num_steps={num_steps}, n_obs={n_obs}, n_act={n_act}")
    print(f"Minibatch size: {minibatch_size}")
    print(f"Warmup: {warmup_iters}, Benchmark: {bench_iters} iterations")
    
    # Create test data
    obs = torch.randn(num_envs, n_obs, device=device)
    batch = tensordict.TensorDict({
        "obs": torch.randn(minibatch_size, n_obs, device=device),
        "actions": torch.randn(minibatch_size, n_act, device=device),
        "logprobs": torch.randn(minibatch_size, device=device),
        "advantages": torch.randn(minibatch_size, device=device),
        "returns": torch.randn(minibatch_size, device=device),
        "vals": torch.randn(minibatch_size, device=device),
    }, batch_size=[minibatch_size])
    
    cfg = MockConfig()
    
    # ============================================================
    # Policy Inference Benchmark
    # ============================================================
    print("\n" + "=" * 70)
    print("POLICY INFERENCE BENCHMARK (num_envs observations)")
    print("=" * 70)
    
    results_policy = []
    
    # 1. No optimization
    agent = Agent(n_obs, n_act, device=device)
    compile_t, run_t = benchmark_policy_inference(agent, obs, warmup_iters, bench_iters, False, False)
    results_policy.append(("No optimization", compile_t, run_t))
    print(f"[1] No optimization:        compile={compile_t:>8.2f}ms, runtime={run_t:.3f}ms")
    baseline_policy = run_t
    
    # 2. torch.compile only
    agent = Agent(n_obs, n_act, device=device)
    compile_t, run_t = benchmark_policy_inference(agent, obs, warmup_iters, bench_iters, True, False)
    results_policy.append(("torch.compile", compile_t, run_t))
    print(f"[2] torch.compile:          compile={compile_t:>8.2f}ms, runtime={run_t:.3f}ms  ({baseline_policy/run_t:.2f}x)")
    
    # 3. torch.compile + CudaGraphModule
    agent = Agent(n_obs, n_act, device=device)
    compile_t, run_t = benchmark_policy_inference(agent, obs, warmup_iters, bench_iters, True, True)
    results_policy.append(("compile + CudaGraph", compile_t, run_t))
    print(f"[3] compile + CudaGraph:    compile={compile_t:>8.2f}ms, runtime={run_t:.3f}ms  ({baseline_policy/run_t:.2f}x)")
    
    # ============================================================
    # PPO Update Benchmark
    # ============================================================
    print("\n" + "=" * 70)
    print("PPO UPDATE BENCHMARK (minibatch)")
    print("=" * 70)
    
    results_update = []
    
    # 1. No optimization
    agent = Agent(n_obs, n_act, device=device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.ppo.learning_rate, fused=True)
    compile_t, run_t = benchmark_update(agent, optimizer, cfg, batch, warmup_iters, bench_iters, False)
    results_update.append(("No optimization", compile_t, run_t))
    print(f"[1] No optimization:        compile={compile_t:>8.2f}ms, runtime={run_t:.3f}ms")
    baseline_update = run_t
    
    # 2. torch.compile (reduce-overhead)
    agent = Agent(n_obs, n_act, device=device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.ppo.learning_rate, fused=True)
    compile_t, run_t = benchmark_update(agent, optimizer, cfg, batch, warmup_iters, bench_iters, True)
    results_update.append(("torch.compile", compile_t, run_t))
    print(f"[2] torch.compile:          compile={compile_t:>8.2f}ms, runtime={run_t:.3f}ms  ({baseline_update/run_t:.2f}x)")
    
    # ============================================================
    # Full Iteration Estimate
    # ============================================================
    print("\n" + "=" * 70)
    print("FULL ITERATION ESTIMATE (policy * num_steps + 32 updates)")
    print("=" * 70)
    
    # Estimate without optimization
    no_opt_time = baseline_policy * num_steps + baseline_update * 32  # 32 minibatches per epoch * 4 epochs / simplified
    print(f"No optimization:       {no_opt_time:.2f} ms/iteration")
    
    # Estimate with optimization
    opt_policy = results_policy[2][2]  # compile + cudagraph
    opt_update = results_update[1][2]  # compile
    opt_time = opt_policy * num_steps + opt_update * 32
    print(f"With optimization:     {opt_time:.2f} ms/iteration")
    print(f"Speedup:               {no_opt_time/opt_time:.2f}x")


if __name__ == "__main__":
    main()
