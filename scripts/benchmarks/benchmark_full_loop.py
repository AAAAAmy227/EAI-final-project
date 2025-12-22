"""
Benchmark PPO training loop with a minimal fake environment.
This isolates RL algorithm performance from simulator overhead.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from functools import partial
from collections import deque

import sys
sys.path.insert(0, "/home/admin/Desktop/eai-final-project")

from scripts.training.agent import Agent
from scripts.training.ppo_utils import optimized_gae, make_ppo_update_fn
from tensordict import from_module
from tensordict.nn import CudaGraphModule
import tensordict


class FakeEnv:
    """Minimal fake environment with almost zero overhead."""
    
    def __init__(self, num_envs, n_obs, n_act, device):
        self.num_envs = num_envs
        self.n_obs = n_obs
        self.n_act = n_act
        self.device = device
        
        # Pre-allocate tensors
        self._obs = torch.randn(num_envs, n_obs, device=device)
        self._reward = torch.randn(num_envs, device=device)
        self._done = torch.zeros(num_envs, device=device, dtype=torch.bool)
        
    def reset(self):
        return self._obs.clone(), {}
    
    def step(self, action):
        # Almost no computation - just return pre-allocated tensors
        return self._obs, self._reward, self._done, self._done, {}


class MockConfig:
    class PPO:
        clip_coef = 0.2
        ent_coef = 0.0
        vf_coef = 0.5
        max_grad_norm = 0.5
        learning_rate = 3e-4
        gamma = 0.8
        gae_lambda = 0.9
        target_kl = 0.2
        update_epochs = 4
        num_minibatches = 32
    ppo = PPO()


def benchmark_training_loop(use_compile: bool, use_cudagraph: bool, num_iterations: int = 20):
    """Benchmark a stripped-down training loop."""
    device = torch.device("cuda")
    
    # Config
    num_envs = 2048
    num_steps = 50
    n_obs = 49
    n_act = 6
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // 32
    
    cfg = MockConfig()
    
    # Setup
    env = FakeEnv(num_envs, n_obs, n_act, device)
    agent = Agent(n_obs, n_act, device=device)
    
    if use_cudagraph:
        agent_inference = Agent(n_obs, n_act, device=device)
        from_module(agent).data.to_module(agent_inference)
    else:
        agent_inference = agent
    
    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(cfg.ppo.learning_rate, device=device),
        fused=True,
    )
    
    # Setup functions
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value
    gae_fn = partial(optimized_gae, gamma=cfg.ppo.gamma, gae_lambda=cfg.ppo.gae_lambda)
    update_fn = make_ppo_update_fn(agent, optimizer, cfg)
    
    if use_compile:
        if use_cudagraph:
            policy = torch.compile(policy)
        else:
            policy = torch.compile(policy, mode="reduce-overhead")
        get_value = torch.compile(get_value)
        update_fn = torch.compile(update_fn, mode="reduce-overhead")
    
    if use_cudagraph:
        policy = CudaGraphModule(policy)
    
    # Pre-allocate storage
    storage = tensordict.TensorDict({
        "obs": torch.zeros((num_steps, num_envs, n_obs), device=device),
        "dones": torch.zeros((num_steps, num_envs), device=device, dtype=torch.bool),
        "vals": torch.zeros((num_steps, num_envs), device=device),
        "actions": torch.zeros((num_steps, num_envs, n_act), device=device),
        "logprobs": torch.zeros((num_steps, num_envs), device=device),
        "rewards": torch.zeros((num_steps, num_envs), device=device),
    }, batch_size=[num_steps, num_envs])
    
    # Initial reset
    obs, _ = env.reset()
    done = torch.zeros(num_envs, device=device, dtype=torch.bool)
    
    # Warmup (includes compilation)
    print("  Warming up (includes compilation)...")
    torch.cuda.synchronize()
    warmup_start = time.perf_counter()
    
    for iteration in range(3):
        if use_compile:
            torch.compiler.cudagraph_mark_step_begin()
        
        # Rollout
        for step in range(num_steps):
            with torch.no_grad():
                action, logprob, _, value = policy(obs=obs)
            
            next_obs, reward, next_done, _, _ = env.step(action)
            
            storage[step]["obs"] = obs
            storage[step]["dones"] = next_done
            storage[step]["vals"] = value.flatten()
            storage[step]["actions"] = action
            storage[step]["logprobs"] = logprob
            storage[step]["rewards"] = reward
            
            obs = next_obs
            done = next_done
        
        # GAE
        with torch.no_grad():
            next_value = get_value(obs)
        advs, rets = gae_fn(storage["rewards"], storage["vals"], storage["dones"], next_value, done)
        storage["advantages"] = advs
        storage["returns"] = rets
        
        # PPO Update
        container_flat = storage.view(-1)
        for epoch in range(cfg.ppo.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(minibatch_size)
            for b in b_inds:
                _ = update_fn(container_flat[b], tensordict_out=tensordict.TensorDict())
        
        # Sync (if cudagraph)
        if use_cudagraph:
            from_module(agent).data.to_module(agent_inference)
    
    torch.cuda.synchronize()
    warmup_time = time.perf_counter() - warmup_start
    print(f"  Warmup time: {warmup_time:.2f}s")
    
    # Benchmark
    print(f"  Running {num_iterations} iterations...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for iteration in range(num_iterations):
        if use_compile:
            torch.compiler.cudagraph_mark_step_begin()
        
        # Rollout
        for step in range(num_steps):
            with torch.no_grad():
                action, logprob, _, value = policy(obs=obs)
            
            next_obs, reward, next_done, _, _ = env.step(action)
            
            storage[step]["obs"] = obs
            storage[step]["dones"] = next_done
            storage[step]["vals"] = value.flatten()
            storage[step]["actions"] = action
            storage[step]["logprobs"] = logprob
            storage[step]["rewards"] = reward
            
            obs = next_obs
            done = next_done
        
        # GAE
        with torch.no_grad():
            next_value = get_value(obs)
        advs, rets = gae_fn(storage["rewards"], storage["vals"], storage["dones"], next_value, done)
        storage["advantages"] = advs
        storage["returns"] = rets
        
        # PPO Update
        container_flat = storage.view(-1)
        for epoch in range(cfg.ppo.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(minibatch_size)
            for b in b_inds:
                _ = update_fn(container_flat[b], tensordict_out=tensordict.TensorDict())
        
        # Sync (if cudagraph)
        if use_cudagraph:
            from_module(agent).data.to_module(agent_inference)
    
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    
    total_steps = num_iterations * batch_size
    sps = total_steps / total_time
    avg_iter_time = total_time / num_iterations * 1000
    
    return sps, avg_iter_time, warmup_time


def main():
    print("=" * 70)
    print("PPO Training Loop Benchmark (Fake Environment)")
    print("=" * 70)
    print("\nConfig: num_envs=2048, num_steps=50, n_obs=49, n_act=6")
    print("        update_epochs=4, num_minibatches=32")
    print()
    
    num_iterations = 20
    
    results = []
    
    # 1. No optimization
    print("[1] No optimization")
    sps, iter_time, warmup = benchmark_training_loop(False, False, num_iterations)
    results.append(("No optimization", sps, iter_time, warmup))
    print(f"    SPS: {sps:,.0f}, Iter time: {iter_time:.2f}ms\n")
    baseline_sps = sps
    
    # 2. torch.compile only (reduce-overhead)
    print("[2] torch.compile (reduce-overhead)")
    sps, iter_time, warmup = benchmark_training_loop(True, False, num_iterations)
    results.append(("torch.compile", sps, iter_time, warmup))
    print(f"    SPS: {sps:,.0f}, Iter time: {iter_time:.2f}ms, Speedup: {sps/baseline_sps:.2f}x\n")
    
    # 3. torch.compile + CudaGraphModule
    print("[3] torch.compile + CudaGraphModule")
    sps, iter_time, warmup = benchmark_training_loop(True, True, num_iterations)
    results.append(("compile + CudaGraph", sps, iter_time, warmup))
    print(f"    SPS: {sps:,.0f}, Iter time: {iter_time:.2f}ms, Speedup: {sps/baseline_sps:.2f}x\n")
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'SPS':<15} {'Iter (ms)':<12} {'Warmup (s)':<12} {'Speedup':<10}")
    print("-" * 80)
    for name, sps, iter_time, warmup in results:
        speedup = sps / baseline_sps
        print(f"{name:<30} {sps:<15,.0f} {iter_time:<12.2f} {warmup:<12.2f} {speedup:.2f}x")


if __name__ == "__main__":
    main()
