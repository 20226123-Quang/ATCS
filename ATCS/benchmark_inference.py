"""Benchmark inference time for ACAC model."""

import time
import torch
import numpy as np
from pathlib import Path
from acac import (
    SinusoidalPositionalEncoding,
    CentralizedCritic,
    AgentHistoryEncoder,
    MacroActor,
    load_model_config
)

_cfg = load_model_config()


def benchmark_inference(num_agents=1, obs_dim=5, hidden_dim=64, time_embed_dim=16, 
                        num_heads=4, action_dim=1, num_iterations=1000, device="cpu"):
    """
    Benchmark the inference time for ACAC components.
    
    Args:
        num_agents: Number of agents (traffic lights)
        obs_dim: Observation dimension
        hidden_dim: Hidden dimension for neural networks
        time_embed_dim: Time embedding dimension
        num_heads: Number of attention heads for critic
        action_dim: Action dimension (output)
        num_iterations: Number of iterations to benchmark
        device: 'cpu' or 'cuda'
    """
    
    print(f"\n{'='*70}")
    print(f"ACAC Model Inference Benchmark")
    print(f"{'='*70}")
    print(f"Device:          {device}")
    print(f"Num Agents:      {num_agents}")
    print(f"Obs Dim:         {obs_dim}")
    print(f"Hidden Dim:      {hidden_dim}")
    print(f"Time Embed Dim:  {time_embed_dim}")
    print(f"Num Heads:       {num_heads}")
    print(f"Iterations:      {num_iterations}")
    print(f"{'='*70}\n")
    
    # Initialize models
    print("Initializing models...")
    time_encoder = SinusoidalPositionalEncoding(time_embed_dim).to(device)
    encoders = [AgentHistoryEncoder(obs_dim, time_embed_dim, hidden_dim).to(device) 
                for _ in range(num_agents)]
    actors = [MacroActor(hidden_dim, action_dim, min_action=0.0, max_action=1.0).to(device)
              for _ in range(num_agents)]
    critic = CentralizedCritic(hidden_dim, num_heads).to(device)
    
    # Set to eval mode (no dropout, batch norm uses running stats)
    time_encoder.eval()
    for enc in encoders:
        enc.eval()
    for act in actors:
        act.eval()
    critic.eval()
    
    print("Models initialized successfully.\n")
    
    # Benchmark each component
    # 1. Time encoding
    print(f"{'Component':<30} {'Time (ms)':<15} {'Std Dev':<15}")
    print("-" * 60)
    
    times_time_enc = []
    with torch.no_grad():
        for t in range(num_iterations):
            time_step = torch.tensor([t % 100], dtype=torch.long).to(device)
            start = time.perf_counter()
            _ = time_encoder(time_step)
            end = time.perf_counter()
            times_time_enc.append((end - start) * 1000)
    
    times_time_enc = np.array(times_time_enc)
    print(f"{'Time Encoding':<30} {times_time_enc.mean():<15.4f} {times_time_enc.std():<15.4f}")
    
    # 2. Agent History Encoder (per-agent)
    times_enc = []
    with torch.no_grad():
        obs = torch.randn(obs_dim).to(device)
        p_t = torch.randn(time_embed_dim).to(device)
        h_prev = torch.zeros(hidden_dim).to(device)
        
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = encoders[0](obs, p_t, h_prev)
            end = time.perf_counter()
            times_enc.append((end - start) * 1000)
    
    times_enc = np.array(times_enc)
    print(f"{'Encoder (per-agent)':<30} {times_enc.mean():<15.4f} {times_enc.std():<15.4f}")
    
    # 3. Actor (per-agent)
    times_actor = []
    with torch.no_grad():
        h = torch.randn(hidden_dim).to(device)
        
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = actors[0](h)
            end = time.perf_counter()
            times_actor.append((end - start) * 1000)
    
    times_actor = np.array(times_actor)
    print(f"{'Actor (per-agent)':<30} {times_actor.mean():<15.4f} {times_actor.std():<15.4f}")
    
    # 4. Critic
    times_critic = []
    with torch.no_grad():
        h_all = torch.randn(num_agents, hidden_dim).to(device)
        
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = critic(h_all)
            end = time.perf_counter()
            times_critic.append((end - start) * 1000)
    
    times_critic = np.array(times_critic)
    print(f"{'Critic':<30} {times_critic.mean():<15.4f} {times_critic.std():<15.4f}")
    
    # 5. Full inference pipeline for one step
    print("\n" + "="*60)
    print("Full Inference Pipeline (All Agents + Critic):")
    print("="*60)
    
    times_full = []
    with torch.no_grad():
        obs_all = torch.randn(num_agents, obs_dim).to(device)
        h_states = [torch.zeros(hidden_dim).to(device) for _ in range(num_agents)]
        
        for t in range(num_iterations):
            start = time.perf_counter()
            
            # Get time encoding
            time_step = torch.tensor([t % 100], dtype=torch.long).to(device)
            p_t = time_encoder(time_step)  # [time_embed_dim]
            
            # Process each agent
            h_next = []
            for agent_idx in range(num_agents):
                h = encoders[agent_idx](obs_all[agent_idx], p_t, h_states[agent_idx])
                h_next.append(h)
            
            h_next = torch.stack(h_next, dim=0)  # [num_agents, hidden_dim]
            
            # Get critic value
            _ = critic(h_next)
            
            # Get actor actions
            for agent_idx in range(num_agents):
                _ = actors[agent_idx](h_next[agent_idx])
            
            end = time.perf_counter()
            times_full.append((end - start) * 1000)
    
    times_full = np.array(times_full)
    print(f"\n{'Full Pipeline':<30} {times_full.mean():<15.4f} {times_full.std():<15.4f}")
    print(f"  Min: {times_full.min():.4f} ms")
    print(f"  Max: {times_full.max():.4f} ms")
    print(f"  P95: {np.percentile(times_full, 95):.4f} ms")
    
    print(f"\n{'='*70}")
    print(f"Summary: Average inference time = {times_full.mean():.4f} ± {times_full.std():.4f} ms")
    print(f"         For {num_agents} agent(s), observations per step = {obs_dim} dims")
    print(f"{'='*70}\n")
    
    return times_full


if __name__ == "__main__":
    # Test on CPU
    print("\n" + "="*70)
    print("BENCHMARKING ON CPU")
    print("="*70)
    times_cpu = benchmark_inference(
        num_agents=1,
        obs_dim=5,
        hidden_dim=64,
        time_embed_dim=16,
        num_heads=4,
        action_dim=1,
        num_iterations=1000,
        device="cpu"
    )
    
    # Test on GPU if available
    if torch.cuda.is_available():
        print("\n" + "="*70)
        print("BENCHMARKING ON GPU")
        print("="*70)
        torch.cuda.synchronize()
        times_gpu = benchmark_inference(
            num_agents=1,
            obs_dim=5,
            hidden_dim=64,
            time_embed_dim=16,
            num_heads=4,
            action_dim=1,
            num_iterations=1000,
            device="cuda"
        )
        torch.cuda.synchronize()
        
        print("\n" + "="*70)
        print("COMPARISON: CPU vs GPU")
        print("="*70)
        print(f"CPU Average: {times_cpu.mean():.4f} ms")
        print(f"GPU Average: {times_gpu.mean():.4f} ms")
        print(f"Speedup:     {times_cpu.mean() / times_gpu.mean():.2f}x")
        print(f"{'='*70}\n")
    
    # Test with multiple agents (more realistic scenario)
    print("\n" + "="*70)
    print("MULTI-AGENT SCENARIO (4 Traffic Lights)")
    print("="*70)
    benchmark_inference(
        num_agents=4,
        obs_dim=20,  # 4 lanes * 5
        hidden_dim=128,
        time_embed_dim=16,
        num_heads=4,
        action_dim=1,
        num_iterations=500,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
