"""
Compare Fixed-Time Baseline vs RL Agent (ACAC) across 6 scenarios.
Fixed-Time logic is imported directly from examples/fixed_time_baseline.py.
Both methods run for exactly 600 simulation seconds.
"""

import os
import sys
import importlib.util
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

# --- Import fixed-time baseline action logic from examples/fixed_time_baseline.py ---
baseline_path = Path(__file__).resolve().parent / "examples" / "fixed_time_baseline.py"
spec = importlib.util.spec_from_file_location("fixed_time_baseline", str(baseline_path))
fixed_time_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed_time_module)

# --- Import ACAC modules ---
from acac import (
    SinusoidalPositionalEncoding,
    CentralizedCritic,
    AgentHistoryEncoder,
    MacroActor,
    AsyncTrajectoryBuffer,
    SyncTrajectoryBuffer,
    ACACTrainer,
    load_model_config,
)
from atcs.environment import TrafficEnvironment

_cfg = load_model_config()

SIMULATION_TIME = 600  # seconds

# ================= Scenarios =================
scenarios = [
    (
        "crowded_2intersection",
        "../SimulationData/Evaluate/Crowded/2Intersection/config.sumocfg",
    ),
    (
        "crowded_3intersection",
        "../SimulationData/Evaluate/Crowded/3Intersection/config.sumocfg",
    ),
    (
        "crowded_4intersection",
        "../SimulationData/Evaluate/Crowded/4Intersection/config.sumocfg",
    ),
    (
        "normal_2intersection",
        "../SimulationData/Evaluate/Normal/2Intersection/config.sumocfg",
    ),
    (
        "normal_3intersection",
        "../SimulationData/Evaluate/Normal/3Intersection/config.sumocfg",
    ),
    (
        "normal_4intersection",
        "../SimulationData/Evaluate/Normal/4Intersection/config.sumocfg",
    ),
]


# ================= Fixed-Time Evaluation =================
def evaluate_fixed_time(sumocfg_path: str) -> dict:
    """
    Run Fixed-Time baseline (extension = 30s for every intersection)
    exactly like examples/fixed_time_baseline.py, for SIMULATION_TIME seconds.
    """
    env = TrafficEnvironment(sumocfg_path=sumocfg_path, use_gui=False)
    obs, reward, done, info = env.reset()

    t = 0
    total_delay = 0.0
    total_queue = 0.0
    total_saturation = 0.0

    while not done and t < SIMULATION_TIME:
        # Same action logic as fixed_time_baseline.py line 36
        required_intersections = info.get("intersection_require_action", [])
        action = {tls_id: 30 for tls_id in required_intersections}

        obs, reward, done, info = env.step(action)
        dt = info["delta_t"]
        total_delay += reward[:, :, 0].mean() * dt
        total_queue += reward[:, :, 1].mean() * dt
        total_saturation += reward[:, :, 2].mean() * dt
        t += dt

    env.close()
    return {
        "delay": total_delay / max(t, 1),
        "queue": total_queue / max(t, 1),
        "saturation": total_saturation / max(t, 1),
    }


# ================= RL Agent Evaluation =================
def initialize_acac(obs_dim, action_dim, tls_names, device="cpu"):
    num_agents = len(tls_names)
    time_encoder = SinusoidalPositionalEncoding(16).to(device)
    encoders = [
        AgentHistoryEncoder(obs_dim, 16, 128).to(device) for _ in range(num_agents)
    ]
    actors = [
        MacroActor(128, action_dim, min_action=0.0, max_action=1.0).to(device)
        for _ in range(num_agents)
    ]
    critic = CentralizedCritic(128, 4).to(device)
    agents_buffer = AsyncTrajectoryBuffer(capacity=100, num_agents=num_agents)
    critic_buffer = SyncTrajectoryBuffer(capacity=100)
    actor_optimizers = [
        torch.optim.Adam(a.parameters(), lr=_cfg.training.actor_lr) for a in actors
    ]
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=_cfg.training.critic_lr)
    trainer = ACACTrainer(
        actors=actors,
        encoders=encoders,
        critic=critic,
        agents_buffer=agents_buffer,
        critic_buffer=critic_buffer,
        optimizers={"actor": actor_optimizers, "critic": critic_optimizer},
        time_encoder=time_encoder,
        tls_names=tls_names,
        device=device,
    )
    return trainer


def evaluate_rl_agent(sumocfg_path: str, checkpoint_path: str) -> dict:
    """Run RL agent for SIMULATION_TIME seconds."""
    device = torch.device("cpu")

    # Get obs_dim and tls_names
    env = TrafficEnvironment(sumocfg_path=sumocfg_path, use_gui=False)
    obs, reward, done, info = env.reset()
    obs_dim = obs.shape[1] * obs.shape[2] + 2
    tls_names = env.tls_ids
    env.close()

    trainer = initialize_acac(obs_dim, 1, tls_names, device)
    trainer.load_model(checkpoint_path)
    for a in trainer.actors:
        a.eval()
    for e in trainer.encoders:
        e.eval()
    trainer.critic.eval()

    env = TrafficEnvironment(sumocfg_path=sumocfg_path, use_gui=False)
    with torch.no_grad():
        metrics = trainer.evaluate_model(env, max_steps=SIMULATION_TIME)
    env.close()
    return metrics


# ================= Main =================
print(f"=== Evaluation: Fixed-Time vs RL Agent ({SIMULATION_TIME}s) ===\n")

results_baseline = []
results_rl = []

for name, cfg_path in scenarios:
    abs_cfg = os.path.abspath(cfg_path)
    ckpt_path = f"checkpoint/{name}_checkpoint.pt"
    label = name.replace("_", " ").title()

    # --- Fixed-Time ---
    print(f"[Fixed-Time] {label} ...")
    ft = evaluate_fixed_time(abs_cfg)
    results_baseline.append(
        {
            "Scenario": label,
            "Wait Time (s)": ft["delay"],
            "Queue Length (m)": ft["queue"],
            "Degree of Saturation": ft["saturation"],
        }
    )
    print(
        f"  -> delay={ft['delay']:.4f}, queue={ft['queue']:.4f}, sat={ft['saturation']:.4f}"
    )

    # --- RL Agent ---
    print(f"[RL Agent]   {label} ...")
    rl = evaluate_rl_agent(abs_cfg, ckpt_path)
    results_rl.append(
        {
            "Scenario": label,
            "Wait Time (s)": rl["delay"],
            "Queue Length (m)": rl["queue"],
            "Degree of Saturation": rl["saturation"],
        }
    )
    print(
        f"  -> delay={rl['delay']:.4f}, queue={rl['queue']:.4f}, sat={rl['saturation']:.4f}\n"
    )

df_baseline = pd.DataFrame(results_baseline)
df_rl = pd.DataFrame(results_rl)
df_baseline.to_csv("checkpoint/kpi_baseline_results.csv", index=False)
df_rl.to_csv("checkpoint/kpi_evaluation_results.csv", index=False)

# ================= Generate 6 comparison plots =================
for idx in range(len(scenarios)):
    scenario = df_rl.iloc[idx]["Scenario"]
    row_base = df_baseline.iloc[idx]
    row_rl = df_rl.iloc[idx]

    metrics = ["Wait Time (s)", "Queue Length (m)", "Degree of Saturation"]
    base_vals = [float(row_base[m]) for m in metrics]
    rl_vals = [float(row_rl[m]) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(
        x - width / 2, base_vals, width, label="Fixed Time", color="lightcoral"
    )
    rects2 = ax.bar(
        x + width / 2, rl_vals, width, label="RL Agent (ACAC)", color="mediumseagreen"
    )

    ax.set_ylabel("KPI Value")
    ax.set_title(
        f"Comparison: Fixed Time vs RL Agent ({SIMULATION_TIME}s)\nScenario: {scenario}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    fname = f"checkpoint/compare_{scenario.replace(' ', '_')}.png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

print("\nDone!")
