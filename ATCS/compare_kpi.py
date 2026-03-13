"""Compare RL (ACAC) vs Fixed-Time on 3 KPIs with normalized saturation."""

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from acac import (
    ACACTrainer,
    AgentHistoryEncoder,
    AsyncTrajectoryBuffer,
    CentralizedCritic,
    MacroActor,
    SinusoidalPositionalEncoding,
    SyncTrajectoryBuffer,
    load_model_config,
)
from atcs.environment import TrafficEnvironment

_cfg = load_model_config()


def initialize_acac(obs_dim, action_dim, min_action, max_action, tls_names, device="cpu"):
    num_agents = len(tls_names)
    time_encoder = SinusoidalPositionalEncoding(_cfg.model.time_embed_dim).to(device)
    encoders = [
        AgentHistoryEncoder(obs_dim, _cfg.model.time_embed_dim, _cfg.model.hidden_dim).to(
            device
        )
        for _ in range(num_agents)
    ]
    actors = [
        MacroActor(_cfg.model.hidden_dim, action_dim, min_action=0.0, max_action=1.0).to(
            device
        )
        for _ in range(num_agents)
    ]
    critic = CentralizedCritic(_cfg.model.hidden_dim, _cfg.model.num_heads).to(device)

    agents_buffer = AsyncTrajectoryBuffer(
        capacity=_cfg.training.buffer_size, num_agents=num_agents
    )
    critic_buffer = SyncTrajectoryBuffer(capacity=_cfg.training.buffer_size)

    all_params = list(critic.parameters())
    for actor in actors:
        all_params += list(actor.parameters())
    for encoder in encoders:
        all_params += list(encoder.parameters())

    opt = torch.optim.Adam(all_params, lr=_cfg.training.actor_lr)

    trainer = ACACTrainer(
        actors=actors,
        encoders=encoders,
        critic=critic,
        agents_buffer=agents_buffer,
        critic_buffer=critic_buffer,
        optimizers={"combined": opt},
        time_encoder=time_encoder,
        tls_names=tls_names,
        device=device,
        gamma=_cfg.training.gamma,
        lam=_cfg.training.lam,
        eps_clip=_cfg.training.eps_clip,
    )
    return trainer


def _normalize_saturation(sat_raw: float, sat_clip_max: float) -> float:
    if sat_clip_max <= 0:
        return 0.0
    sat_clipped = min(max(float(sat_raw), 0.0), sat_clip_max)
    return sat_clipped / sat_clip_max


@torch.no_grad()
def run_acac_episode(env, trainer, max_steps, sat_clip_max):
    obs, reward, done, info = env.reset()
    trainer._reset_hidden()
    t = 0
    step_count = 0

    kpis = {
        "delay": [],
        "queue": [],
        "saturation_raw": [],
        "saturation_norm": [],
    }

    while not done and step_count < max_steps:
        requiring = info.get("intersection_require_action", [])
        action_dict = {}

        for name in requiring:
            i = trainer.tls_index[name]
            z_it = trainer._obs_to_tensor(obs, i).unsqueeze(0)
            eff_range = info.get("effective_action_range", {}).get(
                name, (0.0, info["max_green"] - info["min_green"])
            )
            z_it = torch.cat(
                [
                    z_it,
                    torch.tensor([eff_range], dtype=torch.float32).to(trainer.device),
                ],
                dim=-1,
            )
            p_it = trainer.time_encoder(t).to(trainer.device).unsqueeze(0)
            h_prev = trainer.hidden_states[i].unsqueeze(0)
            trainer.hidden_states[i] = trainer.encoders[i](z_it, p_it, h_prev).squeeze(0)
            actor_out, _ = trainer.actors[i].sample(trainer.hidden_states[i].unsqueeze(0))
            actor_val = float(actor_out.detach().item())
            action_dict[name] = trainer._scale_action(actor_val, eff_range[0], eff_range[1])

        next_obs, reward, done, info = env.step(action_dict)

        avg_delay = float(-reward[:, :, 0].mean())
        avg_queue = float(-reward[:, :, 1].mean())
        avg_sat_raw = float(-reward[:, :, 2].mean())
        avg_sat_norm = _normalize_saturation(avg_sat_raw, sat_clip_max)

        kpis["delay"].append(avg_delay)
        kpis["queue"].append(avg_queue)
        kpis["saturation_raw"].append(avg_sat_raw)
        kpis["saturation_norm"].append(avg_sat_norm)

        obs = next_obs
        t += info["delta_t"]
        step_count += 1

    return kpis


def run_fixed_time_episode(env, max_steps, sat_clip_max, fixed_extension=30):
    obs, reward, done, info = env.reset()
    step_count = 0

    kpis = {
        "delay": [],
        "queue": [],
        "saturation_raw": [],
        "saturation_norm": [],
    }

    while not done and step_count < max_steps:
        requiring = info.get("intersection_require_action", [])
        action_dict = {tls_id: fixed_extension for tls_id in requiring}

        next_obs, reward, done, info = env.step(action_dict)

        avg_delay = float(-reward[:, :, 0].mean())
        avg_queue = float(-reward[:, :, 1].mean())
        avg_sat_raw = float(-reward[:, :, 2].mean())
        avg_sat_norm = _normalize_saturation(avg_sat_raw, sat_clip_max)

        kpis["delay"].append(avg_delay)
        kpis["queue"].append(avg_queue)
        kpis["saturation_raw"].append(avg_sat_raw)
        kpis["saturation_norm"].append(avg_sat_norm)

        obs = next_obs
        step_count += 1

    return kpis


def _mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def summarize_kpis(kpis):
    return {
        "delay": _mean_or_nan(kpis["delay"]),
        "queue": _mean_or_nan(kpis["queue"]),
        "saturation_raw": _mean_or_nan(kpis["saturation_raw"]),
        "saturation_norm": _mean_or_nan(kpis["saturation_norm"]),
    }


def _improvement_pct(lower_is_better_rl: float, lower_is_better_fixed: float) -> float:
    if abs(lower_is_better_fixed) < 1e-12:
        return float("nan")
    return (lower_is_better_fixed - lower_is_better_rl) / abs(lower_is_better_fixed) * 100.0


def _build_comparison(scenario_name, acac_summary, fixed_summary):
    delay_improve = _improvement_pct(acac_summary["delay"], fixed_summary["delay"])
    queue_improve = _improvement_pct(acac_summary["queue"], fixed_summary["queue"])
    satn_improve = _improvement_pct(
        acac_summary["saturation_norm"], fixed_summary["saturation_norm"]
    )

    avg_improve = float(np.mean([delay_improve, queue_improve, satn_improve]))

    return {
        "scenario": scenario_name,
        "rl_delay": acac_summary["delay"],
        "fixed_delay": fixed_summary["delay"],
        "improve_delay_pct": delay_improve,
        "rl_queue": acac_summary["queue"],
        "fixed_queue": fixed_summary["queue"],
        "improve_queue_pct": queue_improve,
        "rl_sat_raw": acac_summary["saturation_raw"],
        "fixed_sat_raw": fixed_summary["saturation_raw"],
        "rl_sat_norm": acac_summary["saturation_norm"],
        "fixed_sat_norm": fixed_summary["saturation_norm"],
        "improve_sat_norm_pct": satn_improve,
        "avg_3kpi_improve_pct": avg_improve,
    }


def plot_comparison(
    acac_kpis,
    fixed_kpis,
    scenario_name,
    output_dir,
    total_steps_count,
    sat_clip_max,
):
    acac_summary = summarize_kpis(acac_kpis)
    fixed_summary = summarize_kpis(fixed_kpis)
    comparison = _build_comparison(scenario_name, acac_summary, fixed_summary)

    metric_specs = [
        ("Wait Time", "delay", "Average Wait Time (s)"),
        ("Queue Length", "queue", "Average Queue Length (m)"),
        (
            f"Saturation (Norm, clip={sat_clip_max:g})",
            "saturation_norm",
            "Average Saturation (0-1)",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#e57373", "#45b06c"]

    for ax, (title, key, ylabel) in zip(axes, metric_specs):
        fixed_val = fixed_summary[key]
        rl_val = acac_summary[key]
        bars = ax.bar(["Fixed Time", "RL (ACAC)"], [fixed_val, rl_val], color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(
        f"RL vs Fixed-Time on 3 KPI ({scenario_name}, {total_steps_count}s)", y=1.03
    )
    fig.tight_layout()

    output_path = os.path.join(output_dir, f"{scenario_name}_compare_kpi3_satnorm.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(
        f"[{scenario_name}] Wait improvement: {comparison['improve_delay_pct']:.2f}% | "
        f"Queue improvement: {comparison['improve_queue_pct']:.2f}% | "
        f"Sat(norm) improvement: {comparison['improve_sat_norm_pct']:.2f}%"
    )
    print(
        f"[{scenario_name}] Avg 3-KPI improvement (equal weight): "
        f"{comparison['avg_3kpi_improve_pct']:.2f}%"
    )
    print(
        f"[{scenario_name}] Saturation raw mean (diagnostic): "
        f"RL={comparison['rl_sat_raw']:.2f}, Fixed={comparison['fixed_sat_raw']:.2f}"
    )
    print(f"Saved 3-KPI plot: {output_path}")

    return comparison


def plot_multi_scenario_3kpi(results, output_dir, total_steps_count, sat_clip_max):
    if not results:
        return None

    scenario_labels = [r["scenario"] for r in results]
    x = np.arange(len(scenario_labels))
    width = 0.35

    metric_specs = [
        ("Wait Time", "fixed_delay", "rl_delay", "Average Wait Time (s)"),
        ("Queue Length", "fixed_queue", "rl_queue", "Average Queue Length (m)"),
        (
            f"Saturation (Norm, clip={sat_clip_max:g})",
            "fixed_sat_norm",
            "rl_sat_norm",
            "Average Saturation (0-1)",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (title, fixed_key, rl_key, ylabel) in zip(axes, metric_specs):
        fixed_vals = [r[fixed_key] for r in results]
        rl_vals = [r[rl_key] for r in results]

        bars_fixed = ax.bar(x - width / 2, fixed_vals, width, label="Fixed Time", color="#e57373")
        bars_rl = ax.bar(x + width / 2, rl_vals, width, label="RL (ACAC)", color="#45b06c")

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        for bars in (bars_fixed, bars_rl):
            for b in bars:
                h = b.get_height()
                ax.annotate(
                    f"{h:.2f}",
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    axes[0].legend(loc="upper left")
    fig.suptitle(
        f"RL vs Fixed-Time on 3 KPI ({' & '.join(scenario_labels)}, {total_steps_count}s)",
        y=1.02,
    )
    fig.tight_layout()

    safe_name = "_".join(scenario_labels)
    output_path = os.path.join(
        output_dir, f"kpi3_compare_{safe_name}_rl_vs_fixed_satnorm.png"
    )
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multi-scenario plot: {output_path}")
    return output_path


def save_summary_csv(results, output_dir):
    if not results:
        return None

    output_path = Path(output_dir) / "kpi3_satnorm_summary.csv"
    fieldnames = [
        "scenario",
        "fixed_delay",
        "rl_delay",
        "improve_delay_pct",
        "fixed_queue",
        "rl_queue",
        "improve_queue_pct",
        "fixed_sat_norm",
        "rl_sat_norm",
        "improve_sat_norm_pct",
        "avg_3kpi_improve_pct",
        "fixed_sat_raw",
        "rl_sat_raw",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"Saved summary CSV: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=int, default=600, help="Max steps to simulate for comparison"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--sat-clip-max",
        type=float,
        default=2.0,
        help="Clip saturation at this value, then normalize to [0,1] for compare",
    )
    parser.add_argument(
        "--fixed-extension",
        type=float,
        default=30.0,
        help="Fixed extension seconds used by fixed-time baseline",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="",
        help="Comma-separated scenario names to run. Empty means all.",
    )
    args = parser.parse_args()

    output_dir = Path("checkpoints/compare_kpi")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_data_dir = Path(__file__).resolve().parents[1] / "SimulationData" / "Evaluate"

    scenarios = [
        (
            "normal_2intersection",
            base_data_dir / "Normal/2Intersection/config.sumocfg",
            "checkpoints/normal_2intersection/normal_2intersection_checkpoint.pt",
        ),
        (
            "normal_3intersection",
            base_data_dir / "Normal/3Intersection/config.sumocfg",
            "checkpoints/normal_3intersection/normal_3intersection_checkpoint.pt",
        ),
        (
            "normal_4intersection",
            base_data_dir / "Normal/4Intersection/config.sumocfg",
            "checkpoints/normal_4intersection/normal_4intersection_checkpoint.pt",
        ),
        (
            "crowded_2intersection",
            base_data_dir / "Crowded/2Intersection/config.sumocfg",
            "checkpoints/crowded_2intersection/crowded_2intersection_checkpoint.pt",
        ),
        (
            "crowded_3intersection",
            base_data_dir / "Crowded/3Intersection/config.sumocfg",
            "checkpoints/crowded_3intersection/crowded_3intersection_checkpoint.pt",
        ),
        (
            "crowded_4intersection",
            base_data_dir / "Crowded/4Intersection/config.sumocfg",
            "checkpoints/crowded_4intersection/crowded_4intersection_checkpoint.pt",
        ),
    ]

    selected = None
    if args.scenarios.strip():
        selected = {x.strip() for x in args.scenarios.split(",") if x.strip()}
        scenarios = [s for s in scenarios if s[0] in selected]

    if not scenarios:
        print("No scenarios selected. Exiting.")
        return

    results = []

    for scenario_name, sumocfg_path, checkpoint_path in scenarios:
        print("\n=========================================")
        print(f"Evaluating Scenario: {scenario_name}")
        print("=========================================")

        sumocfg_path = str(sumocfg_path)
        checkpoint_path = str(checkpoint_path)

        if not Path(checkpoint_path).exists():
            print(f"[Warning] Checkpoint missing: {checkpoint_path}. Skipping...")
            continue

        env = TrafficEnvironment(sumocfg_path=sumocfg_path, use_gui=False)
        obs, _, _, _ = env.reset()
        obs_dim = obs.shape[1] * obs.shape[2]
        obs_encoder_dim = obs_dim + 2
        tls_names = env.tls_ids
        env.close()

        trainer = initialize_acac(
            obs_dim=obs_encoder_dim,
            action_dim=1,
            min_action=0.0,
            max_action=1.0,
            tls_names=tls_names,
            device=args.device,
        )
        trainer.load_model(checkpoint_path)
        for actor in trainer.actors:
            actor.eval()
        for encoder in trainer.encoders:
            encoder.eval()
        trainer.critic.eval()

        print("Running ACAC Evaluation...")
        env = TrafficEnvironment(sumocfg_path=sumocfg_path, use_gui=False)
        acac_kpis = run_acac_episode(env, trainer, args.steps, sat_clip_max=args.sat_clip_max)
        env.close()

        print("Running Fixed Time Baseline Evaluation...")
        env = TrafficEnvironment(sumocfg_path=sumocfg_path, use_gui=False)
        fixed_kpis = run_fixed_time_episode(
            env,
            args.steps,
            sat_clip_max=args.sat_clip_max,
            fixed_extension=args.fixed_extension,
        )
        env.close()

        comp = plot_comparison(
            acac_kpis,
            fixed_kpis,
            scenario_name,
            output_dir,
            args.steps,
            sat_clip_max=args.sat_clip_max,
        )
        results.append(comp)

    if not results:
        print("No scenario completed.")
        return

    save_summary_csv(results, output_dir)
    plot_multi_scenario_3kpi(results, output_dir, args.steps, args.sat_clip_max)

    avg_delay = float(np.mean([r["improve_delay_pct"] for r in results]))
    avg_queue = float(np.mean([r["improve_queue_pct"] for r in results]))
    avg_sat_norm = float(np.mean([r["improve_sat_norm_pct"] for r in results]))
    avg_3kpi = float(np.mean([r["avg_3kpi_improve_pct"] for r in results]))

    print("\n============= Overall Improvement (RL vs Fixed) =============")
    print(f"Avg Wait-Time improvement      : {avg_delay:.2f}%")
    print(f"Avg Queue-Length improvement   : {avg_queue:.2f}%")
    print(f"Avg Saturation(norm) improvement: {avg_sat_norm:.2f}%")
    print(f"Avg 3-KPI improvement (equal)  : {avg_3kpi:.2f}%")
    print("============================================================")


if __name__ == "__main__":
    main()
