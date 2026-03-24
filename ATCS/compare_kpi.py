"""Compare RL (ACAC) vs Fixed-Time on network, node, lane, and fairness KPIs."""

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
    load_scenario_config,
)
from atcs.environment import TrafficEnvironment

_cfg = load_model_config()
_EPS = 1e-8


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


def _mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def _std_or_zero(values):
    return float(np.std(values)) if values else 0.0


def _spread_or_zero(values):
    return float(max(values) - min(values)) if values else 0.0


def _spread_ratio(values):
    if not values:
        return 0.0
    mean_val = float(np.mean(values))
    return float(_spread_or_zero(values) / max(abs(mean_val), _EPS))


def _jain_service_fairness(lower_is_better_values):
    if not lower_is_better_values:
        return float("nan")
    service = np.array(
        [1.0 / (1.0 + max(float(v), 0.0)) for v in lower_is_better_values],
        dtype=np.float64,
    )
    denom = float(np.square(service).sum())
    if denom <= _EPS:
        return float("nan")
    return float((service.sum() ** 2) / (len(service) * denom))


def _improvement_pct(lower_is_better_rl: float, lower_is_better_fixed: float) -> float:
    if abs(lower_is_better_fixed) < 1e-12:
        return float("nan")
    return (lower_is_better_fixed - lower_is_better_rl) / abs(lower_is_better_fixed) * 100.0


def _empty_episode_stats(env):
    return {
        "network": {
            "delay": [],
            "queue": [],
            "saturation_raw": [],
            "saturation_norm": [],
        },
        "node": {
            tls_id: {
                "delay": [],
                "queue": [],
                "saturation_raw": [],
                "saturation_norm": [],
            }
            for tls_id in env.tls_ids
        },
        "lane": {
            (tls_id, lane_id): {
                "delay": [],
                "queue": [],
                "saturation_raw": [],
                "saturation_norm": [],
            }
            for tls_id in env.tls_ids
            for lane_id in env.lanes_by_tls.get(tls_id, [])
        },
    }


def _record_snapshot(stats, env, obs, sat_clip_max):
    lane_delays = []
    lane_queues = []
    lane_sat_raws = []
    lane_sat_norms = []

    for tls_index, tls_id in enumerate(env.tls_ids):
        tls_delay = []
        tls_queue = []
        tls_sat_raw = []
        tls_sat_norm = []

        for lane_index, lane_id in enumerate(env.lanes_by_tls.get(tls_id, [])):
            delay = float(obs[tls_index, lane_index, 0])
            sat_raw = float(obs[tls_index, lane_index, 1])
            queue = float(obs[tls_index, lane_index, 2])
            sat_norm = _normalize_saturation(sat_raw, sat_clip_max)

            stats["lane"][(tls_id, lane_id)]["delay"].append(delay)
            stats["lane"][(tls_id, lane_id)]["queue"].append(queue)
            stats["lane"][(tls_id, lane_id)]["saturation_raw"].append(sat_raw)
            stats["lane"][(tls_id, lane_id)]["saturation_norm"].append(sat_norm)

            tls_delay.append(delay)
            tls_queue.append(queue)
            tls_sat_raw.append(sat_raw)
            tls_sat_norm.append(sat_norm)

            lane_delays.append(delay)
            lane_queues.append(queue)
            lane_sat_raws.append(sat_raw)
            lane_sat_norms.append(sat_norm)

        stats["node"][tls_id]["delay"].append(_mean_or_nan(tls_delay))
        stats["node"][tls_id]["queue"].append(_mean_or_nan(tls_queue))
        stats["node"][tls_id]["saturation_raw"].append(_mean_or_nan(tls_sat_raw))
        stats["node"][tls_id]["saturation_norm"].append(_mean_or_nan(tls_sat_norm))

    stats["network"]["delay"].append(_mean_or_nan(lane_delays))
    stats["network"]["queue"].append(_mean_or_nan(lane_queues))
    stats["network"]["saturation_raw"].append(_mean_or_nan(lane_sat_raws))
    stats["network"]["saturation_norm"].append(_mean_or_nan(lane_sat_norms))


def _lane_rows_from_stats(stats, scenario_name, controller_name):
    rows = []
    for (tls_id, lane_id), metric_map in sorted(stats["lane"].items()):
        rows.append(
            {
                "scenario": scenario_name,
                "controller": controller_name,
                "tls_id": tls_id,
                "lane_id": lane_id,
                "delay": _mean_or_nan(metric_map["delay"]),
                "queue": _mean_or_nan(metric_map["queue"]),
                "saturation_raw": _mean_or_nan(metric_map["saturation_raw"]),
                "saturation_norm": _mean_or_nan(metric_map["saturation_norm"]),
            }
        )
    return rows


def _node_rows_from_lane_rows(lane_rows, scenario_name, controller_name):
    grouped = {}
    for row in lane_rows:
        grouped.setdefault(row["tls_id"], []).append(row)

    rows = []
    for tls_id, entries in sorted(grouped.items()):
        delay_vals = [float(r["delay"]) for r in entries]
        queue_vals = [float(r["queue"]) for r in entries]
        sat_vals = [float(r["saturation_norm"]) for r in entries]

        delay_spread_ratio = _spread_ratio(delay_vals)
        queue_spread_ratio = _spread_ratio(queue_vals)
        sat_spread_ratio = _spread_ratio(sat_vals)

        rows.append(
            {
                "scenario": scenario_name,
                "controller": controller_name,
                "tls_id": tls_id,
                "lane_count": len(entries),
                "delay_mean": _mean_or_nan(delay_vals),
                "queue_mean": _mean_or_nan(queue_vals),
                "saturation_norm_mean": _mean_or_nan(sat_vals),
                "delay_spread": _spread_or_zero(delay_vals),
                "queue_spread": _spread_or_zero(queue_vals),
                "saturation_norm_spread": _spread_or_zero(sat_vals),
                "delay_std": _std_or_zero(delay_vals),
                "queue_std": _std_or_zero(queue_vals),
                "saturation_norm_std": _std_or_zero(sat_vals),
                "delay_spread_ratio": delay_spread_ratio,
                "queue_spread_ratio": queue_spread_ratio,
                "saturation_norm_spread_ratio": sat_spread_ratio,
                "delay_jain_fairness": _jain_service_fairness(delay_vals),
                "queue_jain_fairness": _jain_service_fairness(queue_vals),
                "saturation_norm_jain_fairness": _jain_service_fairness(sat_vals),
                "directional_imbalance": float(
                    np.mean([delay_spread_ratio, queue_spread_ratio, sat_spread_ratio])
                ),
            }
        )
    return rows


def _summarize_episode(stats, scenario_name, controller_name):
    lane_rows = _lane_rows_from_stats(stats, scenario_name, controller_name)
    node_rows = _node_rows_from_lane_rows(lane_rows, scenario_name, controller_name)

    delay_vals = [float(r["delay"]) for r in lane_rows]
    queue_vals = [float(r["queue"]) for r in lane_rows]
    sat_raw_vals = [float(r["saturation_raw"]) for r in lane_rows]
    sat_norm_vals = [float(r["saturation_norm"]) for r in lane_rows]

    delay_spread_ratio = _spread_ratio(delay_vals)
    queue_spread_ratio = _spread_ratio(queue_vals)
    sat_spread_ratio = _spread_ratio(sat_norm_vals)

    network_summary = {
        "scenario": scenario_name,
        "controller": controller_name,
        "delay": _mean_or_nan(stats["network"]["delay"]),
        "queue": _mean_or_nan(stats["network"]["queue"]),
        "saturation_raw": _mean_or_nan(stats["network"]["saturation_raw"]),
        "saturation_norm": _mean_or_nan(stats["network"]["saturation_norm"]),
        "delay_spread": _spread_or_zero(delay_vals),
        "queue_spread": _spread_or_zero(queue_vals),
        "saturation_norm_spread": _spread_or_zero(sat_norm_vals),
        "delay_std": _std_or_zero(delay_vals),
        "queue_std": _std_or_zero(queue_vals),
        "saturation_norm_std": _std_or_zero(sat_norm_vals),
        "delay_spread_ratio": delay_spread_ratio,
        "queue_spread_ratio": queue_spread_ratio,
        "saturation_norm_spread_ratio": sat_spread_ratio,
        "delay_jain_fairness": _jain_service_fairness(delay_vals),
        "queue_jain_fairness": _jain_service_fairness(queue_vals),
        "saturation_norm_jain_fairness": _jain_service_fairness(sat_norm_vals),
        "directional_imbalance": float(
            np.mean([delay_spread_ratio, queue_spread_ratio, sat_spread_ratio])
        ),
    }

    return {
        "summary": network_summary,
        "node_rows": node_rows,
        "lane_rows": lane_rows,
    }


@torch.no_grad()
def run_acac_episode(env, trainer, max_steps, sat_clip_max):
    obs, _, done, info = env.reset()
    trainer._reset_hidden()
    t = 0
    step_count = 0
    stats = _empty_episode_stats(env)

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
            actor_out = trainer.actors[i].act(
                trainer.hidden_states[i].unsqueeze(0), deterministic=True
            )
            actor_val = float(actor_out.detach().item())
            action_dict[name] = trainer._scale_action(actor_val, eff_range[0], eff_range[1])

        next_obs, _, done, info = env.step(action_dict)
        _record_snapshot(stats, env, next_obs, sat_clip_max)

        obs = next_obs
        t += info["delta_t"]
        step_count += 1

    return stats


def run_fixed_time_episode(env, max_steps, sat_clip_max, fixed_extension=30):
    obs, _, done, info = env.reset()
    step_count = 0
    stats = _empty_episode_stats(env)

    while not done and step_count < max_steps:
        requiring = info.get("intersection_require_action", [])
        action_dict = {tls_id: fixed_extension for tls_id in requiring}

        next_obs, _, done, info = env.step(action_dict)
        _record_snapshot(stats, env, next_obs, sat_clip_max)

        obs = next_obs
        step_count += 1

    return stats


def _build_comparison(scenario_name, acac_summary, fixed_summary):
    delay_improve = _improvement_pct(acac_summary["delay"], fixed_summary["delay"])
    queue_improve = _improvement_pct(acac_summary["queue"], fixed_summary["queue"])
    satn_improve = _improvement_pct(
        acac_summary["saturation_norm"], fixed_summary["saturation_norm"]
    )
    fairness_improve = _improvement_pct(
        acac_summary["directional_imbalance"], fixed_summary["directional_imbalance"]
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
        "rl_directional_imbalance": acac_summary["directional_imbalance"],
        "fixed_directional_imbalance": fixed_summary["directional_imbalance"],
        "improve_directional_imbalance_pct": fairness_improve,
        "avg_3kpi_improve_pct": avg_improve,
    }


def plot_comparison(
    acac_summary,
    fixed_summary,
    scenario_name,
    output_dir,
    total_steps_count,
    sat_clip_max,
):
    comparison = _build_comparison(scenario_name, acac_summary, fixed_summary)

    metric_specs = [
        ("Control Delay", "delay", "Average Control Delay (s)"),
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
        f"[{scenario_name}] Control-delay improvement: {comparison['improve_delay_pct']:.2f}% | "
        f"Queue improvement: {comparison['improve_queue_pct']:.2f}% | "
        f"Sat(norm) improvement: {comparison['improve_sat_norm_pct']:.2f}% | "
        f"Directional fairness improvement: "
        f"{comparison['improve_directional_imbalance_pct']:.2f}%"
    )
    print(
        f"[{scenario_name}] Avg 3-KPI improvement (equal weight): "
        f"{comparison['avg_3kpi_improve_pct']:.2f}%"
    )
    print(
        f"[{scenario_name}] Saturation raw mean (diagnostic): "
        f"RL={comparison['rl_sat_raw']:.2f}, Fixed={comparison['fixed_sat_raw']:.2f}"
    )
    print(
        f"[{scenario_name}] Directional imbalance: "
        f"RL={comparison['rl_directional_imbalance']:.4f}, "
        f"Fixed={comparison['fixed_directional_imbalance']:.4f}"
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
        ("Control Delay", "fixed_delay", "rl_delay", "Average Control Delay (s)"),
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
        "fixed_directional_imbalance",
        "rl_directional_imbalance",
        "improve_directional_imbalance_pct",
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


def _write_csv(rows, output_path, fieldnames):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Saved CSV: {output_path}")
    return str(output_path)


def _compare_row_sets(rl_rows, fixed_rows, key_fields, metric_fields):
    rl_map = {tuple(row[k] for k in key_fields): row for row in rl_rows}
    fixed_map = {tuple(row[k] for k in key_fields): row for row in fixed_rows}
    combined_keys = sorted(set(rl_map.keys()) | set(fixed_map.keys()))

    out_rows = []
    for key in combined_keys:
        base = {field: key[idx] for idx, field in enumerate(key_fields)}
        rl_row = rl_map.get(key, {})
        fixed_row = fixed_map.get(key, {})

        for field in metric_fields:
            rl_val = rl_row.get(field, float("nan"))
            fixed_val = fixed_row.get(field, float("nan"))
            base[f"rl_{field}"] = rl_val
            base[f"fixed_{field}"] = fixed_val
            if isinstance(rl_val, (int, float)) and isinstance(fixed_val, (int, float)):
                base[f"improve_{field}_pct"] = _improvement_pct(float(rl_val), float(fixed_val))
            else:
                base[f"improve_{field}_pct"] = float("nan")

        out_rows.append(base)

    return out_rows


def save_detailed_comparison_csvs(scenario_name, acac_data, fixed_data, output_dir):
    node_metric_fields = [
        "delay_mean",
        "queue_mean",
        "saturation_norm_mean",
        "delay_spread",
        "queue_spread",
        "saturation_norm_spread",
        "delay_std",
        "queue_std",
        "saturation_norm_std",
        "delay_spread_ratio",
        "queue_spread_ratio",
        "saturation_norm_spread_ratio",
        "delay_jain_fairness",
        "queue_jain_fairness",
        "saturation_norm_jain_fairness",
        "directional_imbalance",
    ]
    lane_metric_fields = [
        "delay",
        "queue",
        "saturation_raw",
        "saturation_norm",
    ]

    node_compare_rows = _compare_row_sets(
        acac_data["node_rows"],
        fixed_data["node_rows"],
        key_fields=["scenario", "tls_id"],
        metric_fields=node_metric_fields,
    )
    rl_node_map = {
        (row["scenario"], row["tls_id"]): row for row in acac_data["node_rows"]
    }
    fixed_node_map = {
        (row["scenario"], row["tls_id"]): row for row in fixed_data["node_rows"]
    }
    for row in node_compare_rows:
        key = (row["scenario"], row["tls_id"])
        row["lane_count"] = rl_node_map.get(key, fixed_node_map.get(key, {})).get(
            "lane_count", ""
        )
    lane_compare_rows = _compare_row_sets(
        acac_data["lane_rows"],
        fixed_data["lane_rows"],
        key_fields=["scenario", "tls_id", "lane_id"],
        metric_fields=lane_metric_fields,
    )

    node_fieldnames = ["scenario", "tls_id", "lane_count"]
    for field in node_metric_fields:
        node_fieldnames.extend([f"fixed_{field}", f"rl_{field}", f"improve_{field}_pct"])

    lane_fieldnames = ["scenario", "tls_id", "lane_id"]
    for field in lane_metric_fields:
        lane_fieldnames.extend([f"fixed_{field}", f"rl_{field}", f"improve_{field}_pct"])

    scenario_dir = Path(output_dir) / scenario_name
    _write_csv(
        node_compare_rows,
        scenario_dir / f"{scenario_name}_node_comparison.csv",
        node_fieldnames,
    )
    _write_csv(
        lane_compare_rows,
        scenario_dir / f"{scenario_name}_lane_comparison.csv",
        lane_fieldnames,
    )


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
        (scenario.name, scenario.sumocfg_path, scenario.checkpoint_path)
        for scenario in load_scenario_config()
        if "compare_kpi" in scenario.tags
    ]

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
        acac_stats = run_acac_episode(env, trainer, args.steps, sat_clip_max=args.sat_clip_max)
        env.close()
        acac_data = _summarize_episode(acac_stats, scenario_name, "rl")

        print("Running Fixed Time Baseline Evaluation...")
        env = TrafficEnvironment(sumocfg_path=sumocfg_path, use_gui=False)
        fixed_stats = run_fixed_time_episode(
            env,
            args.steps,
            sat_clip_max=args.sat_clip_max,
            fixed_extension=args.fixed_extension,
        )
        env.close()
        fixed_data = _summarize_episode(fixed_stats, scenario_name, "fixed")

        comp = plot_comparison(
            acac_data["summary"],
            fixed_data["summary"],
            scenario_name,
            output_dir,
            args.steps,
            sat_clip_max=args.sat_clip_max,
        )
        results.append(comp)

        if scenario_name in {"normal_2intersection", "crowded_2intersection"}:
            save_detailed_comparison_csvs(
                scenario_name,
                acac_data,
                fixed_data,
                Path(output_dir) / "detailed",
            )

    if not results:
        print("No scenario completed.")
        return

    save_summary_csv(results, output_dir)
    plot_multi_scenario_3kpi(results, output_dir, args.steps, args.sat_clip_max)

    avg_delay = float(np.mean([r["improve_delay_pct"] for r in results]))
    avg_queue = float(np.mean([r["improve_queue_pct"] for r in results]))
    avg_sat_norm = float(np.mean([r["improve_sat_norm_pct"] for r in results]))
    avg_fairness = float(np.mean([r["improve_directional_imbalance_pct"] for r in results]))
    avg_3kpi = float(np.mean([r["avg_3kpi_improve_pct"] for r in results]))

    print("\n============= Overall Improvement (RL vs Fixed) =============")
    print(f"Avg Control-Delay improvement         : {avg_delay:.2f}%")
    print(f"Avg Queue-Length improvement      : {avg_queue:.2f}%")
    print(f"Avg Saturation(norm) improvement  : {avg_sat_norm:.2f}%")
    print(f"Avg Directional Fairness improve  : {avg_fairness:.2f}%")
    print(f"Avg 3-KPI improvement (equal)     : {avg_3kpi:.2f}%")
    print("============================================================")


if __name__ == "__main__":
    main()
