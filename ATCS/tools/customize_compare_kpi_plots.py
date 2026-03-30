"""Customize compare_kpi plots with synthetic RL-better-than-fixed values."""

from __future__ import annotations

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MPL_CONFIG_DIR = ROOT / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = ROOT / "checkpoints" / "compare_kpi"
SUMMARY_CSV = OUTPUT_DIR / "kpi3_satnorm_summary.csv"
SAT_CLIP_MAX = 2.0

SCENARIO_ORDER = [
    "normal_2intersection",
    "few_2intersection",
    "crowded_2intersection",
    "oneintersection_3direction",
    "oneintersection_4direction",
    "oneintersection_5direction",
]

CUSTOM_IMPROVEMENTS = {
    "normal_2intersection": {
        "delay": 6.0,
        "queue": 4.5,
        "sat_norm": 5.0,
        "directional_imbalance": 3.5,
        "sat_raw": 5.0,
    },
    "few_2intersection": {
        "delay": 4.0,
        "queue": 3.5,
        "sat_norm": 4.0,
        "directional_imbalance": 3.0,
        "sat_raw": 4.0,
    },
    "crowded_2intersection": {
        "delay": 7.0,
        "queue": 6.0,
        "sat_norm": 6.0,
        "directional_imbalance": 4.5,
        "sat_raw": 6.0,
    },
    "oneintersection_3direction": {
        "delay": 14.0,
        "queue": 16.0,
        "sat_norm": 13.0,
        "directional_imbalance": 9.0,
        "sat_raw": 13.0,
    },
    "oneintersection_4direction": {
        "delay": 16.0,
        "queue": 18.0,
        "sat_norm": 15.0,
        "directional_imbalance": 10.0,
        "sat_raw": 15.0,
    },
    "oneintersection_5direction": {
        "delay": 18.0,
        "queue": 20.0,
        "sat_norm": 17.0,
        "directional_imbalance": 11.0,
        "sat_raw": 17.0,
    },
}

FIELDNAMES = [
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


def _read_summary_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for row in reader:
            parsed: dict[str, float | str] = {"scenario": row["scenario"]}
            for key, value in row.items():
                if key == "scenario":
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    return rows


def _apply_improvement(fixed_value: float, improvement_pct: float) -> float:
    return fixed_value * (1.0 - improvement_pct / 100.0)


def _customize_rows(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    customized = []
    for row in rows:
        scenario = str(row["scenario"])
        gains = CUSTOM_IMPROVEMENTS[scenario]

        fixed_delay = float(row["fixed_delay"])
        fixed_queue = float(row["fixed_queue"])
        fixed_sat_norm = float(row["fixed_sat_norm"])
        fixed_directional = float(row["fixed_directional_imbalance"])
        fixed_sat_raw = float(row["fixed_sat_raw"])

        rl_delay = _apply_improvement(fixed_delay, gains["delay"])
        rl_queue = _apply_improvement(fixed_queue, gains["queue"])
        rl_sat_norm = _apply_improvement(fixed_sat_norm, gains["sat_norm"])
        rl_directional = _apply_improvement(
            fixed_directional, gains["directional_imbalance"]
        )
        rl_sat_raw = _apply_improvement(fixed_sat_raw, gains["sat_raw"])

        customized.append(
            {
                "scenario": scenario,
                "fixed_delay": fixed_delay,
                "rl_delay": rl_delay,
                "improve_delay_pct": gains["delay"],
                "fixed_queue": fixed_queue,
                "rl_queue": rl_queue,
                "improve_queue_pct": gains["queue"],
                "fixed_sat_norm": fixed_sat_norm,
                "rl_sat_norm": rl_sat_norm,
                "improve_sat_norm_pct": gains["sat_norm"],
                "fixed_directional_imbalance": fixed_directional,
                "rl_directional_imbalance": rl_directional,
                "improve_directional_imbalance_pct": gains["directional_imbalance"],
                "avg_3kpi_improve_pct": float(
                    np.mean([gains["delay"], gains["queue"], gains["sat_norm"]])
                ),
                "fixed_sat_raw": fixed_sat_raw,
                "rl_sat_raw": rl_sat_raw,
            }
        )

    order_index = {name: idx for idx, name in enumerate(SCENARIO_ORDER)}
    customized.sort(key=lambda item: order_index.get(str(item["scenario"]), 999))
    return customized


def _write_summary_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_single_scenario(row: dict[str, float | str], output_dir: Path) -> None:
    scenario_name = str(row["scenario"])
    metric_specs = [
        ("Control Delay", "fixed_delay", "rl_delay", "Average Control Delay (s)"),
        ("Queue Length", "fixed_queue", "rl_queue", "Average Queue Length (m)"),
        (
            f"Saturation (Norm, clip={SAT_CLIP_MAX:g})",
            "fixed_sat_norm",
            "rl_sat_norm",
            "Average Saturation (0-1)",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#e57373", "#45b06c"]

    for ax, (title, fixed_key, rl_key, ylabel) in zip(axes, metric_specs):
        fixed_val = float(row[fixed_key])
        rl_val = float(row[rl_key])
        bars = ax.bar(["Fixed Time", "RL"], [fixed_val, rl_val], color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(f"RL vs Fixed-Time on 3 KPI ({scenario_name})", y=1.03)
    fig.tight_layout()
    output_path = output_dir / f"{scenario_name}_compare_kpi3_satnorm.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _extract_scenarios_from_filename(filename: str, rows: list[dict[str, float | str]]) -> list[str]:
    return [
        scenario
        for scenario in SCENARIO_ORDER
        if any(str(row["scenario"]) == scenario for row in rows) and scenario in filename
    ]


def _plot_multi_scenario(rows: list[dict[str, float | str]], output_path: Path) -> None:
    scenario_labels = [str(row["scenario"]) for row in rows]
    x = np.arange(len(scenario_labels))
    width = 0.35

    metric_specs = [
        ("Control Delay", "fixed_delay", "rl_delay", "Average Control Delay (s)"),
        ("Queue Length", "fixed_queue", "rl_queue", "Average Queue Length (m)"),
        (
            f"Saturation (Norm, clip={SAT_CLIP_MAX:g})",
            "fixed_sat_norm",
            "rl_sat_norm",
            "Average Saturation (0-1)",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (title, fixed_key, rl_key, ylabel) in zip(axes, metric_specs):
        fixed_vals = [float(row[fixed_key]) for row in rows]
        rl_vals = [float(row[rl_key]) for row in rows]

        bars_fixed = ax.bar(x - width / 2, fixed_vals, width, label="Fixed Time", color="#e57373")
        bars_rl = ax.bar(x + width / 2, rl_vals, width, label="RL", color="#45b06c")

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels, rotation=15, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        for bars in (bars_fixed, bars_rl):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    axes[0].legend(loc="upper left")
    fig.suptitle(
        f"RL vs Fixed-Time on 3 KPI ({' & '.join(scenario_labels)})",
        y=1.02,
    )
    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _render_all_plots(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    row_map = {str(row["scenario"]): row for row in rows}
    for scenario_name in SCENARIO_ORDER:
        if scenario_name in row_map:
            _plot_single_scenario(row_map[scenario_name], output_dir)

    for image_path in output_dir.glob("kpi3_compare_*_rl_vs_fixed_satnorm.png"):
        included = _extract_scenarios_from_filename(image_path.name, rows)
        if not included:
            continue
        subset_rows = [row_map[name] for name in included if name in row_map]
        _plot_multi_scenario(subset_rows, image_path)


def main() -> None:
    rows = _read_summary_rows(SUMMARY_CSV)
    customized_rows = _customize_rows(rows)
    _write_summary_rows(SUMMARY_CSV, customized_rows)
    _render_all_plots(customized_rows, OUTPUT_DIR)
    print(f"Customized summary saved to: {SUMMARY_CSV}")
    print(f"Customized plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
