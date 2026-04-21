"""Plot 9 bar charts from the latest manually entered comparison sheet."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict

_MPL_DIR = Path(__file__).resolve().parent / ".mplconfig_quang_9charts"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR.resolve()))

import matplotlib.pyplot as plt
import numpy as np


DIRECTIONS = ["Nga ba", "Nga tu", "Nga nam"]
DENSITIES = ["crowded", "normal", "few"]
METHODS = ["Fixed time", "Inductive", "Adaptive", "AI"]
METRICS = {
    "queue_length": ("Queue length", "vehicle"),
    "avg_wait_time": ("Avg wait time", "s"),
}

# Fixed time uses the "fixed time (cua Quang)" row in the new sheet.
DATA: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
    "Nga ba": {
        "crowded": {
            "queue_length": {"Fixed time": 8.3799, "Inductive": 8.2686, "Adaptive": 7.5867, "AI": 8.0136},
            "avg_wait_time": {"Fixed time": 29.37, "Inductive": 27.6556, "Adaptive": 26.4125, "AI": 26.702},
        },
        "normal": {
            "queue_length": {"Fixed time": 3.6716, "Inductive": 3.324222222, "Adaptive": 3.368888889, "AI": 3.1041},
            "avg_wait_time": {"Fixed time": 18.52, "Inductive": 16.59822222, "Adaptive": 18.02222222, "AI": 16.84},
        },
        "few": {
            "queue_length": {"Fixed time": 0.8865, "Inductive": 0.9134482759, "Adaptive": 1.138695652, "AI": 0.897},
            "avg_wait_time": {"Fixed time": 14.11, "Inductive": 12.56034483, "Adaptive": 14.66, "AI": 12.83},
        },
    },
    "Nga tu": {
        "crowded": {
            "queue_length": {"Fixed time": 4.0985, "Inductive": 4.884333333, "Adaptive": 4.0028, "AI": 3.5869},
            "avg_wait_time": {"Fixed time": 10.54, "Inductive": 11.08566667, "Adaptive": 9.8907, "AI": 10.25},
        },
        "normal": {
            "queue_length": {"Fixed time": 1.6427, "Inductive": 1.8762, "Adaptive": 1.7690, "AI": 1.1398},
            "avg_wait_time": {"Fixed time": 7.75, "Inductive": 9.230666667, "Adaptive": 9.4096, "AI": 6.53},
        },
        "few": {
            "queue_length": {"Fixed time": 0.1118, "Inductive": 0.6857, "Adaptive": 0.5893, "AI": 0.0946},
            "avg_wait_time": {"Fixed time": 6.36, "Inductive": 7.6551, "Adaptive": 6.8080, "AI": 6.99},
        },
    },
    "Nga nam": {
        "crowded": {
            "queue_length": {"Fixed time": 25.534, "Inductive": 24.6693, "Adaptive": 24.4618, "AI": 23.412},
            "avg_wait_time": {"Fixed time": 105.4, "Inductive": 109.7478, "Adaptive": 109.3127, "AI": 100.0},
        },
        "normal": {
            "queue_length": {"Fixed time": 7.7770, "Inductive": 7.6153, "Adaptive": 7.9367, "AI": 7.1461},
            "avg_wait_time": {"Fixed time": 55.8, "Inductive": 53.6867, "Adaptive": 51.1920, "AI": 50.7},
        },
        "few": {
            "queue_length": {"Fixed time": 1.7858, "Inductive": 1.8747, "Adaptive": 1.7555, "AI": 1.2530},
            "avg_wait_time": {"Fixed time": 39.7, "Inductive": 44.2812, "Adaptive": 36.8510, "AI": 33.67},
        },
    },
}


def _improvement_percent(baseline: float, ai_value: float) -> float:
    return (baseline - ai_value) / abs(baseline) * 100.0


def _slug(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("nga_ba", "3direction")
        .replace("nga_tu", "4direction")
        .replace("nga_nam", "5direction")
    )


def _plot_case(direction: str, density: str, output_path: Path) -> None:
    metric_keys = list(METRICS.keys())
    x = np.arange(len(metric_keys))
    width = 0.19
    colors = ["#5b8def", "#f2a65a", "#6abf69", "#d95f59"]
    metric_max = {
        metric: max(DATA[direction][density][metric][method] for method in METHODS)
        for metric in metric_keys
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for index, method in enumerate(METHODS):
        offsets = x + (index - 1.5) * width
        raw_values = [DATA[direction][density][metric][method] for metric in metric_keys]
        plot_values = [
            DATA[direction][density][metric][method] / metric_max[metric]
            if metric_max[metric] > 0.0
            else 0.0
            for metric in metric_keys
        ]
        bars = ax.bar(offsets, plot_values, width, label=method, color=colors[index])
        for bar, value in zip(bars, raw_values):
            ax.annotate(
                f"{value:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(f"{direction} - {density.upper()}")
    ax.set_xticks(x)
    ax.set_xticklabels([METRICS[metric][0] for metric in metric_keys])
    ax.set_ylabel("Normalized height within each metric")
    ax.set_ylim(0.0, 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_summary_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for direction in DIRECTIONS:
        for density in DENSITIES:
            for metric, (metric_label, unit) in METRICS.items():
                ai_value = DATA[direction][density][metric]["AI"]
                row = {
                    "direction": direction,
                    "density": density,
                    "metric": metric_label,
                    "unit": unit,
                    "ai_value": f"{ai_value:.6f}",
                }
                for baseline in ("Fixed time", "Inductive", "Adaptive"):
                    baseline_value = DATA[direction][density][metric][baseline]
                    row[f"{baseline}_value"] = f"{baseline_value:.6f}"
                    row[f"AI_vs_{baseline}_improvement_percent"] = f"{_improvement_percent(baseline_value, ai_value):.2f}"
                rows.append(row)
    return rows


def _write_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    fieldnames = [
        "direction",
        "density",
        "metric",
        "unit",
        "ai_value",
        "Fixed time_value",
        "AI_vs_Fixed time_improvement_percent",
        "Inductive_value",
        "AI_vs_Inductive_improvement_percent",
        "Adaptive_value",
        "AI_vs_Adaptive_improvement_percent",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_rollup(rows: list[dict[str, str]]) -> None:
    print("Average AI improvement, lower is better:")
    for baseline in ("Fixed time", "Inductive", "Adaptive"):
        values = [float(row[f"AI_vs_{baseline}_improvement_percent"]) for row in rows]
        print(f"- AI vs {baseline}: {np.mean(values):.2f}%")

    for metric_label, _ in METRICS.values():
        metric_rows = [row for row in rows if row["metric"] == metric_label]
        print(f"\n{metric_label}:")
        for baseline in ("Fixed time", "Inductive", "Adaptive"):
            values = [float(row[f"AI_vs_{baseline}_improvement_percent"]) for row in metric_rows]
            print(f"- AI vs {baseline}: {np.mean(values):.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "quang_sheet_9charts_new"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for direction in DIRECTIONS:
        for density in DENSITIES:
            output_path = output_dir / f"{_slug(direction)}_{density}_comparison.png"
            _plot_case(direction, density, output_path)
            print(f"Saved chart: {output_path}")

    rows = _build_summary_rows()
    summary_path = output_dir / "ai_improvement_summary.csv"
    _write_summary(rows, summary_path)
    print(f"Saved summary CSV: {summary_path}")
    _print_rollup(rows)


if __name__ == "__main__":
    main()
