"""Export a sheet-friendly KPI table for Quang's report.

Current repo coverage:
- full few/normal/crowded support for 2Intersection
- other topology groups are skipped when fixed-time config or checkpoint is missing
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from plot_crowded_cycle_kpi import (
    CycleKPICollector,
    InstrumentedTrafficEnvironment,
    _aggregate_summary,
    _run_fixed,
    _run_rl,
    initialize_acac,
    load_kpi_config,
)


def _improvement_pct(ai_value: float, fixed_value: float) -> float | None:
    if abs(fixed_value) < 1e-12:
        return None
    return (fixed_value - ai_value) / abs(fixed_value) * 100.0


def _format_metric(metric_key: str, value: float | None) -> str:
    if value is None:
        return "N/A"
    if metric_key == "avg_wait_time_s":
        return f"{value:.2f}s"
    return f"{value:.4f}"


def _format_improvement(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def _scenario_family(scenario_name: str) -> str:
    parts = scenario_name.split("_", 1)
    return parts[1] if len(parts) > 1 else scenario_name


def _evaluate_scenario(
    *,
    scenario_name: str,
    sumocfg_path: str,
    checkpoint_path: str,
    seconds: int,
    device: str,
    fixed_extension: float,
):
    kpi_config = load_kpi_config()

    fixed_rows = _run_fixed(
        InstrumentedTrafficEnvironment(
            sumocfg_path=sumocfg_path,
            use_gui=False,
            max_episode_seconds=seconds,
            collector=CycleKPICollector(kpi_config.constants, "fixed_time_quang"),
        ),
        max_seconds=seconds,
        fixed_extension=fixed_extension,
    )

    shape_env = InstrumentedTrafficEnvironment(
        sumocfg_path=sumocfg_path,
        use_gui=False,
        max_episode_seconds=seconds,
    )
    obs, _, _, _ = shape_env.reset()
    obs_dim = obs.shape[1] * obs.shape[2]
    tls_names = shape_env.tls_ids
    shape_env.close()

    trainer = initialize_acac(
        obs_dim=obs_dim + 2,
        action_dim=1,
        tls_names=tls_names,
        device=device,
    )
    trainer.load_model(checkpoint_path)
    for actor in trainer.actors:
        actor.eval()
    for encoder in trainer.encoders:
        encoder.eval()
    trainer.critic.eval()

    rl_rows = _run_rl(
        InstrumentedTrafficEnvironment(
            sumocfg_path=sumocfg_path,
            use_gui=False,
            max_episode_seconds=seconds,
            collector=CycleKPICollector(kpi_config.constants, "ai_rl"),
        ),
        trainer=trainer,
        max_seconds=seconds,
    )

    return {
        "scenario": scenario_name,
        "fixed": _aggregate_summary(fixed_rows, "fixed_time_quang"),
        "ai": _aggregate_summary(rl_rows, "ai_rl"),
    }


def _write_long_summary(results, output_path: Path) -> None:
    fieldnames = [
        "scenario",
        "density",
        "family",
        "fixed_queue_length_m",
        "ai_queue_length_m",
        "improve_queue_pct",
        "fixed_avg_wait_time_s",
        "ai_avg_wait_time_s",
        "improve_avg_wait_pct",
        "fixed_saturation_x",
        "ai_saturation_x",
        "improve_saturation_pct",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            scenario_name = result["scenario"]
            density = scenario_name.split("_", 1)[0]
            family = _scenario_family(scenario_name)
            fixed = result["fixed"]
            ai = result["ai"]
            writer.writerow(
                {
                    "scenario": scenario_name,
                    "density": density,
                    "family": family,
                    "fixed_queue_length_m": fixed["queue_length_m"],
                    "ai_queue_length_m": ai["queue_length_m"],
                    "improve_queue_pct": _improvement_pct(
                        ai["queue_length_m"], fixed["queue_length_m"]
                    ),
                    "fixed_avg_wait_time_s": fixed["avg_wait_time_s"],
                    "ai_avg_wait_time_s": ai["avg_wait_time_s"],
                    "improve_avg_wait_pct": _improvement_pct(
                        ai["avg_wait_time_s"], fixed["avg_wait_time_s"]
                    ),
                    "fixed_saturation_x": fixed["saturation_x"],
                    "ai_saturation_x": ai["saturation_x"],
                    "improve_saturation_pct": _improvement_pct(
                        ai["saturation_x"], fixed["saturation_x"]
                    ),
                }
            )


def _build_sheet_rows(results):
    metrics = [
        ("queue length", "queue_length_m"),
        ("avg wait time", "avg_wait_time_s"),
        ("saturation", "saturation_x"),
    ]
    densities = ["few", "normal", "crowded"]

    family_map = {}
    for result in results:
        scenario_name = result["scenario"]
        density = scenario_name.split("_", 1)[0]
        family = _scenario_family(scenario_name)
        family_map.setdefault(family, {})[density] = result

    sheet_rows = []
    for family in sorted(family_map):
        density_results = family_map[family]
        for metric_label, metric_key in metrics:
            fixed_row = {"scenario_family": family, "metric": metric_label, "series": "fixed time (Quang)"}
            ai_row = {"scenario_family": family, "metric": metric_label, "series": "AI"}
            diff_row = {"scenario_family": family, "metric": metric_label, "series": "AI vs Fixed time"}
            for density in densities:
                result = density_results.get(density)
                if result is None:
                    fixed_row[density] = "N/A"
                    ai_row[density] = "N/A"
                    diff_row[density] = "N/A"
                    continue
                fixed_value = float(result["fixed"][metric_key])
                ai_value = float(result["ai"][metric_key])
                fixed_row[density] = _format_metric(metric_key, fixed_value)
                ai_row[density] = _format_metric(metric_key, ai_value)
                diff_row[density] = _format_improvement(
                    _improvement_pct(ai_value, fixed_value)
                )
            sheet_rows.extend([fixed_row, ai_row, diff_row])
    return sheet_rows


def _write_sheet_csv(rows, output_path: Path) -> None:
    fieldnames = ["scenario_family", "metric", "series", "few", "normal", "crowded"]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_sheet_table(rows, output_path: Path) -> None:
    fieldnames = ["scenario_family", "metric", "series", "few", "normal", "crowded"]
    cell_text = [[row.get(field, "") for field in fieldnames] for row in rows]

    fig_height = max(4, 0.36 * (len(rows) + 1))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=fieldnames,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)

    for row_index in range(len(rows) + 1):
        for col_index in range(len(fieldnames)):
            cell = table[(row_index, col_index)]
            if row_index == 0:
                cell.set_facecolor("#d9ead3")
                cell.set_text_props(weight="bold")
            elif rows[row_index - 1]["series"] == "AI vs Fixed time":
                cell.set_facecolor("#f4f4f4")

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=400)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fixed-extension", type=float, default=30.0)
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "checkpoints" / "quang_sheet_cycle_kpi_400s"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_specs = [
        {
            "scenario_name": "few_2intersection",
            "sumocfg_path": str(
                repo_root / "SimulationData" / "Evaluate" / "Few" / "2Intersection" / "config.sumocfg"
            ),
            "checkpoint_path": str(
                repo_root / "checkpoints" / "few_2intersection" / "few_2intersection_checkpoint.pt"
            ),
        },
        {
            "scenario_name": "normal_2intersection",
            "sumocfg_path": str(
                repo_root / "SimulationData" / "Evaluate" / "Normal" / "2Intersection" / "config.sumocfg"
            ),
            "checkpoint_path": str(
                repo_root / "checkpoints" / "normal_2intersection" / "normal_2intersection_checkpoint.pt"
            ),
        },
        {
            "scenario_name": "crowded_2intersection",
            "sumocfg_path": str(
                repo_root / "SimulationData" / "Evaluate" / "Crowded" / "2Intersection" / "config.sumocfg"
            ),
            "checkpoint_path": str(
                repo_root / "checkpoints" / "crowded_2intersection" / "crowded_2intersection_checkpoint.pt"
            ),
        },
    ]

    results = []
    for spec in scenario_specs:
        print(f"Evaluating {spec['scenario_name']} ...")
        results.append(
            _evaluate_scenario(
                scenario_name=spec["scenario_name"],
                sumocfg_path=spec["sumocfg_path"],
                checkpoint_path=spec["checkpoint_path"],
                seconds=args.seconds,
                device=args.device,
                fixed_extension=args.fixed_extension,
            )
        )

    long_summary_path = output_dir / "quang_sheet_cycle_kpi_long.csv"
    sheet_csv_path = output_dir / "quang_sheet_cycle_kpi_sheet.csv"
    sheet_png_path = output_dir / "quang_sheet_cycle_kpi_sheet.png"

    _write_long_summary(results, long_summary_path)
    sheet_rows = _build_sheet_rows(results)
    _write_sheet_csv(sheet_rows, sheet_csv_path)
    _render_sheet_table(sheet_rows, sheet_png_path)

    print(f"Saved long summary: {long_summary_path}")
    print(f"Saved sheet CSV: {sheet_csv_path}")
    print(f"Saved sheet image: {sheet_png_path}")


if __name__ == "__main__":
    main()
