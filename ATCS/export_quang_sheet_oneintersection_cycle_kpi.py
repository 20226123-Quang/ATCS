"""Export a Quang-style sheet for oneintersection 3/4/5 direction with few/normal/crowded columns.

The repository only provides topology-specific one-intersection scenarios:
- OneIntersection/3Direction
- OneIntersection/4Direction
- OneIntersection/5Direction

To build few/normal/crowded columns, this script derives density variants from each
base topology route by scaling vehicle departure times. The scaling ratios are inferred
from the available 2Intersection few/normal/crowded datasets within the same evaluation
window.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt

from export_quang_sheet_cycle_kpi import _evaluate_scenario, _improvement_pct

_DENSITY_ORDER = ("few", "normal", "crowded")


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


def _count_departures_in_window(route_path: Path, seconds: int) -> int:
    root = ET.parse(route_path).getroot()
    count = 0
    for element in root:
        if element.tag not in {"vehicle", "trip"}:
            continue
        depart_raw = element.attrib.get("depart")
        if depart_raw is None:
            continue
        try:
            depart = float(depart_raw)
        except ValueError:
            continue
        if depart <= float(seconds):
            count += 1
    return count


def _derive_density_factors(repo_root: Path, seconds: int) -> dict[str, float]:
    route_paths = {
        "few": repo_root / "SimulationData" / "Evaluate" / "Few" / "2Intersection" / "route.rou.xml",
        "normal": repo_root / "SimulationData" / "Evaluate" / "Normal" / "2Intersection" / "route.rou.xml",
        "crowded": repo_root / "SimulationData" / "Evaluate" / "Crowded" / "2Intersection" / "route.rou.xml",
    }
    counts = {density: _count_departures_in_window(path, seconds) for density, path in route_paths.items()}
    normal_count = max(counts.get("normal", 0), 1)
    return {
        "few": counts.get("few", 0) / normal_count,
        "normal": 1.0,
        "crowded": counts.get("crowded", 0) / normal_count,
    }


def _build_variant_route(
    *,
    source_route_path: Path,
    output_route_path: Path,
    density: str,
    demand_factor: float,
    evaluation_seconds: int,
) -> int:
    tree = ET.parse(source_route_path)
    root = tree.getroot()
    scale = max(float(demand_factor), 1e-9)
    within_window = 0

    for element in root:
        if element.tag not in {"vehicle", "trip"}:
            continue
        depart_raw = element.attrib.get("depart")
        if depart_raw is None:
            continue
        try:
            depart = float(depart_raw)
        except ValueError:
            continue
        scaled_depart = depart / scale
        element.set("depart", f"{scaled_depart:.2f}")
        if scaled_depart <= float(evaluation_seconds):
            within_window += 1

    try:
        ET.indent(tree, space="    ")
    except AttributeError:
        pass

    output_route_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_route_path, encoding="utf-8", xml_declaration=True)
    return within_window


def _prepare_density_variant(
    *,
    base_scenario_dir: Path,
    density: str,
    demand_factor: float,
    output_root: Path,
    evaluation_seconds: int,
) -> tuple[Path, int]:
    variant_dir = output_root / base_scenario_dir.name.lower() / density
    variant_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(base_scenario_dir / "config.sumocfg", variant_dir / "config.sumocfg")
    shutil.copy2(base_scenario_dir / "network.net.xml", variant_dir / "network.net.xml")

    route_count = _build_variant_route(
        source_route_path=base_scenario_dir / "route.rou.xml",
        output_route_path=variant_dir / "route.rou.xml",
        density=density,
        demand_factor=demand_factor,
        evaluation_seconds=evaluation_seconds,
    )
    return variant_dir / "config.sumocfg", route_count


def _write_density_metadata(rows, output_path: Path) -> None:
    fieldnames = ["family", "density", "demand_factor_vs_normal", "vehicles_within_window"]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_sheet_rows(results):
    metrics = [
        ("queue length", "queue_length_m"),
        ("avg wait time", "avg_wait_time_s"),
        ("saturation", "saturation_x"),
    ]
    families = ["oneintersection_3direction", "oneintersection_4direction", "oneintersection_5direction"]

    grouped: dict[str, dict[str, dict]] = {}
    for result in results:
        grouped.setdefault(result["family"], {})[result["density"]] = result

    sheet_rows = []
    for family in families:
        density_results = grouped.get(family, {})
        for metric_label, metric_key in metrics:
            fixed_row = {
                "scenario_family": family,
                "metric": metric_label,
                "series": "fixed time (Quang)",
            }
            ai_row = {
                "scenario_family": family,
                "metric": metric_label,
                "series": "AI",
            }
            diff_row = {
                "scenario_family": family,
                "metric": metric_label,
                "series": "AI vs Fixed time",
            }
            for density in _DENSITY_ORDER:
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
                diff_row[density] = _format_improvement(_improvement_pct(ai_value, fixed_value))
            sheet_rows.extend([fixed_row, ai_row, diff_row])
    return sheet_rows


def _write_sheet_csv(rows, output_path: Path) -> None:
    fieldnames = ["scenario_family", "metric", "series", "few", "normal", "crowded"]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_long_summary(results, output_path: Path) -> None:
    fieldnames = [
        "scenario_family",
        "density",
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
            fixed = result["fixed"]
            ai = result["ai"]
            writer.writerow(
                {
                    "scenario_family": result["family"],
                    "density": result["density"],
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


def _render_sheet_table(rows, output_path: Path) -> None:
    fieldnames = ["scenario_family", "metric", "series", "few", "normal", "crowded"]
    cell_text = [[row.get(field, "") for field in fieldnames] for row in rows]

    fig, ax = plt.subplots(figsize=(12.5, max(6, 0.38 * (len(rows) + 1))))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=fieldnames,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.22)

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
        default=str(repo_root / "checkpoints" / "quang_sheet_oneintersection_cycle_kpi_400s"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    topology_specs = [
        {
            "family": "oneintersection_3direction",
            "scenario_dir": repo_root / "SimulationData" / "Evaluate" / "OneIntersection" / "3Direction",
            "checkpoint_path": repo_root / "checkpoints" / "oneintersection_3direction" / "oneintersection_3direction_checkpoint.pt",
        },
        {
            "family": "oneintersection_4direction",
            "scenario_dir": repo_root / "SimulationData" / "Evaluate" / "OneIntersection" / "4Direction",
            "checkpoint_path": repo_root / "checkpoints" / "oneintersection_4direction" / "oneintersection_4direction_checkpoint.pt",
        },
        {
            "family": "oneintersection_5direction",
            "scenario_dir": repo_root / "SimulationData" / "Evaluate" / "OneIntersection" / "5Direction",
            "checkpoint_path": repo_root / "checkpoints" / "oneintersection_5direction" / "oneintersection_5direction_checkpoint.pt",
        },
    ]

    density_factors = _derive_density_factors(repo_root, args.seconds)
    generated_root = output_dir / "generated_density_variants"

    density_metadata = []
    results = []
    for topology in topology_specs:
        for density in _DENSITY_ORDER:
            factor = density_factors[density]
            variant_sumocfg, route_count = _prepare_density_variant(
                base_scenario_dir=topology["scenario_dir"],
                density=density,
                demand_factor=factor,
                output_root=generated_root,
                evaluation_seconds=args.seconds,
            )
            density_metadata.append(
                {
                    "family": topology["family"],
                    "density": density,
                    "demand_factor_vs_normal": factor,
                    "vehicles_within_window": route_count,
                }
            )
            scenario_name = f"{density}_{topology['family']}"
            print(f"Evaluating {scenario_name} ...")
            summary = _evaluate_scenario(
                scenario_name=scenario_name,
                sumocfg_path=str(variant_sumocfg),
                checkpoint_path=str(topology["checkpoint_path"]),
                seconds=args.seconds,
                device=args.device,
                fixed_extension=args.fixed_extension,
            )
            summary["family"] = topology["family"]
            summary["density"] = density
            results.append(summary)

    metadata_path = output_dir / "quang_sheet_oneintersection_density_metadata.csv"
    long_summary_path = output_dir / "quang_sheet_oneintersection_cycle_kpi_long.csv"
    sheet_csv_path = output_dir / "quang_sheet_oneintersection_cycle_kpi_sheet.csv"
    sheet_png_path = output_dir / "quang_sheet_oneintersection_cycle_kpi_sheet.png"

    _write_density_metadata(density_metadata, metadata_path)
    _write_long_summary(results, long_summary_path)
    sheet_rows = _build_sheet_rows(results)
    _write_sheet_csv(sheet_rows, sheet_csv_path)
    _render_sheet_table(sheet_rows, sheet_png_path)

    print(f"Saved density metadata: {metadata_path}")
    print(f"Saved long summary: {long_summary_path}")
    print(f"Saved sheet CSV: {sheet_csv_path}")
    print(f"Saved sheet image: {sheet_png_path}")


if __name__ == "__main__":
    main()
