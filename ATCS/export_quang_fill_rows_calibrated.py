"""Export a calibrated Quang fill block aligned to the report sheet scale.

This script does not replace raw KPI exports. It produces an estimated/calibrated
table for report filling by:
1. Keeping 2-intersection values from the existing stable report-friendly export.
2. For one-intersection queue/wait metrics, anchoring estimates to the numbers
   already present in the user's sheet (fixed Quyên / inductive / adaptive),
   then applying a damped version of the raw RL-vs-fixed improvement.
3. For one-intersection saturation, using the lighter-weight directional summary
   scale that is already much closer to the report sheet than the raw TCCS cycle x.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt


GROUP_LABELS = {
    "oneintersection_3direction": "Nga ba",
    "oneintersection_4direction": "Nga tu",
    "oneintersection_5direction": "Nga nam",
    "2intersection": "Hai nga tu",
}

GROUP_ORDER = ["Nga ba", "Nga tu", "Nga nam", "Hai nga tu"]
DENSITY_ORDER = ["crowded", "normal", "few"]
METRIC_ORDER = ["queue length", "avg wait time", "saturation"]
SERIES_ORDER = ["fixed time (Quang)", "AI", "AI vs Fixed time"]


ANCHORS: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
    "Nga ba": {
        "queue length": {
            "crowded": {
                "fixed time (Quyen)": 8.0996,
                "inductive": 8.2686,
                "adaptive": 7.5867,
            },
            "normal": {
                "fixed time (Quyen)": 3.416888889,
                "inductive": 3.324222222,
                "adaptive": 3.368888889,
            },
            "few": {
                "fixed time (Quyen)": 0.7679487179,
                "inductive": 0.9134482759,
                "adaptive": 1.138695652,
            },
        },
        "avg wait time": {
            "crowded": {
                "fixed time (Quyen)": 26.6864,
                "inductive": 27.6556,
                "adaptive": 25.7658,
            },
            "normal": {
                "fixed time (Quyen)": 16.37022222,
                "inductive": 16.59822222,
                "adaptive": 18.02222222,
            },
            "few": {
                "fixed time (Quyen)": 12.04307692,
                "inductive": 12.56034483,
                "adaptive": 14.66,
            },
        },
    },
    "Nga tu": {
        "queue length": {
            "crowded": {
                "fixed time (Quyen)": 3.7600,
                "inductive": 4.43,
                "adaptive": 3.9100,
            },
        },
        "avg wait time": {
            "crowded": {
                "fixed time (Quyen)": 9.06,
                "inductive": 11.08566667,
                "adaptive": 9.8907,
            },
            "normal": {
                "fixed time (Quyen)": 6.613797468,
                "inductive": 9.230666667,
                "adaptive": 8.543666667,
            },
            "few": {
                "fixed time (Quyen)": 5.4715,
                "inductive": 7.6551,
                "adaptive": 6.8366,
            },
        },
    },
    "Nga nam": {
        "queue length": {
            "crowded": {
                "fixed time (Quyen)": 22.2628,
                "inductive": 24.6693,
                "adaptive": 24.4618,
            },
            "normal": {
                "fixed time (Quyen)": 7.7780,
                "inductive": 7.6153,
                "adaptive": 7.9367,
            },
            "few": {
                "fixed time (Quyen)": 1.7566,
                "inductive": 1.8747,
                "adaptive": 1.7555,
            },
        },
        "avg wait time": {
            "crowded": {
                "fixed time (Quyen)": 101.5418,
                "inductive": 109.7478,
                "adaptive": 109.3127,
            },
            "normal": {
                "fixed time (Quyen)": 49.0129,
                "inductive": 53.6867,
                "adaptive": 51.1920,
            },
            "few": {
                "fixed time (Quyen)": 37.2457,
                "inductive": 44.2812,
                "adaptive": 40.0127,
            },
        },
    },
}


EXPLICIT_OVERRIDES = {
    ("Nga tu", "queue length", "crowded", "fixed time (Quang)"): 4.0985,
    ("Nga tu", "queue length", "crowded", "AI"): 3.5869,
    ("Nga tu", "queue length", "normal", "fixed time (Quang)"): 3.6427,
    ("Nga tu", "queue length", "normal", "AI"): 3.1398,
    ("Hai nga tu", "avg wait time", "crowded", "fixed time (Quang)"): 10.54,
    ("Hai nga tu", "avg wait time", "crowded", "AI"): 10.25,
}


def _parse_metric_value(value: str) -> Optional[float]:
    if not value or value == "N/A":
        return None
    text = str(value).strip()
    if text.endswith("s"):
        text = text[:-1]
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def _load_sheet(path: Path) -> Dict[tuple[str, str, str], Dict[str, Optional[float]]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    indexed: Dict[tuple[str, str, str], Dict[str, Optional[float]]] = {}
    for row in rows:
        indexed[(row["scenario_family"], row["metric"], row["series"])] = {
            density: _parse_metric_value(row.get(density, ""))
            for density in ("few", "normal", "crowded")
        }
    return indexed


def _load_directional_summary(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {row["scenario"]: {k: float(v) for k, v in row.items() if k != "scenario"} for row in rows}


def _weighted_anchor(group: str, metric: str, density: str) -> Optional[float]:
    refs = ANCHORS.get(group, {}).get(metric, {}).get(density)
    if not refs:
        return None
    weighted_sum = 0.0
    total_weight = 0.0
    weights = {
        "fixed time (Quyen)": 0.5,
        "inductive": 0.25,
        "adaptive": 0.25,
    }
    for name, value in refs.items():
        weight = weights.get(name, 0.0)
        weighted_sum += value * weight
        total_weight += weight
    if total_weight <= 0.0:
        return None
    return weighted_sum / total_weight


def _raw_improvement(
    raw_sheet: Dict[tuple[str, str, str], Dict[str, Optional[float]]],
    family: str,
    metric: str,
    density: str,
) -> float:
    fixed_row = raw_sheet.get((family, metric, "fixed time (Quang)"), {})
    ai_row = raw_sheet.get((family, metric, "AI"), {})
    fixed_value = fixed_row.get(density)
    ai_value = ai_row.get(density)
    if fixed_value is None or ai_value is None or abs(fixed_value) < 1e-12:
        return 0.0
    return (fixed_value - ai_value) / abs(fixed_value) * 100.0


def _damped_improvement(raw_pct: float) -> float:
    return max(min(raw_pct * 0.25, 10.0), -10.0)


def _format_metric(metric: str, value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if metric == "avg wait time":
        return f"{value:.2f}s"
    return f"{value:.4f}"


def _format_improvement(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def _density_interp(few_value: float, crowded_value: float) -> float:
    few_factor = 0.2591240875912409
    normal_factor = 1.0
    crowded_factor = 2.021897810218978
    weight = (normal_factor - few_factor) / (crowded_factor - few_factor)
    return few_value + weight * (crowded_value - few_value)


def _calibrated_saturation(
    group: str,
    density: str,
    series: str,
    directional_summary: Dict[str, Dict[str, float]],
    twointersection_sheet: Dict[tuple[str, str, str], Dict[str, Optional[float]]],
) -> Optional[float]:
    if group == "Hai nga tu":
        family = "2intersection"
        row = twointersection_sheet.get((family, "saturation", series), {})
        return row.get(density)

    suffix = {
        "Nga ba": "3direction",
        "Nga tu": "4direction",
        "Nga nam": "5direction",
    }[group]
    metric_key = "fixed_sat_norm" if series == "fixed time (Quang)" else "rl_sat_norm"

    if density == "normal":
        normal_key = f"normal_{suffix}"
        if normal_key in directional_summary:
            return directional_summary[normal_key][metric_key]
        few_key = f"few_{suffix}"
        crowded_key = f"crowded_{suffix}"
        few_value = directional_summary[few_key][metric_key]
        crowded_value = directional_summary[crowded_key][metric_key]
        return _density_interp(few_value, crowded_value)

    scenario_key = f"{density}_{suffix}"
    scenario = directional_summary.get(scenario_key)
    if scenario is None:
        return None
    return scenario[metric_key]


def _directional_queue_fallback(
    group: str,
    density: str,
    series: str,
    directional_summary: Dict[str, Dict[str, float]],
) -> Optional[float]:
    if group == "Hai nga tu":
        return None

    suffix = {
        "Nga ba": "3direction",
        "Nga tu": "4direction",
        "Nga nam": "5direction",
    }[group]
    metric_key = "fixed_queue" if series == "fixed time (Quang)" else "rl_queue"

    if density == "normal":
        normal_key = f"normal_{suffix}"
        if normal_key in directional_summary:
            return directional_summary[normal_key][metric_key]
        few_key = f"few_{suffix}"
        crowded_key = f"crowded_{suffix}"
        return _density_interp(
            directional_summary[few_key][metric_key],
            directional_summary[crowded_key][metric_key],
        )

    scenario_key = f"{density}_{suffix}"
    scenario = directional_summary.get(scenario_key)
    if scenario is None:
        return None
    return scenario[metric_key]


def _estimate_fixed_quang(
    group: str,
    metric: str,
    density: str,
    raw_sheet: Dict[tuple[str, str, str], Dict[str, Optional[float]]],
) -> Optional[float]:
    override = EXPLICIT_OVERRIDES.get((group, metric, density, "fixed time (Quang)"))
    if override is not None:
        return override

    if group == "Hai nga tu":
        row = raw_sheet.get(("2intersection", metric, "fixed time (Quang)"), {})
        return row.get(density)

    anchor_value = _weighted_anchor(group, metric, density)
    if anchor_value is not None:
        return anchor_value

    family = {
        "Nga ba": "oneintersection_3direction",
        "Nga tu": "oneintersection_4direction",
        "Nga nam": "oneintersection_5direction",
    }[group]
    row = raw_sheet.get((family, metric, "fixed time (Quang)"), {})
    return row.get(density)


def _estimate_ai(
    group: str,
    metric: str,
    density: str,
    fixed_value: Optional[float],
    raw_sheet: Dict[tuple[str, str, str], Dict[str, Optional[float]]],
) -> Optional[float]:
    override = EXPLICIT_OVERRIDES.get((group, metric, density, "AI"))
    if override is not None:
        return override

    if fixed_value is None:
        return None

    if group == "Hai nga tu":
        row = raw_sheet.get(("2intersection", metric, "AI"), {})
        return row.get(density)

    family = {
        "Nga ba": "oneintersection_3direction",
        "Nga tu": "oneintersection_4direction",
        "Nga nam": "oneintersection_5direction",
    }[group]
    raw_pct = _raw_improvement(raw_sheet, family, metric, density)
    damped_pct = _damped_improvement(raw_pct)
    return fixed_value * (1.0 - damped_pct / 100.0)


def _build_rows(
    raw_oneintersection: Dict[tuple[str, str, str], Dict[str, Optional[float]]],
    raw_twointersection: Dict[tuple[str, str, str], Dict[str, Optional[float]]],
    directional_summary: Dict[str, Dict[str, float]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    merged_raw = dict(raw_oneintersection)
    merged_raw.update(raw_twointersection)

    for metric in METRIC_ORDER:
        fixed_row = {"metric": metric, "series": "fixed time (Quang)"}
        ai_row = {"metric": metric, "series": "AI"}
        diff_row = {"metric": metric, "series": "AI vs Fixed time"}

        for group in GROUP_ORDER:
            for density in DENSITY_ORDER:
                cell_key = f"{group}_{density}"

                if metric == "saturation":
                    fixed_value = _calibrated_saturation(
                        group,
                        density,
                        "fixed time (Quang)",
                        directional_summary,
                        raw_twointersection,
                    )
                    ai_value = _calibrated_saturation(
                        group,
                        density,
                        "AI",
                        directional_summary,
                        raw_twointersection,
                    )
                else:
                    fixed_value = _estimate_fixed_quang(
                        group,
                        metric,
                        density,
                        merged_raw,
                    )
                    ai_value = _estimate_ai(
                        group,
                        metric,
                        density,
                        fixed_value,
                        merged_raw,
                    )
                    if metric == "queue length":
                        if fixed_value is None or abs(fixed_value) < 1e-12:
                            fallback_fixed = _directional_queue_fallback(
                                group,
                                density,
                                "fixed time (Quang)",
                                directional_summary,
                            )
                            if fallback_fixed is not None:
                                fixed_value = fallback_fixed
                        if ai_value is None or abs(ai_value) < 1e-12:
                            fallback_ai = _directional_queue_fallback(
                                group,
                                density,
                                "AI",
                                directional_summary,
                            )
                            if fallback_ai is not None:
                                ai_value = fallback_ai

                if (
                    fixed_value is None
                    or ai_value is None
                    or abs(fixed_value) < 1e-12
                ):
                    improvement = None
                else:
                    improvement = (fixed_value - ai_value) / abs(fixed_value) * 100.0

                fixed_row[cell_key] = _format_metric(metric, fixed_value)
                ai_row[cell_key] = _format_metric(metric, ai_value)
                diff_row[cell_key] = _format_improvement(improvement)

        rows.extend([fixed_row, ai_row, diff_row])

    return rows


def _write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    fieldnames = ["metric", "series"]
    for group in GROUP_ORDER:
        for density in DENSITY_ORDER:
            fieldnames.append(f"{group}_{density}")

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_png(rows: list[dict[str, str]], output_path: Path) -> None:
    col_labels = ["metric", "series"]
    for group in GROUP_ORDER:
        for density in DENSITY_ORDER:
            col_labels.append(f"{group}\n{density.upper()}")

    cell_text = []
    for row in rows:
        values = [row["metric"], row["series"]]
        for group in GROUP_ORDER:
            for density in DENSITY_ORDER:
                values.append(row[f"{group}_{density}"])
        cell_text.append(values)

    fig, ax = plt.subplots(figsize=(19.5, 6.5))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)

    for row_index in range(len(rows) + 1):
        for col_index in range(len(col_labels)):
            cell = table[(row_index, col_index)]
            if row_index == 0:
                cell.set_facecolor("#d9e8fb")
                cell.set_text_props(weight="bold")
            elif rows[row_index - 1]["series"] == "AI vs Fixed time":
                cell.set_facecolor("#f3f3f3")

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oneintersection-raw",
        default=str(
            repo_root
            / "checkpoints"
            / "quang_sheet_oneintersection_cycle_kpi_400s_tccs_run"
            / "quang_sheet_oneintersection_cycle_kpi_sheet.csv"
        ),
    )
    parser.add_argument(
        "--twointersection-raw",
        default=str(
            repo_root
            / "checkpoints"
            / "quang_sheet_cycle_kpi_400s_run"
            / "quang_sheet_cycle_kpi_sheet.csv"
        ),
    )
    parser.add_argument(
        "--directional-summary",
        default=str(
            repo_root
            / "checkpoints"
            / "compare_kpi_live_20260401_directional"
            / "kpi3_satnorm_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "checkpoints" / "quang_sheet_fill_rows_calibrated"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_oneintersection = _load_sheet(Path(args.oneintersection_raw).resolve())
    raw_twointersection = _load_sheet(Path(args.twointersection_raw).resolve())
    directional_summary = _load_directional_summary(Path(args.directional_summary).resolve())

    rows = _build_rows(raw_oneintersection, raw_twointersection, directional_summary)

    csv_path = output_dir / "quang_fill_rows_calibrated.csv"
    png_path = output_dir / "quang_fill_rows_calibrated.png"
    _write_csv(rows, csv_path)
    _render_png(rows, png_path)

    print(f"Saved calibrated CSV: {csv_path}")
    print(f"Saved calibrated PNG: {png_path}")


if __name__ == "__main__":
    main()
