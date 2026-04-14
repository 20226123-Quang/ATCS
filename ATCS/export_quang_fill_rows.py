"""Build a paste-ready Quang sheet block for fixed-time Quang and AI rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


GROUP_SPECS = [
    ("Nga ba", "oneintersection_3direction"),
    ("Nga tu", "oneintersection_4direction"),
    ("Nga nam", "oneintersection_5direction"),
    ("Hai nga tu", "2intersection"),
]

DENSITY_ORDER = ["crowded", "normal", "few"]
SERIES_ORDER = ["fixed time (Quang)", "AI", "AI vs Fixed time"]
METRIC_ORDER = ["queue length", "avg wait time", "saturation"]


def _load_sheet_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["scenario_family"], row["metric"], row["series"])
        indexed[key] = row
    return indexed


def _merge_sources(source_paths: list[Path]) -> dict[tuple[str, str, str], dict[str, str]]:
    merged: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in source_paths:
        merged.update(_index_rows(_load_sheet_rows(path)))
    return merged


def _build_rows(indexed: dict[tuple[str, str, str], dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metric in METRIC_ORDER:
        for series in SERIES_ORDER:
            row = {
                "metric": metric,
                "series": series,
            }
            for group_label, family in GROUP_SPECS:
                source = indexed.get((family, metric, series), {})
                for density in DENSITY_ORDER:
                    row[f"{group_label}_{density}"] = source.get(density, "N/A")
            rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    fieldnames = ["metric", "series"]
    for group_label, _ in GROUP_SPECS:
        for density in DENSITY_ORDER:
            fieldnames.append(f"{group_label}_{density}")

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_png(rows: list[dict[str, str]], output_path: Path) -> None:
    fieldnames = ["metric", "series"]
    for group_label, _ in GROUP_SPECS:
        for density in DENSITY_ORDER:
            fieldnames.append(f"{group_label}\n{density.upper()}")

    cell_text = []
    for row in rows:
        values = [row["metric"], row["series"]]
        for group_label, _ in GROUP_SPECS:
            for density in DENSITY_ORDER:
                values.append(row.get(f"{group_label}_{density}", "N/A"))
        cell_text.append(values)

    fig, ax = plt.subplots(figsize=(19, 6.5))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=fieldnames,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)

    for row_index in range(len(rows) + 1):
        for col_index in range(len(fieldnames)):
            cell = table[(row_index, col_index)]
            if row_index == 0:
                cell.set_facecolor("#d9e8fb")
                cell.set_text_props(weight="bold")
                continue
            if rows[row_index - 1]["series"] == "AI vs Fixed time":
                cell.set_facecolor("#f3f3f3")

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oneintersection-sheet",
        default=str(
            repo_root
            / "checkpoints"
            / "quang_sheet_oneintersection_cycle_kpi_400s_tccs_run"
            / "quang_sheet_oneintersection_cycle_kpi_sheet.csv"
        ),
    )
    parser.add_argument(
        "--twointersection-sheet",
        default=str(
            repo_root
            / "checkpoints"
            / "quang_sheet_cycle_kpi_400s_run"
            / "quang_sheet_cycle_kpi_sheet.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "checkpoints" / "quang_sheet_fill_rows"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    indexed = _merge_sources(
        [Path(args.oneintersection_sheet).resolve(), Path(args.twointersection_sheet).resolve()]
    )
    rows = _build_rows(indexed)

    csv_path = output_dir / "quang_fill_rows.csv"
    png_path = output_dir / "quang_fill_rows.png"
    _write_csv(rows, csv_path)
    _render_png(rows, png_path)

    print(f"Saved fill CSV: {csv_path}")
    print(f"Saved fill PNG: {png_path}")


if __name__ == "__main__":
    main()
