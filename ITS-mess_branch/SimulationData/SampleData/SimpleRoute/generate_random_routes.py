"""Generate random bidirectional routes for SimpleRoute network.

Routes follow corridor E0 -> E2 -> E5 and the reverse direction.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


WEST_TO_EAST = ["-E0", "E2", "E5"]
EAST_TO_WEST = ["-E5", "-E2", "E0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate random SUMO routes for SimpleRoute.")
    parser.add_argument("--output", default="random_simple.rou.xml", help="Output route file name.")
    parser.add_argument("--vehicles", type=int, default=40, help="Number of vehicles to create.")
    parser.add_argument("--begin", type=float, default=0.0, help="Simulation start time (s).")
    parser.add_argument("--end", type=float, default=100.0, help="Simulation end time (s).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    parser.add_argument("--east_probability", type=float, default=0.5, help="Probability a vehicle goes west->east.")
    return parser.parse_args()


def build_depart_times(count: int, begin: float, end: float) -> list[float]:
    if end <= begin:
        raise ValueError("end must be greater than begin")
    span = end - begin
    times = [begin + random.random() * span for _ in range(count)]
    times.sort()
    return times


def pick_route(east_prob: float) -> list[str]:
    if not 0.0 <= east_prob <= 1.0:
        raise ValueError("east_probability must be in [0, 1]")
    return WEST_TO_EAST if random.random() < east_prob else EAST_TO_WEST


def write_routes(path: Path, depart_times: list[float], east_prob: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\n")
        f.write("<routes xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" "
                "xsi:noNamespaceSchemaLocation=\"http://sumo.dlr.de/xsd/routes_file.xsd\">\n")
        f.write("    <vType id=\"car\" accel=\"2.6\" decel=\"4.5\" sigma=\"0.5\" length=\"5\" minGap=\"2.5\" maxSpeed=\"16.67\" guiShape=\"passenger\"/>\n")
        for idx, depart in enumerate(depart_times):
            edges = " ".join(pick_route(east_prob))
            f.write(f"    <vehicle id=\"veh{idx}\" type=\"car\" depart=\"{depart:.2f}\">\n")
            f.write(f"        <route edges=\"{edges}\"/>\n")
            f.write("    </vehicle>\n")
        f.write("</routes>\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    depart_times = build_depart_times(args.vehicles, args.begin, args.end)
    out_path = Path(args.output)
    write_routes(out_path, depart_times, args.east_probability)


if __name__ == "__main__":
    main()
