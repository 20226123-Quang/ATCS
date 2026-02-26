#!/usr/bin/env python3
"""
Benchmark scaling for the current SUMO traffic-control loop from 2..N intersections.

What this script does:
1) Generate SUMO scenarios (line grid with N traffic lights).
2) Run the existing `SumoTrafficEnv` loop with a fixed phase-rotation policy.
3) Measure wall time, CPU time, peak RAM, reward, throughput and queue.
4) Export summary table to JSON/CSV and print a markdown table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class SumoPaths:
    sumo_home: Path
    sumo_bin: Path
    netgenerate: Path
    random_trips: Path


def detect_sumo_paths() -> SumoPaths:
    import traci

    traci_file = Path(traci.__file__).resolve()
    guessed_home = traci_file.parents[2]
    sumo_home = Path(os.environ.get("SUMO_HOME", str(guessed_home)))

    sumo_bin = sumo_home / "bin" / "sumo.exe"
    netgenerate = sumo_home / "bin" / "netgenerate.exe"
    random_trips = sumo_home / "tools" / "randomTrips.py"

    missing = [p for p in [sumo_bin, netgenerate, random_trips] if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing SUMO tools: {missing_str}")

    return SumoPaths(
        sumo_home=sumo_home,
        sumo_bin=sumo_bin,
        netgenerate=netgenerate,
        random_trips=random_trips,
    )


def run_cmd(cmd: List[str], env: Dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def write_sumocfg(cfg_path: Path, net_file: Path, route_file: Path, max_time: int) -> None:
    cfg_path.write_text(
        textwrap.dedent(
            f"""\
            <configuration>
              <input>
                <net-file value="{net_file.name}"/>
                <route-files value="{route_file.name}"/>
              </input>
              <time>
                <begin value="0"/>
                <end value="{max_time}"/>
              </time>
            </configuration>
            """
        ),
        encoding="utf-8",
    )


def ensure_scenario(
    node_count: int,
    scenario_root: Path,
    sumo_paths: SumoPaths,
    trip_end_base: int,
    trip_period: float,
    seed: int,
    force: bool,
) -> Path:
    scenario_dir = scenario_root / f"n{node_count}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    net_path = scenario_dir / "network.net.xml"
    trips_path = scenario_dir / "random.trips.xml"
    route_path = scenario_dir / "random.rou.xml"
    cfg_path = scenario_dir / "config.sumocfg"

    regen = force or not (net_path.exists() and route_path.exists() and cfg_path.exists())
    if not regen:
        return cfg_path

    env = os.environ.copy()
    env["SUMO_HOME"] = str(sumo_paths.sumo_home)

    run_cmd(
        [
            str(sumo_paths.netgenerate),
            "--grid",
            "--tls.guess",
            "--tls.guess.threshold",
            "0",
            f"--grid.x-number={node_count}",
            "--grid.y-number=1",
            "--grid.x-length",
            "200",
            "--grid.y-length",
            "200",
            "--grid.attach-length",
            "200",
            "-o",
            str(net_path),
        ],
        env=env,
    )

    # Scale demand with node count so load per intersection stays roughly similar.
    trip_end = trip_end_base * node_count
    run_cmd(
        [
            sys.executable,
            str(sumo_paths.random_trips),
            "-n",
            str(net_path),
            "-o",
            str(trips_path),
            "-r",
            str(route_path),
            "--seed",
            str(seed),
            "-b",
            "0",
            "-e",
            str(trip_end),
            "-p",
            str(trip_period),
            "--validate",
        ],
        env=env,
    )

    write_sumocfg(cfg_path, net_path, route_path, max_time=20000)
    return cfg_path


def run_worker(
    node_count: int,
    repeat_idx: int,
    scenario_cfg: Path,
    sumo_bin: Path,
    max_steps: int,
    green_duration: int,
    delta_time: int,
    yellow_time: int,
    reward_fn: str,
) -> Dict[str, float]:
    import psutil
    import traci
    import traffic.environment as env_module
    from traffic.environment import SumoTrafficEnv

    env_module.SUMO_BINARY = str(sumo_bin)

    proc = psutil.Process(os.getpid())
    start_cpu = proc.cpu_times()
    start_wall = time.perf_counter()
    peak_rss = proc.memory_info().rss

    env = SumoTrafficEnv(
        sumo_config=str(scenario_cfg),
        use_gui=False,
        delta_time=delta_time,
        yellow_time=yellow_time,
        max_steps=max_steps,
        reward_fn=reward_fn,
    )

    total_reward = 0.0
    arrived_vehicles = 0
    queue_sum = 0.0
    steps = 0

    try:
        env.reset()
        while True:
            actions = []
            for tls_id in env.tls_ids:
                if env.needed_action.get(tls_id, True):
                    phase = (env.current_phase[tls_id] + 1) % len(env.tls_phases[tls_id])
                    actions.append([phase, green_duration])
                else:
                    actions.append([-1, -1])

            _, reward, terminated, truncated, _ = env.step(actions)
            steps += 1
            total_reward += float(reward)
            arrived_vehicles += int(traci.simulation.getArrivedNumber())

            step_queue = 0.0
            for edge_id in env.edge_ids:
                try:
                    step_queue += float(traci.edge.getLastStepHaltingNumber(edge_id))
                except Exception:
                    pass
            queue_sum += step_queue

            rss = proc.memory_info().rss
            if rss > peak_rss:
                peak_rss = rss

            if terminated or truncated:
                break
    finally:
        env.close()

    end_wall = time.perf_counter()
    end_cpu = proc.cpu_times()

    wall_sec = end_wall - start_wall
    cpu_sec = (end_cpu.user + end_cpu.system) - (start_cpu.user + start_cpu.system)
    avg_reward = total_reward / steps if steps else 0.0
    avg_queue = queue_sum / steps if steps else 0.0

    return {
        "node_count": node_count,
        "repeat": repeat_idx,
        "steps": steps,
        "wall_time_sec": wall_sec,
        "cpu_time_sec": cpu_sec,
        "peak_rss_mb": peak_rss / (1024 * 1024),
        "total_reward": total_reward,
        "avg_reward": avg_reward,
        "arrived_vehicles": arrived_vehicles,
        "avg_queue_length": avg_queue,
    }


def group_mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def group_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def aggregate(raw_rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[int, List[Dict[str, float]]] = {}
    for row in raw_rows:
        grouped.setdefault(int(row["node_count"]), []).append(row)

    summary: List[Dict[str, float]] = []
    for node_count in sorted(grouped.keys()):
        rows = grouped[node_count]
        summary.append(
            {
                "node_count": node_count,
                "runs": len(rows),
                "wall_time_sec_mean": group_mean([r["wall_time_sec"] for r in rows]),
                "wall_time_sec_std": group_std([r["wall_time_sec"] for r in rows]),
                "cpu_time_sec_mean": group_mean([r["cpu_time_sec"] for r in rows]),
                "peak_rss_mb_mean": group_mean([r["peak_rss_mb"] for r in rows]),
                "total_reward_mean": group_mean([r["total_reward"] for r in rows]),
                "avg_reward_mean": group_mean([r["avg_reward"] for r in rows]),
                "arrived_vehicles_mean": group_mean([r["arrived_vehicles"] for r in rows]),
                "avg_queue_length_mean": group_mean([r["avg_queue_length"] for r in rows]),
                "steps_mean": group_mean([r["steps"] for r in rows]),
            }
        )
    return summary


def write_outputs(output_dir: Path, raw_rows: List[Dict[str, float]], summary: List[Dict[str, float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "benchmark_raw.json").open("w", encoding="utf-8") as f:
        json.dump(raw_rows, f, indent=2)

    with (output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_file = output_dir / "benchmark_summary.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "node_count",
                "runs",
                "wall_time_sec_mean",
                "wall_time_sec_std",
                "cpu_time_sec_mean",
                "peak_rss_mb_mean",
                "total_reward_mean",
                "avg_reward_mean",
                "arrived_vehicles_mean",
                "avg_queue_length_mean",
                "steps_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)


def print_markdown_table(summary: List[Dict[str, float]]) -> None:
    print("\n| Node Count | Time (s) | Result | Resource Usage |")
    print("|---:|---:|---|---|")
    for row in summary:
        result_str = (
            f"reward={row['total_reward_mean']:.1f}, "
            f"arrived={row['arrived_vehicles_mean']:.1f}, "
            f"avg_queue={row['avg_queue_length_mean']:.2f}"
        )
        resource_str = (
            f"CPU={row['cpu_time_sec_mean']:.2f}s, "
            f"Peak RAM={row['peak_rss_mb_mean']:.1f}MB"
        )
        time_str = f"{row['wall_time_sec_mean']:.3f} +/- {row['wall_time_sec_std']:.3f}"
        print(f"| {int(row['node_count'])} | {time_str} | {result_str} | {resource_str} |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SUMO traffic model scaling.")
    parser.add_argument("--nodes", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--green-duration", type=int, default=27)
    parser.add_argument("--delta-time", type=int, default=1)
    parser.add_argument("--yellow-time", type=int, default=3)
    parser.add_argument("--reward-fn", type=str, default="wait_time")
    parser.add_argument("--trip-end-base", type=int, default=300)
    parser.add_argument("--trip-period", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-dir", type=Path, default=Path("benchmark/scenarios"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--force-scenario", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sumo_paths = detect_sumo_paths()

    print(f"SUMO home: {sumo_paths.sumo_home}")
    print(f"Using nodes: {args.nodes}, repeats: {args.repeats}")

    scenario_cfgs: Dict[int, Path] = {}
    for n in args.nodes:
        cfg = ensure_scenario(
            node_count=n,
            scenario_root=args.scenario_dir,
            sumo_paths=sumo_paths,
            trip_end_base=args.trip_end_base,
            trip_period=args.trip_period,
            seed=args.seed,
            force=args.force_scenario,
        )
        scenario_cfgs[n] = cfg

    raw_rows: List[Dict[str, float]] = []
    for n in args.nodes:
        for rep in range(args.repeats):
            print(f"Running benchmark: nodes={n}, repeat={rep + 1}/{args.repeats}")
            row = run_worker(
                node_count=n,
                repeat_idx=rep,
                scenario_cfg=scenario_cfgs[n],
                sumo_bin=sumo_paths.sumo_bin,
                max_steps=args.max_steps,
                green_duration=args.green_duration,
                delta_time=args.delta_time,
                yellow_time=args.yellow_time,
                reward_fn=args.reward_fn,
            )
            raw_rows.append(row)

    summary = aggregate(raw_rows)
    write_outputs(args.output_dir, raw_rows, summary)
    print_markdown_table(summary)
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
