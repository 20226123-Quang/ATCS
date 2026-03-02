"""Evaluate the environment using a Fixed Time (Baseline) strategy."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import sys

# Thêm thư mục gốc ATCS vào sys.path để Python tìm thấy module 'atcs'
sys.path.append(str(Path(__file__).resolve().parents[1]))

from atcs.environment import TrafficEnvironment
from atcs.sumo_parser import TLSProgram, PhaseDefinition


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ATCS with explicit Fixed-Time green actions")
    default_cfg = str(Path(__file__).resolve().parents[2] / "SimulationData" / "SampleData" / "Crowded" / "config.sumocfg")
    parser.add_argument("--sumocfg", default=default_cfg, help="Path to SUMO .sumocfg file")
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    args = parser.parse_args()

    print(f"Loading environment from: {args.sumocfg}")
    env = TrafficEnvironment(sumocfg_path=args.sumocfg, use_gui=args.gui)
    
    obs, reward, done, info = env.reset()
    print(f"\nEnvironment reset successful.")
    print(f"Cycle Length (from config): {env.cycle_length_seconds}s")
    
    step_count = 0
    start_time = time.time()

    while not done:
        required_intersections = info.get("intersection_require_action", [])
        action = {tls_id: 30 for tls_id in required_intersections}
        obs, reward, done, info = env.step(action)
        step_count += 1
        
    execution_time = time.time() - start_time
    env.close()

    print(f"\n--- Simulation Complete ---")
    print(f"Total decision steps: {step_count}")
    print(f"Execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()
