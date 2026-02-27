"""Evaluate the environment using a Fixed Time (Baseline) strategy."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import json
import sys
import xml.etree.ElementTree as ET

# Thêm thư mục gốc ATCS vào sys.path để Python tìm thấy module 'atcs'
sys.path.append(str(Path(__file__).resolve().parents[1]))

from atcs.environment import TrafficEnvironment
from atcs.sumo_parser import _resolve_net_file


def get_fixed_time_plans_from_net(sumocfg_path: str):
    """
    Parse SUMO net file to extract fixed-time signal plans for each traffic light.
    Returns a dict: {tls_id: [(duration, state_string), ...]}
    """
    try:
        net_path = _resolve_net_file(Path(sumocfg_path))
        net_root = ET.parse(net_path).getroot()
    except Exception as e:
        print(f"Error loading net file: {e}")
        return None

    plans = {}
    for tl in net_root.findall('tlLogic'):
        tls_id = tl.get('id')
        phases = []
        for phase in tl.findall('phase'):
            duration_raw = phase.get('duration')
            duration = float(duration_raw) if duration_raw else None
            state = phase.get('state')
            if duration is not None and state:
                phases.append((duration, state))
        if phases:
            plans[tls_id] = phases
    return plans


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ATCS Environment with Fixed Time Control")
    default_cfg = str(Path(__file__).resolve().parents[2] / "SimulationData" / "SampleData" / "Crowded" / "config.sumocfg")
    parser.add_argument("--sumocfg", default=default_cfg, help="Path to SUMO .sumocfg file")
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    parser.add_argument("--metrics_out", default="fixed_time_metrics.json", help="Path to save output metrics")
    args = parser.parse_args()

    print(f"Loading environment from: {args.sumocfg}")
    
    # Extract fixed-time plans from net file to verify we have them
    fixed_time_plans = get_fixed_time_plans_from_net(args.sumocfg)
    if fixed_time_plans:
        print("Successfully extracted fixed-time plans from network:")
        for tls, phases in fixed_time_plans.items():
            print(f"  {tls}: {len(phases)} phases detected.")
            for d, s in phases:
                print(f"    - {d}s: {s}")
    else:
        print("Warning: Could not extract specific fixed-time plans. Relying purely on SUMO's internal cycle.")

    env = TrafficEnvironment(sumocfg_path=args.sumocfg, use_gui=args.gui)
    
    obs, reward, done, info = env.reset()
    print("\nEnvironment reset successful. Starting Fixed-Time baseline...")
    
    step_count = 0
    start_time = time.time()

    while not done:
        # In Fixed-Time mode, the environment's `TrafficEnvironment` class internally runs 
        # the exact phases parsed from the `.net.xml` file (via `sumo_parser.parse_sumo_network`).
        # When `info['intersection_require_action']` asks for an action, we return 0.0 seconds
        # extension so that it moves *exactly* to the next planned phase shown in `FIXED_TIME_PLANS`.
        # This guarantees safety and explicit phase adherence without conflicts.
        
        required_intersections = info.get("intersection_require_action", [])
        action = {tls_id: 0.0 for tls_id in required_intersections}
        
        # Display the controllable green lanes info we built earlier
        controllable_lanes = info.get("controllable_intersections", {})
        if controllable_lanes:
            print(f"[{info['delta_t']}s simulated] Controllable Lights: {controllable_lanes}")
        
        obs, reward, done, info = env.step(action)
        step_count += 1
        
    execution_time = time.time() - start_time
    env.close()

    print(f"\n--- Simulation Complete ---")
    print(f"Total decision steps: {step_count}")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Simulation finished. To view formal metrics, integrate the KPI logs.")

if __name__ == "__main__":
    main()
