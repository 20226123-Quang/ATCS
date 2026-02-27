"""Evaluate the environment using a Fixed Time (Baseline) strategy."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import json
import sys

# Thêm thư mục gốc ATCS vào sys.path để Python tìm thấy module 'atcs'
sys.path.append(str(Path(__file__).resolve().parents[1]))

from atcs.environment import TrafficEnvironment
from atcs.sumo_parser import TLSProgram, PhaseDefinition


def inject_fixed_time_plans(env: TrafficEnvironment):
    """
    Override the environment's parsed TLS programs with the conflict-free 
    plans defined in kpi_config.json -> fixed_time_plans.
    """
    plans_dict = env.kpi_config.fixed_time_plans
    if not plans_dict:
        print("Warning: No 'fixed_time_plans' found in KPI config. Relying on default SUMO net phases.")
        return

    for tls_id, plan in plans_dict.items():
        if tls_id not in env.tls_programs:
            continue
            
        phases = []
        for idx, (duration, state) in enumerate(plan):
            phase_type = 'green' if 'g' in state.lower() else ('yellow' if 'y' in state.lower() else 'red')
            phases.append(
                PhaseDefinition(
                    index=idx,
                    duration_seconds=int(duration),
                    state=state,
                    phase_type=phase_type
                )
            )
            
        first_green_index = next((p.index for p in phases if p.phase_type == "green"), 0)
        base_cycle_seconds = sum(p.duration_seconds for p in phases)
        
        env.tls_programs[tls_id] = TLSProgram(
            tls_id=tls_id,
            phases=tuple(phases),
            base_cycle_seconds=base_cycle_seconds,
            first_green_index=first_green_index
        )
    print("Injected conflict-free fixed_time_plans from config into the environment.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ATCS Environment with Fixed Time Control")
    default_cfg = str(Path(__file__).resolve().parents[2] / "SimulationData" / "SampleData" / "Crowded" / "config.sumocfg")
    parser.add_argument("--sumocfg", default=default_cfg, help="Path to SUMO .sumocfg file")
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    parser.add_argument("--metrics_out", default="fixed_time_metrics.json", help="Path to save output metrics")
    args = parser.parse_args()

    print(f"Loading environment from: {args.sumocfg}")
    env = TrafficEnvironment(sumocfg_path=args.sumocfg, use_gui=args.gui)
    
    # Inject the conflict-free plans BEFORE resetting the environment
    inject_fixed_time_plans(env)
    
    obs, reward, done, info = env.reset()
    print("\nEnvironment reset successful. Starting Fixed-Time baseline...")
    
    step_count = 0
    start_time = time.time()

    while not done:
        # In Fixed-Time mode, the environment's TrafficEnvironment class internally runs 
        # the exact phases we just injected from the KPI config file.
        # When `info['intersection_require_action']` asks for an action, we return 0.0 seconds
        # extension so that it moves *exactly* to the next planned phase.
        
        required_intersections = info.get("intersection_require_action", [])
        action = {tls_id: 0.0 for tls_id in required_intersections}
        
        # Display the controllable green lanes info
        controllable_lanes = info.get("controllable_intersections", {})
        remaining_cycle = info.get("remaining_cycle", {})
        
        if controllable_lanes:
            print(f"[{info['delta_t']}s simulated] Controllable Lights: {controllable_lanes}, Remaining Cycle: {remaining_cycle}")
        
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
