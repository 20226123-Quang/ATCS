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


def compute_fixed_times(env: TrafficEnvironment) -> tuple[int, int]:
    """Derive hardcoded green and yellow durations from config cycle_length_seconds.
    
    Formula: cycle = (green + yellow) * num_green_phases
    => green = (cycle / num_green_phases) - yellow
    """
    cycle = env.kpi_config.simulation.cycle_length_seconds
    yellow = env.kpi_config.simulation.yellow_fallback_seconds
    
    # Count the number of green phases in the plan (from fixed_time_plans in config)
    plans = env.kpi_config.fixed_time_plans
    num_green_phases = 0
    if plans:
        # Take the first junction's plan to infer structure
        first_plan = next(iter(plans.values()))
        num_green_phases = sum(1 for (_, state) in first_plan if any(c in state for c in "Gg") and not any(c in state for c in "Yy"))
    num_green_phases = max(num_green_phases, 1)
    
    green = max((cycle // num_green_phases) - yellow, 1)
    return int(green), int(yellow)

def inject_fixed_time_plans(env: TrafficEnvironment):
    """
    Override the environment's TLS programs with the conflict-free phase 
    states from kpi_config.json.
    Green phase durations = 0 (env will immediately ask for action, 
    and the script will provide the explicit FIXED_GREEN_TIME).
    Yellow phase durations = FIXED_YELLOW_TIME (auto-handled by env).
    """
    plans_dict = env.kpi_config.fixed_time_plans
    if not plans_dict:
        print("Warning: No 'fixed_time_plans' found in KPI config.")
        return

    for tls_id, plan in plans_dict.items():
        if tls_id not in env.tls_programs:
            continue
            
        phases = []
        for idx, (duration, state) in enumerate(plan):
            has_green = any(c in state for c in "Gg")
            has_yellow = any(c in state for c in "Yy")
            
            if has_green and not has_yellow:
                phase_type = 'green'
            elif has_yellow:
                phase_type = 'yellow'
            else:
                phase_type = 'red'
                
            phases.append(
                PhaseDefinition(
                    index=idx,
                    duration_seconds=int(duration),
                    state=state,
                    phase_type=phase_type
                )
            )
            
        first_green_index = next((p.index for p in phases if p.phase_type == "green"), 0)
        base_cycle_seconds = env.cycle_length_seconds
        
        env.tls_programs[tls_id] = TLSProgram(
            tls_id=tls_id,
            phases=tuple(phases),
            base_cycle_seconds=base_cycle_seconds,
            first_green_index=first_green_index
        )
    print(f"Injected fixed_time_plans (Cycle={env.cycle_length_seconds}s from config). Green/yellow times derived in main.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ATCS with explicit Fixed-Time green actions")
    default_cfg = str(Path(__file__).resolve().parents[2] / "SimulationData" / "SampleData" / "Crowded" / "config.sumocfg")
    parser.add_argument("--sumocfg", default=default_cfg, help="Path to SUMO .sumocfg file")
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    args = parser.parse_args()

    print(f"Loading environment from: {args.sumocfg}")
    env = TrafficEnvironment(sumocfg_path=args.sumocfg, use_gui=args.gui)
    
    # Inject conflict-free phase states BEFORE reset
    inject_fixed_time_plans(env)
    
    # Compute green/yellow times from cycle_length_seconds in config
    FIXED_GREEN_TIME, FIXED_YELLOW_TIME = compute_fixed_times(env)
    
    obs, reward, done, info = env.reset()
    print(f"\nEnvironment reset successful.")
    print(f"Cycle Length (from config): {env.cycle_length_seconds}s")
    print(f"Fixed-Time derived: Green={FIXED_GREEN_TIME}s/phase, Yellow={FIXED_YELLOW_TIME}s/phase")
    print(f"---")
    
    step_count = 0
    start_time = time.time()
    
    # Tracker: nếu ngã tư đã được cấp xanh rồi → lần hỏi tiếp theo trả 0 để chuyển pha
    phase_extended = set()

    while not done:
        required_intersections = info.get("intersection_require_action", [])
        remaining_cycle = info.get("remaining_cycle", {})
        
        # Xây dựng action: nếu chưa cấp xanh → cấp FIXED_GREEN_TIME, nếu cấp rồi → trả 0
        action = {}
        for tls_id in required_intersections:
            cycle_len = info.get("cycle_length", {})
            rem = remaining_cycle.get(tls_id, "?")
            clen = cycle_len.get(tls_id, "?") if isinstance(cycle_len, dict) else cycle_len
            
            if tls_id not in phase_extended:
                # Lần đầu hỏi cho pha xanh này → cấp đủ số giây xanh
                action[tls_id] = FIXED_GREEN_TIME
                phase_extended.add(tls_id)
                print(f"[{env.simulation_time}s][{tls_id}] GREEN {FIXED_GREEN_TIME}s | remaining={rem}s / cycle={clen}s")
            else:
                # Lần thứ 2 hỏi → hết xanh, chuyển sang pha vàng
                action[tls_id] = 0
                phase_extended.discard(tls_id)
                print(f"[{env.simulation_time}s][{tls_id}] YELLOW  0s | remaining={rem}s / cycle={clen}s")
        
        obs, reward, done, info = env.step(action)
        step_count += 1
        
    execution_time = time.time() - start_time
    env.close()

    print(f"\n--- Simulation Complete ---")
    print(f"Total decision steps: {step_count}")
    print(f"Execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()
