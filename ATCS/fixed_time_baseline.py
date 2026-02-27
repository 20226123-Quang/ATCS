"""Evaluate the environment using a Fixed Time (Baseline) strategy."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import sys

from atcs.environment import TrafficEnvironment
from atcs.sumo_parser import TLSProgram, PhaseDefinition


# ===================== THÔNG SỐ FIXED-TIME HARDCODE =====================
FIXED_GREEN_TIME = 30   # Số giây đèn xanh cố định (giây)
FIXED_YELLOW_TIME = 3   # Số giây đèn vàng cố định (giây)
# ========================================================================


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
        base_cycle_seconds = (FIXED_GREEN_TIME + FIXED_YELLOW_TIME) * (len(phases) // 2)
        
        env.tls_programs[tls_id] = TLSProgram(
            tls_id=tls_id,
            phases=tuple(phases),
            base_cycle_seconds=base_cycle_seconds,
            first_green_index=first_green_index
        )
    print(f"Injected fixed_time_plans from config (Green={FIXED_GREEN_TIME}s, Yellow={FIXED_YELLOW_TIME}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ATCS with explicit Fixed-Time green actions")
    default_cfg = str(Path(__file__).resolve().parents[1] / "SimulationData" / "SampleData" / "Crowded" / "config.sumocfg")
    parser.add_argument("--sumocfg", default=default_cfg, help="Path to SUMO .sumocfg file")
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    args = parser.parse_args()

    print(f"Loading environment from: {args.sumocfg}")
    env = TrafficEnvironment(sumocfg_path=args.sumocfg, use_gui=args.gui)
    
    # Inject conflict-free phase states BEFORE reset
    inject_fixed_time_plans(env)
    
    obs, reward, done, info = env.reset()
    print(f"\nEnvironment reset successful.")
    print(f"Fixed-Time Baseline: Green={FIXED_GREEN_TIME}s, Yellow={FIXED_YELLOW_TIME}s")
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
            if tls_id not in phase_extended:
                # Lần đầu hỏi cho pha xanh này → cấp đủ số giây xanh
                action[tls_id] = FIXED_GREEN_TIME
                phase_extended.add(tls_id)
                print(f"[{env.simulation_time}s][{tls_id}] Action = {FIXED_GREEN_TIME}s (GREEN), Remaining Cycle: {remaining_cycle.get(tls_id, '?')}")
            else:
                # Lần thứ 2 hỏi → hết xanh, chuyển sang pha vàng
                action[tls_id] = 0
                phase_extended.discard(tls_id)
                print(f"[{env.simulation_time}s][{tls_id}] Action = 0 (chuyển YELLOW)")
        
        obs, reward, done, info = env.step(action)
        step_count += 1
        
    execution_time = time.time() - start_time
    env.close()

    print(f"\n--- Simulation Complete ---")
    print(f"Total decision steps: {step_count}")
    print(f"Execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()
