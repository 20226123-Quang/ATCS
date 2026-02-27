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
import argparse
from pathlib import Path
import time
import json
import sys

# Thêm thư mục gốc ATCS vào sys.path để Python tìm thấy module 'atcs'
sys.path.append(str(Path(__file__).resolve().parents[1]))

import traci
from atcs.environment import TrafficEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ATCS Environment with explicit Fixed Time actions")
    default_cfg = str(Path(__file__).resolve().parents[2] / "SimulationData" / "SampleData" / "Crowded" / "config.sumocfg")
    parser.add_argument("--sumocfg", default=default_cfg, help="Path to SUMO .sumocfg file")
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    args = parser.parse_args()

    print(f"Loading environment from: {args.sumocfg}")
    # We still use TrafficEnvironment to initialize traci, KPI engines, etc.
    env = TrafficEnvironment(sumocfg_path=args.sumocfg, use_gui=args.gui)
    
    plans = env.kpi_config.fixed_time_plans
    if not plans:
        print("Error: No fixed_time_plans found in KPI config.")
        return

    # Khởi tạo state mô phỏng
    obs, reward, done, info = env.reset()
    print("\nEnvironment reset successful. Starting Custom Fixed-Time baseline...")
    
    # Custom Tracker cho mỗi ngã tư
    tls_trackers = {}
    for tls_id in env.tls_ids:
        if tls_id in plans:
            tls_trackers[tls_id] = {
                "phases": plans[tls_id],
                "current_idx": 0,
                "elapsed": 0,
                "duration": plans[tls_id][0][0],  # Số giây hardcode (VD: 27.5 hoặc 30)
                "state": plans[tls_id][0][1]      # Chuỗi pha (VD: grrgGrgrrgGr)
            }
            traci.trafficlight.setRedYellowGreenState(tls_id, tls_trackers[tls_id]["state"])
            print(f"[{tls_id}] Init Phase 0: {tls_trackers[tls_id]['duration']}s -> {tls_trackers[tls_id]['state']}")

    step_count = 0
    start_time = time.time()
    
    # Vòng lặp mô phỏng tùy chỉnh
    while True:
        try:
            # Check done condition (hết xe hoăc hết thời gian)
            if env.simulation_time >= env.max_episode_seconds or traci.simulation.getMinExpectedNumber() <= 0:
                break
                
            traci.simulationStep()
            env.simulation_time += env.step_length_seconds
            
            # Action: Mỗi ngã tư tự kiểm tra xem đã hết thời gian của phase hiện tại chưa
            for tls_id, tracker in tls_trackers.items():
                tracker["elapsed"] += env.step_length_seconds
                
                # Nếu đã hết duration của pha này, chuyển sang pha tiếp theo trong mảng
                if tracker["elapsed"] >= tracker["duration"]:
                    tracker["current_idx"] = (tracker["current_idx"] + 1) % len(tracker["phases"])
                    tracker["duration"] = tracker["phases"][tracker["current_idx"]][0]
                    tracker["state"] = tracker["phases"][tracker["current_idx"]][1]
                    tracker["elapsed"] = 0
                    
                    traci.trafficlight.setRedYellowGreenState(tls_id, tracker["state"])
                    print(f"[{env.simulation_time}s][{tls_id}] Action - Chuyển sang phase {tracker['current_idx']}: {tracker['duration']}s -> {tracker['state']}")

            step_count += 1
            
        except traci.exceptions.FatalTraCIError:
            print("SUMO Connection closed.")
            break
        
    execution_time = time.time() - start_time
    env.close()

    print(f"\n--- Simulation Complete ---")
    print(f"Total simulation steps: {step_count}")
    print(f"Execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()
