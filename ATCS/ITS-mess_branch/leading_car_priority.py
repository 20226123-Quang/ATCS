"""
SUMO Traffic Light Priority Control for LEADING CAR
- Detects when LEADING_CAR approaches an intersection
- Sets green light only for LEADING_CAR's direction for 100 seconds
- Returns to fixed-time/adaptive control after that
"""

import sys
import json
from datetime import datetime

import traci

from traffic.config import (
    SUMO_CMD,
    LEADING_CAR_ID,
    PRIORITY_DURATION,
    DETECTION_DISTANCE,
    FIXED_TIME_PLANS,
)
from traffic.controllers.tuc_controller import TUCController
from traffic.traffic_light_controller import TrafficLightController
from traffic.metrics.metrics_collector import MetricsCollector


def main():
    print("Starting SUMO simulation with LEADING_CAR priority control...")
    print(f"LEADING_CAR ID: {LEADING_CAR_ID}")
    print(f"Priority duration: {PRIORITY_DURATION} seconds")
    print(f"Detection distance: {DETECTION_DISTANCE} meters")
    print(f"Signal Control: TUC Adaptive Control (4-phase with protected TL)")
    print("-" * 60)
    print(f"Fixed-time plans for traffic lights:", FIXED_TIME_PLANS)

    try:
        traci.start(SUMO_CMD)

        signal_controller = TUCController(min_phase=5, max_phase=60, max_cycle=122)
        controller = TrafficLightController(signal_controller=signal_controller)
        metrics_collector = MetricsCollector()

        step = 0
        max_steps = 5000  # Safety limit to prevent infinite loop

        print("Running simulation until all vehicles exit the network...")

        while step < max_steps:
            traci.simulationStep()
            current_time = traci.simulation.getTime()

            controller.control_traffic_lights(current_time)
            metrics_collector.update_metrics(current_time)

            vehicle_list = traci.vehicle.getIDList()
            num_vehicles = len(vehicle_list)

            if step % 100 == 0:
                leading_car_status = "ACTIVE" if LEADING_CAR_ID in vehicle_list else "NOT IN NETWORK"
                priority_status = ", ".join([
                    f"{tls}:{'ON' if controller.priority_status[tls]['active'] else 'OFF'}"
                    for tls in FIXED_TIME_PLANS
                ])
                print(
                    f"[Step {step:4d}] Time: {current_time:6.1f}s | Vehicles: {num_vehicles:3d} | "
                    f"Leading Car: {leading_car_status} | Priority: {priority_status}"
                )

            if num_vehicles == 0 and step > 100:
                print(
                    f"\nAll vehicles have exited the network at step {step}, time {current_time:.1f}s"
                )
                break

            step += 1

        print("-" * 60)
        print("Simulation completed")

        avg_wait_time = metrics_collector.get_average_wait_time()
        avg_queue_length = metrics_collector.get_average_queue_length()
        max_queue_length = metrics_collector.get_max_queue_length()
        avg_travel_time = metrics_collector.get_average_travel_time()
        max_travel_time = metrics_collector.get_max_travel_time()
        min_travel_time = metrics_collector.get_min_travel_time()

        print(f"\n===== METRICS RESULTS =====")
        print(f"Average Wait Time (PCU-weighted): {avg_wait_time:.2f} seconds")
        print(f"Total vehicles waited: {len(metrics_collector.completed_wait_times)}")
        print(f"\nAverage Queue Length (PCU-weighted): {avg_queue_length:.2f}")
        print(f"Maximum Queue Length (PCU): {max_queue_length:.2f}")
        print(f"Queue measurements taken: {len(metrics_collector.queue_lengths)}")
        print(f"\nAverage Travel Time: {avg_travel_time:.2f} seconds")
        print(f"Minimum Travel Time: {min_travel_time:.2f} seconds")
        print(f"Maximum Travel Time: {max_travel_time:.2f} seconds")
        print(f"Total vehicles traveled: {len(metrics_collector.completed_travel_times)}")

        metrics_file = "traffic_metrics.json"
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'average_wait_time': avg_wait_time,
            'total_vehicles_waited': len(metrics_collector.completed_wait_times),
            'average_queue_length': avg_queue_length,
            'max_queue_length': max_queue_length,
            'queue_measurements': len(metrics_collector.queue_lengths),
            'average_travel_time': avg_travel_time,
            'min_travel_time': min_travel_time,
            'max_travel_time': max_travel_time,
            'total_vehicles_traveled': len(metrics_collector.completed_travel_times),
            'wait_times_detail': metrics_collector.completed_wait_times,
            'queue_lengths_detail': metrics_collector.queue_lengths,
            'travel_times_detail': metrics_collector.completed_travel_times,
        }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"\nMetrics saved to {metrics_file}")

        traci.close()

    except Exception as e:
        print(f"Error occurred: {e}")
        if traci.isLoaded():
            traci.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
