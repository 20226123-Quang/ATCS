"""Quick smoke run for the new ATCS TrafficEnvironment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Thêm thư mục gốc ATCS vào sys.path để Python tìm thấy module 'atcs'
sys.path.append(str(Path(__file__).resolve().parents[1]))

from atcs.environment import TrafficEnvironment

def main() -> None:
    parser = argparse.ArgumentParser()
    default_cfg = str(Path(__file__).resolve().parents[2] / "SimulationData" / "SampleData" / "Crowded" / "config.sumocfg")
    parser.add_argument("--sumocfg", default=default_cfg, help="Path to SUMO .sumocfg file")
    parser.add_argument("--steps", type=int, default=1000, help="Number of decision steps")
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    args = parser.parse_args()

    env = TrafficEnvironment(sumocfg_path=args.sumocfg, use_gui=args.gui)
    obs, reward, done, info = env.reset()
    print("reset:")
    print(f"  obs shape   = {obs.shape}")
    print(f"  reward shape= {reward.shape}")
    print(f"  done        = {done}")
    print(f"  info        = {info}")

    for step_idx in range(args.steps):
        required = info["intersection_require_action"]
        controllable = info.get("controllable_intersections", {})
        
        action = {tls_id: 5.0 for tls_id in required}
        obs, reward, done, info = env.step(action)
        
        print(f"step {step_idx + 1}: done={done}, controllable_lanes={controllable} (require_action={required}), delta_t={info['delta_t']}")
        if done:
            break

    env.close()


if __name__ == "__main__":
    main()
