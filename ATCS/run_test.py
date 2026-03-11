import sys

sys.path.append("/data/EGEN2025/Philosophi/NewWork/ATCS")
from atcs.config_loader import load_kpi_config
from atcs.environment import TrafficEnvironment


def run_ep(actions, name):
    print(f"\n--- {name} ---")
    env = TrafficEnvironment(
        sumocfg_path="/data/EGEN2025/Philosophi/NewWork/SimulationData/SampleData/OneIntersect/config_one_car_no_delay.sumocfg",
        kpi_config_path="config/kpi_config.json",
        use_gui=False,
    )
    obs, reward, done, info = env.reset()
    total_reward = 0
    t = 0
    for step, act in enumerate(actions):
        if done:
            break
        obs, reward, done, info = env.step({"J1": act})

        # Calculate exactly like trainer.py
        valid_lanes = 0
        global_r = 0.0
        for l_idx in range(reward.shape[1]):
            r_delay = reward[0, l_idx, 0]
            if r_delay != 0:
                print(f"   Step {step} Lane {l_idx} Delay Reward: {r_delay:.2f}")
            global_r += r_delay
            valid_lanes += 1

        step_r = global_r / valid_lanes if valid_lanes > 0 else 0
        total_reward += step_r
        print(f"   Step {step} Total Mean Reward: {step_r:.2f}")

    print(f"Final Episode Mean Reward sum: {total_reward:.2f}")
    env.close()


run_ep([56.18, 51.41, 60.0], "Ep 14 (Long)")
run_ep([15.12, 23.53, 60.0], "Ep 15 (Short)")
