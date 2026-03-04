"""Entry-point script for training the ACAC model."""

import argparse
from pathlib import Path
import time
import psutil
import csv
import sys
import subprocess

import matplotlib.pyplot as plt
import torch
from acac import (
    SinusoidalPositionalEncoding,
    CentralizedCritic,
    AgentHistoryEncoder,
    MacroActor,
    AsyncTrajectoryBuffer,
    SyncTrajectoryBuffer,
    ACACTrainer,
    load_model_config,
)
from atcs.environment import TrafficEnvironment

_cfg = load_model_config()


def initialize_acac(
    obs_dim,
    action_dim,
    min_action,
    max_action,
    tls_names,
    hidden_dim=128,
    time_embed_dim=16,
    num_heads=4,
    buffer_size=1000,
    actor_lr=1e-4,
    critic_lr=1e-3,
    device="cpu",
):
    """
    Khởi tạo toàn bộ ACAC từ config.
    """
    num_agents = len(tls_names)

    # ---- Time encoder ----
    time_encoder = SinusoidalPositionalEncoding(time_embed_dim).to(device)

    # ---- Per-agent encoders ----
    encoders = [
        AgentHistoryEncoder(obs_dim, time_embed_dim, hidden_dim).to(device)
        for _ in range(num_agents)
    ]

    # ---- Per-agent actors ----
    actors = [
        MacroActor(hidden_dim, action_dim, min_action=0.0, max_action=1.0).to(device)
        for _ in range(num_agents)
    ]

    # ---- Centralized critic ----
    critic = CentralizedCritic(hidden_dim, num_heads).to(device)

    # ---- Buffers ----
    agents_buffer = AsyncTrajectoryBuffer(capacity=buffer_size, num_agents=num_agents)
    critic_buffer = SyncTrajectoryBuffer(capacity=buffer_size)

    # ---- Optimizers ----
    actor_optimizers = [
        torch.optim.Adam(actor.parameters(), lr=actor_lr) for actor in actors
    ]
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    optimizers = {
        "actor": actor_optimizers,
        "critic": critic_optimizer,
    }

    # ---- Trainer ----
    trainer = ACACTrainer(
        actors=actors,
        encoders=encoders,
        critic=critic,
        agents_buffer=agents_buffer,
        critic_buffer=critic_buffer,
        optimizers=optimizers,
        time_encoder=time_encoder,
        tls_names=tls_names,
        device=device,
    )

    return trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    default_cfg = str(
        Path(__file__).resolve().parents[1]
        / "SimulationData"
        / "SampleData"
        / "OneIntersect"
        / "config_one_car_no_delay.sumocfg"
    )
    parser.add_argument(
        "--sumocfg", default=default_cfg, help="Path to SUMO .sumocfg file"
    )
    parser.add_argument(
        "--episodes", type=int, default=2000, help="Number of training episodes"
    )
    parser.add_argument(
        "--steps", type=int, default=600, help="Number of decision steps per episode"
    )
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run on (cpu, cuda, mps)"
    )
    args = parser.parse_args()

    # --- TẠO THƯ MỤC VÀ TÊN FILE TỰ ĐỘNG ---
    save_dir = Path("checkpoint")
    save_dir.mkdir(parents=True, exist_ok=True)  # Tạo folder "checkpoint" nếu chưa có

    sumocfg_path = Path(args.sumocfg)
    parent_dir = sumocfg_path.parent.name.lower()
    grandparent_dir = sumocfg_path.parent.parent.name.lower()

    prefix_name = f"{grandparent_dir}_{parent_dir}"

    # Các file sẽ được đặt chung vào trong thư mục checkpoint/
    checkpoint_filepath = save_dir / f"{prefix_name}_checkpoint.pt"
    plot_filepath = save_dir / f"{prefix_name}_reward_plot.png"
    log_filepath = save_dir / f"{prefix_name}_training_log.csv"
    # ---------------------------------------------------------

    env = TrafficEnvironment(sumocfg_path=args.sumocfg, use_gui=args.gui)

    # Reset environment and get observation dimension
    obs, reward, done, info = env.reset()
    obs_dim = obs.shape[1] * obs.shape[2]
    obs_encoder_dim = obs_dim + 2
    print(f"Observation dimension: {obs_dim} (encoder input: {obs_encoder_dim})")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    trainer = initialize_acac(
        obs_dim=obs_encoder_dim,
        action_dim=1,
        min_action=0.0,
        max_action=1.0,
        tls_names=env.tls_ids,
        buffer_size=_cfg.training.buffer_size,
        actor_lr=_cfg.training.actor_lr,
        critic_lr=_cfg.training.critic_lr,
        device=device,
    )

    # Khởi tạo file log CSV và ghi Header (tiêu đề cột)
    with open(log_filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Episode",
                "Reward",
                "Critic_Loss",
                "Time_Sec",
                "CPU_Percent",
                "RAM_Percent",
                "GPU_VRAM_MB",
            ]
        )

    episode_rewards = []
    total_start_time = time.time()

    # ================= QUÁ TRÌNH TRAINING =================
    for ep in range(args.episodes):
        ep_start_time = time.time()

        print(f"\n--- Episode {ep + 1}/{args.episodes} ---")
        metrics = trainer.train_episode(env, max_steps=args.steps)

        ep_end_time = time.time()
        ep_duration = ep_end_time - ep_start_time

        # Đo lường hiệu suất
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        gpu_vram = 0.0
        if torch.cuda.is_available():
            gpu_vram = torch.cuda.memory_allocated(device) / (1024**2)  # Tính bằng MB

        ep_reward = metrics["reward"]
        episode_rewards.append(ep_reward)

        print(
            f"Reward: {ep_reward:.2f} | Time: {ep_duration:.2f}s | CPU: {cpu_usage}% | RAM: {ram_usage}% | GPU: {gpu_vram:.2f} MB"
        )
        print(f"Critic Loss: {metrics['critic_loss']:.4f}")

        # 1. Lưu dữ liệu log vào file CSV
        with open(log_filepath, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    ep + 1,
                    ep_reward,
                    metrics["critic_loss"],
                    round(ep_duration, 2),
                    cpu_usage,
                    ram_usage,
                    round(gpu_vram, 2),
                ]
            )

        # 2. Cập nhật và lưu biểu đồ
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, ep + 2), episode_rewards, marker="o", linestyle="-")
        plt.title(f"Training Reward - {prefix_name.upper()}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.savefig(plot_filepath)
        plt.close()

        # 3. LƯU MODEL CHECKPOINT NGAY LẬP TỨC (Nằm trong vòng lặp)
        trainer.save_model(str(checkpoint_filepath))

    # ================= KẾT THÚC TRAINING =================
    total_duration = time.time() - total_start_time
    print(f"\nTraining complete in {total_duration / 60:.2f} minutes.")

    env.close()

    print(f"Final model saved to {checkpoint_filepath}")
    print(f"Saved reward plot to {plot_filepath}")
    print(f"Saved training log to {log_filepath}")

    # ---- Auto-evaluate với GUI sau khi training xong ----
    print("\nLaunching GUI evaluation...")
    evaluate_script = str(Path(__file__).resolve().parent / "evaluate.py")
    subprocess.run(
        [
            sys.executable,
            evaluate_script,
            "--sumocfg",
            args.sumocfg,
            "--checkpoint",
            str(
                checkpoint_filepath
            ),  # Truyền đường dẫn file checkpoint chính xác vào evaluate
            "--steps",
            str(args.steps),
        ],
        check=False,
    )


if __name__ == "__main__":
    main()
