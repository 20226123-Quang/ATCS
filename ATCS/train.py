"""Entry-point script for training the ACAC model."""

import argparse
import csv
import os
import time
from pathlib import Path

import psutil
import torch

from acac import (
    ACACTrainer,
    AgentHistoryEncoder,
    AsyncTrajectoryBuffer,
    CentralizedCritic,
    MacroActor,
    SinusoidalPositionalEncoding,
    SyncTrajectoryBuffer,
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
    gamma=0.99,
    lam=0.95,
    eps_clip=0.2,
    device="cpu",
):
    """
    Initialize the ACAC stack from config.
    obs_dim: flattened observation size (max_lanes * features)
    action_dim: 1 (green extension seconds)
    """
    num_agents = len(tls_names)

    time_encoder = SinusoidalPositionalEncoding(time_embed_dim).to(device)

    encoders = [
        AgentHistoryEncoder(obs_dim, time_embed_dim, hidden_dim).to(device)
        for _ in range(num_agents)
    ]

    actors = [
        MacroActor(hidden_dim, action_dim, min_action=0.0, max_action=1.0).to(device)
        for _ in range(num_agents)
    ]

    critic = CentralizedCritic(hidden_dim, num_heads).to(device)

    agents_buffer = AsyncTrajectoryBuffer(capacity=buffer_size, num_agents=num_agents)
    critic_buffer = SyncTrajectoryBuffer(capacity=buffer_size)

    all_params = list(critic.parameters())
    for actor in actors:
        all_params += list(actor.parameters())
    for encoder in encoders:
        all_params += list(encoder.parameters())

    optimizers = {
        "combined": torch.optim.Adam(all_params, lr=actor_lr),
    }

    return ACACTrainer(
        actors=actors,
        encoders=encoders,
        critic=critic,
        agents_buffer=agents_buffer,
        critic_buffer=critic_buffer,
        optimizers=optimizers,
        time_encoder=time_encoder,
        tls_names=tls_names,
        device=device,
        gamma=gamma,
        lam=lam,
        eps_clip=eps_clip,
    )


def load_existing_training_history(log_file: Path):
    """Return reward history and the last logged episode number."""
    if not log_file.exists():
        return [], 0

    rewards = []
    last_episode = 0
    with open(log_file, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rewards.append(float(row["Reward"]))
                last_episode = max(last_episode, int(row["Episode"]))
            except (KeyError, TypeError, ValueError):
                continue

    return rewards, last_episode


def initialize_log_file(log_file: Path) -> None:
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Episode",
                "Reward",
                "Critic_Loss",
                "Time_Sec",
                "CPU_Percent",
                "RAM_Percent",
            ]
        )


def _should_log_lane_width_factor(scenario_name: str) -> bool:
    return scenario_name == "oneintersection_4direction"


def main() -> None:
    parser = argparse.ArgumentParser()
    default_cfg = str(
        Path(__file__).resolve().parents[1]
        / "SimulationData"
        / "SampleData"
        / "OneIntersect"
        / "config_one_car_delay_40_5_cars.sumocfg"
    )
    parser.add_argument("--sumocfg", default=default_cfg, help="Path to SUMO .sumocfg file")
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Total target training episodes. With --resume, training continues up to this total.",
    )
    parser.add_argument(
        "--steps", type=int, default=600, help="Number of decision steps per episode"
    )
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run on (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional path to checkpoint (.pt) to continue/fine-tune training",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint/log/plot instead of resetting them",
    )
    args = parser.parse_args()

    cfg_path = Path(args.sumocfg)
    intersection_name = cfg_path.parent.name.lower()
    condition_name = cfg_path.parent.parent.name.lower()
    scenario_name = f"{condition_name}_{intersection_name}"

    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_dir = repo_root / "checkpoints" / scenario_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_file = checkpoint_dir / f"{scenario_name}_training_log.csv"
    model_file = checkpoint_dir / f"{scenario_name}_checkpoint.pt"
    plot_file = checkpoint_dir / f"{scenario_name}_reward_plot.png"
    latest_model_file = model_file

    mpl_config_dir = checkpoint_dir / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir.resolve())

    print(f"[{scenario_name}] Output directory: {checkpoint_dir}")

    env = TrafficEnvironment(
        sumocfg_path=args.sumocfg,
        use_gui=args.gui,
        log_lane_width_adjustment_factor=_should_log_lane_width_factor(scenario_name),
    )

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
        hidden_dim=_cfg.model.hidden_dim,
        time_embed_dim=_cfg.model.time_embed_dim,
        num_heads=_cfg.model.num_heads,
        buffer_size=_cfg.training.buffer_size,
        actor_lr=_cfg.training.actor_lr,
        critic_lr=_cfg.training.critic_lr,
        device=device,
        gamma=_cfg.training.gamma,
        lam=_cfg.training.lam,
        eps_clip=_cfg.training.eps_clip,
    )

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = (repo_root / ckpt_path).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"Loading checkpoint for continued training: {ckpt_path}")
        trainer.load_model(str(ckpt_path))
        print("Checkpoint loaded successfully.")

    import matplotlib.pyplot as plt
    import subprocess
    import sys

    for ep in range(start_episode, args.episodes):
        print(f"\n--- Episode {ep + 1}/{args.episodes} ---")
        start_time = time.time()
        metrics = trainer.train_episode(env, max_steps=args.steps)
        end_time = time.time()

        ep_time = end_time - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent

        ep_reward = metrics["reward"]
        critic_loss = metrics["critic_loss"]
        episode_rewards.append(ep_reward)

        print(f"Reward: {ep_reward:.2f}")
        print(f"Critic Loss: {critic_loss:.4f}")
        for i, aloss in metrics["actor_losses"].items():
            print(f"  Actor {i} Loss: {aloss:.4f}")

        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    ep + 1,
                    round(ep_reward, 2),
                    round(critic_loss, 4),
                    round(ep_time, 2),
                    cpu_percent,
                    ram_percent,
                ]
            )

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(episode_rewards) + 1),
            episode_rewards,
            marker="o",
            linestyle="-",
        )
        plt.title(f"Training Reward over Episodes ({scenario_name})")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.savefig(plot_file)
        plt.close()

        latest_model_file = Path(trainer.save_model(str(model_file)))

    print(f"\nTraining complete. Model saved to {latest_model_file}")
    env.close()

    if args.gui:
        print("\nLaunching GUI evaluation...")
        evaluate_script = str(Path(__file__).resolve().parent / "evaluate.py")
        subprocess.run(
            [
                sys.executable,
                evaluate_script,
                "--sumocfg",
                args.sumocfg,
                "--checkpoint",
                str(latest_model_file),
                "--steps",
                str(args.steps),
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
