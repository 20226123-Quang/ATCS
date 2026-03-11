"""Compare KPIs between Trained ACAC and Fixed-Time Baseline across scenarios."""

import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from acac import (
    load_model_config,
    SinusoidalPositionalEncoding,
    CentralizedCritic,
    AgentHistoryEncoder,
    MacroActor,
    AsyncTrajectoryBuffer,
    SyncTrajectoryBuffer,
    ACACTrainer,
)
from atcs.environment import TrafficEnvironment

_cfg = load_model_config()


def initialize_acac(
    obs_dim, action_dim, min_action, max_action, tls_names, device="cpu"
):
    num_agents = len(tls_names)
    time_encoder = SinusoidalPositionalEncoding(_cfg.model.time_embed_dim).to(device)
    encoders = [
        AgentHistoryEncoder(
            obs_dim, _cfg.model.time_embed_dim, _cfg.model.hidden_dim
        ).to(device)
        for _ in range(num_agents)
    ]
    actors = [
        MacroActor(
            _cfg.model.hidden_dim, action_dim, min_action=0.0, max_action=1.0
        ).to(device)
        for _ in range(num_agents)
    ]
    critic = CentralizedCritic(_cfg.model.hidden_dim, _cfg.model.num_heads).to(device)

    agents_buffer = AsyncTrajectoryBuffer(
        capacity=_cfg.training.buffer_size, num_agents=num_agents
    )
    critic_buffer = SyncTrajectoryBuffer(capacity=_cfg.training.buffer_size)

    all_params = list(critic.parameters())
    for actor in actors:
        all_params += list(actor.parameters())
    for encoder in encoders:
        all_params += list(encoder.parameters())

    opt = torch.optim.Adam(all_params, lr=_cfg.training.actor_lr)

    trainer = ACACTrainer(
        actors=actors,
        encoders=encoders,
        critic=critic,
        agents_buffer=agents_buffer,
        critic_buffer=critic_buffer,
        optimizers={"combined": opt},
        time_encoder=time_encoder,
        tls_names=tls_names,
        device=device,
        gamma=_cfg.training.gamma,
        lam=_cfg.training.lam,
        eps_clip=_cfg.training.eps_clip,
    )
    return trainer


@torch.no_grad()
def run_acac_episode(env, trainer, max_steps):
    obs, reward, done, info = env.reset()
    trainer._reset_hidden()
    t = 0
    step_count = 0

    kpis = {"delay": [], "queue": [], "saturation": []}

    while not done and step_count < max_steps:
        requiring = info.get("intersection_require_action", [])
        action_dict = {}

        for name in requiring:
            i = trainer.tls_index[name]
            z_it = trainer._obs_to_tensor(obs, i).unsqueeze(0)
            eff_range = info.get("effective_action_range", {}).get(
                name, (info["min_green"], info["max_green"])
            )
            z_it = torch.cat(
                [
                    z_it,
                    torch.tensor([eff_range], dtype=torch.float32).to(trainer.device),
                ],
                dim=-1,
            )
            p_it = trainer.time_encoder(t).to(trainer.device).unsqueeze(0)
            h_prev = trainer.hidden_states[i].unsqueeze(0)
            trainer.hidden_states[i] = trainer.encoders[i](z_it, p_it, h_prev).squeeze(
                0
            )
            actor_out, _ = trainer.actors[i].sample(
                trainer.hidden_states[i].unsqueeze(0)
            )
            actor_val = float(actor_out.detach().item())
            action_dict[name] = trainer._scale_action(
                actor_val, eff_range[0], eff_range[1]
            )

        next_obs, reward, done, info = env.step(action_dict)

        avg_delay = -reward[:, :, 0].mean()
        avg_queue = -reward[:, :, 1].mean()
        avg_sat = -reward[:, :, 2].mean()

        kpis["delay"].append(avg_delay)
        kpis["queue"].append(avg_queue)
        kpis["saturation"].append(avg_sat)

        obs = next_obs
        t += info["delta_t"]
        step_count += 1

    return kpis


def run_fixed_time_episode(env, max_steps, fixed_extension=30):
    obs, reward, done, info = env.reset()
    step_count = 0

    kpis = {"delay": [], "queue": [], "saturation": []}

    while not done and step_count < max_steps:
        requiring = info.get("intersection_require_action", [])
        action_dict = {tls_id: fixed_extension for tls_id in requiring}

        next_obs, reward, done, info = env.step(action_dict)

        avg_delay = -reward[:, :, 0].mean()
        avg_queue = -reward[:, :, 1].mean()
        avg_sat = -reward[:, :, 2].mean()

        kpis["delay"].append(avg_delay)
        kpis["queue"].append(avg_queue)
        kpis["saturation"].append(avg_sat)

        obs = next_obs
        step_count += 1

    return kpis


def plot_comparison(
    acac_kpis, fixed_kpis, scenario_name, output_dir, total_steps_count
):
    # Calculate averages over the episode
    acac_avg = [
        np.mean(acac_kpis["delay"]),
        np.mean(acac_kpis["queue"]),
        np.mean(acac_kpis["saturation"]),
    ]
    fixed_avg = [
        np.mean(fixed_kpis["delay"]),
        np.mean(fixed_kpis["queue"]),
        np.mean(fixed_kpis["saturation"]),
    ]

    # Format scenario name (e.g. "crowded_2intersection" -> "Crowded 2Intersection")
    parts = scenario_name.split("_")
    formatted_scenario = " ".join([p.capitalize() for p in parts])

    labels = ["Wait Time (s)", "Queue Length (m)", "Degree of Saturation"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(
        x - width / 2, fixed_avg, width, label="Fixed Time", color="#f08080"
    )
    rects2 = ax.bar(
        x + width / 2, acac_avg, width, label="RL Agent (ACAC)", color="#3cb371"
    )

    ax.set_ylabel("KPI Value")
    ax.set_title(
        f"Comparison: Fixed Time vs RL Agent ({total_steps_count}s)\nScenario: {formatted_scenario}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    output_path = os.path.join(output_dir, f"{scenario_name}_compare_kpi.png")
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved bar plot: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=int, default=600, help="Max steps to simulate for comparison"
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    output_dir = Path("checkpoints/compare_kpi")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_data_dir = Path("/data/EGEN2025/Philosophi/NewWork/SimulationData/Evaluate")

    scenarios = [
        (
            "normal_2intersection",
            base_data_dir / "Normal/2Intersection/config.sumocfg",
            "checkpoints/normal_2intersection/normal_2intersection_checkpoint.pt",
        ),
        (
            "normal_3intersection",
            base_data_dir / "Normal/3Intersection/config.sumocfg",
            "checkpoints/normal_3intersection/normal_3intersection_checkpoint.pt",
        ),
        (
            "normal_4intersection",
            base_data_dir / "Normal/4Intersection/config.sumocfg",
            "checkpoints/normal_4intersection/normal_4intersection_checkpoint.pt",
        ),
        (
            "crowded_2intersection",
            base_data_dir / "Crowded/2Intersection/config.sumocfg",
            "checkpoints/crowded_2intersection/crowded_2intersection_checkpoint.pt",
        ),
        (
            "crowded_3intersection",
            base_data_dir / "Crowded/3Intersection/config.sumocfg",
            "checkpoints/crowded_3intersection/crowded_3intersection_checkpoint.pt",
        ),
        (
            "crowded_4intersection",
            base_data_dir / "Crowded/4Intersection/config.sumocfg",
            "checkpoints/crowded_4intersection/crowded_4intersection_checkpoint.pt",
        ),
    ]

    for scenario_name, sumocfg_path, checkpoint_path in scenarios:
        print("\n=========================================")
        print(f"Evaluating Scenario: {scenario_name}")
        print("=========================================")

        sumocfg_path = str(sumocfg_path)
        checkpoint_path = str(checkpoint_path)

        if not Path(checkpoint_path).exists():
            print(f"[Warning] Checkpoint missing: {checkpoint_path}. Skipping...")
            continue

        env = TrafficEnvironment(sumocfg_path=sumocfg_path, use_gui=False)
        obs, _, _, _ = env.reset()
        obs_dim = obs.shape[1] * obs.shape[2]
        obs_encoder_dim = obs_dim + 2
        tls_names = env.tls_ids
        env.close()

        trainer = initialize_acac(
            obs_dim=obs_encoder_dim,
            action_dim=1,
            min_action=0.0,
            max_action=1.0,
            tls_names=tls_names,
            device=args.device,
        )
        trainer.load_model(checkpoint_path)
        for actor in trainer.actors:
            actor.eval()
        for encoder in trainer.encoders:
            encoder.eval()
        trainer.critic.eval()

        print("Running ACAC Evaluation...")
        env = TrafficEnvironment(sumocfg_path=sumocfg_path, use_gui=False)
        acac_kpis = run_acac_episode(env, trainer, args.steps)
        env.close()

        print("Running Fixed Time Baseline Evaluation...")
        env = TrafficEnvironment(sumocfg_path=sumocfg_path, use_gui=False)
        fixed_kpis = run_fixed_time_episode(env, args.steps)
        env.close()

        plot_comparison(acac_kpis, fixed_kpis, scenario_name, output_dir, args.steps)


if __name__ == "__main__":
    main()
