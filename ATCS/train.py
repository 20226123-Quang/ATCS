"""Entry-point script for training the ACAC model."""

import argparse
from pathlib import Path

import torch
from acac import (
	SinusoidalPositionalEncoding, CentralizedCritic, AgentHistoryEncoder, MacroActor,
	AsyncTrajectoryBuffer, SyncTrajectoryBuffer,
	ACACTrainer,
	load_model_config
)
from atcs.environment import TrafficEnvironment

_cfg = load_model_config()


def initialize_acac(obs_dim, action_dim, min_action, max_action, tls_names,
					hidden_dim=128, time_embed_dim=16, num_heads=4,
					buffer_size=1000, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, lam=0.95, eps_clip=0.2, device="cpu"):
	"""
	Khởi tạo toàn bộ ACAC từ config.
	obs_dim  : kích thước obs đã flatten (max_lanes * 5)
	action_dim: 1 (thời gian extend, scalar)
	Returns: ACACTrainer instance
	"""
	num_agents = len(tls_names)

	# ---- Time encoder ----
	time_encoder = SinusoidalPositionalEncoding(time_embed_dim).to(device)

	# ---- Per-agent encoders ----
	encoders = [
		AgentHistoryEncoder(obs_dim, time_embed_dim, hidden_dim).to(device)
		for _ in range(num_agents)
	]

	# ---- Per-agent actors (output [0, 1], trainer scales to effective range) ----
	actors = [
		MacroActor(hidden_dim, action_dim, min_action=0.0, max_action=1.0).to(device)
		for _ in range(num_agents)
	]

	# ---- Centralized critic ----
	critic = CentralizedCritic(hidden_dim, num_heads).to(device)

	# ---- Buffers ----
	agents_buffer = AsyncTrajectoryBuffer(capacity=buffer_size, num_agents=num_agents)
	critic_buffer = SyncTrajectoryBuffer(capacity=buffer_size)

	# ---- Combined Optimizer for BPTT ----
	all_params = list(critic.parameters())
	for actor in actors:
		all_params += list(actor.parameters())
	for encoder in encoders:
		all_params += list(encoder.parameters())
		
	# Mặc định lấy actor_lr (thường bằng critic_lr hoặc gần bằng)
	opt = torch.optim.Adam(all_params, lr=actor_lr)

	optimizers = {
		"combined": opt,
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
		gamma=gamma,
		lam=lam,
		eps_clip=eps_clip,
	)

	return trainer

def main() -> None:
	parser = argparse.ArgumentParser()
	default_cfg = str(Path(__file__).resolve().parents[1] / "SimulationData" / "SampleData" / "OneIntersect" / "config_one_car_delay_120.sumocfg")
	parser.add_argument("--sumocfg", default=default_cfg, help="Path to SUMO .sumocfg file")
	parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
	parser.add_argument("--steps", type=int, default=300, help="Number of decision steps per episode")
	parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
	parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu, cuda, mps)")
	args = parser.parse_args()

	env = TrafficEnvironment(sumocfg_path=args.sumocfg, use_gui=args.gui)

	# Reset environment and get observation dimension
	obs, reward, done, info = env.reset()
	obs_dim = obs.shape[1] * obs.shape[2]  # max_lanes * 5
	obs_encoder_dim = obs_dim + 2  # +2 for eff_range (min_green, max_green) concat
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

	import matplotlib.pyplot as plt
	import subprocess
	import sys

	episode_rewards = []
	for ep in range(args.episodes):
		print(f"\n--- Episode {ep+1}/{args.episodes} ---")
		metrics = trainer.train_episode(env, max_steps=args.steps)
		ep_reward = metrics['reward']
		episode_rewards.append(ep_reward)
		print(f"Reward: {ep_reward:.2f}")
		print(f"Critic Loss: {metrics['critic_loss']:.4f}")
		for i, aloss in metrics['actor_losses'].items():
			print(f"  Actor {i} Loss: {aloss:.4f}")

		# Plot rewards after each episode
		plt.figure(figsize=(10, 6))
		plt.plot(range(1, ep + 2), episode_rewards, marker='o', linestyle='-')
		plt.title("Training Reward over Episodes")
		plt.xlabel("Episode")
		plt.ylabel("Total Reward")
		plt.grid(True)
		plt.savefig("reward_plot.png")
		plt.close()

	print("\nTraining complete. Saving model to acac_checkpoint.pt")
	trainer.save_model("acac_checkpoint.pt")
	env.close()
	print("Saved reward plot to reward_plot.png")

	# ---- Auto-evaluate với GUI sau khi training xong ----
	print("\nLaunching GUI evaluation...")
	evaluate_script = str(Path(__file__).resolve().parent / "evaluate.py")
	subprocess.run([
		sys.executable, evaluate_script,
		"--sumocfg", args.sumocfg,
		"--checkpoint", "acac_checkpoint.pt",
		"--steps", str(args.steps),
	], check=False)

if __name__ == "__main__":
	main()
