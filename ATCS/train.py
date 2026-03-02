"""Entry-point script for training the ACAC model."""

import argparse
from pathlib import Path

import torch
from model import (
	SinusoidalPositionalEncoding, CentralizedCritic, AgentHistoryEncoder, MacroActor,
	AsyncTrajectoryBuffer, SyncTrajectoryBuffer,
	ACACTrainer,
	load_model_config
)
from atcs.environment import TrafficEnvironment

_cfg = load_model_config()


def initialize_acac(obs_dim, action_dim, min_action, max_action, tls_names,
					hidden_dim=128, time_embed_dim=16, num_heads=4,
					buffer_size=1000, actor_lr=1e-4, critic_lr=1e-3, device="cpu"):
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

	# ---- Optimizers ----
	actor_optimizers = [
		torch.optim.Adam(actor.parameters(), lr=actor_lr)
		for actor in actors
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
	default_cfg = str(Path(__file__).resolve().parents[1] / "SimulationData" / "SampleData" / "SimpleRoute" / "config.sumocfg")
	parser.add_argument("--sumocfg", default=default_cfg, help="Path to SUMO .sumocfg file")
	parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
	parser.add_argument("--steps", type=int, default=1000, help="Number of decision steps per episode")
	parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
	parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu, cuda, mps)")
	args = parser.parse_args()

	env = TrafficEnvironment(sumocfg_path=args.sumocfg, use_gui=args.gui)

	# Reset environment and get observation dimension
	obs, reward, done, info = env.reset()
	obs_dim = obs.shape[1] * obs.shape[2]
	print(f"Observation dimension: {obs_dim}")

	device = torch.device(args.device)
	print(f"Using device: {device}")

	trainer = initialize_acac(
		obs_dim=obs_dim,
		action_dim=1,
		min_action=0.0,
		max_action=1.0,
		tls_names=env.tls_ids,
		buffer_size=_cfg.training.buffer_size,
		actor_lr=_cfg.training.actor_lr,
		critic_lr=_cfg.training.critic_lr,
		device=device
	)

	import matplotlib.pyplot as plt

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

		print("\nTraining complete. Saving model to acac_checkpoint.pt")
		trainer.save_model("acac_checkpoint.pt")
		env.close()

		# Plot rewards
		plt.figure(figsize=(10, 6))
		plt.plot(range(1, ep + 2), episode_rewards, marker='o', linestyle='-')
		plt.title("Training Reward over Episodes")
		plt.xlabel("Episode")
		plt.ylabel("Total Reward")
		plt.grid(True)
		plt.savefig("reward_plot.png")
		print("Saved reward plot to reward_plot.png")
		# plt.show()
		plt.close()

if __name__ == "__main__":
	main()
