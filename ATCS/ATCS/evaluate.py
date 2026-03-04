"""Entry-point script for evaluating the trained ACAC model with SUMO GUI."""

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
					buffer_size=100, actor_lr=1e-4, critic_lr=1e-3, device="cpu"):
	"""
	Khởi tạo toàn bộ ACAC từ config (dùng cho evaluation, buffer nhỏ).
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

	# ---- Buffers (nhỏ, chỉ cần để khởi tạo) ----
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


def print_metrics(metrics: dict) -> None:
	print("\n" + "=" * 50)
	print("         EVALUATION RESULTS")
	print("=" * 50)
	print(f"  Total Reward     : {metrics['total_reward']:.4f}")
	print(f"  Avg Reward/step  : {metrics['avg_reward']:.6f}")
	print(f"  Avg Delay        : {metrics['delay']:.4f} s")
	print(f"  Avg Queue Length : {metrics['queue']:.4f} m")
	print("=" * 50 + "\n")


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate trained ACAC model with SUMO GUI")
	default_cfg = str(
		Path(__file__).resolve().parents[1]
		/ "SimulationData" / "SampleData" / "OneIntersect" / "config_one_car_no_delay.sumocfg"
	)
	parser.add_argument("--sumocfg", default=default_cfg, help="Path to SUMO .sumocfg file")
	parser.add_argument("--checkpoint", default="acac_checkpoint.pt", help="Path to model checkpoint")
	parser.add_argument("--steps", type=int, default=300, help="Max decision steps per episode")
	parser.add_argument("--no-gui", action="store_true", help="Disable SUMO GUI (run headless)")
	parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")
	args = parser.parse_args()

	use_gui = not args.no_gui

	print(f"Loading environment from: {args.sumocfg}")
	print(f"GUI mode: {use_gui}")

	env = TrafficEnvironment(sumocfg_path=args.sumocfg, use_gui=use_gui)

	# Reset để lấy obs_dim và tls_names
	obs, reward, done, info = env.reset()
	obs_dim = obs.shape[1] * obs.shape[2]
	obs_encoder_dim = obs_dim + 2  # +2 for eff_range concat
	tls_names = env.tls_ids
	print(f"Observation dimension : {obs_dim} (encoder input: {obs_encoder_dim})")
	print(f"Traffic lights        : {tls_names}")
	env.close()  # Đóng để trainer khởi tạo lại khi evaluate

	device = torch.device(args.device)
	print(f"Using device          : {device}")

	# Khởi tạo trainer
	trainer = initialize_acac(
		obs_dim=obs_encoder_dim,
		action_dim=1,
		min_action=0.0,
		max_action=1.0,
		tls_names=tls_names,
		buffer_size=100,
		actor_lr=_cfg.training.actor_lr,
		critic_lr=_cfg.training.critic_lr,
		device=device,
	)

	# Load checkpoint
	checkpoint_path = Path(args.checkpoint)
	if not checkpoint_path.exists():
		# Thử tìm trong thư mục hiện tại hoặc thư mục script
		alt_path = Path(__file__).resolve().parent / args.checkpoint
		if alt_path.exists():
			checkpoint_path = alt_path
		else:
			print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
			print("        Please train first with: python train.py")
			return

	print(f"Loading checkpoint    : {checkpoint_path}")
	trainer.load_model(str(checkpoint_path))
	print("Checkpoint loaded successfully.")

	# Set tất cả networks về eval mode
	for actor in trainer.actors:
		actor.eval()
	for encoder in trainer.encoders:
		encoder.eval()
	trainer.critic.eval()

	# Tạo lại env với GUI
	print("\nStarting SUMO simulation with GUI...")
	env = TrafficEnvironment(sumocfg_path=args.sumocfg, use_gui=use_gui)

	with torch.no_grad():
		metrics = trainer.evaluate_model(env, max_steps=args.steps)

	env.close()

	print_metrics(metrics)


if __name__ == "__main__":
	main()
