import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

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
	num_agents = len(tls_names)
	time_encoder = SinusoidalPositionalEncoding(time_embed_dim).to(device)
	encoders = [AgentHistoryEncoder(obs_dim, time_embed_dim, hidden_dim).to(device) for _ in range(num_agents)]
	actors = [MacroActor(hidden_dim, action_dim, min_action=0.0, max_action=1.0).to(device) for _ in range(num_agents)]
	critic = CentralizedCritic(hidden_dim, num_heads).to(device)
	agents_buffer = AsyncTrajectoryBuffer(capacity=buffer_size, num_agents=num_agents)
	critic_buffer = SyncTrajectoryBuffer(capacity=buffer_size)
	actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=actor_lr) for actor in actors]
	critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
	optimizers = {"actor": actor_optimizers, "critic": critic_optimizer}
	
	trainer = ACACTrainer(
		actors=actors, encoders=encoders, critic=critic,
		agents_buffer=agents_buffer, critic_buffer=critic_buffer,
		optimizers=optimizers, time_encoder=time_encoder,
		tls_names=tls_names, device=device
	)
	return trainer

checkpoints = [
	("crowded_2intersection", "../SimulationData/Evaluate/Crowded/2Intersection/config.sumocfg"),
	("crowded_3intersection", "../SimulationData/Evaluate/Crowded/3Intersection/config.sumocfg"),
	("crowded_4intersection", "../SimulationData/Evaluate/Crowded/4Intersection/config.sumocfg"),
	("normal_2intersection", "../SimulationData/Evaluate/Normal/2Intersection/config.sumocfg"),
	("normal_3intersection", "../SimulationData/Evaluate/Normal/3Intersection/config.sumocfg"),
	("normal_4intersection", "../SimulationData/Evaluate/Normal/4Intersection/config.sumocfg")
]

results = []

print("Starting evaluations...")
for name, cfg_path in checkpoints:
	ckpt_path = f"checkpoint/{name}_checkpoint.pt"
	csv_path = f"checkpoint/{name}_training_log.csv"
	abs_cfg = os.path.abspath(cfg_path)
	
	print(f"\\nEvaluating {name}...")
	
	try:
		env = TrafficEnvironment(sumocfg_path=abs_cfg, use_gui=False)
		obs, reward, done, info = env.reset()
		obs_dim = obs.shape[1] * obs.shape[2]
		obs_encoder_dim = obs_dim + 2
		tls_names = env.tls_ids
		env.close()
		
		device = torch.device("cpu")
		trainer = initialize_acac(
			obs_dim=obs_encoder_dim, action_dim=1, min_action=0.0, max_action=1.0,
			tls_names=tls_names, buffer_size=100, actor_lr=_cfg.training.actor_lr,
			critic_lr=_cfg.training.critic_lr, device=device
		)
		
		trainer.load_model(ckpt_path)
		for actor in trainer.actors: actor.eval()
		for encoder in trainer.encoders: encoder.eval()
		trainer.critic.eval()
		
		env = TrafficEnvironment(sumocfg_path=abs_cfg, use_gui=False)
		# Changed to large max_steps to let env run until done, which matches typical eval
		with torch.no_grad():
			metrics = trainer.evaluate_model(env, max_steps=3600)
		env.close()
		
		results.append({
			"Scenario": name.replace('_', ' ').title(),
			"Wait Time (s)": metrics["delay"],
			"Queue Length (m)": metrics["queue"],
			"Degree of Saturation": metrics["saturation"]
		})
		print(f"Metrics: {metrics}")
	except Exception as e:
		print(f"Error evaluating {name}: {e}")

# Save KPI plots
df_kpi = pd.DataFrame(results)
df_kpi.to_csv("checkpoint/kpi_evaluation_results.csv", index=False)
print("\\nKPI Results:")
print(df_kpi)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
scenarios = df_kpi["Scenario"]

axes[0].bar(scenarios, df_kpi["Wait Time (s)"], color='skyblue')
axes[0].set_title("Wait Time (Delay)")
axes[0].set_ylabel("Seconds")
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(scenarios, df_kpi["Queue Length (m)"], color='lightgreen')
axes[1].set_title("Queue Length")
axes[1].set_ylabel("Meters")
axes[1].tick_params(axis='x', rotation=45)

axes[2].bar(scenarios, df_kpi["Degree of Saturation"], color='salmon')
axes[2].set_title("Degree of Saturation")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("checkpoint/kpi_comparison_plot.png")
print("Saved KPI plots to checkpoint/kpi_comparison_plot.png")

# Plot CPU Usage
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
axes2 = axes2.flatten()

for i, (name, _) in enumerate(checkpoints):
	csv_path = f"checkpoint/{name}_training_log.csv"
	try:
		df_train = pd.read_csv(csv_path)
		episodes = df_train["Episode"]
		cpu = df_train["CPU_Percent"]
		axes2[i].plot(episodes, cpu, color='orange')
		axes2[i].set_title(f"CPU Usage: {name.replace('_', ' ').title()}")
		axes2[i].set_xlabel("Episode")
		axes2[i].set_ylabel("CPU (%)")
		axes2[i].grid(True)
	except Exception as e:
		print(f"Could not plot CPU for {name}: {e}")

plt.tight_layout()
plt.savefig("checkpoint/cpu_usage_plot.png")
print("Saved CPU Usage plots to checkpoint/cpu_usage_plot.png")

