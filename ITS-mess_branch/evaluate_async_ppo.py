"""
Evaluate trained asynchronous multi-agent PPO models.
"""
import os
import numpy as np
import torch
from traffic.env_wrappers import AsynchronousMultiAgentWrapper, PhaseDurationEnv
from RLAlgorithm.PPO.PPO_continuous import PPOAgent
from traffic.config import SUMO_CONFIG_PATH
from traffic.metrics.metrics_collector import MetricsCollector
import traci
import json


def evaluate_async_ppo(
	model_paths: dict,  # {tls_id: model_path}
	sumo_config=None,
	tls_ids=None,
	min_green=10.0,
	max_green=60.0,
	delta_time=1,
	yellow_time=3,
	max_steps_per_episode=3600,
	reward_fn='wait_time',
	num_episodes=5,
	device='cpu',
use_gui=False,
):
	"""
	Evaluate trained models.
	
	Args:
		model_paths: Dict mapping tls_id to model file path
		tls_ids: List of TLS IDs to evaluate
		num_episodes: Number of evaluation episodes
	"""
	
	if tls_ids is None:
		tls_ids = list(model_paths.keys())
	
	# Create environment
	env = AsynchronousMultiAgentWrapper(
		PhaseDurationEnv(
			sumo_config=sumo_config,
			controlled_tls_index=0,
			min_green=min_green,
			max_green=max_green,
			delta_time=delta_time,
			yellow_time=yellow_time,
			max_steps=max_steps_per_episode,
			reward_fn=reward_fn,
			use_gui=use_gui,
		)
	)
	
	# Get observation dimensions
	obs_dim = env.obs_dim_per_agent
	if obs_dim is None:
		obs_dim = env.env.observation_space.shape[0] // len(tls_ids)
	
	# Load agents
	agents = {}
	for tls in tls_ids:
		if tls in model_paths:
			agent = PPOAgent(
				obs_dim=obs_dim,
				act_dim=1,
				action_low=[min_green],
				action_high=[max_green],
				device=device,
			)
			agent.load_model(model_paths[tls])
			agents[tls] = agent
	
	# Evaluation
	all_rewards = []
	all_wait_times = []
	all_queue_lengths = []
	all_travel_times = []
	best_episode = None
	best_reward = float('-inf')
	best_metrics_data = None
	
	for ep in range(num_episodes):
		# Reset environment and metrics
		obs = env.reset()
		metrics = MetricsCollector()
		metrics.reset()
		episode_rewards = {tls: 0.0 for tls in agents.keys()}
		step_count = 0
		
		while step_count < max_steps_per_episode:
			ready_agents = env.get_agents_ready_for_action()
			callback = lambda current_time, step_idx, dt: metrics.update_metrics(current_time)
			if not ready_agents:
				obs, rewards, dones, infos = env.step({}, metrics_callback=callback)
				for tls in rewards:
					if tls in episode_rewards:
						reward_value = rewards[tls]
						if isinstance(reward_value, dict):
							reward_value = reward_value.get('reward', 0.0)
						episode_rewards[tls] += reward_value
				step_count += 1
				continue
			
			# Get actions from agents
			actions = {}
			for tls in ready_agents:
				if tls in agents:
					action, _, _ = agents[tls].select_action(obs[tls])
					actions[tls] = float(action[0])
			
			# Step environment
			next_obs, rewards, dones, infos = env.step(actions, metrics_callback=callback)
			
			# Update rewards
			for tls in ready_agents:
				if tls in episode_rewards:
					reward_value = rewards.get(tls, 0.0)
					if isinstance(reward_value, dict):
						reward_value = reward_value.get('reward', 0.0)
					episode_rewards[tls] += reward_value
			
			obs = next_obs
			step_count += 1
			
			if any(dones.values()):
				break
		
		total_reward = sum(episode_rewards.values())
		avg_reward = total_reward / len(episode_rewards)
		all_rewards.append(avg_reward)
		# Compute metric summaries
		avg_wait = metrics.get_average_wait_time()
		avg_queue = metrics.get_average_queue_length()
		max_queue = metrics.get_max_queue_length()
		avg_travel = metrics.get_average_travel_time()
		max_travel = metrics.get_max_travel_time()
		min_travel = metrics.get_min_travel_time()
		
		# Check if this is the best episode
		if total_reward > best_reward:
			best_reward = total_reward
			best_episode = ep + 1
			best_metrics_data = {
				'episode': ep+1,
				'total_reward': float(total_reward),
				'avg_reward': float(avg_reward),
				'avg_wait_time': float(avg_wait),
				'avg_queue_length': float(avg_queue),
				'max_queue_length': float(max_queue),
				'avg_travel_time': float(avg_travel),
				'max_travel_time': float(max_travel),
				'min_travel_time': float(min_travel),
				'wait_times_detail': [wt if isinstance(wt, (int, float)) else wt.get('wait_time', 0.0) for wt in metrics.completed_wait_times.copy()],
				'queue_lengths_detail': [ql if isinstance(ql, (int, float)) else ql.get('queue_length', 0.0) for ql in metrics.queue_lengths.copy()],
				'travel_times_detail': [tt if isinstance(tt, (int, float)) else tt.get('travel_time', 0.0) for tt in metrics.completed_travel_times.copy()],
			}
		
		print(f"Episode {ep+1}: Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}")
		# Use MetricsCollector's built-in print helper
		metrics.print_metrics()

		# Collect detailed metrics
		all_wait_times.extend([wt if isinstance(wt, (int, float)) else wt.get('wait_time', 0.0) for wt in metrics.completed_wait_times])
		all_queue_lengths.extend([ql if isinstance(ql, (int, float)) else ql.get('queue_length', 0.0) for ql in metrics.queue_lengths])
		all_travel_times.extend([tt if isinstance(tt, (int, float)) else tt.get('travel_time', 0.0) for tt in metrics.completed_travel_times])
		metrics_data = {
			'episode': ep+1,
			'total_reward': float(total_reward),
			'avg_reward': float(avg_reward),
			'avg_wait_time': float(avg_wait),
			'avg_queue_length': float(avg_queue),
			'max_queue_length': float(max_queue),
			'avg_travel_time': float(avg_travel),
			'max_travel_time': float(max_travel),
			'min_travel_time': float(min_travel),
		}
		with open(os.path.join('evaluation_results', f'metrics_ep{ep+1}.json'), 'w') as mf:
			json.dump(metrics_data, mf, indent=2)
	
	env.close()
	
	# Summary
	mean_reward = np.mean(all_rewards)
	std_reward = np.std(all_rewards)
	print(f"\nEvaluation Results:")
	print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
	if best_episode is not None:
		print(f"Best Episode: {best_episode} with Total Reward: {best_reward:.2f}")
		# Save best episode metrics
		with open('best_episode_metrics.json', 'w') as f:
			json.dump(best_metrics_data, f, indent=2)
		print(f"Best episode metrics saved to best_episode_metrics.json")
	
	# Save aggregated metrics like leading_car
	from datetime import datetime
	metrics_file = "traffic_metrics.json"
	if best_metrics_data is not None:
		aggregated_data = {
			'timestamp': datetime.now().isoformat(),
			'episode': best_metrics_data['episode'],
			'total_reward': best_metrics_data['total_reward'],
			'avg_reward': best_metrics_data['avg_reward'],
			'average_wait_time': best_metrics_data['avg_wait_time'],
			'total_vehicles_waited': len(best_metrics_data['wait_times_detail']),
			'average_queue_length': best_metrics_data['avg_queue_length'],
			'max_queue_length': best_metrics_data['max_queue_length'],
			'queue_measurements': len(best_metrics_data['queue_lengths_detail']),
			'average_travel_time': best_metrics_data['avg_travel_time'],
			'min_travel_time': best_metrics_data['min_travel_time'],
			'max_travel_time': best_metrics_data['max_travel_time'],
			'total_vehicles_traveled': len(best_metrics_data['travel_times_detail']),
			'wait_times_detail': best_metrics_data['wait_times_detail'],
			'queue_lengths_detail': best_metrics_data['queue_lengths_detail'],
			'travel_times_detail': best_metrics_data['travel_times_detail'],
		}
	else:
		# Fallback to aggregated if no best
		aggregated_data = {
			'timestamp': datetime.now().isoformat(),
			'num_episodes': num_episodes,
			'mean_reward': float(mean_reward),
			'std_reward': float(std_reward),
			'all_rewards': [float(r) for r in all_rewards],
			'average_wait_time': float(np.mean(all_wait_times)) if all_wait_times else 0.0,
			'total_vehicles_waited': len(all_wait_times),
			'average_queue_length': float(np.mean(all_queue_lengths)) if all_queue_lengths else 0.0,
			'max_queue_length': float(max(all_queue_lengths)) if all_queue_lengths else 0.0,
			'queue_measurements': len(all_queue_lengths),
			'average_travel_time': float(np.mean(all_travel_times)) if all_travel_times else 0.0,
			'min_travel_time': float(min(all_travel_times)) if all_travel_times else float('inf'),
			'max_travel_time': float(max(all_travel_times)) if all_travel_times else 0.0,
			'total_vehicles_traveled': len(all_travel_times),
			'wait_times_detail': all_wait_times,
			'queue_lengths_detail': all_queue_lengths,
			'travel_times_detail': all_travel_times,
		}
	with open(metrics_file, 'w') as f:
		json.dump(aggregated_data, f, indent=2)
	print(f"\nAggregated metrics saved to {metrics_file}")
	
	return mean_reward, std_reward


if __name__ == '__main__':
	# Example: Evaluate models for J1 and J3
	model_paths = {
		'J1': 'models/ppo_async_J1_ep300.pt',
		'J3': 'models/ppo_async_J3_ep300.pt',
	}
	
	evaluate_async_ppo(
		model_paths=model_paths,
		sumo_config=SUMO_CONFIG_PATH,
		tls_ids=['J1', 'J3'],
		num_episodes=3,
		device='cuda' if torch.cuda.is_available() else 'cpu',
		use_gui=False,
	)