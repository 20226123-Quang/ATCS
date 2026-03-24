"""
Train asynchronous multi-agent PPO for traffic light control.

Each agent (TLS) predicts green duration when its current phase expires.
Uses AsynchronousMultiAgentWrapper for environment interaction.
"""
import os
import time
import numpy as np
import torch
import json
from typing import Dict, List

from traffic.env_wrappers import AsynchronousMultiAgentWrapper
from traffic.environment import SumoTrafficEnv
from RLAlgorithm.PPO.PPO_continuous import PPOAgent
from traffic.config import SUMO_CONFIG_PATH


class CentralizedPPOTrainer:
	"""
	Trainer for asynchronous multi-agent PPO.
	Each agent has its own PPO model and buffer.
	"""
		
	def __init__(
		self,
		env,
		agent: Dict[str, PPOAgent],
		device='cpu',
		max_steps_per_episode=3600,
	):
		self.env = env
		self.agent = agent
		self.device = device
		self.max_steps_per_episode = max_steps_per_episode
		
	def get_tls_one_hot(self, tls_id: str) -> np.ndarray:
		"""
		Get one-hot encoding of traffic light state.
		Args:
			tls_id: Traffic light ID
		Returns:
			one_hot: One-hot encoded numpy array
		"""
		idx = self.env.tls_ids.index(tls_id)
		num_tls = len(self.env.tls_ids)
		one_hot = np.zeros(num_tls, dtype=np.float32);
		one_hot[idx] = 1.0
		return one_hot

	def train_episode(self) -> Dict[str, float]:
		"""
		Run one training episode.
		
		Returns:
			episode_rewards: Dict of total rewards per agent
		"""
		obs = self.env.reset()
		episode_rewards = {tls: 0.0 for tls in self.env.tls_ids}
		step_count = 0
		
		while step_count < self.max_steps_per_episode:
			# Get agents ready for action
			ready_agents = self.env.get_agents_ready_for_action()
			
			if not ready_agents:
				# No agents ready, step environment without actions
				obs, rewards, dones, infos = self.env.step({})
				for tls in rewards:
					episode_rewards[tls] += rewards[tls]
				step_count += 1
				continue
			
			# Collect actions from ready agents
			actions = {}
			# track buffer indices for each stored transition so we can update rewards correctly
			indices = {}
			for tls in ready_agents:
				tls_one_hot = self.get_tls_one_hot(tls)
				# Combine observation with one-hot encoding
				obs_input = np.concatenate([obs[tls], tls_one_hot], axis=0)
				action, logprob, value = self.agent.select_action(obs_input)
				actions[tls] = float(action[0])  # Duration is scalar
			
				# Store transition (will finish path later) and record its index in buffer
				self.agent.store(obs_input, action, logprob, 0, False, value)  # Reward added later
				indices[tls] = len(self.agent.buffer.memory) - 1
			
			# Step environment with actions
			next_obs, rewards, dones, infos = self.env.step(actions)
			
			# Update rewards and store for ready agents
			for tls in ready_agents:
				reward = rewards.get(tls, 0.0)
				episode_rewards[tls] += reward
				# Update the stored transition at the saved index
				idx = indices.get(tls)
				if idx is not None and idx < len(self.agent.buffer.memory):
					tr = list(self.agent.buffer.memory[idx])
					tr[3] = reward  # reward index
					self.agent.buffer.memory[idx] = tuple(tr)
			
			obs = next_obs
			step_count += 1
			
			# Check if episode done
			if any(dones.values()):
				break
		
		# Finish paths for all agents
		if len(self.agent.buffer) > 0:
			# Bootstrap with 0 (or could use value estimate)
			batch = self.agent.finish_path(last_value=0, buffer=self.agent.buffer)
			self.agent.update(batch)

		return episode_rewards


def create_multi_agent_env(
	sumo_config=None,
	min_green=10.0,
	max_green=60.0,
	delta_time=1,
	yellow_time=3,
	max_steps=3600,
	reward_fn='wait_time',
	use_gui=False,
):
	"""
	Create asynchronous multi-agent environment.
	
	For now, uses PhaseDurationEnv as base (single agent) and wraps it.
	TODO: Implement true multi-agent SumoTrafficEnv.
	"""
	# Temporary: Use single PhaseDurationEnv wrapped for multi-agent
	# In future, replace with true multi-agent env
	base_env = SumoTrafficEnv(
		sumo_config=sumo_config,
		delta_time=delta_time,
		yellow_time=yellow_time,
		max_steps=max_steps,
		reward_fn=reward_fn,
		use_gui=use_gui,
	)
	
	# Wrap for asynchronous multi-agent
	env = AsynchronousMultiAgentWrapper(base_env)
	return env


def train_async_ppo(
	config_path='ppo_config.json',
	sumo_config=None,
	save_dir='models',
):
	"""
	Main training function for asynchronous multi-agent PPO.
	
	Args:
		config_path: Path to JSON config file
		sumo_config: Override SUMO config path
		save_dir: Directory to save models
	"""
	
	# Load config
	with open(config_path, 'r') as f:
		config = json.load(f)
	
	ppo_config = config['ppo']
	env_config = config['env']
	training_config = config['training']
	tls_ids = config['tls_ids']
	
	if sumo_config is None:
		sumo_config = SUMO_CONFIG_PATH
	
	# Determine device
	device = training_config['device']
	if device == 'auto':
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
	env = create_multi_agent_env(
		sumo_config=sumo_config,
		min_green=env_config['min_green'],
		max_green=env_config['max_green'],
		delta_time=env_config['delta_time'],
		yellow_time=env_config['yellow_time'],
		max_steps=env_config['max_steps_per_episode'],
		reward_fn=env_config['reward_fn'],
		use_gui=False,
	)
	
	# Get observation dimensions
	obs_dim = env.obs_dim_per_agent + len(env.tls_ids)
	if obs_dim is None:
		# Fallback: assume from base env
		obs_dim = env.env.observation_space.shape[0] // len(tls_ids)
	
	# Create PPO agents for each TLS
	agent = PPOAgent(
		obs_dim=obs_dim,
		act_dim=1,  # Duration scalar
		action_low=[env_config['min_green']],
		action_high=[env_config['max_green']],
		lr=ppo_config.get('lr', 1e-4),
		gamma=ppo_config.get('gamma', 0.99),
		lam=ppo_config.get('lam', 0.95),
		clip_ratio=ppo_config.get('clip_ratio', 0.2),
		epochs=ppo_config.get('epochs', 10),
		batch_size=ppo_config.get('batch_size', 64),
		minibatch_size=ppo_config.get('minibatch_size', 16),
		entropy_coef=ppo_config.get('entropy_coef', 0.0),
		device=device,
	)
	
	# Create trainer
	trainer = CentralizedPPOTrainer(
		env=env,
		agent=agent,
		device=device,
		max_steps_per_episode=env_config['max_steps_per_episode'],
	)
	
	# Training loop
	os.makedirs(save_dir, exist_ok=True)
	
	for ep in range(training_config['episodes']):
		start_time = time.time()
		
		# Train one episode
		episode_rewards = trainer.train_episode()
		
		# Log progress
		total_reward = sum(episode_rewards.values())
		avg_reward = total_reward / len(episode_rewards)
		duration = time.time() - start_time
		
		print(f"Episode {ep+1}/{training_config['episodes']} | Total Reward: {total_reward:.2f} | Avg Reward: {avg_reward:.2f} | Time: {duration:.2f}s")
		
		# Save models periodically
		if (ep + 1) % training_config['save_interval'] == 0:
			model_path = os.path.join(save_dir, f'ppo_centralized_ep{ep+1}.pt')
			agent.save_model(model_path)
			print(f"Models saved at episode {ep+1}")
	
	# Final save
	model_path = os.path.join(save_dir, f'ppo_centralized_final.pt')
	agent.save_model(model_path)
	
	env.close()
	print("Training completed!")


if __name__ == '__main__':
	# Example usage
	train_async_ppo(
		config_path='ppo_config.json',
	)