import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _to_tensor(x, dtype=torch.float32):
	if isinstance(x, np.ndarray):
		return torch.from_numpy(x).to(dtype)
	return torch.tensor(x, dtype=dtype)


class ActorCritic(nn.Module):
	def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 128):
		super().__init__()
		self.pi_body = nn.Sequential(
			nn.Linear(obs_dim, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, hidden_size),
			nn.Tanh(),
		)
		self.v_body = nn.Sequential(
			nn.Linear(obs_dim, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, hidden_size),
			nn.Tanh(),
		)
		self.pi_head = nn.Linear(hidden_size, act_dim)
		self.v_head = nn.Linear(hidden_size, 1)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		# Handle both [batch, obs_dim] and [batch, steps, obs_dim] formats
		original_shape = x.shape
		if len(x.shape) == 3:
			# [batch, steps, obs_dim] -> [batch * steps, obs_dim]
			batch_size, steps_size, obs_dim = x.shape
			x = x.reshape(batch_size * steps_size, obs_dim)
			reshaped = True
		else:
			reshaped = False
		
		pi_h = self.pi_body(x)
		v_h = self.v_body(x)
		logits = self.pi_head(pi_h)
		value = self.v_head(v_h).squeeze(-1)
		
		# Reshape back if needed
		if reshaped:
			logits = logits.reshape(batch_size, steps_size, -1)
			value = value.reshape(batch_size, steps_size)
		
		return logits, value

	def act(self, x: torch.Tensor) -> Tuple[int, float, float]:
		logits, value = self.forward(x)
		dist = torch.distributions.Categorical(logits=logits)
		action = dist.sample()
		logprob = dist.log_prob(action)
		return int(action.item()), float(logprob.item()), float(value.item())

	def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		# Handle both [batch, obs_dim] and [batch, steps, obs_dim] formats
		original_shape = x.shape
		if len(x.shape) == 3:
			# [batch, steps, obs_dim] -> [batch * steps, obs_dim]
			batch_size, steps_size, obs_dim = x.shape
			x = x.reshape(batch_size * steps_size, obs_dim)
			actions = actions.reshape(batch_size * steps_size)
			reshaped = True
		else:
			reshaped = False
		
		logits, value = self.forward(x)
		dist = torch.distributions.Categorical(logits=logits)
		logprobs = dist.log_prob(actions)
		entropy = dist.entropy()
		
		# Reshape back if needed
		if reshaped:
			logprobs = logprobs.reshape(batch_size, steps_size)
			value = value.reshape(batch_size, steps_size)
			entropy = entropy.reshape(batch_size, steps_size)
		
		return logprobs, value, entropy


class RolloutBuffer:
	def __init__(self, max_size: int = 2048, max_episode_length: int = 500):
		self.episodes = []  # List of episode dicts
		self.max_size = max_size
		self.max_episode_length = max_episode_length

	def add_episode(self, states, actions, logprobs, rewards, dones, values):
		"""Add a complete episode to buffer, padded to max_episode_length."""
		ep_len = len(states)
		
		# Pad episode to max_episode_length if needed
		if ep_len < self.max_episode_length:
			pad_len = self.max_episode_length - ep_len
			
			# Pad states (assuming states are arrays)
			states = states + [np.zeros_like(states[0])] * pad_len
			actions = actions + [0] * pad_len
			logprobs = logprobs + [0.0] * pad_len
			rewards = rewards + [0.0] * pad_len
			dones = dones + [1.0] * pad_len  # Mark padded steps as done
			values = values + [0.0] * pad_len
		elif ep_len > self.max_episode_length:
			# Truncate if episode is too long
			states = states[:self.max_episode_length]
			actions = actions[:self.max_episode_length]
			logprobs = logprobs[:self.max_episode_length]
			rewards = rewards[:self.max_episode_length]
			dones = dones[:self.max_episode_length]
			values = values[:self.max_episode_length]
		
		episode = {
			'states': states,
			'actions': actions,
			'logprobs': logprobs,
			'rewards': rewards,
			'dones': dones,
			'values': values,
			'ep_len': min(ep_len, self.max_episode_length),  # Store actual length
		}
		self.episodes.append(episode)
		if len(self.episodes) > self.max_size:
			self.episodes.pop(0)

	def get_minibatch_episodes(self, minibatch_size: int):
		"""Yield minibatches with data stacked as [batch_size, steps_size, ...]"""
		num_episodes = len(self.episodes)
		if num_episodes == 0:
			return
		indices = np.arange(num_episodes)
		np.random.shuffle(indices)
		batch_indices = indices[:min(minibatch_size, num_episodes)]
		batch_episodes = [self.episodes[idx] for idx in batch_indices]

		batch = {
			'states': np.array([ep['states'] for ep in batch_episodes], dtype=np.float32),
			'actions': np.array([ep['actions'] for ep in batch_episodes], dtype=np.int64),
			'logprobs': np.array([ep['logprobs'] for ep in batch_episodes], dtype=np.float32),
			'rewards': np.array([ep['rewards'] for ep in batch_episodes], dtype=np.float32),
			'dones': np.array([ep['dones'] for ep in batch_episodes], dtype=np.float32),
			'values': np.array([ep['values'] for ep in batch_episodes], dtype=np.float32),
		}
		return batch
		
	def clear(self):
		self.episodes = []


@dataclass
class PPOConfig:
	gamma: float = 0.99
	lam: float = 0.95
	clip_ratio: float = 0.2
	lr: float = 3e-4
	train_epochs: int = 4
	batch_size: int = 256
	minibatch_size: int = 64
	entropy_coef: float = 0.0
	value_coef: float = 0.5
	max_grad_norm: float = 0.5
	device: str = "cuda:0"


class PPOAgent:
	def __init__(self, obs_dim: int, act_dim: int, config: PPOConfig = PPOConfig()):
		self.config = config
		self.device = torch.device(config.device)
		self.net = ActorCritic(obs_dim, act_dim).to(self.device)
		self.opt = optim.Adam(self.net.parameters(), lr=config.lr)
		self.buffer = RolloutBuffer()

	def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
		s = _to_tensor(state).to(self.device)
		action, logprob, value = self.net.act(s)
		return action, logprob, value

	def store(self, state, action, logprob, reward, done, value):
		self.buffer.add(state, action, logprob, reward, done, value)

	def store_episode(self, states, actions, logprobs, rewards, dones, values):
		"""Store a complete episode."""
		self.buffer.add_episode(states, actions, logprobs, rewards, dones, values)

	def compute_returns_advantages(self, batch, gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Compute returns and advantages for a batch of episodes on GPU.
		
		Args:
			batch: Dict with batched arrays shape [batch_size, steps_size]
			gamma: Discount factor
			lam: GAE lambda
			
		Returns:
			returns: Returns tensor on GPU, shape [batch_size, steps_size]
			advantages: Normalized advantages tensor on GPU, shape [batch_size, steps_size]
		"""
		# Get batched data [batch_size, steps_size]
		rewards = batch['rewards']  # already numpy array
		dones = batch['dones']
		values = batch['values']
		
		num_episodes = rewards.shape[0]
		ep_length = rewards.shape[1]
		
		# Create CUDA tensors directly
		rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
		dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
		values = torch.tensor(values, dtype=torch.float32, device=self.device)
		
		# Add next value (bootstrap) for each episode
		next_values = torch.zeros((num_episodes, 1), dtype=torch.float32, device=self.device)
		values_with_next = torch.cat([values, next_values], dim=1)
		
		# Compute GAE on GPU
		advantages = torch.zeros_like(rewards, device=self.device)
		gae = torch.zeros(num_episodes, dtype=torch.float32, device=self.device)
		
		for t in reversed(range(ep_length)):
			delta = rewards[:, t] + gamma * values_with_next[:, t + 1] * (1.0 - dones[:, t]) - values[:, t]
			gae = delta + gamma * lam * (1.0 - dones[:, t]) * gae
			advantages[:, t] = gae
		
		returns = advantages + values
		
		# Normalize advantages on GPU
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
		
		return returns, advantages

	def update(self):
		"""Update using minibatches of episodes on GPU."""
		cfg = self.config

		# Iterate through minibatches of episodes
		for epoch in range(cfg.train_epochs):
			batch = self.buffer.get_minibatch_episodes(cfg.minibatch_size)
			returns, advantages = self.compute_returns_advantages(batch, cfg.gamma, cfg.lam)
			
			# Convert to tensors on GPU, then flatten for training
			states = torch.tensor(batch['states'], dtype=torch.float32, device=self.device)
			actions = torch.tensor(batch['actions'], dtype=torch.int64, device=self.device)
			logprobs = torch.tensor(batch['logprobs'], dtype=torch.float32, device=self.device)
			
			# Forward pass on GPU
			new_logprobs, values, entropy = self.net.evaluate_actions(states, actions)

			#print(new_logprobs.shape, logprobs.shape, advantages.shape, returns.shape, values.shape)

			ratio = (new_logprobs - logprobs).exp()
			surr1 = ratio * advantages
			surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantages
			policy_loss = -torch.min(surr1, surr2)

			value_loss = nn.functional.mse_loss(values, returns, reduction='none')
			entropy_loss = -entropy

			#print(policy_loss.shape, value_loss.shape, entropy_loss.shape)

			loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss
			loss = loss.mean()
			# Backward pass
			self.opt.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
			self.opt.step()
		self.buffer.clear()

	def save(self, path: str):
		torch.save(self.net.state_dict(), path)

	def load(self, path: str):
		self.net.load_state_dict(torch.load(path, map_location=self.device))

