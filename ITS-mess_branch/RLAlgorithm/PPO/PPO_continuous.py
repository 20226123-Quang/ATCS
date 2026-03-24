class TrajectoryBuffer:
	def __init__(self):
		self.memory = []

	def store(self, *transition):
		self.memory.append(transition)

	def clear(self):
		self.memory = []

	def get(self):
		return self.memory

	def __len__(self):
		return len(self.memory)
"""
PPO for CartPole Continuous (continuous action space)
- Uses a Gaussian policy for continuous actions
- Compatible with Gymnasium's Pendulum-v1, MountainCarContinuous-v0, ...
- Action output is squashed and scaled to env.action_space
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
	def __init__(self, obs_dim, act_dim, action_low, action_high, hidden_size=64):
		super().__init__()
		self.actor = nn.Sequential(
			nn.Linear(obs_dim, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, act_dim)
		)
		self.log_std = nn.Parameter(torch.zeros(act_dim))
		self.critic = nn.Sequential(
			nn.Linear(obs_dim, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, 1)
		)
		self.register_buffer('action_low', torch.tensor(action_low, dtype=torch.float32))
		self.register_buffer('action_high', torch.tensor(action_high, dtype=torch.float32))

	def forward(self, x):
		mu = self.actor(x)
		std = torch.exp(self.log_std)
		value = self.critic(x).squeeze(-1)
		return mu, std, value

	def act(self, x):
		mu, std, value = self.forward(x)
		print(f"DEBUG: ActorCritic.act() mu={mu}, std={std}, value={value}")
		action_low = self.action_low.to(x.device)
		action_high = self.action_high.to(x.device)
		dist = torch.distributions.Normal(mu, std)
		action = dist.rsample()  # for reparameterization
		action_tanh = torch.tanh(action)
		action_scaled = action_low + (action_tanh + 1) * 0.5 * (action_high - action_low)
		# Clamp action to valid range
		action_scaled = torch.clamp(action_scaled, action_low, action_high)
		# Correct tanh squash for logprob
		logprob = dist.log_prob(action).sum(-1)
		logprob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(-1)
		return action_scaled.detach().cpu().numpy().astype(np.float32), logprob.detach().cpu().numpy(), value.detach().cpu().numpy()

	def evaluate_actions(self, x, actions):
		mu, std, value = self.forward(x)
		#print(f"DEBUG: ActorCritic.evaluate_actions() mu={mu}, std={std}, value={value}")
		#print(f"x shape: {x.shape}, actions shape: {actions.shape}")
		action_low = self.action_low.to(x.device)
		action_high = self.action_high.to(x.device)
		# Inverse scale and atanh for actions
		action_tanh = 2 * (torch.tensor(actions, device=x.device, dtype=torch.float32) - action_low) / (action_high - action_low) - 1
		action_tanh = torch.clamp(action_tanh, -0.999999, 0.999999)
		action_raw = torch.atanh(action_tanh)
		dist = torch.distributions.Normal(mu, std)
		logprobs = dist.log_prob(action_raw).sum(-1)
		logprobs -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(-1)
		entropy = dist.entropy().sum(-1)
		return logprobs, value, entropy

class PPOAgent:

	def __init__(self, obs_dim, act_dim, action_low, action_high, lr=1e-4, gamma=0.99, lam=0.95, clip_ratio=0.12, epochs=10, batch_size=4096, minibatch_size=128, entropy_coef=0.005, device="cpu"):
		self.gamma = gamma
		self.lam = lam
		self.clip_ratio = clip_ratio
		self.epochs = epochs
		self.batch_size = batch_size
		self.minibatch_size = minibatch_size
		self.entropy_coef = entropy_coef
		self.device = torch.device(device)
		self.net = ActorCritic(obs_dim, act_dim, action_low, action_high).to(self.device)
		self.opt = optim.Adam(self.net.parameters(), lr=lr)
		self.buffer = TrajectoryBuffer()


	def select_action(self, state):
		state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
		action, logprob, value = self.net.act(state)
		return action[0], logprob[0], value[0]


	def store(self, *transition):
		self.buffer.store(*transition)

	def finish_path(self, last_value=0, buffer=None):
		if buffer is None:
			raise ValueError("Buffer must be provided explicitly.")
		memory = buffer.get()
		states, actions, logprobs, rewards, dones, values = zip(*memory)
		#print("-*" * 20)
		#print(states, actions, logprobs, rewards, dones, values)
		rewards = np.array(rewards, dtype=np.float32)
		values = np.array(values + (last_value,), dtype=np.float32)
		dones = np.array(dones, dtype=np.float32)
		advs = np.zeros_like(rewards)
		lastgaelam = 0
		for t in reversed(range(len(rewards))):
			nonterminal = 1.0 - dones[t]
			delta = rewards[t] + self.gamma * values[t+1] * nonterminal - values[t]
			lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
			advs[t] = lastgaelam
		returns = advs + values[:-1]
		#print("-" * 30)
		#print(states, actions, logprobs, returns, advs)
		batch = dict(
			states=np.array(states, dtype=np.float32),
			actions=np.array(actions, dtype=np.float32),
			logprobs=np.array(logprobs, dtype=np.float32),
			returns=returns,
			advantages=(advs - advs.mean()) / (advs.std() + 1e-8)
		)
		buffer.clear()
		return batch

	def update(self, batch):
		states = torch.tensor(batch['states'], dtype=torch.float32, device=self.device)
		actions = torch.tensor(batch['actions'], dtype=torch.float32, device=self.device)
		old_logprobs = torch.tensor(batch['logprobs'], dtype=torch.float32, device=self.device)
		returns = torch.tensor(batch['returns'], dtype=torch.float32, device=self.device)
		advantages = torch.tensor(batch['advantages'], dtype=torch.float32, device=self.device)
		n = states.size(0)
		idxs = np.arange(n)
		print(f"Updating PPO with {n} samples.")
		for _ in range(self.epochs):
			np.random.shuffle(idxs)
			for start in range(0, n, self.minibatch_size):
				end = start + self.minibatch_size
				mb_idx = idxs[start:end]
				mb_states = states[mb_idx]
				mb_actions = actions[mb_idx]
				mb_old_logprobs = old_logprobs[mb_idx]
				mb_returns = returns[mb_idx]
				mb_advantages = advantages[mb_idx]
				logprobs, values, entropy = self.net.evaluate_actions(mb_states, mb_actions)
				ratio = (logprobs - mb_old_logprobs).exp()
				surr1 = ratio * mb_advantages
				surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
				policy_loss = -torch.min(surr1, surr2).mean()
				value_loss = nn.functional.mse_loss(values, mb_returns)
				entropy_loss = -entropy.mean()
				loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss
				self.opt.zero_grad()
				loss.backward()
				self.opt.step()
				
	def save_model(self, path):
		"""Lưu trọng số model vào file."""
		torch.save(self.net.state_dict(), path)

	def load_model(self, path, map_location=None):
		"""Tải trọng số model từ file."""
		state_dict = torch.load(path, map_location=map_location or self.device)
		self.net.load_state_dict(state_dict)
