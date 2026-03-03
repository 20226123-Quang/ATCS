"""Neural network modules for ACAC."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .config_loader import load_model_config

_cfg = load_model_config()
_EPS = _cfg.training.eps
_TANH_CLAMP = 1 - _EPS


class SinusoidalPositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=10000):
		super().__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
		)

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer("pe", pe)

	def forward(self, t):
		if isinstance(t, int):
			return self.pe[t]
		return self.pe[t.long()]


class AgentHistoryEncoder(nn.Module):
	def __init__(self, obs_dim, time_dim, hidden_dim):
		super().__init__()

		self.mlp = nn.Sequential(
			nn.Linear(obs_dim + time_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU()
		)

		self.gru = nn.GRUCell(hidden_dim, hidden_dim)

	def forward(self, z_it, p_it, h_prev):
		"""
		z_it : [obs_dim]
		p_it : [time_dim]
		h_prev : [hidden_dim]
		"""
		x = torch.cat([z_it, p_it], dim=-1)
		x = self.mlp(x)
		h_it = self.gru(x, h_prev)
		return h_it


class MacroActor(nn.Module):
	def __init__(self, hidden_dim, macro_action_dim, min_action, max_action):
		super().__init__()

		self.min_action = min_action
		self.max_action = max_action
		self.macro_action_dim = macro_action_dim

		self.net = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
		)

		self.mean = nn.Linear(hidden_dim, macro_action_dim)
		self.log_std = nn.Linear(hidden_dim, macro_action_dim)
		print(f"MacroActor initialized with min_action: {min_action}, max_action: {max_action}, macro_action_dim: {macro_action_dim}")

	def forward(self, h_it):
		x = self.net(h_it)
		mean = self.mean(x)
		log_std = self.log_std(x).clamp(-20, 20)
		std = log_std.exp().clamp(min=_EPS)
		return mean, std

	def _squash_and_scale(self, raw_action):
		squashed = torch.tanh(raw_action)
		action = self.min_action + (squashed + 1.0) * 0.5 * (
			self.max_action - self.min_action
		)
		return action

	def _unsquash_and_unscale(self, action):
		squashed = (action - self.min_action) / (self.max_action - self.min_action) * 2.0 - 1.0
		squashed = squashed.clamp(-_TANH_CLAMP, _TANH_CLAMP)
		return torch.atanh(squashed)

	def sample(self, h_it):
		"""
		Stochastic action (PPO rollout / training)
		Returns:
			action    : scaled action in [min_action, max_action]
			log_prob  : log π(a|s) with tanh Jacobian correction
		"""
		mean, std = self.forward(h_it)
		dist = Normal(mean, std)
		raw_action = dist.rsample()  # shape: [macro_action_dim] or [T, macro_action_dim]

		log_prob = dist.log_prob(raw_action).sum(-1)
		log_prob -= torch.sum(
			torch.log(1 - torch.tanh(raw_action) ** 2 + _EPS),
			dim=-1
		)
		action = self._squash_and_scale(raw_action)

		return action, log_prob

	def evaluate(self, h_it, action):
		mean, std = self.forward(h_it)
		dist = Normal(mean, std)

		raw_action = self._unsquash_and_unscale(action)  # shape: [T, macro_action_dim]

		log_prob = dist.log_prob(raw_action).sum(-1)
		log_prob -= torch.sum(
			torch.log(1 - torch.tanh(raw_action) ** 2 + _EPS),
			dim=-1
		)

		entropy = dist.entropy().sum(-1)
		return log_prob, entropy

class AgentAttentionAggregator(nn.Module):
	def __init__(self, hidden_dim, num_heads):
		super().__init__()
		self.attn = nn.MultiheadAttention(
			embed_dim=hidden_dim,
			num_heads=num_heads,
			batch_first=True
		)

	def forward(self, h_all):
		attn_out, _ = self.attn(h_all, h_all, h_all)
		h_global = attn_out.mean(dim=1)
		return h_global


class CentralizedCritic(nn.Module):
	def __init__(self, hidden_dim, num_heads):
		super().__init__()

		self.aggregator = AgentAttentionAggregator(
			hidden_dim, num_heads
		)

		self.value_head = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)

	def forward(self, h_all):
		h_global = self.aggregator(h_all)
		value = self.value_head(h_global)
		return value.squeeze(-1)
