"""ACAC Trainer: rollout collection, advantage computation, actor/critic updates."""

import torch
import torch.nn.functional as F
from .config_loader import load_model_config

_cfg = load_model_config()


class ACACTrainer:
	def __init__(
		self,
		actors,
		encoders,
		time_encoder,
		critic,
		agents_buffer,
		critic_buffer,
		optimizers,
		tls_names,
		gamma=0.99,
		lam=0.95,
		eps_clip=0.2,
		device="cpu"
	):
		"""
		actors        : list[MacroActor]
		encoders      : list[AgentHistoryEncoder]
		critic        : CentralizedCritic
		agents_buffer : AsyncTrajectoryBuffer
		critic_buffer : SyncTrajectoryBuffer
		optimizers    : dict {"actor": list[Optimizer], "critic": Optimizer}
		tls_names     : list[str] — tên các TLS theo thứ tự index trong obs
		"""
		# ===== Core components =====
		self.actors = actors
		self.encoders = encoders
		self.critic = critic
		self.agents_buffer = agents_buffer
		self.critic_buffer = critic_buffer
		self.optimizers = optimizers
		self.time_encoder = time_encoder

		# ===== TLS mapping =====
		self.tls_names = tls_names
		self.tls_index = {name: i for i, name in enumerate(tls_names)}

		# ===== Hyper-parameters =====
		self.gamma = gamma
		self.lam = lam
		self.eps_clip = eps_clip
		self.device = device

		# ===== Book-keeping =====
		self.num_agents = len(actors)

		# Hidden states — infer hidden_dim from GRUCell
		hidden_dim = encoders[0].gru.hidden_size
		self.hidden_states = [
			torch.zeros(hidden_dim, device=device)
			for _ in range(self.num_agents)
		]

		assert len(optimizers["actor"]) == self.num_agents, \
			"Number of actor optimizers must match number of agents"

	# =====================================================================
	# Internal helpers
	# =====================================================================

	def _reset_hidden(self):
		for h in self.hidden_states:
			h.zero_()

	def _get_current_global_state(self):
		"""Stack hidden states: [num_agents, hidden_dim]"""
		return torch.stack([h.detach() for h in self.hidden_states])

	def _obs_to_tensor(self, obs, tls_index):
		"""Flatten obs[tls_index] từ [max_lanes, 5] → [max_lanes * 5]"""
		return torch.tensor(
			obs[tls_index].flatten(), dtype=torch.float32, device=self.device
		)

	def _reward_scalar(self, reward, tls_index):
		"""Aggregate reward[tls_index] [max_lanes, 2] → scalar (âm vì minimize)"""
		return -float(reward[tls_index].mean())

	def _global_reward_scalar(self, reward):
		"""Global reward scalar từ toàn bộ reward [n, max_lanes, 2]"""
		return -float(reward.mean())

	def _scale_action(self, actor_output, min_ext, max_ext):
		"""
		Scale actor output từ [0, 1] sang [min_ext, max_ext].
		actor_output : float trong [0, 1] (output của MacroActor với min_action=0, max_action=1)
		Returns: float extension seconds trong [min_ext, max_ext]
		"""
		return min_ext + actor_output * (max_ext - min_ext)

	# =====================================================================
	# Rollout
	# =====================================================================

	def collect_async_rollout(self, env, max_steps=1000):
		"""
		Thu thập rollout bất đồng bộ theo API môi trường mới.
		Môi trường tự quản lý timing, step() chạy đến khi có nút cần action.
		"""
		obs, reward, done, info = env.reset()
		self._reset_hidden()
		t = 0

		# Pending buffer entries cho mỗi agent (chờ reward_seq)
		pending_entries = [None] * self.num_agents
		reward_seqs = [[] for _ in range(self.num_agents)]
		active = [False] * self.num_agents

		while not done and t < max_steps:
			requiring = info["intersection_require_action"]  # list[str]
			action_dict = {}

			for name in requiring:
				i = self.tls_index[name]

				# Finalize entry từ lần action trước (nếu có)
				if active[i] and pending_entries[i] is not None:
					pending_entries[i]["reward_seq"] = reward_seqs[i]
					self.agents_buffer.store(i, pending_entries[i])
					reward_seqs[i] = []
					pending_entries[i] = None

				# Encode quan sát mới
				z_it = self._obs_to_tensor(obs, i)
				p_it = self.time_encoder(t).to(self.device)
				self.hidden_states[i] = self.encoders[i](z_it, p_it, self.hidden_states[i])

				# Sample hành động (actor output trong [0, 1])
				actor_out, old_log_prob = self.actors[i].sample(self.hidden_states[i])
				actor_val = float(actor_out.detach().item())  # [0, 1]

				# Scale sang kự năng có hiệu lực [min_ext, max_ext]
				eff_range = info.get("effective_action_range", {}).get(name, (info["min_green"], info["max_green"]))
				print(f"eff_range: {eff_range}")
				env_action = self._scale_action(actor_val, eff_range[0], eff_range[1])
				action_dict[name] = env_action

				# Lưu pending entry (raw_action là actor output [0,1], nhất quán với evaluate)
				active[i] = True
				pending_entries[i] = {
					"agent_id": i,
					"t": t,
					"h": self.hidden_states[i].detach(),
					"global_h": None,
					"raw_action": actor_out.detach(),  # [0, 1]
					"old_log_prob": old_log_prob.detach(),
					"reward_seq": [],
				}

			# Cập nhật global_h sau khi tất cả actor trong step này đã encode
			global_h = self._get_current_global_state()
			for name in requiring:
				i = self.tls_index[name]
				if pending_entries[i] is not None:
					pending_entries[i]["global_h"] = global_h.detach()

			# Bước môi trường
			print(f"action dict: {action_dict}")
			next_obs, reward, done, info = env.step(action_dict)

			# Lưu critic buffer
			self.critic_buffer.store({
				"t": t,
				"global_h": self._get_current_global_state().detach(),
				"reward": self._global_reward_scalar(reward),
			})

			# Tích lũy reward cho từng agent đang active
			for i in range(self.num_agents):
				if active[i]:
					reward_seqs[i].append(self._reward_scalar(reward, i))

			obs = next_obs
			t += info["delta_t"]  # thời gian thực env đã chạy (giây)

		# Finalize tất cả macros chưa kết thúc
		for i in range(self.num_agents):
			if pending_entries[i] is not None:
				pending_entries[i]["reward_seq"] = reward_seqs[i]
				self.agents_buffer.store(i, pending_entries[i])

	# =====================================================================
	# Evaluation
	# =====================================================================

	def evaluate_model(self, env, max_steps=1000):
		"""
		Đánh giá model không lưu buffer.
		Returns: {"total_reward": float, "avg_reward": float}
		"""
		obs, reward, done, info = env.reset()
		self._reset_hidden()
		t = 0
		total_reward = 0.0

		while not done and t < max_steps:
			requiring = info["intersection_require_action"]
			action_dict = {}

			for name in requiring:
				i = self.tls_index[name]
				z_it = self._obs_to_tensor(obs, i)
				p_it = self.time_encoder(t).to(self.device)
				self.hidden_states[i] = self.encoders[i](z_it, p_it, self.hidden_states[i])
				actor_out, _ = self.actors[i].sample(self.hidden_states[i])
				actor_val = float(actor_out.detach().item())
				eff_range = info.get("effective_action_range", {}).get(name, (0.0, info["max_green"]))
				action_dict[name] = self._scale_action(actor_val, eff_range[0], eff_range[1])

			next_obs, reward, done, info = env.step(action_dict)
			total_reward += self._global_reward_scalar(reward)
			obs = next_obs
			t += info["delta_t"]

		return {
			"total_reward": total_reward,
			"avg_reward": total_reward / max(t, 1),
		}

	# =====================================================================
	# Training
	# =====================================================================

	def compute_advantages(self):
		for i in range(self.num_agents):
			traj = self.agents_buffer.get_agent_traj(i)
			if len(traj) == 0:
				continue

			states = torch.stack([e["global_h"] for e in traj]).to(self.device)

			with torch.no_grad():
				values = self.critic(states)

			gae = 0
			next_value = 0

			for k in reversed(range(len(traj))):
				rewards = traj[k]["reward_seq"]
				R = 0
				for r in reversed(rewards):
					R = r + self.gamma * R

				delta = R + (self.gamma ** len(rewards)) * next_value - values[k]
				gae = delta + (self.gamma ** len(rewards)) * self.lam * gae

				traj[k]["advantage"] = gae.detach()
				traj[k]["return"] = (gae + values[k]).detach()
				next_value = values[k]

	def compute_returns(self):
		traj = self.critic_buffer.buffers
		for k in reversed(range(len(traj))):
			reward = traj[k]["reward"]
			next_value = 0 if k == len(traj) - 1 else traj[k + 1]["return"]
			traj[k]["return"] = reward + self.gamma * next_value

	def update_critic(self):
		states = torch.stack([e["global_h"] for e in self.critic_buffer.buffers]).to(self.device)
		returns = torch.tensor(
			[e["return"] for e in self.critic_buffer.buffers],
			dtype=torch.float32, device=self.device
		)

		values = self.critic(states)
		value_loss = F.mse_loss(values, returns)

		self.optimizers["critic"].zero_grad()
		value_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
		self.optimizers["critic"].step()

		return value_loss.item()

	def update_actors(self, clip_eps=0.2):
		actor_losses = {}

		for i in range(self.num_agents):
			agent_entries = self.agents_buffer.get_agent_traj(i)
			if len(agent_entries) == 0:
				continue

			hs = torch.stack([e["h"] for e in agent_entries]).to(self.device)
			actions = torch.stack([e["raw_action"] for e in agent_entries]).to(self.device)
			advantages = torch.stack([e["advantage"] for e in agent_entries]).to(self.device)
			old_log_probs = torch.stack([e["old_log_prob"] for e in agent_entries]).to(self.device).squeeze(-1)

			log_probs = self.actors[i].evaluate(hs, actions)

			ratio = torch.exp(log_probs - old_log_probs)
			surr1 = ratio * advantages
			surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
			loss = -torch.min(surr1, surr2).mean()

			self.optimizers["actor"][i].zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
			self.optimizers["actor"][i].step()

			actor_losses[i] = loss.item()

		return actor_losses

	def train_episode(self, env, max_steps=None):
		self.collect_async_rollout(env, max_steps=max_steps or 1000)
		self.compute_advantages()
		self.compute_returns()
		critic_loss = self.update_critic()
		actor_losses = self.update_actors()

		total_reward = sum(e["reward"] for e in self.critic_buffer.buffers)

		self.agents_buffer.clear()
		self.critic_buffer.clear()

		return {
			"reward": total_reward,
			"critic_loss": critic_loss,
			"actor_losses": actor_losses,
		}

	# =====================================================================
	# Persistence
	# =====================================================================

	def save_model(self, path):
		state = {
			"agents": [
				{
					"actor_state_dict": self.actors[i].state_dict(),
					"encoder_state_dict": self.encoders[i].state_dict(),
				}
				for i in range(self.num_agents)
			],
			"critic_state_dict": self.critic.state_dict(),
			"optimizers": {
				"actor": [opt.state_dict() for opt in self.optimizers["actor"]],
				"critic": self.optimizers["critic"].state_dict(),
			},
		}
		torch.save(state, path)

	def load_model(self, path):
		state = torch.load(path, map_location=self.device)
		for i in range(self.num_agents):
			self.actors[i].load_state_dict(state["agents"][i]["actor_state_dict"])
			self.encoders[i].load_state_dict(state["agents"][i]["encoder_state_dict"])
		self.critic.load_state_dict(state["critic_state_dict"])
		for i, opt in enumerate(self.optimizers["actor"]):
			opt.load_state_dict(state["optimizers"]["actor"][i])
		self.optimizers["critic"].load_state_dict(state["optimizers"]["critic"])
