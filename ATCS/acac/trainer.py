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
		return torch.stack([h.detach() for h in self.hidden_states])

	def _obs_to_tensor(self, obs, tls_index):
		return torch.tensor(
			obs[tls_index].flatten(), dtype=torch.float32, device=self.device
		)

	def _global_reward_scalar(self, reward):
		return -float(reward[:, :, 1].mean())

	def _scale_action(self, actor_output, min_ext, max_ext):
		"""
		Scale actor output từ [0, 1] sang [min_ext, max_ext].
		actor_output : float trong [0, 1] (output của MacroActor với min_action=0, max_action=1)
		Returns: float extension seconds trong [min_ext, max_ext]
		"""
		return int(min_ext + actor_output * (max_ext - min_ext))

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

		while not done and t < max_steps:
			requiring = info["intersection_require_action"]  # list[str]
			action_dict = {}

			step_entries = []

			for name in requiring:
				i = self.tls_index[name]

				# Encode quan sát mới
				z_it = self._obs_to_tensor(obs, i)
				eff_range = info.get("effective_action_range", {}).get(name, (info["min_green"], info["max_green"]))
				z_it = torch.cat([z_it, torch.tensor(eff_range, dtype=torch.float32).to(self.device)], dim=-1)
				p_it = self.time_encoder(t).to(self.device)
				self.hidden_states[i] = self.encoders[i](z_it, p_it, self.hidden_states[i])

				# Sample hành động (actor output trong [0, 1])
				actor_out, old_log_prob = self.actors[i].sample(self.hidden_states[i])
				actor_val = float(actor_out.detach().item())  # [0, 1]

				# Scale sang kự năng có hiệu lực [min_ext, max_ext]
				print(f"eff_range: {eff_range}")
				env_action = self._scale_action(actor_val, eff_range[0], eff_range[1])
				action_dict[name] = env_action

				step_entries.append({
					"agent_id": i,
					"t": t,
					"h": self.hidden_states[i].detach(),
					"global_h": None,
					"raw_action": actor_out.detach(),  # [0, 1]
					"old_log_prob": old_log_prob.detach(),
				})

			# Cập nhật global_h sau khi tất cả actor trong step này đã encode
			global_h = self._get_current_global_state()
			for entry in step_entries:
				entry["global_h"] = global_h.detach()
				self.agents_buffer.store(entry["agent_id"], entry)

			# Bước môi trường
			print(f"action dict: {action_dict}")
			next_obs, reward, done, info = env.step(action_dict)

			# Lưu critic buffer
			self.critic_buffer.store({
				"t": t,
				"global_h": self._get_current_global_state().detach(),
				"reward": self._global_reward_scalar(reward),
			})

			obs = next_obs
			t += info["delta_t"]  # thời gian thực env đã chạy (giây)

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
		total_delay = 0.0
		total_queue = 0.0

		while not done and t < max_steps:
			requiring = info["intersection_require_action"]
			action_dict = {}

			for name in requiring:
				i = self.tls_index[name]
				z_it = self._obs_to_tensor(obs, i)
				eff_range = info.get("effective_action_range", {}).get(name, (info["min_green"], info["max_green"]))
				z_it = torch.cat([z_it, torch.tensor(eff_range, dtype=torch.float32).to(self.device)], dim=-1)
				p_it = self.time_encoder(t).to(self.device)
				self.hidden_states[i] = self.encoders[i](z_it, p_it, self.hidden_states[i])
				actor_out, _ = self.actors[i].sample(self.hidden_states[i])
				actor_val = float(actor_out.detach().item())
				action_dict[name] = self._scale_action(actor_val, eff_range[0], eff_range[1])

			next_obs, reward, done, info = env.step(action_dict)
			total_delay += reward[:, :, 0].mean()
			total_queue += reward[:, :, 1].mean()
			total_reward += self._global_reward_scalar(reward)
			obs = next_obs
			t += info["delta_t"]

		return {
			"total_reward": total_reward,
			"avg_reward": total_reward / max(t, 1),
			"delay": total_delay / max(t, 1),
			"queue": total_queue / max(t, 1),
		}

	# =====================================================================
	# Training
	# =====================================================================

	def compute_advantages(self):
		traj = self.critic_buffer.buffers
		if len(traj) == 0:
			return

		states = torch.stack([e["global_h"] for e in traj]).to(self.device)

		with torch.no_grad():
			values = self.critic(states).squeeze(-1)

		gae = 0
		ret = 0
		advantage_map = {}

		for k in reversed(range(len(traj))):
			reward = traj[k]["reward"]
			if k == len(traj) - 1:
				next_value = 0
				dt = 1
			else:
				next_value = values[k + 1]
				dt = traj[k + 1]["t"] - traj[k]["t"]

			gamma_dt = self.gamma ** dt
			delta = reward + gamma_dt * next_value - values[k]
			gae = delta + gamma_dt * self.lam * gae
			ret = reward + gamma_dt * ret

			traj[k]["advantage"] = gae.detach()
			traj[k]["return"] = ret if isinstance(ret, torch.Tensor) else torch.tensor(ret, dtype=torch.float32, device=self.device)
			
			advantage_map[traj[k]["t"]] = traj[k]["advantage"]

		# Add logic to assign advantages for actors at their decision timestep
		for i in range(self.num_agents):
			agent_traj = self.agents_buffer.get_agent_traj(i)
			if len(agent_traj) == 0:
				continue
				
			for e in agent_traj:
				t_start = e["t"]
				
				# The actor uses the advantage at the exact timestep it made the decision.
				# A_t computes the advantage AT timestep `t_start` looking forward up to `t_next`.
				# This matches the definition where GAE(t) estimates sum(td_errors).
				if t_start in advantage_map:
					e["advantage"] = advantage_map[t_start]
				else:
					print(f"Warning: No advantage found for timestep {t_start}")
					e["advantage"] = 0.0

	def update_critic(self, n_epochs=4):
		if len(self.critic_buffer.buffers) == 0:
			return 0.0

		states = torch.stack([e["global_h"] for e in self.critic_buffer.buffers]).to(self.device)
		returns = torch.stack([
			e["return"] if isinstance(e["return"], torch.Tensor)
			else torch.tensor(e["return"], dtype=torch.float32)
			for e in self.critic_buffer.buffers
		]).to(self.device).float()

		loss_val = 0.0
		for _ in range(n_epochs):
			values = self.critic(states).squeeze(-1)
			value_loss = F.smooth_l1_loss(values, returns)

			self.optimizers["critic"].zero_grad()
			value_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
			self.optimizers["critic"].step()
			loss_val = value_loss.item()

		return loss_val

	def update_actors(self, clip_eps=0.2, n_epochs=4):
		actor_losses = {}

		for i in range(self.num_agents):
			agent_entries = self.agents_buffer.get_agent_traj(i)
			if len(agent_entries) == 0:
				continue

			hs = torch.stack([e["h"] for e in agent_entries]).to(self.device)
			actions = torch.stack([e["raw_action"] for e in agent_entries]).to(self.device)
			advantages = torch.stack([e["advantage"] for e in agent_entries]).to(self.device)
			if advantages.numel() > 1:
				advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
			old_log_probs = torch.stack([e["old_log_prob"] for e in agent_entries]).to(self.device).squeeze(-1)

			for epoch in range(n_epochs):
				log_probs, entropy = self.actors[i].evaluate(hs, actions)
				ratio = torch.exp(log_probs - old_log_probs)
				print(f"[Agent {i}] epoch={epoch} ratio={ratio}")
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
		# compute_returns is now handled inside compute_advantages
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
