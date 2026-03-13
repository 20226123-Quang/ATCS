"""ACAC Trainer: rollout collection, advantage computation, actor/critic updates."""

import os
import time

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
        device="cpu",
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
        self.vf_coef = _cfg.training.vf_coef  # c1: value function coefficient
        self.ent_coef = _cfg.training.ent_coef  # c2: entropy bonus coefficient
        self.device = device

        # ===== Book-keeping =====
        self.num_agents = len(actors)

        # Hidden states — infer hidden_dim from GRUCell
        hidden_dim = encoders[0].gru.hidden_size
        self.hidden_states = [
            torch.zeros(hidden_dim, device=device) for _ in range(self.num_agents)
        ]

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
        # Return the mean reward across all agents and lanes.
        # Note: The environment should now return negative costs (e.g., -delay).
        return float(reward[:, :, 0].mean())

    def _scale_action(self, actor_output, min_ext, max_ext):
        """
        Scale actor output từ [0, 1] sang extension range [e_min, e_max].
        actor_output : float trong [0, 1] (output của MacroActor với min_action=0, max_action=1)
        Returns: float extension seconds trong [e_min, e_max]
        """
        return float(min_ext + actor_output * (max_ext - min_ext))

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
                z_it = self._obs_to_tensor(obs, i).unsqueeze(0)  # [1, obs_dim]
                eff_range = info.get("effective_action_range", {}).get(
                    name, (0.0, info["max_green"] - info["min_green"])
                )
                z_it = torch.cat(
                    [
                        z_it,
                        torch.tensor([eff_range], dtype=torch.float32).to(self.device),
                    ],
                    dim=-1,
                )
                p_it = (
                    self.time_encoder(t).to(self.device).unsqueeze(0)
                )  # [1, time_dim]

                # Update hidden state with batch dim
                h_prev = self.hidden_states[i].unsqueeze(0)  # [1, hidden_dim]
                self.hidden_states[i] = self.encoders[i](z_it, p_it, h_prev).squeeze(0)

                # Sample hành động (actor output trong [0, 1])
                actor_out, old_log_prob = self.actors[i].sample(
                    self.hidden_states[i].unsqueeze(0)
                )
                actor_out = actor_out.squeeze(0)
                old_log_prob = old_log_prob.squeeze(0)
                actor_val = float(actor_out.detach().item())  # [0, 1]

                # Scale sang extension range hợp lệ [e_min, e_max]
                env_action = self._scale_action(actor_val, eff_range[0], eff_range[1])
                action_dict[name] = env_action
                print(f"action dict: {action_dict}")

                step_entries.append(
                    {
                        "agent_id": i,
                        "t": t,
                        "h": self.hidden_states[i].detach(),
                        "obs": z_it.detach(),  # [1, obs_dim+2] for GRU reconstruction
                        "time_step": t,  # int, to reconstruct p_it
                        "global_h": None,
                        "raw_action": actor_out.detach(),  # [0, 1]
                        "old_log_prob": old_log_prob.detach(),
                    }
                )

            # Cập nhật global_h sau khi tất cả actor trong step này đã encode
            global_h = self._get_current_global_state()
            for entry in step_entries:
                entry["global_h"] = global_h.detach()
                self.agents_buffer.store(entry["agent_id"], entry)

            # Bước môi trường
            # print(f"action dict: {action_dict}")
            next_obs, reward, done, info = env.step(action_dict)

            # Lưu critic buffer
            self.critic_buffer.store(
                {
                    "t": t,
                    "global_h": self._get_current_global_state().detach(),
                    "reward": self._global_reward_scalar(reward),
                }
            )

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
                z_it = self._obs_to_tensor(obs, i).unsqueeze(0)
                eff_range = info.get("effective_action_range", {}).get(
                    name, (0.0, info["max_green"] - info["min_green"])
                )
                z_it = torch.cat(
                    [
                        z_it,
                        torch.tensor([eff_range], dtype=torch.float32).to(self.device),
                    ],
                    dim=-1,
                )
                p_it = self.time_encoder(t).to(self.device).unsqueeze(0)

                h_prev = self.hidden_states[i].unsqueeze(0)
                self.hidden_states[i] = self.encoders[i](z_it, p_it, h_prev).squeeze(0)

                actor_out, _ = self.actors[i].sample(self.hidden_states[i].unsqueeze(0))
                actor_val = float(actor_out.detach().item())
                action_dict[name] = self._scale_action(
                    actor_val, eff_range[0], eff_range[1]
                )

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
            values = self.critic(states).squeeze(-1)  # [T, 1] → [T]

        gae = 0
        ret = 0
        advantage_map = {}

        rewards = [e["reward"] for e in traj]
        # print(f"rewards: {len(rewards)}")
        # time_print = [e["t"] for e in traj]
        # # print(f"time_print: {time_print}")

        for k in reversed(range(len(traj))):
            reward = traj[k]["reward"]
            if k == len(traj) - 1:
                next_value = 0
                dt = 1
            else:
                next_value = values[k + 1]
                dt = traj[k + 1]["t"] - traj[k]["t"]

            gamma_dt = self.gamma**dt
            delta = reward + gamma_dt * next_value - values[k]
            gae = delta + gamma_dt * self.lam * gae
            ret = gae + values[k]  # GAE-based return, nhất quán với bootstrap

            traj[k]["advantage"] = gae.detach()
            traj[k]["return"] = (
                ret
                if isinstance(ret, torch.Tensor)
                else torch.tensor(ret, dtype=torch.float32, device=self.device)
            )

            # print(f"advantage: {gae}, return: {ret}")

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

    def update(self, clip_eps=0.2, n_epochs=4):
        if len(self.critic_buffer.buffers) == 0:
            return 0.0, {}

        critic_returns = (
            torch.stack(
                [
                    e["return"]
                    if isinstance(e["return"], torch.Tensor)
                    else torch.tensor(e["return"], dtype=torch.float32)
                    for e in self.critic_buffer.buffers
                ]
            )
            .to(self.device)
            .float()
        )

        critic_times = [e["t"] for e in self.critic_buffer.buffers]

        # Chuẩn bị data cho các agents
        agent_data = {}
        for i in range(self.num_agents):
            traj = self.agents_buffer.get_agent_traj(i)
            if len(traj) > 0:
                agent_data[i] = {
                    "times": [e["time_step"] for e in traj],
                    "obs": torch.stack([e["obs"] for e in traj])
                    .to(self.device)
                    .squeeze(1),  # [T, obs_dim+2]
                    "actions": torch.stack([e["raw_action"] for e in traj]).to(
                        self.device
                    ),
                    "advantages": torch.stack([e["advantage"] for e in traj]).to(
                        self.device
                    ),
                    "old_log_probs": torch.stack([e["old_log_prob"] for e in traj])
                    .to(self.device)
                    .squeeze(-1),
                }
                if agent_data[i]["advantages"].numel() > 1:
                    adv = agent_data[i]["advantages"]
                    agent_data[i]["advantages"] = (adv - adv.mean()) / (
                        adv.std() + 1e-8
                    )

        total_critic_loss = 0.0
        total_actor_losses = {i: 0.0 for i in agent_data.keys()}

        for epoch in range(n_epochs):
            # 1. Reconstruct hidden states cho mọi agents (có gradient)
            reconstructed_hs = {i: [] for i in range(self.num_agents)}
            reconstructed_hs_tensor = {}

            for i in range(self.num_agents):
                traj = self.agents_buffer.get_agent_traj(i)
                if len(traj) == 0:
                    continue

                h = torch.zeros(1, self.encoders[i].gru.hidden_size, device=self.device)
                for e in traj:
                    obs = e["obs"].to(self.device)
                    p_it = (
                        self.time_encoder(e["time_step"]).to(self.device).unsqueeze(0)
                    )
                    h = self.encoders[i](obs, p_it, h)
                    reconstructed_hs[i].append(h.squeeze(0))
                reconstructed_hs_tensor[i] = torch.stack(
                    reconstructed_hs[i]
                )  # [T_i, hidden_dim]

            # 2. Xây dựng global_h cho Critic bằng Zero-Order Hold interpolation
            global_hs = []
            for critic_t in critic_times:
                # Với mỗi agent, tìm h_t ở thời điểm <= critic_t
                # (Trong continuous time, trạng thái của agent tính đến critic_t là trạng thái mới nhất nó cập nhật)
                agent_hs_at_t = []
                for i in range(self.num_agents):
                    if i not in agent_data:
                        # Agent chưa bao giờ hành động, dùng h=0 hoặc h khởi tạo
                        agent_hs_at_t.append(
                            torch.zeros(
                                self.encoders[i].gru.hidden_size, device=self.device
                            )
                        )
                        continue

                    agent_times = agent_data[i]["times"]
                    # Tìm index cuối cùng mà time <= critic_t
                    idx = -1
                    for k, t in enumerate(agent_times):
                        if t <= critic_t:
                            idx = k
                        else:
                            break

                    if idx == -1:
                        agent_hs_at_t.append(
                            torch.zeros(
                                self.encoders[i].gru.hidden_size, device=self.device
                            )
                        )
                    else:
                        agent_hs_at_t.append(reconstructed_hs_tensor[i][idx])

                global_hs.append(torch.stack(agent_hs_at_t))

            global_h_tensor = torch.stack(
                global_hs
            )  # [T_critic, num_agents, hidden_dim]

            # 3. Tính Critic Loss
            values = self.critic(global_h_tensor).squeeze(-1)  # [T_critic]
            critic_loss = F.mse_loss(values, critic_returns)
            total_critic_loss += critic_loss.item()

            combined_loss = self.vf_coef * critic_loss

            # 4. Tính Actor Loss
            for i, data in agent_data.items():
                hs_i = reconstructed_hs_tensor[i]
                actions = data["actions"]
                advantages = data["advantages"]
                old_log_probs = data["old_log_probs"]

                log_probs, entropy = self.actors[i].evaluate(hs_i, actions)
                ratio = torch.exp(log_probs - old_log_probs)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
                l_clip = torch.min(surr1, surr2).mean()
                l_entropy = entropy.mean()

                actor_loss = -(l_clip + self.ent_coef * l_entropy)
                combined_loss += actor_loss
                total_actor_losses[i] += actor_loss.item()

            # 5. Combined Backward Pass
            self.optimizers["combined"].zero_grad()
            combined_loss.backward()

            # Clip gradients chung
            all_params = []
            all_params += list(self.critic.parameters())
            for i in range(self.num_agents):
                all_params += list(self.actors[i].parameters())
                all_params += list(self.encoders[i].parameters())
            torch.nn.utils.clip_grad_norm_(all_params, 0.5)

            self.optimizers["combined"].step()

        # Average losses over epochs
        return (
            total_critic_loss / n_epochs,
            {i: loss / n_epochs for i, loss in total_actor_losses.items()},
        )

    def train_episode(self, env, max_steps=None):
        self.collect_async_rollout(env, max_steps=max_steps or 1000)
        self.compute_advantages()

        critic_loss, actor_losses = self.update()

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

    def save_model(self, path, retries=5, retry_delay=0.2):
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
                "combined": self.optimizers["combined"].state_dict(),
            },
        }

        last_error = None
        for attempt in range(retries):
            try:
                torch.save(state, path)
                return path
            except RuntimeError as err:
                last_error = err
                if "error code: 1224" not in str(err):
                    raise
                time.sleep(retry_delay * (attempt + 1))

        base, ext = os.path.splitext(path)
        fallback_path = f"{base}_{int(time.time())}{ext}"
        torch.save(state, fallback_path)
        print(
            f"[checkpoint] File lock on {path}. Saved fallback checkpoint to {fallback_path}. "
            f"Original error: {last_error}"
        )
        return fallback_path

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(state["agents"][i]["actor_state_dict"])
            self.encoders[i].load_state_dict(state["agents"][i]["encoder_state_dict"])
        self.critic.load_state_dict(state["critic_state_dict"])
        self.optimizers["combined"].load_state_dict(state["optimizers"]["combined"])
