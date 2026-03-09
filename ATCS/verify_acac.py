"""Quick verification: Pendulum ACAC with reward normalization."""
import os
import sys
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from contextlib import contextmanager
from acac import (
    SinusoidalPositionalEncoding, CentralizedCritic, AgentHistoryEncoder, MacroActor,
    AsyncTrajectoryBuffer, SyncTrajectoryBuffer, ACACTrainer
)


class PendulumACACWrapper:
    """
    Wrap Pendulum-v1 to match the ACAC TrafficEnvironment API.
    
    Env API contract:
        reset()  -> (obs, reward, done, info)
        step(action_dict) -> (obs, reward, done, info)
    
    obs: np.ndarray shape [num_agents, *per_agent_shape]
         trainer uses obs[tls_index].flatten()
    reward: np.ndarray shape [N, M, 2]
         trainer uses  -reward[:, :, 0].mean()  as scalar
    info keys:
         intersection_require_action: list[str]
         effective_action_range: dict[str, tuple]
         delta_t: int
         min_green, max_green: float
    """

    def __init__(self):
        self.env = gym.make("Pendulum-v1")
        self.tls_ids = ["p0"]
        self.tls_index = {"p0": 0}

    def reset(self):
        obs, _ = self.env.reset()
        # obs shape: (3,) -> wrap to (1, 3) so obs[0] = (3,)
        self.obs_mapped = np.array([obs], dtype=np.float32)
        reward = np.zeros((1, 1, 2), dtype=np.float32)
        return self.obs_mapped, reward, False, self._make_info(done=False)

    def step(self, action_dict):
        # Actor output is in [0, 1] (MacroActor range), trainer._scale_action
        # maps it to int(min_ext + val*(max_ext - min_ext)).
        # With eff_range=(-2, 2), that gives int values {-2,-1,0,1,2} — coarse!
        # We accept whatever the trainer sends and pass to Pendulum (expects [-2,2]).
        a = action_dict.get("p0", 0.0)
        a_clipped = np.clip(float(a), -2.0, 2.0)
        obs, reward, terminated, truncated, _ = self.env.step([a_clipped])
        done = terminated or truncated
        self.obs_mapped = np.array([obs], dtype=np.float32)
        
        # Pendulum reward is negative (higher = better, e.g. -0.1 is good).
        # ACAC trainer computes: -reward[:,:,0].mean() as the scalar.
        # We want ACAC to *maximize* the original reward (minimize cost).
        # Put negative-reward in slot 0 so trainer gets: -(-pendulum_reward) = +pendulum_reward
        # → higher pendulum_reward → higher ACAC reward.
        r = np.zeros((1, 1, 2), dtype=np.float32)
        r[0, 0, 0] = (reward + 8.1) / 8.1  # normalize [-16.2, 0] → ~[-1, 1]
        return self.obs_mapped, r, done, self._make_info(done=done)

    def _make_info(self, done=False):
        return {
            "intersection_require_action": ["p0"] if not done else [],
            "effective_action_range": {"p0": (-2.0, 2.0)},
            "delta_t": 1,
            "min_green": -2.0,
            "max_green": 2.0,
        }

    def close(self):
        self.env.close()


def build_trainer(obs_dim=5, device="cpu"):
    """
    obs_dim = 3 (Pendulum obs: cos, sin, angular_vel) + 2 (eff_range) = 5
    """
    hd, td = 64, 16
    te = SinusoidalPositionalEncoding(td).to(device)
    enc = [AgentHistoryEncoder(obs_dim, td, hd).to(device)]
    # MacroActor outputs in [0,1]; trainer._scale_action maps to env range
    act = [MacroActor(hd, 1, min_action=0.0, max_action=1.0).to(device)]
    cri = CentralizedCritic(hd, 1).to(device)
    ab = AsyncTrajectoryBuffer(200, 1)
    cb = SyncTrajectoryBuffer(200)
    # Combined optimizer for BPTT (Encoder + Actor + Critic)
    all_params = list(cri.parameters())
    for a in act:
        all_params += list(a.parameters())
    for e in enc:
        all_params += list(e.parameters())
        
    opt = torch.optim.Adam(all_params, lr=3e-4)

    return ACACTrainer(
        actors=act, encoders=enc, critic=cri,
        agents_buffer=ab, critic_buffer=cb,
        optimizers={"combined": opt},
        time_encoder=te, tls_names=["p0"], device=device,
        gamma=0.99, lam=0.95, eps_clip=0.2,
    )


if __name__ == "__main__":
    NUM_EPISODES = 5000
    MAX_STEPS = 200

    env = PendulumACACWrapper()
    trainer = build_trainer()
    
    episode_rewards = []
    out = open("verify_log.txt", "w", encoding="utf-8")
    
    for ep in range(NUM_EPISODES):
        metrics = trainer.train_episode(env, max_steps=MAX_STEPS)
        ep_reward = metrics["reward"]
        episode_rewards.append(ep_reward)
        
        line = (
            f"Ep {ep+1:03d} | Reward: {ep_reward:8.1f} "
            f"| CriticLoss: {metrics['critic_loss']:8.4f} "
            f"| ActorLoss: {metrics['actor_losses'].get(0, 0):8.4f}"
        )
        print(line)
        out.write(line + "\n")

        # Save reward plot every 50 episodes
        if (ep + 1) % 50 == 0 or ep == NUM_EPISODES - 1:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(episode_rewards) + 1), episode_rewards,
                     alpha=0.4, label="Episode Reward")
            # Smoothed line (moving average window=20)
            if len(episode_rewards) >= 20:
                window = 20
                smoothed = np.convolve(episode_rewards,
                                       np.ones(window) / window, mode="valid")
                plt.plot(range(window, len(episode_rewards) + 1), smoothed,
                         linewidth=2, label=f"MA-{window}")
            plt.title("Pendulum ACAC Training")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("pendulum_acac_reward.png", dpi=100)
            plt.close()
    
    out.close()
    env.close()
    print(f"\nDone. Log saved to verify_log.txt")
    print(f"Reward plot saved to pendulum_acac_reward.png")
