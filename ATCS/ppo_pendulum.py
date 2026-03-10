import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters for faster convergence
GAMMA = 0.99
LAMBDA = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 10
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
MAX_STEPS = 200
TOTAL_EPISODES = 500
UPDATE_TIMESTEP = 2048
MINI_BATCH_SIZE = 64

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.mu = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        x = self.fc(state)
        mu = torch.tanh(self.mu(x)) * 2.0
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std

    def get_action(self, state):
        mu, std = self.forward(state)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach().cpu().numpy(), log_prob.detach()

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.fc(state)

class PPO:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.MseLoss = nn.MSELoss()

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        return self.actor.get_action(state)

    def update(self, buffer):
        old_states = torch.tensor(np.array(buffer['states']), dtype=torch.float32).to(self.device)
        old_actions = torch.tensor(np.array(buffer['actions']), dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(np.array(buffer['log_probs']), dtype=torch.float32).to(self.device)
        rewards = buffer['rewards']
        dones = buffer['dones']

        returns = []
        advantages = []
        gae = 0
        with torch.no_grad():
            values = self.critic(old_states).squeeze().cpu().numpy()
            next_value = 0
            for i in reversed(range(len(rewards))):
                delta = rewards[i] + GAMMA * next_value * (1 - dones[i]) - values[i]
                gae = delta + GAMMA * LAMBDA * (1 - dones[i]) * gae
                advantages.insert(0, gae)
                next_value = values[i]
                returns.insert(0, gae + values[i])

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0
        update_count = 0

        dataset_size = len(old_states)
        for _ in range(K_EPOCHS):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                idx = indices[start:end]
                
                batch_states = old_states[idx]
                batch_actions = old_actions[idx]
                batch_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                mu, std = self.actor(batch_states)
                dist = Normal(mu, std)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                
                ratio = torch.exp(log_probs - batch_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
                critic_loss = self.MseLoss(self.critic(batch_states).squeeze(), batch_returns)

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                update_count += 1

        return total_actor_loss / update_count, total_critic_loss / update_count

def main():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPO(state_dim, action_dim)
    episode_rewards = []
    buffer = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []}
    time_step = 0
    latest_losses = (0, 0)

    print("Starting training with optimized PPO...")
    for episode in range(TOTAL_EPISODES):
        state, _ = env.reset()
        ep_reward = 0
        
        for step in range(MAX_STEPS):
            time_step += 1
            action, log_prob = agent.get_action(state.reshape(1, -1))
            
            next_state, reward, terminated, truncated, _ = env.step(action[0])
            done = terminated or truncated
            
            buffer['states'].append(state)
            buffer['actions'].append(action[0])
            buffer['log_probs'].append(log_prob.item())
            # Improved reward normalization for Pendulum
            buffer['rewards'].append((reward + 8.1) / 8.1)
            buffer['dones'].append(float(done))
            
            state = next_state
            ep_reward += reward
            
            if time_step % UPDATE_TIMESTEP == 0:
                latest_losses = agent.update(buffer)
                buffer = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []}

            if done:
                break
        
        episode_rewards.append(ep_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Ep {episode+1:03d} | Reward: {avg_reward:8.1f} | ActorLoss: {latest_losses[0]:8.4f} | CriticLoss: {latest_losses[1]:8.4f}")

        if (episode + 1) % 50 == 0:
            plt.figure(figsize=(10, 5))
            plt.plot(episode_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Raw Reward')
            plt.title('Optimized PPO Pendulum Training')
            plt.savefig('ppo_pendulum_reward_optimized.png')
            plt.close()
            
            if avg_reward > -250:
                print(f"Solved at episode {episode+1}!")
                break

    print("Training finished. Chart saved to ppo_pendulum_reward_optimized.png")
    torch.save(agent.actor.state_dict(), "ppo_pendulum_optimized.pt")

if __name__ == "__main__":
    main()
