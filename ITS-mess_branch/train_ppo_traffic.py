"""Train PPO agent to predict green durations for one intersection.

This script uses `traffic.env_wrappers.PhaseDurationEnv` and
`RLAlgorithm.PPO.PPO_continuous.PPOAgent` to run a basic training loop.
"""
import os
import time
import numpy as np

from traffic.env_wrappers import PhaseDurationEnv
from RLAlgorithm.PPO.PPO_continuous import PPOAgent
from traffic.metrics.metrics_collector import MetricsCollector


def train(
    sumo_config=None,
    controlled_tls_index=0,
    min_green=10.0,
    max_green=60.0,
    delta_time=1,
    yellow_time=3,
    max_steps=10000,
    reward_fn='wait_time',
    episodes=100,
    batch_size=64,
    device='cpu',
):
    env = PhaseDurationEnv(
        sumo_config=sumo_config,
        controlled_tls_index=controlled_tls_index,
        min_green=min_green,
        max_green=max_green,
        delta_time=delta_time,
        yellow_time=yellow_time,
        max_steps=max_steps,
        reward_fn=reward_fn,
        use_gui=False,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = 1
    action_low = [min_green]
    action_high = [max_green]

    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low=action_low,
        action_high=action_high,
        device=device,
        batch_size=batch_size,
        minibatch_size=min(16, batch_size),
    )

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            # Agent selects continuous duration
            action, logprob, value = agent.select_action(obs)
            # Ensure correct shape
            action = np.asarray(action, dtype=np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, logprob, reward, float(done), value)

            obs = next_obs
            ep_reward += reward
            steps += 1

            # If buffer has reached batch_size, finish path and update
            if len(agent.buffer) >= batch_size:
                batch = agent.finish_path(last_value=0, buffer=agent.buffer)
                agent.update(batch)

        # End of episode: flush buffer
        if len(agent.buffer) > 0:
            batch = agent.finish_path(last_value=0, buffer=agent.buffer)
            agent.update(batch)

        print(f"Episode {ep+1}/{episodes} reward={ep_reward:.2f} steps={steps}")
        # Optionally save model checkpoint
        model_path = os.path.join('models', f'ppo_phase_dur_ep{ep+1}.pt')
        os.makedirs('models', exist_ok=True)
        agent.save_model(model_path)

    env.close()


if __name__ == '__main__':
    # Minimal entry to start training quickly for smoke test
    from traffic.config import SUMO_CONFIG_PATH

    # Example usage: train
    train(
        sumo_config=SUMO_CONFIG_PATH,
        controlled_tls_index=0,
        episodes=1000,
        batch_size=64,
		max_steps=10000,
    )
