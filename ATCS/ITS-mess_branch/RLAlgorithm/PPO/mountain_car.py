import time
import math
import numpy as np
import os
import torch

try:
	import gymnasium as gym
except ImportError:
	import gym

from ppo import PPOAgent, PPOConfig


def _env_reset(env):
	out = env.reset()
	if isinstance(out, tuple):
		return out[0]
	return out


def _env_step(env, action):
	out = env.step(action)
	if len(out) == 5:
		obs, reward, terminated, truncated, info = out
		done = terminated or truncated
		return obs, reward, done, info
	obs, reward, done, info = out
	return obs, reward, done, info


def train_mountain_car(
	env_id: str = "CartPole-v1",
	total_steps: int = 40_000,
	update_every: int = 1024,
	seed: int = 0,
):
	env = gym.make(env_id)
	try:
		env.reset(seed=seed)
	except Exception:
		pass

	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.n

	cfg = PPOConfig(
		gamma=0.99,
		lam=0.95,
		clip_ratio=0.2,
		lr=3e-4,
		train_epochs=6,
		batch_size=update_every,
		minibatch_size=256,
		entropy_coef=0.0,
		value_coef=0.5,
		max_grad_norm=0.5,
		device="cuda:0" if torch.cuda.is_available() else "cpu",
	)
	agent = PPOAgent(obs_dim, act_dim, cfg)

	state = _env_reset(env)
	ep_reward = 0.0
	ep_count = 0
	steps = 0
	start_time = time.time()
	
	# Episode collection
	episode_states = []
	episode_actions = []
	episode_logprobs = []
	episode_rewards = []
	episode_dones = []
	episode_values = []
	
	while steps < total_steps:
		action, logprob, value = agent.select_action(state.astype(np.float32))
		next_state, reward, done, _ = _env_step(env, action)
		
		# Collect transitions for current episode
		episode_states.append(state.astype(np.float32))
		episode_actions.append(action)
		episode_logprobs.append(logprob)
		episode_rewards.append(reward)
		episode_dones.append(done)
		episode_values.append(value)
		
		ep_reward += reward
		state = next_state
		steps += 1

		if done:
			print(f"episode={ep_count} reward={ep_reward:.2f}")
			# Store completed episode
			agent.store_episode(episode_states, episode_actions, episode_logprobs, episode_rewards, episode_dones, episode_values)
			# Clear episode data
			episode_states = []
			episode_actions = []
			episode_logprobs = []
			episode_rewards = []
			episode_dones = []
			episode_values = []
			ep_count += 1
			state = _env_reset(env)
			ep_reward = 0.0

		if steps % update_every == 0:
			agent.update()

		if steps % 5000 == 0:
			elapsed = time.time() - start_time
			print(f"steps={steps} elapsed={elapsed:.1f}s")

	agent.save("mountaincar_ppo.pt")
	env.close()
	return agent


def record_video_episodes(agent: PPOAgent, env_id: str, seed: int, video_dir: str = "videos", episodes: int = 3):
	os.makedirs(video_dir, exist_ok=True)
	use_wrapper = True
	try:
		env = gym.make(env_id, render_mode="rgb_array")
	except TypeError:
		env = gym.make(env_id)
	# Try to use built-in RecordVideo; on failure fall back to manual encoding
	try:
		env = gym.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda ep_id: True)
	except Exception:
		use_wrapper = False
		print("Video recording wrapper unavailable. Using fallback encoder.")

	try:
		env.reset(seed=seed)
	except Exception:
		pass

	# Lazy import imageio for fallback
	writer_available = False
	if not use_wrapper:
		try:
			import imageio
			writer_available = True
		except Exception:
			print("Fallback requires 'imageio' (and optionally 'imageio-ffmpeg'). Install them to record videos.")

	for ep in range(episodes):
		state = _env_reset(env)
		ep_reward = 0.0
		done = False
		frames = []
		while not done:
			action, _, _ = agent.select_action(np.asarray(state, dtype=np.float32))
			# Capture frame for fallback
			if not use_wrapper and writer_available:
				try:
					frame = env.render()
					if frame is not None:
						frames.append(frame)
				except Exception:
					pass
			state, reward, done, _ = _env_step(env, action)
			ep_reward += reward
		print(f"[EVAL] episode={ep+1} reward={ep_reward:.2f}")

		# Write video via imageio if not using wrapper
		if not use_wrapper and writer_available and frames:
			try:
				import imageio
				out_path = os.path.join(video_dir, f"{env_id}_ep{ep+1}.mp4")
				imageio.mimsave(out_path, frames, fps=30)
				print(f"Saved video: {out_path}")
			except Exception as e:
				print(f"Failed to save video via imageio: {e}")

	env.close()


if __name__ == "__main__":
	# Allow quick overrides via environment variables for faster sanity checks
	env_id = os.getenv("PPO_ENV_ID", "CartPole-v1")
	total = int(os.getenv("PPO_TOTAL_STEPS", "40000"))
	upd = int(os.getenv("PPO_UPDATE_EVERY", "2048"))
	agent = train_mountain_car(env_id=env_id, total_steps=total, update_every=upd)
	# After training, record a few evaluation episodes as videos
	record_video_episodes(agent=agent, env_id=env_id, seed=0)

