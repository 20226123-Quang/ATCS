import gymnasium as gym
import numpy as np
from PPO_continuous import PPOAgent
import torch
import os
from PPO_continuous import TrajectoryBuffer
import imageio

# Tên môi trường
ENV_NAME = "Pendulum-v1"  # hoặc "Pendulum-v0" nếu dùng gym cũ


# Video save setup
video_dir = "videos"
os.makedirs(video_dir, exist_ok=True)
# Hàm lưu video thủ công
def save_video(frames, path, fps=30):
    imageio.mimsave(path, frames, fps=fps)


# Tạo env gốc để lấy thông tin không bị ảnh hưởng bởi wrapper
env = gym.make(ENV_NAME, render_mode="rgb_array", max_episode_steps=2000)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
action_low = env.action_space.low
action_high = env.action_space.high

# Tạo agent và buffer riêng biệt
agent = PPOAgent(
    obs_dim, act_dim,
    np.array(action_low, dtype=np.float32),
    np.array(action_high, dtype=np.float32),
    lr=3e-4, gamma=0.99, lam=0.95, clip_ratio=0.12,
    epochs=20, batch_size=8192, minibatch_size=128, entropy_coef=0.01, device="cpu"
)
buffer = TrajectoryBuffer()

max_steps = 2000  # Số bước tối đa mỗi episode
num_episodes = 80


for episode in range(num_episodes):
    state, _ = env.reset() if hasattr(env, 'reset') and len(env.reset()) == 2 else (env.reset(), None)
    ep_reward = 0
    frames = []
    buffer.clear()
    for t in range(max_steps):
        action, logprob, value = agent.select_action(state)
        step_result = env.step(action)
        # Lưu frame nếu là mỗi 20 episode
        if episode % 20 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        next_state, reward, done, truncated, info = step_result
        done = bool(done)
        truncated = bool(truncated)
        buffer.store(state, action, logprob, reward, done or truncated, value)
        state = next_state
        ep_reward += reward
        if done or truncated:
            print(f"Episode finished after {t+1} steps.")
            break
    # Lưu video nếu là mỗi 20 episode
    if episode % 20 == 0 and len(frames) > 0:
        video_path = os.path.join(video_dir, f"manual_ep{episode+1}.mp4")
        save_video(frames, video_path)
    # Tính advantage và update
    last_value = agent.net.forward(torch.tensor(state, dtype=torch.float32).unsqueeze(0))[2].item()
    batch = agent.finish_path(last_value, buffer)
    agent.update(batch)
    print(f"Episode {episode+1}, Reward: {ep_reward:.2f}")

env.close()
print(f"Videos saved to {video_dir}/ (every 20 steps)")
