"""
Wrapper to integrate RL environments with PPO and other RL algorithms
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from traffic.environment import SumoTrafficEnv
import gym
from gymnasium import spaces
from traffic.config import SUMO_CONFIG_PATH
import traci


class SingleAgentWrapper:
	"""
	Wrapper for TrafficEnvironment to work with single PPO agent.
	Converts dict actions to array and handles state preprocessing.
	"""
		
	def __init__(self, env):
		self.env = env
		self.tls_ids = getattr(env, 'tls_ids', [])
		# derive dims from observation/action spaces if available
		self.obs_dim = getattr(env, 'observation_space', None)
		if hasattr(self.obs_dim, 'shape'):
			self.obs_dim = self.obs_dim.shape[0]
		else:
			self.obs_dim = None
		self.act_dim = getattr(env, 'action_space', None)
		if hasattr(self.act_dim, 'shape'):
			self.act_dim = int(np.prod(self.act_dim.shape))
		elif hasattr(self.act_dim, 'n'):
			self.act_dim = int(self.act_dim.n)
		else:
			self.act_dim = None
		
	def reset(self) -> np.ndarray:
		"""Reset and return flattened observation."""
		return self.env.reset()
		
	def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
		"""
		Step with array action, convert to dict for environment.
		
		Args:
			action: numpy array of shape (num_agents,) with green durations
		
		Returns:
			obs, reward, done, info
		"""
		# Convert array to dict
		action_dict = {
			tls_id: float(action[i])
			for i, tls_id in enumerate(self.tls_ids)
		}
		
		return self.env.step(action_dict)
		
	def close(self):
		self.env.close()
		
	def get_observation_space_dim(self):
		return self.obs_dim
		
	def get_action_space_dim(self):
		return self.act_dim


class MultiAgentWrapper:
	"""
	Wrapper for MultiAgentTrafficEnvironment to handle asynchronous agents.
	Provides simplified interface for training independent agents.
	"""
		
	def __init__(self, env):
		self.env = env
		self.tls_ids = getattr(env, 'tls_ids', [])
		self.obs_dim_per_agent = getattr(env, 'obs_dim_per_agent', None)
		
		# Store last observations and info
		self.last_obs = None
		self.last_infos = None
		
	def reset(self) -> Dict[str, np.ndarray]:
		"""Reset and return observations dict."""
		obs_flat, info = self.env.reset()
		self.last_obs = self._parse_observations(obs_flat)
		return self.last_obs
		
	def step(self, actions: Dict[str, float]) -> Tuple[
		Dict[str, np.ndarray], 
		Dict[str, float], 
		Dict[str, bool], 
		Dict[str, Dict]
	]:
		"""
		Step with actions for agents that need them.
		
		Args:
			actions: Dict mapping agent ID to action (only for agents needing action)
		
		Returns:
			observations, rewards, dones, infos
		"""
		obs_flat, rewards, dones, infos = self.env.step(actions)
		self.last_obs = self._parse_observations(obs_flat)
		self.last_infos = infos
		return self.last_obs, rewards, dones, infos
		
	def _parse_observations(self, obs_flat):
		"""Parse flat observation vector into dict of per-agent observations."""
		obs_dict = {}
		obs_size_per_agent = self.obs_dim_per_agent
		
		for i, tls_id in enumerate(self.tls_ids):
			start_idx = i * obs_size_per_agent
			end_idx = start_idx + obs_size_per_agent
			obs_dict[tls_id] = obs_flat[start_idx:end_idx]
		
		return obs_dict
		
	def get_agents_needing_action(self) -> List[str]:
		"""Get list of agents that need to provide actions."""
		return self.env.get_agents_needing_action()
		
	def close(self):
		self.env.close()


class AsynchronousMultiAgentWrapper:
	"""
	Wrapper for asynchronous multi-agent traffic control.
	Each agent (TLS) acts independently when its current phase duration expires.
	Agents predict green duration for the next phase, including fixed yellow time.
	"""
		
	def __init__(self, env):
		self.env = env
		# For PhaseDurationEnv, tls_ids is the controlled_tls
		if hasattr(env, 'controlled_tls') and env.controlled_tls:
			self.tls_ids = [env.controlled_tls]
		else:
			self.tls_ids = getattr(env, 'tls_ids', [])
		# self.obs_dim_per_agent = getattr(env, 'obs_size_per_tls', None)
		self.obs_dim_per_agent = getattr(env, 'observation_space', None)

		if hasattr(self.obs_dim_per_agent, 'shape'):
			self.obs_dim_per_agent = self.obs_dim_per_agent.shape[0]
		
		# Track phase timing per agent
		self.current_phase_duration = {tls: 0 for tls in self.tls_ids}
		self.last_phase_change = {tls: 0 for tls in self.tls_ids}
		self.yellow_time = getattr(env, 'yellow_time', 3)  # Fixed yellow time
		
		# Store last observations and info
		self.last_obs = None
		self.last_infos = None
		
	def reset(self) -> Dict[str, np.ndarray]:
		"""Reset environment and return initial observations."""
		obs_flat, _ = self.env.reset()
		self.last_obs = self._parse_observations(obs_flat)
		self.current_phase_duration = {tls: 0 for tls in self.tls_ids}
		self.last_phase_change = {tls: 0 for tls in self.tls_ids}
		return self.last_obs
		
	def step(self, actions: Dict[str, float], metrics_callback=None) -> Tuple[
		Dict[str, np.ndarray], 
		Dict[str, float], 
		Dict[str, bool], 
		Dict[str, Dict]
	]:
		"""
		Step environment asynchronously.
		Only agents ready for action (phase expired) will apply new actions.
		
		Args:
			actions: Dict mapping agent ID to green duration (only for ready agents)
		
		Returns:
			observations, rewards, dones, infos
		"""
		# For now, assume single agent env (PhaseDurationEnv)
		# TODO: Adapt for true multi-agent env
		
		ready_agents = self.get_agents_ready_for_action()
		#valid_actions = {tls: actions[tls] for tls in ready_agents if tls in actions}
		action_value = np.array([[-1, -1] for tls in self.tls_ids])
		for i, tls in enumerate(self.tls_ids):
			if tls in ready_agents and tls in actions:
				action_value[i] = [self.get_next_phase(tls), actions[tls]]

		#print(action_value)
		obs_flat, reward, terminated, truncated, info = self.env.step(action_value, metrics_callback=metrics_callback)
		rewards = {tls: reward for tls in self.tls_ids}
		dones = {tls: terminated or truncated for tls in self.tls_ids}
		infos = {tls: info for tls in self.tls_ids}
		# if valid_actions:
		# 	# Take action for the single agent
		# 	action_value = list(valid_actions.values())[0]  # Assume one agent
		# 	obs_flat, reward, terminated, truncated, info = self.env.step(np.array([action_value]), metrics_callback=metrics_callback)
		# 	rewards = {self.tls_ids[0]: reward}
		# 	dones = {self.tls_ids[0]: terminated or truncated}
		# 	infos = {self.tls_ids[0]: info}
		# else:
		# 	# No action, step with zero
		# 	obs_flat, reward, terminated, truncated, info = self.env.step(np.array([0.0]), metrics_callback=metrics_callback)
		# 	rewards = {self.tls_ids[0]: reward}
		# 	dones = {self.tls_ids[0]: terminated or truncated}
		# 	infos = {self.tls_ids[0]: info}
		
		# Update phase durations
		for tls in self.tls_ids:
			if self.env.needed_action[tls]:
				self.current_phase_duration[tls] = 0
				self.last_phase_change[tls] = getattr(self.env, 'sumo_step', 0)
			else:
				self.current_phase_duration[tls] += getattr(self.env, 'delta_time', 1)
		
		self.last_obs = self._parse_observations(obs_flat)
		self.last_infos = infos
		
		return self.last_obs, rewards, dones, infos
		
	def _parse_observations(self, obs_flat):
		"""Parse flat observation vector into dict of per-agent observations."""
		# For single agent, just return {tls_id: obs_flat}
		# obs_dict = {}
		# # print(obs_flat)
		# obs_size_per_agent = self.obs_dim_per_agent
		# # print(obs_size_per_agent)
		# for i, tls_id in enumerate(self.tls_ids):
		# 	start_idx = i * obs_size_per_agent
		# 	end_idx = start_idx + obs_size_per_agent
		# 	obs_dict[tls_id] = obs_flat[start_idx:end_idx]
		# # 	print("TLS ID:", tls_id, "start:", start_idx, "end:", end_idx)
		# # 	print("Obs:", obs_dict[tls_id], "Obs flat:", obs_flat[start_idx:end_idx])
		# # print(obs_dict)
		obs_dict = {tls: obs_flat for tls in self.tls_ids}
		return obs_dict
		
	def get_next_phase(self, tls_id: str) -> int:
		"""
		Get the next phase index for the given
		traffic light signal (TLS) ID.
		Assumes phases are ordered in pairs: green phase followed by yellow phase.
		"""
		return (self.env.current_phase[tls_id] + 1) % (len(self.env.tls_phases[tls_id]))
	def get_agents_ready_for_action(self) -> List[str]:
		"""
		Get list of agents ready for action (current phase duration >= min green + yellow).
		Agents are ready when their current green + yellow phase has expired.
		"""
		ready = []
		for tls in self.tls_ids:
			if self.env.needed_action[tls]:
				ready.append(tls)
		return ready
		
	def close(self):
		self.env.close()


class SynchronousMultiAgentWrapper:
	"""
	Wrapper that forces all agents to act synchronously.
	Useful when you want decentralized control but synchronized actions.
		
	All agents provide actions at every step, but each maintains its own
	phase duration internally.
	"""
		
	def __init__(self, env):
		self.env = env
		self.tls_ids = getattr(env, 'tls_ids', [])
		self.obs_dim_per_agent = getattr(env, 'obs_dim_per_agent', None)
		
		# Track when each agent last acted
		self.last_action = {tls_id: None for tls_id in self.tls_ids}
		
	def reset(self) -> Dict[str, np.ndarray]:
		"""Reset environment."""
		obs = self.env.reset()
		self.last_action = {tls_id: None for tls_id in self.tls_ids}
		return obs
		
	def step(self, actions: Dict[str, float]) -> Tuple[
		Dict[str, np.ndarray],
		Dict[str, float],
		Dict[str, bool],
		Dict[str, Dict]
	]:
		"""
		Step with all agents providing actions synchronously.
		Only applies actions to agents that need them.
		
		Args:
			actions: Dict with actions for all agents
		
		Returns:
			observations, rewards, dones, infos
		"""
		# Get agents that actually need actions
		agents_need_action = self.env.get_agents_needing_action()
		
		# Filter actions to only those that are needed
		filtered_actions = {
			tls_id: actions[tls_id] 
			for tls_id in agents_need_action 
			if tls_id in actions
		}
		
		# Store the actions
		for tls_id, action in filtered_actions.items():
			self.last_action[tls_id] = action
		
		# Step environment
		obs, rewards, dones, infos = self.env.step(filtered_actions)
		
		return obs, rewards, dones, infos
		
	def close(self):
		self.env.close()


class VectorizedEnvWrapper:
	"""
	Wrapper to run multiple environments in parallel.
	Useful for faster training with vectorized PPO.
	"""
		
	def __init__(
		self, 
		num_envs: int,
		env_fn,
		**env_kwargs
	):
		"""
		Args:
			num_envs: Number of parallel environments
			env_fn: Function that creates an environment
			**env_kwargs: Arguments to pass to env_fn
		"""
		self.num_envs = num_envs
		self.envs = [env_fn(**env_kwargs) for _ in range(num_envs)]
		
		# Assume all envs have same dimensions
		self.obs_dim = self.envs[0].get_observation_space_dim()
		self.act_dim = self.envs[0].get_action_space_dim()
		
	def reset(self) -> np.ndarray:
		"""
		Reset all environments.
		
		Returns:
			Stacked observations of shape (num_envs, obs_dim)
		"""
		obs_list = [env.reset() for env in self.envs]
		return np.array(obs_list)
		
	def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
		"""
		Step all environments with vectorized actions.
		
		Args:
			actions: Array of shape (num_envs, act_dim)
		
		Returns:
			observations: (num_envs, obs_dim)
			rewards: (num_envs,)
			dones: (num_envs,)
			infos: List of info dicts
		"""
		results = [
			env.step(actions[i]) 
			for i, env in enumerate(self.envs)
		]
		
		obs = np.array([r[0] for r in results])
		rewards = np.array([r[1] for r in results])
		dones = np.array([r[2] for r in results])
		infos = [r[3] for r in results]
		
		# Auto-reset done environments
		for i, done in enumerate(dones):
			if done:
				obs[i] = self.envs[i].reset()
		
		return obs, rewards, dones, infos
		
	def close(self):
		for env in self.envs:
			env.close()
		
	def get_observation_space_dim(self):
		return self.obs_dim
		
	def get_action_space_dim(self):
		return self.act_dim


def create_training_env(
	mode: str = "single",
	gui: bool = False,
	num_parallel: int = 1,
	**env_kwargs
):
	"""
	Factory function to create appropriate environment for training.
		
	Args:
		mode: "single" for centralized, "multi" for decentralized asynchronous,
			  "sync" for decentralized synchronous
		gui: Whether to use SUMO GUI
		num_parallel: Number of parallel environments (if > 1)
		**env_kwargs: Additional arguments for environment
		
	Returns:
		Wrapped environment ready for training
	"""
	if mode == "phase":
		return PhaseDurationEnv(**env_kwargs)

	if mode == "single":
		# return raw SumoTrafficEnv for ad-hoc usage
		return SumoTrafficEnv(**env_kwargs)

	raise NotImplementedError(f"create_training_env supports 'single' or 'phase' modes only")


class PhaseDurationEnv(gym.Env):
	"""
	Wrapper environment where agent predicts the green duration (continuous)
	for a single controlled intersection (TLS). The other TLS remain on
	fixed-time plans defined in `traffic/config.py`.

	Decision points occur at the start of each green phase for the controlled
	TLS. The agent predicts a duration (seconds) which the wrapper enforces
	by stepping the underlying `SumoTrafficEnv` for the requested duration.
	"""

	def __init__(
		self,
		sumo_config=None,
		controlled_tls_index: int = 0,
		min_green: float = 5.0,
		max_green: float = 60.0,
		delta_time: int = 1,
		yellow_time: int = 3,
		max_steps: int = 3600,
		reward_fn: str = 'wait_time',
		use_gui: bool = False,
	):
		super().__init__()
		# Create underlying SumoTrafficEnv with fine-grained delta_time
		self.env = SumoTrafficEnv(
			sumo_config=sumo_config,
			use_gui=use_gui,
			delta_time=delta_time,
			yellow_time=yellow_time,
			max_steps=max_steps,
			reward_fn=reward_fn,
		)

		self.controlled_index = controlled_tls_index
		self.controlled_tls = None
		if len(self.env.tls_ids) > 0:
			if controlled_tls_index < len(self.env.tls_ids):
				self.controlled_tls = self.env.tls_ids[controlled_tls_index]
			else:
				self.controlled_tls = self.env.tls_ids[0]

		# Observation: use the underlying observation (flat vector)
		self.observation_space = self.env.observation_space

		# Action: continuous scalar = green duration in seconds
		self.min_green = min_green
		self.max_green = max_green
		self.action_space = spaces.Box(low=float(min_green), high=float(max_green), shape=(1,), dtype=float)

	def reset(self, **kwargs):
		obs, _ = self.env.reset()
		obs = self.env._get_observation()
		return obs

	def _get_tls_state(self, tls_id):
		import traci
		return traci.trafficlight.getRedYellowGreenState(tls_id)

	def step(self, action, metrics_callback=None):
		"""Apply predicted green duration (seconds) for the current green phase.

		Args:
			action: array-like with shape (1,) giving desired green duration.
		Returns:
			obs, reward, terminated, truncated, info
		"""
		# Clip action
		dur = float(action[0])
		dur = max(self.min_green, min(self.max_green, dur))

		# Identify current green phase index for controlled tls
		tls = self.controlled_tls
		try:
			current_state = self._get_tls_state(tls)
		except Exception as e:
			# If TRACI not connected, return done
			return self.env._get_observation(), 0.0, True, False, {}

		phases = self.env.tls_phases.get(tls, [])
		if self.env.needed_action.get(tls, True):
			# If action needed, set next phase
			phase_idx = (self.env.current_phase[tls] + 1) % len(phases)
		else:
			# Continue current phase
			if current_state in phases:
				phase_idx = phases.index(current_state)
			else:
				phase_idx = 0

		# Create action for SumoTrafficEnv: list of [phase_id, duration] for all tls_ids
		action_for_env = [[-1, -1] for _ in self.env.tls_ids]
		action_for_env[self.env.tls_ids.index(tls)] = [phase_idx, int(dur)]

		for tls_id in self.env.tls_ids:
			if tls_id != self.controlled_tls and self.env.needed_action.get(tls_id, True):
				# Fixed-time plans for other TLS
				fixed_plan = self.env.tls_phases.get(tls_id, [])
				if fixed_plan:
					next_phase_idx = (self.env.current_phase[tls_id] + 1) % len(fixed_plan)
					action_for_env[self.env.tls_ids.index(tls_id)] = [next_phase_idx, 27.5]

		# Step the underlying env to set the action
		#print("Action for env:", action_for_env)
		obs, reward, terminated, truncated, info = self.env.step(action_for_env, metrics_callback=metrics_callback)
		total_reward = reward

		# Continue stepping until the full phase cycle (green + yellow) expires
		while not self.env.needed_action.get(tls, True):
			no_action = [[-1, -1] for _ in self.env.tls_ids]
			for tls_id in self.env.tls_ids:
				if tls_id != self.controlled_tls and self.env.needed_action.get(tls_id, True):
					# Fixed-time plans for other TLS
					fixed_plan = self.env.tls_phases.get(tls_id, [])
					if fixed_plan:
						next_phase_idx = (self.env.current_phase[tls_id] + 1) % len(fixed_plan)
						no_action[self.env.tls_ids.index(tls_id)] = [next_phase_idx, 27.5]
			#print("No action for env:", no_action)
			obs, r, terminated, truncated, info = self.env.step(no_action, metrics_callback=metrics_callback)
			total_reward += r
			if terminated or truncated:
				break

		return obs, total_reward, terminated, truncated, info

	def close(self):
		self.env.close()


# Example usage
if __name__ == "__main__":
	print("Testing environment wrappers...")
		
	# Test single agent wrapper
	print("\n1. Single-Agent Wrapper:")
	env = create_training_env(mode="single", gui=False, max_steps=100)
	print(f"Obs dim: {env.get_observation_space_dim()}")
	print(f"Act dim: {env.get_action_space_dim()}")
		
	obs = env.reset()
	action = np.random.uniform(10, 60, size=env.get_action_space_dim())
	obs, reward, done, info = env.step(action)
	print(f"Step reward: {reward:.2f}")
	env.close()
		
	# Test multi-agent wrapper
	print("\n2. Multi-Agent Wrapper:")
	env = create_training_env(mode="multi", gui=False, max_steps=100)
	obs_dict = env.reset()
	print(f"Agents: {list(obs_dict.keys())}")
		
	agents_need_action = env.get_agents_needing_action()
	actions = {tls_id: np.random.uniform(10, 60) for tls_id in agents_need_action}
	obs_dict, rewards, dones, infos = env.step(actions)
	print(f"Rewards: {rewards}")
	env.close()
		
	# Test vectorized wrapper
	print("\n3. Vectorized Wrapper (4 parallel envs):")
	env = create_training_env(mode="single", gui=False, max_steps=100, num_parallel=4)
	obs = env.reset()
	print(f"Obs shape: {obs.shape}")
		
	actions = np.random.uniform(10, 60, size=(4, env.get_action_space_dim()))
	obs, rewards, dones, infos = env.step(actions)
	print(f"Rewards shape: {rewards.shape}")
	print(f"Rewards: {rewards}")
	env.close()
		
	print("\nAll tests completed!")
