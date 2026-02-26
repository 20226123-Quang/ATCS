"""
SUMO Traffic Light RL Environment
Flexible environment that works with any SUMO config file
"""
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import xml.etree.ElementTree as ET
from traffic.config import FIXED_TIME_PLANS, SUMO_BINARY


class SumoTrafficEnv(gym.Env):
	"""
	SUMO-based traffic light control environment for RL training.
		
	Args:
		sumo_config: Path to SUMO config file (.sumocfg)
		use_gui: Whether to use SUMO GUI (default: False)
		delta_time: Fixed time duration for each action (default: 5 seconds)
		yellow_time: Duration of yellow phase (default: 3 seconds)
		max_steps: Maximum simulation steps (default: 3600)
		reward_fn: Reward function name ('queue', 'wait_time', 'throughput')
	"""
		
	def __init__(
		self,
		sumo_config=None,
		use_gui=False,
		delta_time=5,
		yellow_time=3,
		max_steps=3600,
		reward_fn='wait_time',
		min_green_time=5,
	):
		super().__init__()
		
		# Load config
		if sumo_config is None:
			from traffic.config import SUMO_CONFIG_PATH
			sumo_config = SUMO_CONFIG_PATH
		
		self.sumo_config = sumo_config
		self.use_gui = use_gui
		self.delta_time = delta_time
		self.yellow_time = yellow_time
		self.max_steps = max_steps
		self.reward_fn_name = reward_fn
		self.min_green_time = min_green_time
		
		# Parse SUMO network to get traffic light info
		self._parse_network()
		
		# Simulation state
		self.current_step = 0
		self.sumo_step = 0
		self.connection_label = f"sumo_{id(self)}"
		self.is_connected = False
		
		# Track phase timing
		self.current_phase_duration = {tls: 0 for tls in self.tls_ids}
		self.last_phase_change = {tls: 0 for tls in self.tls_ids}
		self.remain_phase_duration = {tls: 0 for tls in self.tls_ids}
		self.remain_yellow_duration = {tls: 0 for tls in self.tls_ids}
		self.needed_action = {tls: True for tls in self.tls_ids}
		self.current_phase = {tls: 0 for tls in self.tls_ids}
		
		# Define action and observation spaces
		self._setup_spaces()
		
		# Metrics tracking
		self.total_waiting_time = 0
		self.total_throughput = 0
		self.episode_reward = 0
		
	def _parse_network(self):
		"""Parse SUMO config and network files to extract traffic light information"""
		# Get network file path from config
		config_dir = os.path.dirname(self.sumo_config)
		tree = ET.parse(self.sumo_config)
		root = tree.getroot()
		
		net_file = None
		for input_elem in root.findall('input'):
			for net_elem in input_elem.findall('net-file'):
				net_file = net_elem.get('value')
		
		if not net_file:
			for net_elem in root.findall('.//net-file'):
				net_file = net_elem.get('value')
		
		if not net_file:
			raise ValueError(f"Could not find net-file in {self.sumo_config}")
		
		net_path = os.path.join(config_dir, net_file)
		if not os.path.exists(net_path):
			raise FileNotFoundError(f"Network file not found: {net_path}")
		
		# Parse network file
		net_tree = ET.parse(net_path)
		net_root = net_tree.getroot()
		
		# Get traffic light IDs and their phases
		self.tls_ids = []
		self.tls_phases = {}  # {tls_id: [phase_states]}
		self.tls_phases_full = {}  # {tls_id: all phases including yellow}
		self.tls_controlled_lanes = {}  # {tls_id: [lane_ids]}
		
		for tl in net_root.findall('tlLogic'):
			tls_id = tl.get('id')
			if tls_id:
				self.tls_ids.append(tls_id)
				
				# Use FIXED_TIME_PLANS from config for green phases
				if tls_id in FIXED_TIME_PLANS:
					# Get only green phases (even indices: 0, 2, 4, 6)
					green_phases = []
					for i, (duration, state) in enumerate(FIXED_TIME_PLANS[tls_id]):
						if i % 2 == 0:  # Even index = green phase
							green_phases.append(state)
					self.tls_phases[tls_id] = green_phases
					self.tls_phases_full[tls_id] = FIXED_TIME_PLANS[tls_id]
				else:
					# Fallback: parse from network file
					all_phases = []
					all_phases_full = []
					phase_idx = 0
					for phase in tl.findall('phase'):
						state = phase.get('state')
						if state:
							all_phases_full.append((phase.get('duration', '0'), state))
							if phase_idx % 2 == 0 and ('G' in state or 'g' in state) and 'y' not in state:
								all_phases.append(state)
						phase_idx += 1
					
					self.tls_phases_full[tls_id] = all_phases_full
					self.tls_phases[tls_id] = all_phases[:4] if len(all_phases) >= 4 else all_phases if all_phases else ['']
		
		# Get controlled lanes for each traffic light from connections
		for tls_id in self.tls_ids:
			controlled_lanes = set()
			for conn in net_root.findall('connection'):
				if conn.get('tl') == tls_id:
					from_lane = conn.get('from')
					if from_lane:
						# Get the lane ID (edge + lane index)
						to_lane = conn.get('toLane')
						if to_lane:
							controlled_lanes.add(to_lane.rsplit('_', 1)[0] + '_' + conn.get('fromLane', '0'))
			
			self.tls_controlled_lanes[tls_id] = list(controlled_lanes) if controlled_lanes else []
		
		# Get all edges for metrics
		self.edge_ids = []
		for edge in net_root.findall('edge'):
			eid = edge.get('id')
			if eid and not edge.get('function'):  # Skip internal edges
				self.edge_ids.append(eid)
		
		print(f"Loaded network: {len(self.tls_ids)} traffic lights")
		for tls_id in self.tls_ids:
			print(f"  {tls_id}: {len(self.tls_phases[tls_id])} green phases (from FIXED_TIME_PLANS)")
			for i, phase in enumerate(self.tls_phases[tls_id]):
				print(f"	Phase {i}: {phase}")
		
	def _setup_spaces(self):
		"""Setup observation and action spaces based on network topology"""
		# Action space: for each traffic light, select which phase to activate
		# Action is a list of phase indices, one per traffic light
		if len(self.tls_ids) == 1:
			# Single traffic light: discrete action space
			tls_id = self.tls_ids[0]
			self.action_space = spaces.Discrete(len(self.tls_phases[tls_id]))
		else:
			# Multiple traffic lights: multi-discrete action space
			self.action_space = spaces.MultiDiscrete(
				[len(self.tls_phases[tls_id]) for tls_id in self.tls_ids]
			)
		
		# Observation space: queue length, waiting time, and current phase for each TLS
		# For simplicity, we use a fixed-size observation vector
		# Features per TLS: 
		#   - queue length on each controlled lane (normalized)
		#   - average waiting time on each controlled lane (normalized)
		#   - current phase (one-hot encoded)
		#   - time since last phase change (normalized)
		
		max_lanes = max(len(lanes) for lanes in self.tls_controlled_lanes.values()) if self.tls_controlled_lanes else 4
		max_phases = max(len(phases) for phases in self.tls_phases.values()) if self.tls_phases else 4
		
		obs_size_per_tls = max_lanes * 2 + max_phases + 1  # queue + wait + phase_onehot + time_since_change
		total_obs_size = obs_size_per_tls * len(self.tls_ids)
		
		self.observation_space = spaces.Box(
			low=0,
			high=1,
			shape=(total_obs_size,),
			dtype=np.float32
		)
		
		self.max_lanes = max_lanes
		self.max_phases = max_phases
		self.obs_size_per_tls = obs_size_per_tls
		
	def _start_sumo(self):
		"""Start SUMO simulation"""
		if self.is_connected:
			try:
				traci.close()
			except:
				pass
		# Determine SUMO binary to run. Prefer configured SUMO_BINARY.
		sumo_bin = None
		try:
			if self.use_gui:
				# Prefer sumo-gui in same directory as configured SUMO_BINARY
				if SUMO_BINARY:
					gui_candidate = None
					try:
						base_dir = os.path.dirname(SUMO_BINARY)
						gui_candidate = os.path.join(base_dir, 'sumo-gui.exe')
						if not os.path.exists(gui_candidate):
							# Try without .exe or different naming
							gui_candidate = os.path.join(base_dir, 'sumo-gui')
						if os.path.exists(gui_candidate):
							sumo_bin = gui_candidate
					except Exception:
						sumo_bin = None
				# fallback to 'sumo-gui' on PATH
				if sumo_bin is None:
					sumo_bin = 'sumo-gui'
			else:
				sumo_bin = SUMO_BINARY if SUMO_BINARY else 'sumo'
		except Exception:
			sumo_bin = 'sumo-gui' if self.use_gui else 'sumo'

		sumo_cmd = [
			sumo_bin,
			'-c', self.sumo_config,
			'--no-step-log', 'true',
			'--waiting-time-memory', '1000',
			'--no-warnings', 'true',
			'--duration-log.disable', 'true',
			'--step-length', '1',
		]

		traci.start(sumo_cmd, label=self.connection_label)
		self.is_connected = True
		traci.switch(self.connection_label)
		
	def _get_observation(self):
		"""Get current observation from SUMO"""
		obs = []
		
		for tls_id in self.tls_ids:
			# Get controlled lanes
			try:
				controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
				controlled_lanes = list(set(controlled_lanes))  # Remove duplicates
			except:
				controlled_lanes = []
			
			# Queue lengths (number of halted vehicles)
			queue_lengths = []
			for lane in controlled_lanes[:self.max_lanes]:
				try:
					queue = traci.lane.getLastStepHaltingNumber(lane)
					queue_lengths.append(min(queue / 10.0, 1.0))  # Normalize to [0, 1]
				except:
					queue_lengths.append(0.0)
			
			# Pad if fewer lanes
			while len(queue_lengths) < self.max_lanes:
				queue_lengths.append(0.0)
			
			# Waiting times
			wait_times = []
			for lane in controlled_lanes[:self.max_lanes]:
				try:
					wait = traci.lane.getWaitingTime(lane)
					wait_times.append(min(wait / 100.0, 1.0))  # Normalize to [0, 1]
				except:
					wait_times.append(0.0)
			
			# Pad if fewer lanes
			while len(wait_times) < self.max_lanes:
				wait_times.append(0.0)
			
			# Current phase (one-hot encoded)
			try:
				current_state = traci.trafficlight.getRedYellowGreenState(tls_id)
				if current_state in self.tls_phases[tls_id]:
					phase_idx = self.tls_phases[tls_id].index(current_state)
				else:
					phase_idx = 0
			except:
				phase_idx = 0
			
			phase_onehot = [0.0] * self.max_phases
			if phase_idx < self.max_phases:
				phase_onehot[phase_idx] = 1.0
			
			# Time since last phase change (normalized to [0, 1])
			time_since_change = min(self.current_phase_duration[tls_id] / 60.0, 1.0)
			
			# Combine features
			tls_obs = queue_lengths + wait_times + phase_onehot + [time_since_change]
			obs.extend(tls_obs)
		
		return np.array(obs, dtype=np.float32)
		
	def _compute_reward(self):
		"""Compute reward based on the selected reward function"""
		if self.reward_fn_name == 'queue':
			# Negative of total queue length
			total_queue = 0
			for tls_id in self.tls_ids:
				try:
					controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
					for lane in set(controlled_lanes):
						total_queue += traci.lane.getLastStepHaltingNumber(lane)
				except:
					pass
			return -total_queue
		
		elif self.reward_fn_name == 'wait_time':
			# Negative of total waiting time
			total_wait = 0
			for tls_id in self.tls_ids:
				try:
					controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
					for lane in set(controlled_lanes):
						total_wait += traci.lane.getWaitingTime(lane)
				except:
					pass
			return -total_wait / 100.0  # Scale down
		
		elif self.reward_fn_name == 'throughput':
			# Number of vehicles that passed
			throughput = 0
			for edge_id in self.edge_ids:
				try:
					throughput += traci.edge.getLastStepVehicleNumber(edge_id)
				except:
					pass
			return throughput
		
		else:
			# Default: negative waiting time
			total_wait = 0
			for tls_id in self.tls_ids:
				try:
					controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
					for lane in set(controlled_lanes):
						total_wait += traci.lane.getWaitingTime(lane)
				except:
					pass
			return -total_wait / 100.0
		
	# def _apply_action(self, action):
	# 	"""Apply action to traffic lights with 4 green phases"""
	# 	# Convert action to list if single value
	# 	if isinstance(action, (int, np.integer)):
	# 		action = [action]
		
	# 	print(action)
	# 	print(self.tls_phases)
	# 	for i, tls_id in enumerate(self.tls_ids):
	# 		if i >= len(action):
	# 			break
			
	# 		phase_idx = int(action[i])
			
	# 		# Check if enough time has passed since last change
	# 		if self.current_phase_duration[tls_id] < self.min_green_time:
	# 			continue  # Don't change phase yet
			
	# 		# Get target phase (should be 0-3 for 4 green phases)
	# 		if phase_idx >= len(self.tls_phases[tls_id]):
	# 			phase_idx = 0
			
	# 		target_phase = self.tls_phases[tls_id][phase_idx]
			
	# 		# Get current state
	# 		try:
	# 			current_state = traci.trafficlight.getRedYellowGreenState(tls_id)
				
	# 			# If changing phase, insert yellow phase first
	# 			if current_state != target_phase:
	# 				# Create yellow phase (replace G/g with y)
	# 				yellow_state = ''.join(['y' if c in 'Gg' else c for c in current_state])
	# 				traci.trafficlight.setRedYellowGreenState(tls_id, yellow_state)
					
	# 				# Wait for yellow time (run simulation steps)
	# 				for _ in range(self.yellow_time):
	# 					traci.simulationStep()
	# 					self.sumo_step += 1
					
	# 				# Set target phase
	# 				traci.trafficlight.setRedYellowGreenState(tls_id, target_phase)
	# 				self.current_phase_duration[tls_id] = 0
	# 				self.last_phase_change[tls_id] = self.sumo_step
					
	# 		except Exception as e:
	# 			print(f"Error applying action to {tls_id}: {e}")
		
	def reset(self, seed=None, options=None):
		"""Reset the environment"""
		super().reset(seed=seed)
		
		# Close previous connection
		if self.is_connected:
			try:
				traci.close()
				self.is_connected = False
			except:
				pass
		
		# Start new simulation
		self._start_sumo()
		
		# Reset counters
		self.current_step = 0
		self.sumo_step = 0
		self.current_phase_duration = {tls: 0 for tls in self.tls_ids}
		self.last_phase_change = {tls: 0 for tls in self.tls_ids}
		self.remain_phase_duration = {tls: 0 for tls in self.tls_ids}
		self.remain_yellow_duration = {tls: 0 for tls in self.tls_ids}
		self.needed_action = {tls: True for tls in self.tls_ids}
		self.total_waiting_time = 0
		self.total_throughput = 0
		self.episode_reward = 0
		
		# Get initial observation
		obs = self._get_observation()
		
		return obs, {}
	
	def apply_action(self, action):
		for i, tls_id in enumerate(self.tls_ids):
			if self.needed_action[tls_id] == False and action[i][0] != -1:
				print("Skipping action application as not needed yet");
				continue;
			if self.needed_action[tls_id] == True and action[i][0] == -1:
				print("WARNING: Action needed but no action provided for tls ", tls_id);
				continue;
			if self.needed_action[tls_id] == False and action[i][0] == -1:
				continue;
			phase_id, duration = action[i]
			self.current_phase[tls_id] = int(phase_id)
			self.remain_phase_duration[tls_id] = int(duration)
			self.remain_yellow_duration[tls_id] = self.yellow_time
			self.needed_action[tls_id] = False
			target_phase = self.tls_phases[tls_id][phase_id]
			traci.trafficlight.setRedYellowGreenState(tls_id, target_phase)
				
	def step(self, action, metrics_callback=None):
		"""
		Execute one step in the environment.
		
		Args:
			action: Action to take (phase index) (example [[-1, -1], [0, 3], [2, 2]] for 3 TLS)
			metrics_callback: Optional callback function called after each traci step.
							Receives: (current_time, step_within_action, total_delta_time)
		
		Returns:
			obs, reward, terminated, truncated, info
		"""
		# Apply action
		self.apply_action(action)
		
		# Simulate for delta_time steps
		for step_idx in range(self.delta_time):
			traci.simulationStep()
			self.sumo_step += 1
			
			# Call metrics callback if provided
			if metrics_callback is not None:
				try:
					current_time = traci.simulation.getTime()
					metrics_callback(current_time, step_idx, self.delta_time)
				except Exception as e:
					print(f"Error in metrics callback: {e}")
			
			# Update phase durations
			for tls_id in self.tls_ids:
				self.current_phase_duration[tls_id] += 1
				if(self.remain_phase_duration[tls_id] == 0):
					if(self.remain_yellow_duration[tls_id] > 0):
						self.remain_yellow_duration[tls_id] = max(0, self.remain_yellow_duration[tls_id] - 1)
					if(self.remain_yellow_duration[tls_id] == 1):
						self.needed_action[tls_id] = True
				if(self.remain_phase_duration[tls_id] == 1):
					traci.trafficlight.setRedYellowGreenState(tls_id, ''.join(['y' if c in 'Gg' else c for c in traci.trafficlight.getRedYellowGreenState(tls_id)]))	
				if(self.remain_phase_duration[tls_id] > 0):
					self.remain_phase_duration[tls_id] = max(0, self.remain_phase_duration[tls_id] - 1)
				
		
		self.current_step += 1
		
		# Get new observation
		obs = self._get_observation()
		
		# Calculate reward
		reward = self._compute_reward()
		self.episode_reward += reward
		
		# Check if done
		terminated = self.sumo_step >= self.max_steps
		truncated = False
		
		# Check if simulation ended
		try:
			if traci.simulation.getMinExpectedNumber() <= 0:
				terminated = True
		except:
			terminated = True
		
		# Info
		info = {
			'step': self.current_step,
			'sumo_step': self.sumo_step,
			'episode_reward': self.episode_reward,
		}
		
		return obs, reward, terminated, truncated, info
		
	def close(self):
		"""Close the environment"""
		if self.is_connected:
			try:
				traci.close()
				self.is_connected = False
			except:
				pass
		
	def render(self):
		"""Render the environment (handled by SUMO GUI)"""
		pass


if __name__ == "__main__":
	# Test the environment
	from traffic.config import SUMO_CONFIG_PATH
		
	env = SumoTrafficEnv(
		sumo_config=SUMO_CONFIG_PATH,
		use_gui=False,
		delta_time=5,
		yellow_time=3,
		max_steps=1000,
		reward_fn='wait_time'
	)
		
	print(f"Action space: {env.action_space}")
	print(f"Observation space: {env.observation_space}")
	print(f"Traffic lights: {env.tls_ids}")
		
	obs, info = env.reset()
	print(f"Initial observation shape: {obs.shape}")
		
	for i in range(10):
		action = env.action_space.sample()
		obs, reward, terminated, truncated, info = env.step(action)
		print(f"Step {i+1}: reward={reward:.2f}, terminated={terminated}")
		
		if terminated or truncated:
			break
		
	env.close()
	print("Test completed!")
