import traci
from traffic.config import FIXED_TIME_PLANS, PCU_MAPPING, VEHICLE_STOP_THRESHOLD

class MetricsCollector:
	def __init__(self):
		# Vehicle lane tracking
		self.vehicles_in_intersection = {}
		# Completed wait times
		self.completed_wait_times = []
		
		# Queue length tracking
		self.previous_tls_states = {tls_id: None for tls_id in FIXED_TIME_PLANS}
		self.queue_lengths = []
		
		# Travel time tracking
		self.vehicles_in_network = {}
		self.completed_travel_times = []
		
	def print_metrics(self):
		print("\n========== Traffic Metrics ==========")
		print(f"Completed vehicles: {len(self.completed_travel_times)}")
		print(f"Average wait time      : {self.get_average_wait_time():.2f}")
		print(f"Average queue length   : {self.get_average_queue_length():.2f}")
		print(f"Max queue length       : {self.get_max_queue_length():.2f}")
		print(f"Average travel time    : {self.get_average_travel_time():.2f}")
		print(f"Max travel time        : {self.get_max_travel_time():.2f}")
		print(f"Min travel time        : {self.get_min_travel_time():.2f}")
		print("====================================\n")
		
	def get_pcu_value(self, vehicle_id):
		"""Get PCU value for a vehicle based on its type"""
		try:
			vtype = traci.vehicle.getTypeID(vehicle_id)
			if vtype in PCU_MAPPING:
				return PCU_MAPPING[vtype]
			vtype_lower = vtype.lower()
			for vehicle_type, pcu in PCU_MAPPING.items():
				if vehicle_type in vtype_lower:
					return pcu
			return PCU_MAPPING['passenger']
		except:
			return PCU_MAPPING['passenger']
		
	def update_metrics(self, current_time):
		self.update_wait_times(current_time)
		self.update_queue_lengths(current_time)
		self.update_travel_times(current_time)
		
	def update_wait_times(self, current_time):
		current_vehicles = traci.vehicle.getIDList()
		
		for vehicle_id in current_vehicles:
			try:
				speed = traci.vehicle.getSpeed(vehicle_id)
				lane = traci.vehicle.getLaneID(vehicle_id)
				
				# Extract edge from lane ID (format: edgeID_laneIndex)
				edge = lane.rsplit("_", 1)[0] if "_" in lane else lane
				
				# Check if vehicle is in a controlled lane (approaching intersection)
				is_in_controlled_lane = False
				for tls_id in FIXED_TIME_PLANS:
					controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
					if lane in controlled_lanes:
						is_in_controlled_lane = True
						break
				
				# Vehicle enters intersection lane (first time detected in controlled lane)
				if is_in_controlled_lane and vehicle_id not in self.vehicles_in_intersection:
					self.vehicles_in_intersection[vehicle_id] = {
						'in_lane': True,
						'entry_time': None,
						'lane': lane,
						'edge': edge,
						'pcu': self.get_pcu_value(vehicle_id)
					}
				
				# Vehicle is in lane but hasn't stopped yet -> capture first stop time
				elif is_in_controlled_lane and vehicle_id in self.vehicles_in_intersection:
					vehicle_data = self.vehicles_in_intersection[vehicle_id]
					if speed < VEHICLE_STOP_THRESHOLD and vehicle_data['entry_time'] is None:
						vehicle_data['entry_time'] = current_time
				
				# Vehicle leaves intersection (was in controlled lane, now not)
				elif not is_in_controlled_lane and vehicle_id in self.vehicles_in_intersection:
					entry_data = self.vehicles_in_intersection.pop(vehicle_id)
					if entry_data['entry_time'] is not None:
						wait_time = current_time - entry_data['entry_time']
						self.completed_wait_times.append({
							'vehicle_id': vehicle_id,
							'entry_time': entry_data['entry_time'],
							'exit_time': current_time,
							'wait_time': wait_time,
							'pcu': entry_data['pcu']
						})
			except Exception as e:
				print(f"Error in update_wait_times for {vehicle_id}: {e}")
		
	def get_current_waiting_time_at_tls(self, tls_id, current_time):
		total_wait_time = 0.0
		controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
		for lane in controlled_lanes:
			vehicles_in_lane = traci.lane.getLastStepVehicleIDs(lane)
			for vehicle_id in vehicles_in_lane:
				try:
					speed = traci.vehicle.getSpeed(vehicle_id)
					if speed < VEHICLE_STOP_THRESHOLD:
						if vehicle_id in self.vehicles_in_intersection:
							vehicle_data = self.vehicles_in_intersection[vehicle_id]
							if vehicle_data['entry_time'] is not None:
								wait_time = current_time - vehicle_data['entry_time']
								total_wait_time += wait_time * vehicle_data['pcu']
				except Exception as e:
					print(f"Error calculating current wait time for vehicle {vehicle_id}: {e}")
		return total_wait_time
	def get_average_wait_time(self):
		if not self.completed_wait_times:
			return 0.0
		total_weighted_wait = sum(wt['wait_time'] * wt['pcu'] for wt in self.completed_wait_times)
		total_pcu = sum(wt['pcu'] for wt in self.completed_wait_times)
		if total_pcu == 0:
			return 0.0
		return total_weighted_wait / total_pcu
		
	def update_queue_lengths(self, current_time):
		for tls_id in FIXED_TIME_PLANS:
			try:
				current_state = traci.trafficlight.getRedYellowGreenState(tls_id)
				previous_state = self.previous_tls_states[tls_id]
				is_red_to_green = False
				if previous_state is not None:
					for prev_char, curr_char in zip(previous_state, current_state):
						if prev_char == 'r' and curr_char == 'G':
							is_red_to_green = True
							break
				if is_red_to_green:
					queue_length_pcu = self._calculate_queue_length_for_tls(tls_id)
					self.queue_lengths.append({
						'tls_id': tls_id,
						'time': current_time,
						'queue_length_pcu': queue_length_pcu
					})
				self.previous_tls_states[tls_id] = current_state
			except Exception as e:
				print(f"Error updating queue lengths for {tls_id}: {e}")
		
	def _calculate_queue_length_for_tls(self, tls_id):
		try:
			controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
			queue_length_pcu = 0.0
			for lane in controlled_lanes:
				vehicles_in_lane = traci.lane.getLastStepVehicleIDs(lane)
				for vehicle_id in vehicles_in_lane:
					try:
						speed = traci.vehicle.getSpeed(vehicle_id)
						if speed < VEHICLE_STOP_THRESHOLD:
							pcu = self.get_pcu_value(vehicle_id)
							queue_length_pcu += pcu
					except Exception as e:
						print(f"Error calculating queue for vehicle {vehicle_id}: {e}")
			return queue_length_pcu
		except Exception as e:
			print(f"Error calculating queue length for {tls_id}: {e}")
			return 0.0
		
	def get_average_queue_length(self):
		if not self.queue_lengths:
			return 0.0
		total_queue = sum(ql['queue_length_pcu'] for ql in self.queue_lengths)
		return total_queue / len(self.queue_lengths)
		
	def get_max_queue_length(self):
		if not self.queue_lengths:
			return 0.0
		return max(ql['queue_length_pcu'] for ql in self.queue_lengths)
		
	def update_travel_times(self, current_time):
		current_vehicles = traci.vehicle.getIDList()
		for vehicle_id in current_vehicles:
			if vehicle_id not in self.vehicles_in_network:
				try:
					self.vehicles_in_network[vehicle_id] = {
						'entry_time': current_time,
						'pcu': self.get_pcu_value(vehicle_id)
					}
				except Exception as e:
					print(f"Error tracking vehicle {vehicle_id} entry: {e}")
		vehicles_to_remove = []
		for vehicle_id in self.vehicles_in_network:
			if vehicle_id not in current_vehicles:
				travel_data = self.vehicles_in_network[vehicle_id]
				travel_time = current_time - travel_data['entry_time']
				self.completed_travel_times.append({
					'vehicle_id': vehicle_id,
					'entry_time': travel_data['entry_time'],
					'exit_time': current_time,
					'travel_time': travel_time,
					'pcu': travel_data['pcu']
				})
				vehicles_to_remove.append(vehicle_id)
		for vehicle_id in vehicles_to_remove:
			del self.vehicles_in_network[vehicle_id]
		
	def get_average_travel_time(self):
		if not self.completed_travel_times:
			return 0.0
		total_travel = sum(tt['travel_time'] for tt in self.completed_travel_times)
		return total_travel / len(self.completed_travel_times)
		
	def get_max_travel_time(self):
		if not self.completed_travel_times:
			return 0.0
		return max(tt['travel_time'] for tt in self.completed_travel_times)
		
	def get_min_travel_time(self):
		if not self.completed_travel_times:
			return 0.0
		return min(tt['travel_time'] for tt in self.completed_travel_times)
	
	def reset(self):
		self.vehicles_in_intersection = {}
		self.completed_wait_times = []
		self.previous_tls_states = {tls_id: None for tls_id in FIXED_TIME_PLANS}
		self.queue_lengths = []
		self.vehicles_in_network = {}
		self.completed_travel_times = []
	"""Collects and calculates traffic metrics during simulation"""