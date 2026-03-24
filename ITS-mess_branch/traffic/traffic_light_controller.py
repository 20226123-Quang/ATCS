import traci
from traffic.config import (
	FIXED_TIME_PLANS,
	TL_STATES,
	DETECTION_DISTANCE,
	LEADING_CAR_ID,
	PRIORITY_DURATION,
)
from traffic.controllers.fixed_time_controller import FixedTimeController

class TrafficLightController:
	def __init__(self, signal_controller=None):
		self.priority_status = {
			tls_id: {
				'active': False,
				'start_time': 0,
				'direction': None
			}
			for tls_id in FIXED_TIME_PLANS
		}
		self.signal_controller = signal_controller or FixedTimeController()

	def get_green_state_for_direction(self, tls_id, incoming_edge):
		reference_state = FIXED_TIME_PLANS[tls_id][0][1]
		new_state = list("r" * len(reference_state))
		if tls_id in TL_STATES and incoming_edge in TL_STATES[tls_id]:
			indices = TL_STATES[tls_id][incoming_edge]
			for idx in indices:
				new_state[idx] = "G"
			print(f"DEBUG: Setting green for {tls_id} from {incoming_edge}, indices: {indices}, state: {''.join(new_state)}")
		else:
			print(f"WARNING: No mapping found for {tls_id} from {incoming_edge}")
			print(f"Available edges: {list(TL_STATES.get(tls_id, {}).keys())}")
		return "".join(new_state)

	def detect_leading_car_at_intersection(self, current_time):
		try:
			if LEADING_CAR_ID not in traci.vehicle.getIDList():
				return None, None
			vehicle_lane = traci.vehicle.getLaneID(LEADING_CAR_ID)
			vehicle_pos = traci.vehicle.getLanePosition(LEADING_CAR_ID)
			current_edge = vehicle_lane.rsplit("_", 1)[0] if "_" in vehicle_lane else vehicle_lane
			lane_length = traci.lane.getLength(vehicle_lane)
			distance_to_end = lane_length - vehicle_pos
			for tls_id in FIXED_TIME_PLANS:
				incoming_lanes = traci.trafficlight.getControlledLanes(tls_id)
				if not incoming_lanes:
					continue
				for lane in incoming_lanes:
					edge_of_lane = lane.rsplit("_", 1)[0] if "_" in lane else lane
					if current_edge == edge_of_lane and distance_to_end <= DETECTION_DISTANCE:
						print(f"[{current_time:.1f}s] LEADING_CAR detected approaching {tls_id} from {current_edge}")
						return tls_id, current_edge
			return None, None
		except Exception as e:
			print(f"Error detecting vehicle: {e}")
			return None, None

	def activate_priority(self, tls_id, incoming_edge, current_time):
		self.priority_status[tls_id]['active'] = True
		self.priority_status[tls_id]['start_time'] = current_time
		self.priority_status[tls_id]['direction'] = incoming_edge
		print(f"[{current_time:.1f}s] PRIORITY ACTIVATED for {tls_id} from {incoming_edge}")

	def deactivate_priority(self, tls_id, current_time):
		self.priority_status[tls_id]['active'] = False
		self.signal_controller.reset(tls_id, current_time)
		print(f"[{current_time:.1f}s] PRIORITY DEACTIVATED for {tls_id}, returning to signal control")

	def apply_signal_control(self, tls_id, current_time):
		self.signal_controller.apply_control(tls_id, current_time)
		
	def set_signal_controller(self, signal_controller):
		self.signal_controller = signal_controller
		print(f"Signal controller switched to {signal_controller.__class__.__name__}")

	def control_traffic_lights(self, current_time):
		for tls_id in FIXED_TIME_PLANS:
			if self.priority_status[tls_id]['active']:
				elapsed = current_time - self.priority_status[tls_id]['start_time']
				if elapsed >= PRIORITY_DURATION:
					self.deactivate_priority(tls_id, current_time)
					self.apply_signal_control(tls_id, current_time)
				else:
					try:
						priority_direction = self.priority_status[tls_id]['direction']
						green_state = self.get_green_state_for_direction(tls_id, priority_direction)
						traci.trafficlight.setRedYellowGreenState(tls_id, green_state)
					except Exception as e:
						print(f"Error applying priority for {tls_id}: {e}")
					continue
			detected_edge = self.detect_leading_car_at_intersection_for_tls(tls_id, current_time)
			if detected_edge:
				self.activate_priority(tls_id, detected_edge, current_time)
				try:
					green_state = self.get_green_state_for_direction(tls_id, detected_edge)
					traci.trafficlight.setRedYellowGreenState(tls_id, green_state)
				except Exception as e:
					print(f"Error applying priority for {tls_id}: {e}")
			else:
				self.apply_signal_control(tls_id, current_time)

	def detect_leading_car_at_intersection_for_tls(self, tls_id, current_time):
		try:
			if LEADING_CAR_ID not in traci.vehicle.getIDList():
				return None
			vehicle_lane = traci.vehicle.getLaneID(LEADING_CAR_ID)
			vehicle_pos = traci.vehicle.getLanePosition(LEADING_CAR_ID)
			current_edge = vehicle_lane.rsplit("_", 1)[0] if "_" in vehicle_lane else vehicle_lane
			lane_length = traci.lane.getLength(vehicle_lane)
			distance_to_end = lane_length - vehicle_pos
			incoming_lanes = traci.trafficlight.getControlledLanes(tls_id)
			if not incoming_lanes:
				return None
			for lane in incoming_lanes:
				edge_of_lane = lane.rsplit("_", 1)[0] if "_" in lane else lane
				if current_edge == edge_of_lane and distance_to_end <= DETECTION_DISTANCE:
					if not self.priority_status[tls_id]['active']:
						print(f"[{current_time:.1f}s] LEADING_CAR detected approaching {tls_id} from {current_edge}")
					return current_edge
			return None
		except Exception as e:
			print(f"Error detecting vehicle for {tls_id}: {e}")
			return None
