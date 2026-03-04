import traci
from traffic.config import FIXED_TIME_PLANS
from traffic.controllers.signal_controller import SignalController

class FixedTimeController(SignalController):
    """Manages fixed-time signal control for traffic lights"""
    
    def __init__(self):
        self.fixed_time_cycle_start = {tls_id: 0 for tls_id in FIXED_TIME_PLANS}
        self.cycle_time = {tls_id: sum(duration for duration, _ in FIXED_TIME_PLANS[tls_id])
                          for tls_id in FIXED_TIME_PLANS}
    
    def apply_control(self, tls_id, current_time, **kwargs):
        plan = FIXED_TIME_PLANS[tls_id]
        cycle_duration = self.cycle_time[tls_id]
        elapsed = (current_time - self.fixed_time_cycle_start[tls_id]) % cycle_duration
        current_position = 0
        for duration, state in plan:
            if current_position + duration > elapsed:
                try:
                    traci.trafficlight.setRedYellowGreenState(tls_id, state)
                except Exception as e:
                    print(f"Error setting state for {tls_id}: {e}")
                return
            current_position += duration
    
    def reset(self, tls_id, current_time):
        self.fixed_time_cycle_start[tls_id] = current_time
