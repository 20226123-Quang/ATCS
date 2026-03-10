"""KPI computation and input-output queue accumulation engine."""


from __future__ import annotations


from dataclasses import dataclass, field
from typing import Dict, Set


from .config_loader import KPIConstants




@dataclass
class LaneRuntimeStats:
    queue_count: float = 0.0
    queue_integral: float = 0.0
    cycle_steps: int = 0
    cycle_inflow_pcu: float = 0.0
    cycle_outflow_pcu: float = 0.0
    green_seconds: float = 0.0
    initial_cycle_queue: float = 0.0
    previous_vehicle_ids: Set[str] = field(default_factory=set)




@dataclass(frozen=True)
class LaneKPI:
    control_delay_seconds: float
    degree_of_saturation: float
    queue_length_meters: float
    capacity_pcu_per_hour: float
    inflow_pcu_per_hour: float
    average_queue_vehicle: float




class KPIEngine:
    """Implements queue accumulation and KPI formulas from config."""


    def __init__(self, constants: KPIConstants):
        self.constants = constants
        self._lane_stats: Dict[str, LaneRuntimeStats] = {}


    def lane_ids(self):
        return self._lane_stats.keys()


    def get_lane_stats(self, lane_id: str) -> LaneRuntimeStats:
        if lane_id not in self._lane_stats:
            self._lane_stats[lane_id] = LaneRuntimeStats()
        return self._lane_stats[lane_id]


    def reset_lane_state(
        self,
        lane_id: str,
        initial_queue: float,
        previous_vehicle_ids: Set[str],
    ) -> None:
        stats = self.get_lane_stats(lane_id)
        stats.queue_count = max(float(initial_queue), 0.0)
        stats.queue_integral = 0.0
        stats.cycle_steps = 0
        stats.cycle_inflow_pcu = 0.0
        stats.cycle_outflow_pcu = 0.0
        stats.green_seconds = 0.0
        stats.initial_cycle_queue = stats.queue_count
        stats.previous_vehicle_ids = set(previous_vehicle_ids)


    def start_new_cycle(self, lane_id: str) -> None:
        stats = self.get_lane_stats(lane_id)
        stats.queue_integral = 0.0
        stats.cycle_steps = 0
        stats.cycle_inflow_pcu = 0.0
        stats.cycle_outflow_pcu = 0.0
        stats.green_seconds = 0.0
        stats.initial_cycle_queue = stats.queue_count


    def mark_lane_green_seconds(self, lane_id: str, delta_seconds: float = 1.0) -> None:
        stats = self.get_lane_stats(lane_id)
        stats.green_seconds += max(0.0, float(delta_seconds))


    def update_lane(
        self,
        lane_id: str,
        inflow_pcu: float,
        outflow_pcu: float,
        current_vehicle_ids: Set[str],
    ) -> None:
        stats = self.get_lane_stats(lane_id)
        inflow = max(0.0, float(inflow_pcu))
        outflow = max(0.0, float(outflow_pcu))


        # Input-output queue accumulation model:
        # Q(t) = max(0, Q(t-1) + V_in(t) - V_out(t))
        stats.queue_count = max(0.0, stats.queue_count + inflow - outflow)
        stats.queue_integral += stats.queue_count
        stats.cycle_steps += 1
        stats.cycle_inflow_pcu += inflow
        stats.cycle_outflow_pcu += outflow
        stats.previous_vehicle_ids = set(current_vehicle_ids)


    def compute_lane_kpis(self, lane_id: str, cycle_length_seconds: float) -> LaneKPI:
        stats = self.get_lane_stats(lane_id)
        constants = self.constants


        eps = constants.epsilon
        cycle_steps = max(stats.cycle_steps, 1)
        cycle_length = max(float(cycle_length_seconds), 1.0)


        q_avg = stats.queue_integral / cycle_steps
        queue_length_m = q_avg * constants.average_vehicle_space_meter


        inflow_pcu_per_hour = stats.cycle_inflow_pcu * 3600.0 / cycle_steps
        g_effective = max(stats.green_seconds, 1.0)
        g_over_c = min(g_effective / cycle_length, 0.999)


        capacity = constants.saturation_flow_pcu_per_hour_per_lane * g_over_c
        v_over_c = inflow_pcu_per_hour / max(capacity, eps)


        # Microscopic Control Delay calculation:
        # Based on actual accumulated queue time.
        # stats.queue_integral is the total vehicle-seconds spent in queue in this cycle.
        total_vehicles = stats.initial_cycle_queue + stats.cycle_inflow_pcu


        if total_vehicles <= eps:
            control_delay = 0.0
        else:
            control_delay = stats.queue_integral / total_vehicles


        control_delay = min(control_delay, constants.max_control_delay_seconds)


        return LaneKPI(
            control_delay_seconds=float(control_delay),
            degree_of_saturation=float(v_over_c),
            queue_length_meters=float(queue_length_m),
            capacity_pcu_per_hour=float(capacity),
            inflow_pcu_per_hour=float(inflow_pcu_per_hour),
            average_queue_vehicle=float(q_avg),
        )