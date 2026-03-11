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

        # Prevent astronomical flow rate spikes when a vehicle arrives at the start of a cycle
        # by treating the expected flow period as at least the full cycle length.
        # TCCS 24:2018 is a macroscopic formula that expects stable q.
        effective_period = max(float(cycle_steps), cycle_length)
        inflow_pcu_per_hour = stats.cycle_inflow_pcu * 3600.0 / effective_period
        g_effective = max(stats.green_seconds, eps)
        g_over_c = min(g_effective / cycle_length, 0.999)

        S = constants.saturation_flow_pcu_per_hour_per_lane
        capacity = S * g_over_c
        v_over_c = inflow_pcu_per_hour / max(capacity, eps)  # This is g in TCCS 24:2018

        # -------------------------------------------------------------
        # 1. Chiều dài hàng chờ (N_GE) theo HBS 2001 (TCCS 24:2018 F-21)
        # -------------------------------------------------------------
        m_max = float(g_effective * S / 3600.0)
        m_tb = float(inflow_pcu_per_hour * cycle_length / 3600.0)
        g_val = float(v_over_c)

        if g_val <= 0.65:
            n_ge_vehicles = 0.0
        elif g_val <= 0.9:
            val_065 = 0.0
            val_09 = 1.0 / (0.26 + m_tb / 150.0) if (0.26 + m_tb / 150.0) > 0 else 0.0
            n_ge_vehicles = val_065 + (g_val - 0.65) / (0.9 - 0.65) * (val_09 - val_065)
        elif g_val <= 1.0:
            val_09 = 1.0 / (0.26 + m_tb / 150.0) if (0.26 + m_tb / 150.0) > 0 else 0.0
            val_10 = float(0.3476 * (max(m_max, 0.0) ** 0.5))
            n_ge_vehicles = val_09 + (g_val - 0.9) / (1.0 - 0.9) * (val_10 - val_09)
        elif g_val <= 1.2:
            val_10 = float(0.3476 * (max(m_max, 0.0) ** 0.5))
            val_12 = float((m_max * (1.2 - 1.0) + 25.0 - 20.0 * 1.2) / 2.0)
            n_ge_vehicles = val_10 + (g_val - 1.0) / (1.2 - 1.0) * (val_12 - val_10)
        else:
            n_ge_vehicles = float(m_max * (g_val - 1.0) / 2.0)

        n_ge_vehicles = max(0.0, float(n_ge_vehicles))
        queue_length_m = n_ge_vehicles * constants.average_vehicle_space_meter

        # -------------------------------------------------------------
        # 2. Thời gian trễ (Average Control Delay t_w) theo HBS 2001 (TCCS 24:2018)
        # -------------------------------------------------------------
        q_over_S = min(
            inflow_pcu_per_hour / S, 0.999
        )  # limit to avoid div by zero in t_w1

        # t_w1: Thời gian trễ cơ bản
        t_w1 = (cycle_length * ((1.0 - g_over_c) ** 2)) / (
            2.0 * max(1.0 - q_over_S, eps)
        )

        # t_w2: Thời gian trễ do tắc nghẽn
        t_w2 = (3600.0 * n_ge_vehicles) / max(g_over_c * S, eps)

        control_delay = t_w1 + t_w2
        control_delay = min(control_delay, constants.max_control_delay_seconds)

        return LaneKPI(
            control_delay_seconds=float(control_delay),
            degree_of_saturation=float(v_over_c),
            queue_length_meters=float(queue_length_m),
            capacity_pcu_per_hour=float(capacity),
            inflow_pcu_per_hour=float(inflow_pcu_per_hour),
            average_queue_vehicle=float(n_ge_vehicles),
        )
