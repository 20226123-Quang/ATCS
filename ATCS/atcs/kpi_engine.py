"""KPI computation and input-output queue accumulation engine."""

from __future__ import annotations


from dataclasses import dataclass, field
from typing import Dict, Optional, Set


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
    phase_inflow_pcu: float = 0.0
    phase_outflow_pcu: float = 0.0
    phase_steps: int = 0
    phase_queue_start: float = 0.0
    residual_queue_vehicles: float = 0.0
    split_failure_rate: float = 0.0
    pending_split_failure_rate: float = 0.0
    last_green_demand_pcu: float = 0.0
    time_since_last_service_seconds: float = 0.0


@dataclass(frozen=True)
class LaneKPI:
    control_delay_seconds: float
    degree_of_saturation: float
    queue_length_meters: float
    capacity_pcu_per_hour: float
    inflow_pcu_per_hour: float
    average_queue_vehicle: float
    total_demand_pcu_per_hour: float
    residual_n_ge_vehicles: float


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

    def lane_has_current_demand(self, lane_id: str) -> bool:
        stats = self.get_lane_stats(lane_id)
        eps = self.constants.epsilon
        return bool(
            stats.queue_count > eps
            or stats.residual_queue_vehicles > eps
            or len(stats.previous_vehicle_ids) > 0
        )

    def lane_has_reward_demand(self, lane_id: str) -> bool:
        stats = self.get_lane_stats(lane_id)
        eps = self.constants.epsilon
        return bool(
            self.lane_has_current_demand(lane_id)
            or stats.phase_inflow_pcu > eps
            or stats.cycle_inflow_pcu > eps
            or stats.split_failure_rate > eps
            or stats.pending_split_failure_rate > eps
        )

    def lane_demand_weight(self, lane_id: str) -> float:
        stats = self.get_lane_stats(lane_id)
        if not self.lane_has_reward_demand(lane_id):
            return 0.0

        vehicle_count = float(len(stats.previous_vehicle_ids))
        weight = (
            max(float(stats.queue_count), 0.0)
            + max(float(stats.residual_queue_vehicles), 0.0)
            + max(float(stats.phase_inflow_pcu), 0.0)
            + max(float(stats.cycle_inflow_pcu), 0.0)
            + vehicle_count
        )
        return max(weight, 1.0)

    @staticmethod
    def _reset_phase_window(stats: LaneRuntimeStats) -> None:
        stats.phase_inflow_pcu = 0.0
        stats.phase_outflow_pcu = 0.0
        stats.phase_steps = 0
        stats.phase_queue_start = max(float(stats.queue_count), 0.0)

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
        stats.residual_queue_vehicles = 0.0
        stats.split_failure_rate = 0.0
        stats.pending_split_failure_rate = 0.0
        stats.last_green_demand_pcu = 0.0
        stats.time_since_last_service_seconds = 0.0
        self._reset_phase_window(stats)

    def reset_cycle(self, lane_id: str, last_n_ge: Optional[float] = None) -> None:
        stats = self.get_lane_stats(lane_id)
        if last_n_ge is None:
            last_n_ge = stats.residual_queue_vehicles
        stats.residual_queue_vehicles = max(float(last_n_ge), 0.0)
        stats.queue_integral = 0.0
        stats.cycle_steps = 0
        stats.cycle_inflow_pcu = 0.0
        stats.cycle_outflow_pcu = 0.0
        stats.green_seconds = 0.0
        stats.initial_cycle_queue = max(
            float(stats.queue_count),
            float(stats.residual_queue_vehicles),
            0.0,
        )
        self._reset_phase_window(stats)

    def start_new_cycle(self, lane_id: str) -> None:
        self.reset_cycle(lane_id)

    def start_new_phase(self, lane_id: str) -> None:
        stats = self.get_lane_stats(lane_id)
        self._reset_phase_window(stats)

    def mark_lane_green_seconds(self, lane_id: str, delta_seconds: float = 1.0) -> None:
        stats = self.get_lane_stats(lane_id)
        stats.green_seconds += max(0.0, float(delta_seconds))

    def mark_lane_service(
        self, lane_id: str, was_served: bool, delta_seconds: float = 1.0
    ) -> None:
        stats = self.get_lane_stats(lane_id)
        if not self.lane_has_current_demand(lane_id):
            stats.time_since_last_service_seconds = 0.0
            return
        if was_served:
            stats.time_since_last_service_seconds = 0.0
        else:
            stats.time_since_last_service_seconds += max(0.0, float(delta_seconds))

    def snapshot_green_end_queue(self, lane_id: str) -> None:
        stats = self.get_lane_stats(lane_id)
        eps = self.constants.epsilon
        residual_queue = max(float(stats.queue_count), 0.0)
        served_demand = max(
            float(stats.phase_queue_start) + float(stats.phase_inflow_pcu),
            eps,
        )
        split_failure_rate = min(max(residual_queue / served_demand, 0.0), 1.0)

        stats.last_green_demand_pcu = served_demand
        stats.split_failure_rate = split_failure_rate
        stats.pending_split_failure_rate = split_failure_rate

    def consume_pending_split_failure(self, lane_id: str) -> float:
        stats = self.get_lane_stats(lane_id)
        value = float(stats.pending_split_failure_rate)
        stats.pending_split_failure_rate = 0.0
        return value

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
        stats.phase_inflow_pcu += inflow
        stats.phase_outflow_pcu += outflow
        stats.phase_steps += 1
        stats.previous_vehicle_ids = set(current_vehicle_ids)

    @staticmethod
    def _lane_width_adjustment_factor(lane_width_m: Optional[float]) -> float:
        """
        f_b from lane-width nomograph (Figure 37):
        - b <= 2.5 m: f_b = 1.18
        - 2.5 < b < 3.0 m: linear from 1.18 to 1.0
        - b >= 3.0 m: f_b = 1.0
        """
        if lane_width_m is None:
            return 1.0
        b = float(lane_width_m)
        if b <= 2.5:
            return 1.18
        if b < 3.0:
            return 1.0 + (3.0 - b) * (0.18 / 0.5)
        return 1.0

    def compute_lane_kpis(
        self,
        lane_id: str,
        cycle_length_seconds: float,
        green_floor_seconds: float = 0.0,
        lane_width_m: Optional[float] = None,
    ) -> LaneKPI:
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
        total_demand_pcu_per_hour = inflow_pcu_per_hour + (
            max(float(stats.residual_queue_vehicles), 0.0) * 3600.0 / cycle_length
        )

        # Use the actual granted green time. A tiny numerical floor keeps the capacity
        # finite without masking the effect of the agent's green extension decision.
        g_effective = min(
            max(stats.green_seconds, float(green_floor_seconds), eps),
            cycle_length,
        )
        g_over_c = min(g_effective / cycle_length, 0.999)

        # TCCS 24:2018 Appendix F:
        # S = 3600 / t_H, with t_H = f1 * f2 * t_H0,
        # f1 = max(f_b, f_r, f_d), f2 = min(1, f_d)
        t_h0 = max(constants.saturation_headway_base_seconds, eps)
        f_hv = max(constants.saturation_headway_f_hv, eps)
        # f_b is adjusted by actual lane width b (if available), then scaled by config.
        f_b_nomograph = self._lane_width_adjustment_factor(lane_width_m)
        f_b = max(constants.saturation_headway_f_b * f_b_nomograph, eps)
        f_r = max(constants.saturation_headway_f_r, eps)
        f_d = max(constants.saturation_headway_f_d, eps)
        f1 = max(f_b, f_r, f_d)
        f2 = min(1.0, f_d)
        t_h = max(f_hv * f1 * f2 * t_h0, eps)
        S = 3600.0 / t_h
        capacity = S * g_over_c
        v_over_c = total_demand_pcu_per_hour / max(capacity, eps)

        # -------------------------------------------------------------
        # 1. Chiều dài hàng chờ (N_GE) theo HBS 2001 (TCCS 24:2018 F-21)
        # -------------------------------------------------------------
        m_max = float(g_effective * S / 3600.0)
        m_tb = float(total_demand_pcu_per_hour * cycle_length / 3600.0)
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
            total_demand_pcu_per_hour / S, 0.999
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
            total_demand_pcu_per_hour=float(total_demand_pcu_per_hour),
            residual_n_ge_vehicles=float(n_ge_vehicles),
        )
