"""Compatibility adapter for legacy TraCI-based KPI scripts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from .config_loader import KPIConstants, load_kpi_config


def _load_traci():
    import traci  # type: ignore

    return traci


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@dataclass(frozen=True)
class _LegacyConstants:
    pcu_mapping: Dict[str, float]
    default_pcu: float
    t_h0: float
    f_hv: float
    f_b: float
    f_r: float
    f_d: float
    avg_veh_space: float
    epsilon: float
    max_control_delay_seconds: float


def _normalize_constants(
    constants: KPIConstants | Mapping[str, Any] | None,
) -> _LegacyConstants:
    if constants is None:
        constants = load_kpi_config().constants

    if isinstance(constants, KPIConstants):
        return _LegacyConstants(
            pcu_mapping={str(k).lower(): float(v) for k, v in constants.pcu_mapping.items()},
            default_pcu=float(constants.default_pcu),
            t_h0=float(constants.saturation_headway_base_seconds),
            f_hv=float(constants.saturation_headway_f_hv),
            f_b=float(constants.saturation_headway_f_b),
            f_r=float(constants.saturation_headway_f_r),
            f_d=float(constants.saturation_headway_f_d),
            avg_veh_space=float(constants.average_vehicle_space_meter),
            epsilon=float(constants.epsilon),
            max_control_delay_seconds=float(constants.max_control_delay_seconds),
        )

    raw = dict(constants)
    return _LegacyConstants(
        pcu_mapping={
            str(k).lower(): _to_float(v, 1.0)
            for k, v in dict(raw.get("pcu_mapping", {})).items()
        },
        default_pcu=_to_float(raw.get("default_pcu"), 1.0),
        t_h0=_to_float(
            raw.get("t_h0", raw.get("saturation_headway_base_seconds")),
            1.8,
        ),
        f_hv=_to_float(
            raw.get("f_hv", raw.get("saturation_headway_f_hv")),
            1.0,
        ),
        f_b=_to_float(
            raw.get("f_b", raw.get("saturation_headway_f_b")),
            1.0,
        ),
        f_r=_to_float(
            raw.get("f_r", raw.get("saturation_headway_f_r")),
            1.0,
        ),
        f_d=_to_float(
            raw.get("f_d", raw.get("saturation_headway_f_d")),
            1.0,
        ),
        avg_veh_space=_to_float(
            raw.get("avg_veh_space", raw.get("average_vehicle_space_meter")),
            6.5,
        ),
        epsilon=_to_float(raw.get("epsilon"), 1e-6),
        max_control_delay_seconds=_to_float(raw.get("max_control_delay_seconds"), 300.0),
    )


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
    waiting_time_total: float = 0.0
    v_ids_in_cycle: Set[str] = field(default_factory=set)


@dataclass(frozen=True)
class LaneKPI:
    control_delay_seconds: float
    degree_of_saturation: float
    queue_length_meters: float
    inflow_pcu_per_hour: float
    outflow_pcu_per_hour: float
    avg_queue_vehicles: float
    capacity_pcu_h: float
    green_ratio: float


class LegacyKPIAdapter:
    """Legacy KPI API driven directly from TraCI lane state."""

    def __init__(
        self,
        constants: KPIConstants | Mapping[str, Any] | None = None,
        traci_module: Optional[Any] = None,
        one_cycle_observation: bool = True,
    ) -> None:
        self.constants = _normalize_constants(constants)
        self._traci = traci_module
        self.one_cycle_observation = bool(one_cycle_observation)
        self._lane_stats: Dict[str, LaneRuntimeStats] = {}

    @property
    def traci(self):
        if self._traci is None:
            self._traci = _load_traci()
        return self._traci

    def get_lane_stats(self, lane_id: str) -> LaneRuntimeStats:
        if lane_id not in self._lane_stats:
            self._lane_stats[lane_id] = LaneRuntimeStats()
        return self._lane_stats[lane_id]

    def _vehicle_pcu(self, vehicle_id: str) -> float:
        try:
            type_id = str(self.traci.vehicle.getTypeID(vehicle_id)).lower()
        except Exception:
            return self.constants.default_pcu

        for token, pcu in self.constants.pcu_mapping.items():
            if token in type_id:
                return pcu
        return self.constants.default_pcu

    def update_lane(
        self,
        lane_id: str,
        current_vehicle_ids: Iterable[str],
        is_green: bool,
        step_len: float,
    ) -> None:
        stats = self.get_lane_stats(lane_id)
        step_seconds = max(float(step_len), 0.0)
        current_ids = set(current_vehicle_ids)

        stats.cycle_steps += 1
        if is_green:
            stats.green_seconds += step_seconds

        stats.queue_count = float(self.traci.lane.getLastStepHaltingNumber(lane_id))
        stats.queue_integral += stats.queue_count * step_seconds

        new_vehicles = current_ids - stats.previous_vehicle_ids
        for vehicle_id in new_vehicles:
            if vehicle_id in stats.v_ids_in_cycle:
                continue
            stats.cycle_inflow_pcu += self._vehicle_pcu(vehicle_id)
            stats.v_ids_in_cycle.add(vehicle_id)

        exited_vehicles = stats.previous_vehicle_ids - current_ids
        for vehicle_id in exited_vehicles:
            stats.cycle_outflow_pcu += self._vehicle_pcu(vehicle_id)

        stats.previous_vehicle_ids = current_ids
        try:
            stats.waiting_time_total += (
                float(self.traci.lane.getWaitingTime(lane_id)) * step_seconds
            )
        except Exception:
            pass

    def compute_kpi(self, lane_id: str, cycle_len: float) -> LaneKPI:
        stats = self.get_lane_stats(lane_id)
        cycle_length = max(float(cycle_len), 1.0)
        eps = self.constants.epsilon

        f_adj = max(self.constants.f_b, self.constants.f_r, self.constants.f_d)
        t_h = self.constants.f_hv * f_adj * self.constants.t_h0
        saturation = 3600.0 / max(t_h, eps)

        inflow_h = stats.cycle_inflow_pcu * 3600.0 / cycle_length
        outflow_h = stats.cycle_outflow_pcu * 3600.0 / cycle_length
        green_ratio = stats.green_seconds / cycle_length
        capacity = saturation * max(green_ratio, eps)
        x_val = min(inflow_h / max(capacity, eps), 2.0)

        m_max = stats.green_seconds * saturation / 3600.0
        m_tb = inflow_h * cycle_length / 3600.0

        if x_val <= 0.65:
            n_ge = 0.0
        elif x_val <= 0.9:
            val_09 = 1.0 / (0.26 + m_tb / 150.0)
            n_ge = (x_val - 0.65) / 0.25 * val_09
        elif x_val <= 1.0:
            val_09 = 1.0 / (0.26 + m_tb / 150.0)
            coeff_10 = 0.545 if self.one_cycle_observation else 0.3476
            val_10 = coeff_10 * (max(m_max, 0.0) ** 0.5)
            n_ge = val_09 + (x_val - 0.9) / 0.1 * (val_10 - val_09)
        elif x_val <= 1.2:
            coeff_10 = 0.545 if self.one_cycle_observation else 0.3476
            val_10 = coeff_10 * (max(m_max, 0.0) ** 0.5)
            val_12 = (m_max * 0.2 + 5.0) / 2.0
            n_ge = val_10 + (x_val - 1.0) / 0.2 * (val_12 - val_10)
        else:
            n_ge = m_max * (x_val - 1.0) / 2.0

        q_over_s = min(inflow_h / max(saturation, eps), 0.99)
        t_w1 = (cycle_length * (1.0 - green_ratio) ** 2) / (
            2.0 * max(1.0 - q_over_s, eps)
        )
        t_w2 = (3600.0 * n_ge) / max(capacity, eps)
        delay = min(t_w1 + t_w2, self.constants.max_control_delay_seconds)

        return LaneKPI(
            control_delay_seconds=float(delay),
            degree_of_saturation=float(x_val),
            queue_length_meters=float(n_ge * self.constants.avg_veh_space),
            inflow_pcu_per_hour=float(inflow_h),
            outflow_pcu_per_hour=float(outflow_h),
            avg_queue_vehicles=float(stats.queue_integral / cycle_length),
            capacity_pcu_h=float(capacity),
            green_ratio=float(green_ratio),
        )

    def reset_cycle(self, lane_id: str) -> None:
        stats = self.get_lane_stats(lane_id)
        stats.initial_cycle_queue = stats.queue_count
        stats.cycle_inflow_pcu = 0.0
        stats.cycle_outflow_pcu = 0.0
        stats.green_seconds = 0.0
        stats.cycle_steps = 0
        stats.queue_integral = 0.0
        stats.waiting_time_total = 0.0
        stats.v_ids_in_cycle.clear()


KPIEngine = LegacyKPIAdapter


__all__ = ["KPIEngine", "LaneKPI", "LaneRuntimeStats", "LegacyKPIAdapter"]
