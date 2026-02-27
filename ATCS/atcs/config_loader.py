"""Load and validate KPI configuration for ATCS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SimulationSettings:
    default_step_length_seconds: int
    max_episode_seconds: int
    min_green_seconds: int
    max_green_seconds: int
    yellow_fallback_seconds: int
    cycle_length_seconds: int
    use_gui: bool


@dataclass(frozen=True)
class KPIConstants:
    saturation_flow_pcu_per_hour_per_lane: float
    average_vehicle_space_meter: float
    green_wave_pf_default: float
    incremental_delay_k: float
    incremental_delay_power: float
    epsilon: float
    max_control_delay_seconds: float
    v_c_penalty_threshold: float
    v_c_penalty_weight: float
    default_pcu: float
    pcu_mapping: Dict[str, float]


@dataclass(frozen=True)
class KPIConfig:
    path: Path
    simulation: SimulationSettings
    constants: KPIConstants
    formulas: Dict[str, str]
    los_table: List[Dict[str, Any]]
    reward_design: Dict[str, Any]
    fixed_time_plans: Dict[str, List[tuple]]


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "kpi_config.json"


def load_kpi_config(config_path: Optional[str] = None) -> KPIConfig:
    """Load KPI config JSON and convert to typed dataclasses."""
    path = Path(config_path) if config_path else _default_config_path()
    if not path.exists():
        raise FileNotFoundError(f"KPI config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    sim_raw = raw.get("simulation", {})
    simulation = SimulationSettings(
        default_step_length_seconds=int(sim_raw.get("default_step_length_seconds", 1)),
        max_episode_seconds=int(sim_raw.get("max_episode_seconds", 3600)),
        min_green_seconds=int(sim_raw.get("min_green_seconds", 10)),
        max_green_seconds=int(sim_raw.get("max_green_seconds", 60)),
        yellow_fallback_seconds=int(sim_raw.get("yellow_fallback_seconds", 3)),
        cycle_length_seconds=int(sim_raw.get("cycle_length_seconds", 132)),
        use_gui=bool(sim_raw.get("use_gui", False)),
    )

    constants_raw = raw.get("constants", {})
    constants = KPIConstants(
        saturation_flow_pcu_per_hour_per_lane=float(
            constants_raw.get("saturation_flow_pcu_per_hour_per_lane", 1900.0)
        ),
        average_vehicle_space_meter=float(
            constants_raw.get("average_vehicle_space_meter", 6.5)
        ),
        green_wave_pf_default=float(constants_raw.get("green_wave_pf_default", 1.0)),
        incremental_delay_k=float(constants_raw.get("incremental_delay_k", 0.5)),
        incremental_delay_power=float(constants_raw.get("incremental_delay_power", 2.0)),
        epsilon=float(constants_raw.get("epsilon", 1e-6)),
        max_control_delay_seconds=float(constants_raw.get("max_control_delay_seconds", 300.0)),
        v_c_penalty_threshold=float(constants_raw.get("v_c_penalty_threshold", 0.9)),
        v_c_penalty_weight=float(constants_raw.get("v_c_penalty_weight", 12.0)),
        default_pcu=float(constants_raw.get("default_pcu", 1.0)),
        pcu_mapping={
            str(k).lower(): float(v)
            for k, v in constants_raw.get("pcu_mapping", {}).items()
        },
    )

    return KPIConfig(
        path=path,
        simulation=simulation,
        constants=constants,
        formulas={str(k): str(v) for k, v in raw.get("kpi_formulas", {}).items()},
        los_table=list(raw.get("los_table", [])),
        reward_design=dict(raw.get("reward_design", {})),
        fixed_time_plans=dict(raw.get("fixed_time_plans", {})),
    )
