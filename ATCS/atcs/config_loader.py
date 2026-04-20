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
    use_gui: bool


@dataclass(frozen=True)
class KPIConstants:
    saturation_headway_base_seconds: float
    saturation_headway_f_hv: float
    saturation_headway_f_b: float
    saturation_headway_f_r: float
    saturation_headway_f_d: float
    average_vehicle_space_meter: float
    epsilon: float
    max_control_delay_seconds: float
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
        use_gui=bool(sim_raw.get("use_gui", False)),
    )

    constants_raw = raw.get("constants", {})
    constants = KPIConstants(
        saturation_headway_base_seconds=float(
            constants_raw.get("saturation_headway_base_seconds", 1.8)
        ),
        saturation_headway_f_hv=float(
            constants_raw.get("saturation_headway_f_hv", 1.1)
        ),
        saturation_headway_f_b=float(
            constants_raw.get("saturation_headway_f_b", 1.0)
        ),
        saturation_headway_f_r=float(
            constants_raw.get("saturation_headway_f_r", 1.0)
        ),
        saturation_headway_f_d=float(
            constants_raw.get("saturation_headway_f_d", 1.0)
        ),
        average_vehicle_space_meter=float(
            constants_raw.get("average_vehicle_space_meter", 6.5)
        ),
        epsilon=float(constants_raw.get("epsilon", 1e-6)),
        max_control_delay_seconds=float(
            constants_raw.get("max_control_delay_seconds", 300.0)
        ),
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
    )
