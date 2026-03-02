"""Parse SUMO network details needed by the ATCS environment."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class PhaseDefinition:
    index: int
    duration_seconds: int
    state: str
    phase_type: str


@dataclass(frozen=True)
class TLSProgram:
    tls_id: str
    phases: Tuple[PhaseDefinition, ...]
    base_cycle_seconds: int
    first_green_index: int


@dataclass(frozen=True)
class ParsedSUMONetwork:
    sumocfg_path: Path
    net_file_path: Path
    tls_programs: Dict[str, TLSProgram]


def _classify_phase_type(state: str) -> str:
    if any(char in state for char in ("G", "g")):
        return "green"
    if any(char in state for char in ("y", "Y")):
        return "yellow"
    return "red"


def _resolve_net_file(sumocfg_path: Path) -> Path:
    if not sumocfg_path.exists():
        raise FileNotFoundError(f"SUMO config not found: {sumocfg_path}")

    root = ET.parse(sumocfg_path).getroot()
    net_value = None
    for element in root.findall(".//net-file"):
        net_value = element.get("value")
        if net_value:
            break

    if not net_value:
        raise ValueError(f"net-file entry not found in {sumocfg_path}")

    net_file_path = Path(os.path.join(sumocfg_path.parent, net_value)).resolve()
    if not net_file_path.exists():
        raise FileNotFoundError(f"SUMO net file not found: {net_file_path}")
    return net_file_path


def parse_sumo_network(sumocfg_path: str, yellow_fallback_seconds: int = 3) -> ParsedSUMONetwork:
    """Parse SUMO .sumocfg and corresponding .net.xml traffic light programs."""
    cfg_path = Path(sumocfg_path).resolve()
    net_path = _resolve_net_file(cfg_path)

    net_root = ET.parse(net_path).getroot()
    tls_programs: Dict[str, TLSProgram] = {}

    for tl_logic in net_root.findall("tlLogic"):
        tls_id = tl_logic.get("id")
        if not tls_id:
            continue

        phases = []
        for idx, phase in enumerate(tl_logic.findall("phase")):
            state = phase.get("state", "")
            duration_raw = phase.get("duration")
            try:
                duration = int(round(float(duration_raw))) if duration_raw else yellow_fallback_seconds
            except ValueError:
                duration = yellow_fallback_seconds
            duration = max(duration, 0)

            phases.append(
                PhaseDefinition(
                    index=idx,
                    duration_seconds=duration,
                    state=state,
                    phase_type=_classify_phase_type(state),
                )
            )

        if not phases:
            continue

        first_green_index = next(
            (phase.index for phase in phases if phase.phase_type == "green"),
            0,
        )
        base_cycle_seconds = sum(phase.duration_seconds for phase in phases)
        if base_cycle_seconds <= 0:
            base_cycle_seconds = max(len(phases) * yellow_fallback_seconds, 1)

        tls_programs[tls_id] = TLSProgram(
            tls_id=tls_id,
            phases=tuple(phases),
            base_cycle_seconds=base_cycle_seconds,
            first_green_index=first_green_index,
        )

    if not tls_programs:
        raise ValueError(f"No tlLogic entries found in {net_path}")

    ordered_programs = {tls_id: tls_programs[tls_id] for tls_id in sorted(tls_programs)}
    return ParsedSUMONetwork(
        sumocfg_path=cfg_path,
        net_file_path=net_path,
        tls_programs=ordered_programs,
    )