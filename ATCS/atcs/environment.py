"""ATCS traffic environment with KPI-standard observation and reward tensors."""

from __future__ import annotations

import os
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import traci

from .config_loader import KPIConfig, load_kpi_config
from .kpi_engine import KPIEngine
from .sumo_parser import (
    ParsedSUMONetwork,
    TLSProgram,
    parse_sumo_network,
    PhaseDefinition,
    _classify_phase_type,
    _resolve_sumocfg_path,
)


@dataclass
class TLSRuntimeState:
    current_phase_index: int
    remaining_phase_seconds: int
    cycle_elapsed_seconds: int
    cycle_length_seconds: int
    decision_pending: bool = True


class TrafficEnvironment:
    """
    SUMO traffic environment for centralized RL training.

    Required API:
    - step(action_dict) -> (observation, reward, done, information)
    - reset() -> (observation, reward, done, information)
    """

    def __init__(
        self,
        sumocfg_path: str,
        kpi_config_path: Optional[str] = None,
        use_gui: Optional[bool] = None,
        max_episode_seconds: Optional[int] = None,
        sumo_binary: Optional[str] = None,
        log_lane_width_adjustment_factor: bool = False,
    ) -> None:
        self.sumocfg_path = _resolve_sumocfg_path(Path(sumocfg_path).resolve())
        self.kpi_config: KPIConfig = load_kpi_config(kpi_config_path)
        self.network: ParsedSUMONetwork = parse_sumo_network(
            str(self.sumocfg_path),
            yellow_fallback_seconds=self.kpi_config.simulation.yellow_fallback_seconds,
        )

        self.tls_programs: Dict[str, TLSProgram] = self.network.tls_programs
        self.tls_ids: List[str] = list(self.tls_programs.keys())

        sim_cfg = self.kpi_config.simulation
        self.step_length_seconds = max(int(sim_cfg.default_step_length_seconds), 1)
        self.min_green_seconds = int(sim_cfg.min_green_seconds)
        self.max_green_seconds = int(sim_cfg.max_green_seconds)
        self.cycle_length_seconds = 0
        self.max_extension_seconds = max(
            self.max_green_seconds - self.min_green_seconds, 0
        )
        self.use_gui = sim_cfg.use_gui if use_gui is None else bool(use_gui)
        self.max_episode_seconds = (
            int(sim_cfg.max_episode_seconds)
            if max_episode_seconds is None
            else int(max_episode_seconds)
        )

        self.sumo_binary = sumo_binary or self._resolve_sumo_binary(self.use_gui)
        self.connection_label = f"ATCS_{uuid.uuid4().hex[:8]}"
        self.connected = False

        self.kpi_engine = KPIEngine(self.kpi_config.constants)
        self.tls_runtime: Dict[str, TLSRuntimeState] = {}
        self.log_lane_width_adjustment_factor = bool(log_lane_width_adjustment_factor)
        self._lane_width_adjustment_logged = False

        self.lanes_by_tls: Dict[str, List[str]] = {}
        self.lane_link_indices: Dict[str, Dict[str, List[int]]] = {}
        self.lane_width_m: Dict[str, float] = {}
        self.max_lanes = 0

        self.required_action: Set[str] = set()
        self.vehicle_pcu_cache: Dict[str, float] = {}
        self.simulation_time = 0
        self.done = False

    def _resolve_sumo_binary(self, use_gui: bool) -> str:
        if use_gui:
            return os.getenv("SUMO_GUI_BINARY", "sumo-gui")
        return os.getenv("SUMO_BINARY", "sumo")

    def _start_sumo(self) -> None:
        if self.connected:
            self.close()

        sumo_cmd = [
            self.sumo_binary,
            "-c",
            str(self.sumocfg_path),
            "--step-length",
            str(self.step_length_seconds),
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
            "--duration-log.disable",
            "true",
            "--waiting-time-memory",
            "1000",
        ]
        traci.start(sumo_cmd, label=self.connection_label)
        traci.switch(self.connection_label)
        self.connected = True

    def _resolve_scenario_phase_config(self) -> Optional[Path]:
        candidate_names = [
            "2nutgiao_fixedtime(2).tll.xml",
            "2nutgiao.fixedtime.ttl.xml",
        ]
        for name in candidate_names:
            candidate = self.sumocfg_path.parent / name
            if candidate.exists():
                return candidate

        wildcard_candidates = sorted(
            self.sumocfg_path.parent.glob("*fixedtime*.xml")
        )
        return wildcard_candidates[0] if wildcard_candidates else None

    def _load_scenario_phase_states(self) -> Dict[str, List[str]]:
        phase_config_path = self._resolve_scenario_phase_config()
        if phase_config_path is None:
            return {}

        root = ET.parse(phase_config_path).getroot()
        states_by_tls: Dict[str, List[str]] = {}
        for tl_logic in root.findall("tlLogic"):
            tls_id = tl_logic.get("id")
            if not tls_id:
                continue
            states = [
                phase.get("state", "")
                for phase in tl_logic.findall("phase")
                if phase.get("state")
            ]
            if states:
                states_by_tls[tls_id] = states
        return states_by_tls

    def _derive_cycle_length_seconds(self, phases: List[PhaseDefinition]) -> int:
        green_count = sum(1 for phase in phases if phase.phase_type == "green")
        if green_count <= 0:
            return max(sum(phase.duration_seconds for phase in phases), 1)

        yellow_seconds = max(
            int(self.kpi_config.simulation.yellow_fallback_seconds),
            0,
        )
        per_phase_seconds = max(self.min_green_seconds, 0) + yellow_seconds
        return max(green_count * per_phase_seconds, 1)

    def _refresh_tls_programs_from_sumo(self) -> None:
        self._ensure_connection()

        tls_programs: Dict[str, TLSProgram] = {}
        yellow_fallback = self.kpi_config.simulation.yellow_fallback_seconds
        scenario_phase_states = self._load_scenario_phase_states()

        for tls_id in traci.trafficlight.getIDList():
            current_program_id = str(traci.trafficlight.getProgram(tls_id))
            logics = traci.trafficlight.getAllProgramLogics(tls_id)
            selected_logic = None

            for logic in logics:
                logic_program_id = str(getattr(logic, "programID", getattr(logic, "subID", "")))
                if logic_program_id == current_program_id:
                    selected_logic = logic
                    break

            if selected_logic is None and logics:
                selected_logic = logics[0]
            if selected_logic is None:
                continue

            raw_phases = getattr(selected_logic, "phases", None)
            if raw_phases is None and hasattr(selected_logic, "getPhases"):
                raw_phases = selected_logic.getPhases()
            if not raw_phases:
                continue

            phases = []
            for idx, phase in enumerate(raw_phases):
                state = getattr(phase, "state", "")
                phase_type = _classify_phase_type(state)
                duration_raw = getattr(phase, "duration", None)
                try:
                    duration = (
                        int(round(float(duration_raw)))
                        if duration_raw is not None
                        else yellow_fallback
                    )
                except (TypeError, ValueError):
                    duration = yellow_fallback
                duration = max(duration, 0)
                if phase_type == "green":
                    # Green time is decided by ATCS actions / fixed-time controller.
                    duration = 0
                elif phase_type == "red":
                    duration = 0

                phases.append(
                    PhaseDefinition(
                        index=idx,
                        duration_seconds=duration,
                        state=state,
                        phase_type=phase_type,
                    )
                )

            if not phases:
                continue

            override_states = scenario_phase_states.get(tls_id)
            if override_states and len(override_states) == len(phases):
                phases = [
                    PhaseDefinition(
                        index=phase.index,
                        duration_seconds=(
                            yellow_fallback
                            if _classify_phase_type(override_state) == "yellow"
                            else 0
                        ),
                        state=override_state,
                        phase_type=_classify_phase_type(override_state),
                    )
                    for phase, override_state in zip(phases, override_states)
                ]

            first_green_index = next(
                (phase.index for phase in phases if phase.phase_type == "green"),
                0,
            )
            base_cycle_seconds = self._derive_cycle_length_seconds(phases)

            tls_programs[tls_id] = TLSProgram(
                tls_id=tls_id,
                phases=tuple(phases),
                base_cycle_seconds=base_cycle_seconds,
                first_green_index=first_green_index,
            )

        if tls_programs:
            self.tls_programs = {tls_id: tls_programs[tls_id] for tls_id in sorted(tls_programs)}
            self.tls_ids = list(self.tls_programs.keys())
            unique_cycle_lengths = {
                program.base_cycle_seconds for program in self.tls_programs.values()
            }
            if len(unique_cycle_lengths) == 1:
                self.cycle_length_seconds = next(iter(unique_cycle_lengths))
            else:
                self.cycle_length_seconds = max(unique_cycle_lengths)

    def _ensure_connection(self) -> None:
        if not self.connected:
            raise RuntimeError("SUMO connection is not active. Call reset() first.")
        traci.switch(self.connection_label)

    def _build_lane_topology(self) -> None:
        self.lanes_by_tls = {}
        self.lane_link_indices = {}
        self.lane_width_m = {}
        self.max_lanes = 0

        for tls_id in self.tls_ids:
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            unique_lanes = list(dict.fromkeys(controlled_lanes))
            self.lanes_by_tls[tls_id] = unique_lanes
            self.max_lanes = max(self.max_lanes, len(unique_lanes))

            lane_indices: Dict[str, List[int]] = {}
            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
            for link_index, links in enumerate(controlled_links):
                for link in links:
                    incoming_lane = link[0]
                    if incoming_lane:
                        lane_indices.setdefault(incoming_lane, []).append(link_index)
            self.lane_link_indices[tls_id] = lane_indices

            for lane_id in unique_lanes:
                try:
                    self.lane_width_m[lane_id] = float(traci.lane.getWidth(lane_id))
                except Exception:
                    # Fallback to 3.0 m if SUMO does not expose lane width.
                    self.lane_width_m[lane_id] = 3.0

        self.max_lanes = max(self.max_lanes, 1)

    def _initialize_lane_runtime(self) -> None:
        self.vehicle_pcu_cache = {}
        unique_lanes = {lane for lanes in self.lanes_by_tls.values() for lane in lanes}
        for lane_id in unique_lanes:
            vehicle_ids = set(traci.lane.getLastStepVehicleIDs(lane_id))
            initial_queue = float(traci.lane.getLastStepHaltingNumber(lane_id))
            self.kpi_engine.reset_lane_state(lane_id, initial_queue, vehicle_ids)
            for vehicle_id in vehicle_ids:
                self.vehicle_pcu_cache[vehicle_id] = self._vehicle_pcu(vehicle_id)

    def _log_lane_width_adjustment_factors(self) -> None:
        if (
            not self.log_lane_width_adjustment_factor
            or self._lane_width_adjustment_logged
        ):
            return

        print(
            f"[LaneWidthFactor] sumocfg={self.sumocfg_path} "
            f"scenario_dir={self.sumocfg_path.parent.name}"
        )
        for tls_id in self.tls_ids:
            for lane_id in self.lanes_by_tls.get(tls_id, []):
                lane_width_m = self.lane_width_m.get(lane_id)
                f_b_nomograph = KPIEngine._lane_width_adjustment_factor(lane_width_m)
                width_text = "None" if lane_width_m is None else f"{lane_width_m:.3f}"
                print(
                    f"[LaneWidthFactor] tls={tls_id} lane={lane_id} "
                    f"width_m={width_text} f_b_nomograph={f_b_nomograph:.4f}"
                )
        self._lane_width_adjustment_logged = True

    def _prepare_tls_runtime(self) -> None:
        self.tls_runtime = {}
        self.required_action = set()

        for tls_id in self.tls_ids:
            program = self.tls_programs[tls_id]
            phase_index = program.first_green_index
            phase = program.phases[phase_index]

            self.tls_runtime[tls_id] = TLSRuntimeState(
                current_phase_index=phase_index,
                remaining_phase_seconds=max(int(phase.duration_seconds), 0),
                cycle_elapsed_seconds=0,
                cycle_length_seconds=program.base_cycle_seconds,
                decision_pending=True,
            )
            traci.trafficlight.setRedYellowGreenState(tls_id, phase.state)
            self._reset_cycle_lane_metrics(tls_id)
            self._reset_phase_lane_metrics(tls_id)
            self.required_action.add(tls_id)

    def _reset_cycle_lane_metrics(
        self, tls_id: str, last_n_ge_by_lane: Optional[Dict[str, float]] = None
    ) -> None:
        for lane_id in self.lanes_by_tls.get(tls_id, []):
            last_n_ge = None
            if last_n_ge_by_lane is not None:
                last_n_ge = float(last_n_ge_by_lane.get(lane_id, 0.0))
            self.kpi_engine.reset_cycle(lane_id, last_n_ge)

    def _reset_phase_lane_metrics(self, tls_id: str) -> None:
        for lane_id in self.lanes_by_tls.get(tls_id, []):
            self.kpi_engine.start_new_phase(lane_id)

    def _green_lanes_for_phase(self, tls_id: str, phase_state: str) -> Set[str]:
        green_lanes: Set[str] = set()
        lane_index_map = self.lane_link_indices.get(tls_id, {})
        for lane_id, link_indices in lane_index_map.items():
            if any(
                idx < len(phase_state) and phase_state[idx] in ("G", "g")
                for idx in link_indices
            ):
                green_lanes.add(lane_id)
        return green_lanes

    def _lane_mask(self) -> np.ndarray:
        mask = np.zeros((len(self.tls_ids), self.max_lanes), dtype=np.float32)
        for tls_index, tls_id in enumerate(self.tls_ids):
            lane_count = len(self.lanes_by_tls.get(tls_id, []))
            if lane_count > 0:
                mask[tls_index, :lane_count] = 1.0
        return mask

    def _vehicle_pcu(self, vehicle_id: str, fallback_only: bool = False) -> float:
        constants = self.kpi_config.constants
        if fallback_only:
            return constants.default_pcu

        try:
            type_id = traci.vehicle.getTypeID(vehicle_id).lower()
        except Exception:
            return constants.default_pcu

        for token, pcu in constants.pcu_mapping.items():
            if token in type_id:
                return pcu
        return constants.default_pcu

    def _update_lane_accumulation(self) -> None:
        for lane_id in list(self.kpi_engine.lane_ids()):
            lane_stats = self.kpi_engine.get_lane_stats(lane_id)
            current_vehicle_ids = set(traci.lane.getLastStepVehicleIDs(lane_id))

            entered = current_vehicle_ids - lane_stats.previous_vehicle_ids
            exited = lane_stats.previous_vehicle_ids - current_vehicle_ids

            inflow_pcu = 0.0
            for vehicle_id in entered:
                pcu = self._vehicle_pcu(vehicle_id)
                self.vehicle_pcu_cache[vehicle_id] = pcu
                inflow_pcu += pcu

            outflow_pcu = 0.0
            for vehicle_id in exited:
                pcu = self.vehicle_pcu_cache.pop(
                    vehicle_id,
                    self._vehicle_pcu(vehicle_id, fallback_only=True),
                )
                outflow_pcu += pcu

            self.kpi_engine.update_lane(
                lane_id=lane_id,
                inflow_pcu=inflow_pcu,
                outflow_pcu=outflow_pcu,
                current_vehicle_ids=current_vehicle_ids,
            )

    def _update_service_state_for_tls(self, tls_id: str, phase_state: str) -> Set[str]:
        green_lanes = self._green_lanes_for_phase(tls_id, phase_state)
        for lane_id in self.lanes_by_tls.get(tls_id, []):
            lane_has_green = lane_id in green_lanes
            self.kpi_engine.mark_lane_service(
                lane_id, lane_has_green, self.step_length_seconds
            )
            if lane_has_green:
                self.kpi_engine.mark_lane_green_seconds(
                    lane_id, self.step_length_seconds
                )
        return green_lanes

    def _snapshot_green_end_queues(self, tls_id: str, phase_state: str) -> None:
        """Snapshot residual queue for all lanes that were just served green."""
        for lane_id in self._green_lanes_for_phase(tls_id, phase_state):
            self.kpi_engine.snapshot_green_end_queue(lane_id)

    def _finalize_cycle_kpis(self, tls_id: str, cycle_length_seconds: float) -> None:
        last_n_ge_by_lane: Dict[str, float] = {}
        cycle_length = max(float(cycle_length_seconds), float(self.step_length_seconds))
        for lane_id in self.lanes_by_tls.get(tls_id, []):
            lane_kpi = self.kpi_engine.compute_lane_kpis(
                lane_id,
                cycle_length_seconds=cycle_length,
                green_floor_seconds=float(self.step_length_seconds),
                lane_width_m=self.lane_width_m.get(lane_id),
            )
            last_n_ge_by_lane[lane_id] = max(
                float(lane_kpi.residual_n_ge_vehicles),
                0.0,
            )
        self._reset_cycle_lane_metrics(tls_id, last_n_ge_by_lane)

    def _advance_to_next_phase(self, tls_id: str) -> None:
        runtime = self.tls_runtime[tls_id]
        program = self.tls_programs[tls_id]
        phase_count = len(program.phases)
        current_phase = program.phases[runtime.current_phase_index]
        if current_phase.phase_type == "green":
            self._snapshot_green_end_queues(tls_id, current_phase.state)

        for _ in range(max(phase_count, 1)):
            next_index = (runtime.current_phase_index + 1) % phase_count
            wrapped_cycle = next_index <= runtime.current_phase_index
            runtime.current_phase_index = next_index

            if wrapped_cycle:
                self._finalize_cycle_kpis(
                    tls_id,
                    float(runtime.cycle_length_seconds),
                )
                runtime.cycle_elapsed_seconds = 0
                # Reset to the cycle length derived from the active program phases.
                runtime.cycle_length_seconds = program.base_cycle_seconds

            phase = program.phases[next_index]
            runtime.remaining_phase_seconds = max(int(phase.duration_seconds), 0)
            runtime.decision_pending = False
            traci.trafficlight.setRedYellowGreenState(tls_id, phase.state)
            self._reset_phase_lane_metrics(tls_id)

            if runtime.remaining_phase_seconds > 0:
                return

            if phase.phase_type == "green":
                runtime.decision_pending = True
                self.required_action.add(tls_id)
                return

        runtime.remaining_phase_seconds = 1

    def _apply_pending_actions(self, action: Dict[str, float]) -> None:
        pending_tls_ids = list(self.required_action)
        self.required_action.clear()

        for tls_id in pending_tls_ids:
            runtime = self.tls_runtime[tls_id]
            extension_raw = float(action.get(tls_id, 0.0))
            min_ext, max_ext = self._compute_effective_green_range(tls_id)
            extension = min(max(extension_raw, min_ext), max_ext)

            runtime.decision_pending = False

            # Action semantics:
            # - Agent predicts continuous extension e (seconds).
            # - Effective range is [e_min, e_max].
            # - Executed green is absolute x = base_green + e.
            # - Current cycle length grows with the granted green extension.
            base_green = float(self.min_green_seconds)
            green_seconds = int(round(base_green + extension))
            runtime.remaining_phase_seconds = max(green_seconds, 0)
            runtime.cycle_length_seconds += max(int(round(extension)), 0)

            if runtime.remaining_phase_seconds <= 0:
                self._advance_to_next_phase(tls_id)

    def _simulate_until_need_action(self) -> int:
        delta_t = 0
        while not self.done and not self.required_action:
            traci.simulationStep()
            self.simulation_time += self.step_length_seconds
            delta_t += self.step_length_seconds

            self._update_lane_accumulation()

            for tls_id in self.tls_ids:
                runtime = self.tls_runtime[tls_id]
                if runtime.decision_pending:
                    self.required_action.add(tls_id)
                    continue

                program = self.tls_programs[tls_id]
                phase = program.phases[runtime.current_phase_index]
                runtime.cycle_elapsed_seconds += self.step_length_seconds
                self._update_service_state_for_tls(tls_id, phase.state)

                runtime.remaining_phase_seconds -= self.step_length_seconds
                if runtime.remaining_phase_seconds > 0:
                    continue

                # Advance automatically when any phase ends, including extended greens.
                self._advance_to_next_phase(tls_id)

            self.done = self._check_done()

        return delta_t

    def _check_done(self) -> bool:
        if self.simulation_time >= self.max_episode_seconds:
            return True
        try:
            return traci.simulation.getMinExpectedNumber() <= 0
        except Exception:
            return True

    def _build_observation_reward(self, delta_t: int) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.zeros((len(self.tls_ids), self.max_lanes, 9), dtype=np.float32)
        reward = np.zeros((len(self.tls_ids), self.max_lanes, 6), dtype=np.float32)
        eps = self.kpi_config.constants.epsilon
        sat_clip_max = float(
            self.kpi_config.reward_design.get("saturation_norm_clip_max", 2.0)
        )
        split_clip_max = float(
            self.kpi_config.reward_design.get("split_failure_clip_max", 1.0)
        )
        service_age_clip_cycles = float(
            self.kpi_config.reward_design.get("service_age_clip_cycles", 2.0)
        )

        for tls_index, tls_id in enumerate(self.tls_ids):
            runtime = self.tls_runtime[tls_id]
            remaining_cycle = max(
                float(runtime.cycle_length_seconds - runtime.cycle_elapsed_seconds),
                0.0,
            )
            current_phase = float(runtime.current_phase_index)
            cycle_length = max(
                float(runtime.cycle_length_seconds), float(self.step_length_seconds)
            )
            service_age_clip_seconds = max(
                cycle_length * service_age_clip_cycles,
                float(self.step_length_seconds),
            )
            current_green_lanes: Set[str] = set()
            if tls_id in self.required_action:
                phase = self.tls_programs[tls_id].phases[runtime.current_phase_index]
                if phase.phase_type == "green":
                    current_green_lanes = self._green_lanes_for_phase(tls_id, phase.state)

            lane_entries = []
            for lane_index, lane_id in enumerate(self.lanes_by_tls.get(tls_id, [])):
                lane_stats = self.kpi_engine.get_lane_stats(lane_id)
                lane_kpi = self.kpi_engine.compute_lane_kpis(
                    lane_id,
                    cycle_length_seconds=float(runtime.cycle_length_seconds),
                    green_floor_seconds=float(self.step_length_seconds),
                    lane_width_m=self.lane_width_m.get(lane_id),
                )

                sat_norm = min(
                    max(lane_kpi.degree_of_saturation, 0.0),
                    sat_clip_max,
                ) / max(sat_clip_max, eps)
                starvation_norm = min(
                    max(lane_stats.time_since_last_service_seconds, 0.0),
                    service_age_clip_seconds,
                ) / max(service_age_clip_seconds, eps)
                residual_queue_meters = (
                    max(lane_stats.residual_queue_vehicles, 0.0)
                    * self.kpi_config.constants.average_vehicle_space_meter
                )
                split_failure_rate = min(
                    max(self.kpi_engine.consume_pending_split_failure(lane_id), 0.0),
                    split_clip_max,
                ) / max(split_clip_max, eps)

                lane_entries.append(
                    {
                        "lane_index": lane_index,
                        "control_delay": lane_kpi.control_delay_seconds,
                        "saturation_raw": lane_kpi.degree_of_saturation,
                        "queue_length": lane_kpi.queue_length_meters,
                        "is_controllable": 1.0 if lane_id in current_green_lanes else 0.0,
                        "time_since_service": lane_stats.time_since_last_service_seconds,
                        "residual_queue": residual_queue_meters,
                        "phase_demand": lane_stats.phase_inflow_pcu,
                        "sat_norm": sat_norm,
                        "split_failure_rate": split_failure_rate,
                        "starvation_norm": starvation_norm,
                    }
                )

            tls_mean_starvation = (
                float(np.mean([entry["starvation_norm"] for entry in lane_entries]))
                if lane_entries
                else 0.0
            )

            for entry in lane_entries:
                lane_index = entry["lane_index"]
                obs[tls_index, lane_index, 0] = entry["control_delay"]
                obs[tls_index, lane_index, 1] = entry["saturation_raw"]
                obs[tls_index, lane_index, 2] = entry["queue_length"]
                obs[tls_index, lane_index, 3] = remaining_cycle
                obs[tls_index, lane_index, 4] = current_phase
                obs[tls_index, lane_index, 5] = entry["is_controllable"]
                obs[tls_index, lane_index, 6] = entry["time_since_service"]
                obs[tls_index, lane_index, 7] = entry["residual_queue"]
                obs[tls_index, lane_index, 8] = entry["phase_demand"]

                fairness_gap = max(entry["starvation_norm"] - tls_mean_starvation, 0.0)
                reward[tls_index, lane_index, 0] = -entry["control_delay"]
                reward[tls_index, lane_index, 1] = -entry["queue_length"]
                reward[tls_index, lane_index, 2] = -entry["sat_norm"]
                reward[tls_index, lane_index, 3] = -entry["split_failure_rate"]
                reward[tls_index, lane_index, 4] = -entry["starvation_norm"]
                reward[tls_index, lane_index, 5] = -fairness_gap

        return obs, reward

    def _compute_effective_green_range(self, tls_id: str) -> Tuple[float, float]:
        """
        Compute the valid extension range [e_min, e_max] for the current green phase.

        Cycle length is dynamic per scenario and per cycle, so the current phase
        only needs to respect the configured green bounds.
        """
        del tls_id
        base_green = float(self.min_green_seconds)
        max_green = float(self.max_green_seconds)
        min_ext = 0.0
        max_ext = max(0.0, min(float(self.max_extension_seconds), max_green - base_green))
        return min_ext, max_ext

    def _build_info(self, delta_t: int) -> Dict[str, object]:
        cycle_length_map = {
            tls_id: self.tls_runtime[tls_id].cycle_length_seconds
            for tls_id in self.tls_ids
        }
        cycle_length_value: object = cycle_length_map
        if len(cycle_length_map) == 1:
            cycle_length_value = next(iter(cycle_length_map.values()))

        controllable_lanes = {}
        remaining_cycle_info = {}
        residual_nge_info = {}

        for tls_id in self.tls_ids:
            runtime = self.tls_runtime[tls_id]
            remaining = max(
                0, runtime.cycle_length_seconds - runtime.cycle_elapsed_seconds
            )
            remaining_cycle_info[tls_id] = remaining
            residual_nge_info[tls_id] = {
                lane_id: float(self.kpi_engine.get_lane_stats(lane_id).residual_queue_vehicles)
                for lane_id in self.lanes_by_tls.get(tls_id, [])
            }

        for tls_id in self.required_action:
            runtime = self.tls_runtime[tls_id]
            program = self.tls_programs[tls_id]
            phase = program.phases[runtime.current_phase_index]
            if phase.phase_type == "green":
                green_lanes = self._green_lanes_for_phase(tls_id, phase.state)
                if green_lanes:
                    controllable_lanes[tls_id] = sorted(green_lanes)

        return {
            "min_green": self.min_green_seconds,
            "max_green": self.max_green_seconds,
            "cycle_length": cycle_length_value,
            "delta_t": int(delta_t),
            "intersection_require_action": sorted(self.required_action),
            "effective_action_range": {
                tls_id: self._compute_effective_green_range(tls_id)
                for tls_id in self.required_action
            },
            "controllable_lanes": controllable_lanes,
            "controllable_intersections": controllable_lanes,
            "remaining_cycle": remaining_cycle_info,
            "Residual_NGE": residual_nge_info,
            "residual_nge": residual_nge_info,
            "lane_mask": self._lane_mask(),
        }

    def reset(self) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, object]]:
        self.close()
        self._start_sumo()
        self._ensure_connection()
        self._refresh_tls_programs_from_sumo()

        self.simulation_time = 0
        self.done = False
        self.required_action = set()

        self._build_lane_topology()
        self._log_lane_width_adjustment_factors()

        # Run one step to populate lane vehicles at time 0->1.
        traci.simulationStep()
        self.simulation_time += self.step_length_seconds

        self._initialize_lane_runtime()
        self._prepare_tls_runtime()
        self.done = self._check_done()

        observation, reward = self._build_observation_reward(delta_t=0)
        info = self._build_info(delta_t=0)
        return observation, reward, self.done, info

    def step(
        self, action: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, object]]:
        self._ensure_connection()
        if not isinstance(action, dict):
            raise TypeError(
                "action must be a dict: {intersection_id: green_extension_seconds}"
            )

        if self.done:
            observation, reward = self._build_observation_reward(delta_t=0)
            return observation, reward, True, self._build_info(delta_t=0)

        self._apply_pending_actions(action)
        delta_t = self._simulate_until_need_action()
        self.done = self.done or self._check_done()

        observation, reward = self._build_observation_reward(delta_t=delta_t)
        info = self._build_info(delta_t=delta_t)
        return observation, reward, self.done, info

    def close(self) -> None:
        if getattr(self, "connected", False):
            try:
                traci.switch(self.connection_label)
                traci.close()
            except Exception:
                pass
            self.connected = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

