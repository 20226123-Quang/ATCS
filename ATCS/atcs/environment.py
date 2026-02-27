"""ATCS traffic environment with KPI-standard observation and reward tensors."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import traci

from .config_loader import KPIConfig, load_kpi_config
from .kpi_engine import KPIEngine
from .sumo_parser import ParsedSUMONetwork, TLSProgram, parse_sumo_network


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
    ) -> None:
        self.connected = False
        self.sumocfg_path = Path(sumocfg_path).resolve()
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
        self.max_extension_seconds = max(self.max_green_seconds - self.min_green_seconds, 0)
        self.use_gui = sim_cfg.use_gui if use_gui is None else bool(use_gui)
        self.max_episode_seconds = (
            int(sim_cfg.max_episode_seconds)
            if max_episode_seconds is None
            else int(max_episode_seconds)
        )

        self.sumo_binary = sumo_binary or self._resolve_sumo_binary(self.use_gui)
        self.connection_label = f"ATCS_{uuid.uuid4().hex[:8]}"

        self.kpi_engine = KPIEngine(self.kpi_config.constants)
        self.tls_runtime: Dict[str, TLSRuntimeState] = {}

        self.lanes_by_tls: Dict[str, List[str]] = {}
        self.lane_link_indices: Dict[str, Dict[str, List[int]]] = {}
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

    def _ensure_connection(self) -> None:
        if not self.connected:
            raise RuntimeError("SUMO connection is not active. Call reset() first.")
        traci.switch(self.connection_label)

    def _build_lane_topology(self) -> None:
        self.lanes_by_tls = {}
        self.lane_link_indices = {}
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
            self.required_action.add(tls_id)

    def _reset_cycle_lane_metrics(self, tls_id: str) -> None:
        for lane_id in self.lanes_by_tls.get(tls_id, []):
            self.kpi_engine.start_new_cycle(lane_id)

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

    def _mark_green_seconds_for_tls(self, tls_id: str, phase_state: str) -> None:
        lane_index_map = self.lane_link_indices.get(tls_id, {})
        for lane_id in self.lanes_by_tls.get(tls_id, []):
            link_indices = lane_index_map.get(lane_id, [])
            lane_has_green = any(
                idx < len(phase_state) and phase_state[idx] in ("G", "g")
                for idx in link_indices
            )
            if lane_has_green:
                self.kpi_engine.mark_lane_green_seconds(lane_id, self.step_length_seconds)

    def _advance_to_next_phase(self, tls_id: str) -> None:
        runtime = self.tls_runtime[tls_id]
        program = self.tls_programs[tls_id]
        phase_count = len(program.phases)

        for _ in range(max(phase_count, 1)):
            next_index = (runtime.current_phase_index + 1) % phase_count
            wrapped_cycle = next_index <= runtime.current_phase_index
            runtime.current_phase_index = next_index

            if wrapped_cycle:
                runtime.cycle_elapsed_seconds = 0
                runtime.cycle_length_seconds = program.base_cycle_seconds
                self._reset_cycle_lane_metrics(tls_id)

            phase = program.phases[next_index]
            runtime.remaining_phase_seconds = max(int(phase.duration_seconds), 0)
            runtime.decision_pending = False
            traci.trafficlight.setRedYellowGreenState(tls_id, phase.state)

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
            extension = float(action.get(tls_id, 0.0))
            extension = min(max(extension, 0.0), float(self.max_extension_seconds))
            extension_seconds = int(round(extension))

            runtime.decision_pending = False
            if extension_seconds > 0:
                runtime.remaining_phase_seconds += extension_seconds
                runtime.cycle_length_seconds += extension_seconds

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

                if phase.phase_type == "green":
                    self._mark_green_seconds_for_tls(tls_id, phase.state)

                runtime.remaining_phase_seconds -= self.step_length_seconds
                if runtime.remaining_phase_seconds > 0:
                    continue

                if phase.phase_type == "green":
                    runtime.remaining_phase_seconds = 0
                    runtime.decision_pending = True
                    self.required_action.add(tls_id)
                else:
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

    def _build_observation_reward(self) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.zeros((len(self.tls_ids), self.max_lanes, 5), dtype=np.float32)
        reward = np.zeros((len(self.tls_ids), self.max_lanes, 2), dtype=np.float32)

        for tls_index, tls_id in enumerate(self.tls_ids):
            runtime = self.tls_runtime[tls_id]
            remaining_cycle = max(
                float(runtime.cycle_length_seconds - runtime.cycle_elapsed_seconds),
                0.0,
            )
            current_phase = float(runtime.current_phase_index)

            for lane_index, lane_id in enumerate(self.lanes_by_tls.get(tls_id, [])):
                lane_kpi = self.kpi_engine.compute_lane_kpis(
                    lane_id,
                    cycle_length_seconds=float(runtime.cycle_length_seconds),
                )
                obs[tls_index, lane_index, 0] = lane_kpi.control_delay_seconds
                obs[tls_index, lane_index, 1] = lane_kpi.degree_of_saturation
                obs[tls_index, lane_index, 2] = lane_kpi.queue_length_meters
                obs[tls_index, lane_index, 3] = remaining_cycle
                obs[tls_index, lane_index, 4] = current_phase

                reward[tls_index, lane_index, 0] = lane_kpi.control_delay_seconds
                reward[tls_index, lane_index, 1] = lane_kpi.queue_length_meters

        return obs, reward

    def _compute_effective_extension_range(self, tls_id: str) -> Tuple[float, float]:
        """
        Tính khoảng extension [min_ext, max_ext] hợp lệ, theo công thức:

            x = base_current + ext  (tổng thời gian xanh pha hiện tại)

            max_x = min(max_green, T_remain - remain_phase × (min_green + yellow_time))
            min_x = max(min_green, T_remain - remain_phase × (max_green + yellow_time))

            ext ∈ [min_x - base, max_x - base]  ∩  [0, max_extension]

        remain_phase = số pha xanh còn lại SAU pha hiện tại.
        yellow_time  = thời gian yellow trung bình giữa các pha xanh.
        """
        runtime = self.tls_runtime[tls_id]
        program = self.tls_programs[tls_id]

        # Thời gian còn lại trong chu kỳ
        t_remain = float(runtime.cycle_length_seconds - runtime.cycle_elapsed_seconds)

        # Base duration của pha hiện tại (chưa chạy)
        base_current = float(runtime.remaining_phase_seconds)

        # Thống kê các pha SAU pha hiện tại
        phases_after = program.phases[runtime.current_phase_index + 1:]
        remain_phase = sum(1 for p in phases_after if p.phase_type == "green")
        total_yellow_after = sum(
            p.duration_seconds for p in phases_after if p.phase_type != "green"
        )

        # Yellow trung bình mỗi pha xanh còn lại (như công thức cũ)
        yellow_time = total_yellow_after / remain_phase if remain_phase > 0 else 0.0

        # Tổng green cho pha hiện tại phải thỏa:

        print(f"t_remain: {t_remain}, remain_phase: {remain_phase}, yellow_time: {yellow_time}, min_green: {self.min_green_seconds}, max_green: {self.max_green_seconds}, max_extension: {self.max_extension_seconds}")

        max_x = min(
            float(self.max_green_seconds),
            t_remain - remain_phase * (self.min_green_seconds + yellow_time),
        )
        min_x = max(
            float(self.min_green_seconds),
            t_remain - remain_phase * (self.max_green_seconds + yellow_time),
        )

        print("max_x", max_x)
        print("min_x", min_x)
        print("base_current", base_current)
        
        # Convert sang extension (ext = x - base)
        max_ext = min(float(self.max_extension_seconds), max_x - base_current)
        min_ext = max(0.0, min_x - base_current)

        # Đảm bảo khoảng hợp lệ
        min_ext = max(0.0, min_ext)
        max_ext = max(min_ext, max_ext)

        return min_ext, max_ext


    def _build_info(self, delta_t: int) -> Dict[str, object]:
        cycle_length_map = {
            tls_id: self.tls_runtime[tls_id].cycle_length_seconds for tls_id in self.tls_ids
        }
        cycle_length_value: object = cycle_length_map
        if len(cycle_length_map) == 1:
            cycle_length_value = next(iter(cycle_length_map.values()))

        # Determine which lanes are green for the intersections that require action
        controllable_lanes = {}
        remaining_cycle_info = {}
        
        for tls_id in self.tls_ids:
            runtime = self.tls_runtime[tls_id]
            remaining = max(0, runtime.cycle_length_seconds - runtime.cycle_elapsed_seconds)
            remaining_cycle_info[tls_id] = remaining

        for tls_id in self.required_action:
            runtime = self.tls_runtime[tls_id]
            program = self.tls_programs[tls_id]
            phase = program.phases[runtime.current_phase_index]
            
            if phase.phase_type == "green":
                green_lanes = set()
                lane_index_map = self.lane_link_indices.get(tls_id, {})
                for lane_id, link_indices in lane_index_map.items():
                    # If any of the links for this lane have a 'G' or 'g' in the state
                    if any(idx < len(phase.state) and phase.state[idx] in ("G", "g") for idx in link_indices):
                        green_lanes.add(lane_id)
                if green_lanes:
                    controllable_lanes[tls_id] = sorted(list(green_lanes))

        return {
            "min_green": self.min_green_seconds,
            "max_green": self.max_green_seconds,
            "cycle_length": cycle_length_value,
            "delta_t": int(delta_t),
            "intersection_require_action": sorted(self.required_action),
            "effective_action_range": {
                tls_id: self._compute_effective_extension_range(tls_id)
                for tls_id in self.required_action
            },
            "controllable_intersections": controllable_lanes,
            "remaining_cycle": remaining_cycle_info,
        }

    def reset(self) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, object]]:
        self.close()
        self._start_sumo()
        self._ensure_connection()

        self.simulation_time = 0
        self.done = False
        self.required_action = set()

        self._build_lane_topology()

        # Run one step to populate lane vehicles at time 0->1.
        traci.simulationStep()
        self.simulation_time += self.step_length_seconds

        self._initialize_lane_runtime()
        self._prepare_tls_runtime()
        self.done = self._check_done()

        observation, reward = self._build_observation_reward()
        info = self._build_info(delta_t=0)
        return observation, reward, self.done, info

    def step(self, action: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, object]]:
        self._ensure_connection()
        if not isinstance(action, dict):
            raise TypeError("action must be a dict: {intersection_id: green_extension_seconds}")

        if self.done:
            observation, reward = self._build_observation_reward()
            return observation, reward, True, self._build_info(delta_t=0)

        self._apply_pending_actions(action)
        delta_t = self._simulate_until_need_action()
        self.done = self.done or self._check_done()

        observation, reward = self._build_observation_reward()
        info = self._build_info(delta_t=delta_t)
        return observation, reward, self.done, info

    def close(self) -> None:
        if self.connected:
            try:
                traci.switch(self.connection_label)
                traci.close()
            except Exception:
                pass
            self.connected = False

    def __del__(self) -> None:
        self.close()
