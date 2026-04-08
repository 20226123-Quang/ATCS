"""Evaluate crowded fixed-time vs RL with one-cycle KPI logic."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

_MPL_DIR = Path(__file__).resolve().parent / ".mplconfig_crowded_cycle_kpi"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import torch
import traci

from acac import (
    ACACTrainer,
    AgentHistoryEncoder,
    AsyncTrajectoryBuffer,
    CentralizedCritic,
    MacroActor,
    SinusoidalPositionalEncoding,
    SyncTrajectoryBuffer,
    load_model_config,
)
from atcs.config_loader import KPIConstants, load_kpi_config
from atcs.environment import TrafficEnvironment
from atcs.kpi_engine import KPIEngine

_MODEL_CFG = load_model_config()


def initialize_acac(obs_dim, action_dim, tls_names, device="cpu"):
    num_agents = len(tls_names)
    time_encoder = SinusoidalPositionalEncoding(_MODEL_CFG.model.time_embed_dim).to(device)
    encoders = [
        AgentHistoryEncoder(obs_dim, _MODEL_CFG.model.time_embed_dim, _MODEL_CFG.model.hidden_dim).to(device)
        for _ in range(num_agents)
    ]
    actors = [
        MacroActor(_MODEL_CFG.model.hidden_dim, action_dim, min_action=0.0, max_action=1.0).to(device)
        for _ in range(num_agents)
    ]
    critic = CentralizedCritic(_MODEL_CFG.model.hidden_dim, _MODEL_CFG.model.num_heads).to(device)
    agents_buffer = AsyncTrajectoryBuffer(
        capacity=_MODEL_CFG.training.buffer_size, num_agents=num_agents
    )
    critic_buffer = SyncTrajectoryBuffer(capacity=_MODEL_CFG.training.buffer_size)

    all_params = list(critic.parameters())
    for actor in actors:
        all_params += list(actor.parameters())
    for encoder in encoders:
        all_params += list(encoder.parameters())

    optimizer = torch.optim.Adam(all_params, lr=_MODEL_CFG.training.actor_lr)
    return ACACTrainer(
        actors=actors,
        encoders=encoders,
        critic=critic,
        agents_buffer=agents_buffer,
        critic_buffer=critic_buffer,
        optimizers={"combined": optimizer},
        time_encoder=time_encoder,
        tls_names=tls_names,
        device=device,
        gamma=_MODEL_CFG.training.gamma,
        lam=_MODEL_CFG.training.lam,
        eps_clip=_MODEL_CFG.training.eps_clip,
    )


@dataclass
class LaneCycleState:
    previous_vehicle_ids: Set[str] = field(default_factory=set)
    vehicle_link_index: Dict[str, int] = field(default_factory=dict)
    v_ids_in_cycle: Set[str] = field(default_factory=set)
    queue_integral: float = 0.0
    lane_green_seconds: float = 0.0
    waiting_time_total: float = 0.0
    inflow_pcu_by_link: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    outflow_pcu_by_link: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    green_seconds_by_link: Dict[int, float] = field(default_factory=lambda: defaultdict(float))

    def reset_cycle(self) -> None:
        self.v_ids_in_cycle.clear()
        self.queue_integral = 0.0
        self.lane_green_seconds = 0.0
        self.waiting_time_total = 0.0
        self.inflow_pcu_by_link.clear()
        self.outflow_pcu_by_link.clear()
        self.green_seconds_by_link.clear()


@dataclass(frozen=True)
class LaneCycleKPI:
    controller: str
    tls_id: str
    lane_id: str
    cycle_index: int
    cycle_length_seconds: float
    queue_length_meters: float
    control_delay_seconds: float
    degree_of_saturation: float
    q_h_pcu_per_hour: float
    lane_green_ratio: float
    lane_green_seconds: float
    mixed_saturation_flow_pcu_per_hour: float
    outflow_pcu_per_cycle: float
    inflow_pcu_per_cycle: float
    observed_avg_wait_seconds: float


def _vehicle_pcu(constants: KPIConstants, vehicle_id: str, fallback_only: bool = False) -> float:
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


def _vehicle_tls_index(vehicle_id: str, tls_id: str) -> Optional[int]:
    try:
        next_tls = traci.vehicle.getNextTLS(vehicle_id)
    except Exception:
        return None
    for item in next_tls:
        if item and str(item[0]) == str(tls_id):
            try:
                return int(item[1])
            except (TypeError, ValueError):
                return None
    return None


class CycleKPICollector:
    def __init__(self, constants: KPIConstants, controller_name: str):
        self.constants = constants
        self.controller_name = controller_name
        self.step_length_seconds = 1.0
        self.env: Optional[TrafficEnvironment] = None
        self.lane_states: Dict[str, LaneCycleState] = {}
        self.cycle_counter_by_tls: Dict[str, int] = defaultdict(int)
        self.rows: list[LaneCycleKPI] = []

    def attach(self, env: TrafficEnvironment) -> None:
        self.env = env
        self.step_length_seconds = float(env.step_length_seconds)
        self.rows = []
        self.cycle_counter_by_tls = defaultdict(int)
        self.lane_states = {
            lane_id: LaneCycleState(
                previous_vehicle_ids=set(traci.lane.getLastStepVehicleIDs(lane_id))
            )
            for tls_id in env.tls_ids
            for lane_id in env.lanes_by_tls.get(tls_id, [])
        }

    def _lane_saturation_flow(self, lane_id: str) -> float:
        eps = self.constants.epsilon
        lane_width_m = None
        if self.env is not None:
            lane_width_m = self.env.lane_width_m.get(lane_id)
        f_b_nomograph = KPIEngine._lane_width_adjustment_factor(lane_width_m)
        f_b = max(self.constants.saturation_headway_f_b * f_b_nomograph, eps)
        f_r = max(self.constants.saturation_headway_f_r, eps)
        f_d = max(self.constants.saturation_headway_f_d, eps)
        f1 = max(f_b, f_r, f_d)
        f2 = min(1.0, f_d)
        t_h = max(
            self.constants.saturation_headway_f_hv
            * f1
            * f2
            * self.constants.saturation_headway_base_seconds,
            eps,
        )
        return 3600.0 / t_h

    def on_simulation_step(self) -> None:
        if self.env is None:
            return
        for tls_id in self.env.tls_ids:
            runtime = self.env.tls_runtime[tls_id]
            phase_state = self.env.tls_programs[tls_id].phases[runtime.current_phase_index].state
            lane_index_map = self.env.lane_link_indices.get(tls_id, {})
            for lane_id in self.env.lanes_by_tls.get(tls_id, []):
                state = self.lane_states[lane_id]
                current_vehicle_ids = set(traci.lane.getLastStepVehicleIDs(lane_id))
                state.queue_integral += (
                    float(traci.lane.getLastStepHaltingNumber(lane_id)) * self.step_length_seconds
                )
                try:
                    state.waiting_time_total += (
                        float(traci.lane.getWaitingTime(lane_id)) * self.step_length_seconds
                    )
                except Exception:
                    pass

                lane_has_green = False
                for link_index in lane_index_map.get(lane_id, []):
                    if link_index < len(phase_state) and phase_state[link_index] in ("G", "g", "u"):
                        state.green_seconds_by_link[link_index] += self.step_length_seconds
                        lane_has_green = True
                if lane_has_green:
                    state.lane_green_seconds += self.step_length_seconds

                entered = current_vehicle_ids - state.previous_vehicle_ids
                exited = state.previous_vehicle_ids - current_vehicle_ids

                for vehicle_id in entered:
                    if vehicle_id in state.v_ids_in_cycle:
                        continue
                    link_index = _vehicle_tls_index(vehicle_id, tls_id)
                    if link_index is None:
                        lane_links = lane_index_map.get(lane_id, [])
                        if len(lane_links) == 1:
                            link_index = lane_links[0]
                    if link_index is None:
                        continue
                    state.vehicle_link_index[vehicle_id] = link_index
                    state.v_ids_in_cycle.add(vehicle_id)
                    state.inflow_pcu_by_link[link_index] += _vehicle_pcu(self.constants, vehicle_id)

                for vehicle_id in exited:
                    link_index = state.vehicle_link_index.pop(vehicle_id, None)
                    if link_index is None:
                        lane_links = lane_index_map.get(lane_id, [])
                        if len(lane_links) == 1:
                            link_index = lane_links[0]
                    if link_index is None:
                        continue
                    state.outflow_pcu_by_link[link_index] += _vehicle_pcu(
                        self.constants,
                        vehicle_id,
                        fallback_only=True,
                    )

                state.previous_vehicle_ids = current_vehicle_ids

    def on_cycle_end(self, tls_id: str, cycle_length_seconds: float) -> None:
        if self.env is None:
            return
        self._append_cycle_rows(tls_id, cycle_length_seconds)

    def finalize_open_cycles(self) -> None:
        if self.env is None:
            return
        for tls_id in self.env.tls_ids:
            runtime = self.env.tls_runtime.get(tls_id)
            if runtime is None:
                continue
            if self.cycle_counter_by_tls.get(tls_id, 0) > 0:
                continue
            cycle_length = max(float(runtime.cycle_elapsed_seconds), self.step_length_seconds)
            has_cycle_data = any(
                (
                    state.queue_integral > 0.0
                    or state.lane_green_seconds > 0.0
                    or sum(state.inflow_pcu_by_link.values()) > 0.0
                    or sum(state.outflow_pcu_by_link.values()) > 0.0
                )
                for lane_id, state in self.lane_states.items()
                if lane_id in self.env.lanes_by_tls.get(tls_id, [])
            )
            if not has_cycle_data:
                continue
            self._append_cycle_rows(tls_id, cycle_length)

    def _append_cycle_rows(self, tls_id: str, cycle_length_seconds: float) -> None:
        if self.env is None:
            return
        eps = self.constants.epsilon
        cycle_length = max(float(cycle_length_seconds), self.step_length_seconds)
        cycle_index = self.cycle_counter_by_tls[tls_id]
        self.cycle_counter_by_tls[tls_id] += 1

        for lane_id in self.env.lanes_by_tls.get(tls_id, []):
            state = self.lane_states[lane_id]
            total_outflow = float(sum(state.outflow_pcu_by_link.values()))
            total_inflow = float(sum(state.inflow_pcu_by_link.values()))
            q_h = total_outflow * 3600.0 / cycle_length

            if total_outflow > eps:
                ratios = {
                    idx: flow / total_outflow
                    for idx, flow in state.outflow_pcu_by_link.items()
                    if flow > eps
                }
            else:
                flow_base = max(total_inflow, eps)
                ratios = {
                    idx: flow / flow_base
                    for idx, flow in state.inflow_pcu_by_link.items()
                    if flow > eps
                }
            if not ratios:
                lane_links = self.env.lane_link_indices.get(tls_id, {}).get(lane_id, [])
                if lane_links:
                    ratio_value = 1.0 / len(lane_links)
                    ratios = {idx: ratio_value for idx in lane_links}

            lane_sat = self._lane_saturation_flow(lane_id)
            denominator = sum(ratio / max(lane_sat, eps) for ratio in ratios.values())
            mixed_sat = 1.0 / max(denominator, 1.0 / max(lane_sat, eps))
            green_ratio = min(max(state.lane_green_seconds / cycle_length, 0.0), 0.999)
            capacity = mixed_sat * max(green_ratio, eps)
            x_val = q_h / max(capacity, eps)

            m_max = state.lane_green_seconds * mixed_sat / 3600.0
            m_tb = q_h * cycle_length / 3600.0
            if x_val <= 0.65:
                n_ge = 0.0
            elif x_val <= 0.9:
                val_09 = 1.0 / max(0.26 + m_tb / 150.0, eps)
                n_ge = (x_val - 0.65) / 0.25 * val_09
            elif x_val <= 1.0:
                val_09 = 1.0 / max(0.26 + m_tb / 150.0, eps)
                val_10 = 0.545 * (max(m_max, 0.0) ** 0.5)
                n_ge = val_09 + (x_val - 0.9) / 0.1 * (val_10 - val_09)
            elif x_val <= 1.2:
                val_10 = 0.545 * (max(m_max, 0.0) ** 0.5)
                val_12 = (m_max * 0.2 + 5.0) / 2.0
                n_ge = val_10 + (x_val - 1.0) / 0.2 * (val_12 - val_10)
            else:
                n_ge = m_max * (x_val - 1.0) / 2.0

            q_over_s = min(q_h / max(mixed_sat, eps), 0.99)
            t_w1 = (cycle_length * (1.0 - green_ratio) ** 2) / (
                2.0 * max(1.0 - q_over_s, eps)
            )
            t_w2 = (3600.0 * max(n_ge, 0.0)) / max(capacity, eps)
            delay = min(t_w1 + t_w2, self.constants.max_control_delay_seconds)

            self.rows.append(
                LaneCycleKPI(
                    controller=self.controller_name,
                    tls_id=tls_id,
                    lane_id=lane_id,
                    cycle_index=cycle_index,
                    cycle_length_seconds=cycle_length,
                    queue_length_meters=(
                        state.queue_integral / cycle_length * self.constants.average_vehicle_space_meter
                    ),
                    control_delay_seconds=float(delay),
                    degree_of_saturation=float(x_val),
                    q_h_pcu_per_hour=float(q_h),
                    lane_green_ratio=float(green_ratio),
                    lane_green_seconds=float(state.lane_green_seconds),
                    mixed_saturation_flow_pcu_per_hour=float(mixed_sat),
                    outflow_pcu_per_cycle=float(total_outflow),
                    inflow_pcu_per_cycle=float(total_inflow),
                    observed_avg_wait_seconds=float(state.waiting_time_total / max(cycle_length, eps)),
                )
            )
            state.reset_cycle()


class InstrumentedTrafficEnvironment(TrafficEnvironment):
    def __init__(self, *args, collector: Optional[CycleKPICollector] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.collector = collector

    def reset(self):
        result = super().reset()
        if self.collector is not None:
            self.collector.attach(self)
        return result

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
                if self.collector is not None:
                    self.collector.on_cycle_end(tls_id, float(runtime.cycle_length_seconds))
                runtime.cycle_elapsed_seconds = 0
                runtime.cycle_length_seconds = program.base_cycle_seconds
                self._reset_cycle_lane_metrics(tls_id)

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

    def _simulate_until_need_action(self) -> int:
        delta_t = 0
        while not self.done and not self.required_action:
            traci.simulationStep()
            self.simulation_time += self.step_length_seconds
            delta_t += self.step_length_seconds
            self._update_lane_accumulation()
            if self.collector is not None:
                self.collector.on_simulation_step()

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
                self._advance_to_next_phase(tls_id)
            self.done = self._check_done()
        return delta_t


def _run_fixed(env: InstrumentedTrafficEnvironment, max_seconds: int, fixed_extension: float):
    _, _, done, info = env.reset()
    while not done and env.simulation_time < max_seconds:
        action = {tls_id: float(fixed_extension) for tls_id in info.get("intersection_require_action", [])}
        _, _, done, info = env.step(action)
        if env.simulation_time >= max_seconds:
            break
    if env.collector is not None:
        env.collector.finalize_open_cycles()
    env.close()
    return list(env.collector.rows if env.collector is not None else [])


def _run_rl(env: InstrumentedTrafficEnvironment, trainer: ACACTrainer, max_seconds: int):
    obs, _, done, info = env.reset()
    trainer._reset_hidden()
    t = 0
    while not done and env.simulation_time < max_seconds:
        action = {}
        for name in info.get("intersection_require_action", []):
            i = trainer.tls_index[name]
            z_it = trainer._obs_to_tensor(obs, i).unsqueeze(0)
            eff_range = info.get("effective_action_range", {}).get(
                name,
                (0.0, info["max_green"] - info["min_green"]),
            )
            z_it = torch.cat(
                [
                    z_it,
                    torch.tensor([eff_range], dtype=torch.float32).to(trainer.device),
                ],
                dim=-1,
            )
            p_it = trainer.time_encoder(t).to(trainer.device).unsqueeze(0)
            h_prev = trainer.hidden_states[i].unsqueeze(0)
            trainer.hidden_states[i] = trainer.encoders[i](z_it, p_it, h_prev).squeeze(0)
            actor_out = trainer.actors[i].act(
                trainer.hidden_states[i].unsqueeze(0),
                deterministic=True,
            )
            actor_val = float(actor_out.detach().item())
            action[name] = trainer._scale_action(actor_val, eff_range[0], eff_range[1])

        obs, _, done, info = env.step(action)
        t += info["delta_t"]
        if env.simulation_time >= max_seconds:
            break
    if env.collector is not None:
        env.collector.finalize_open_cycles()
    env.close()
    return list(env.collector.rows if env.collector is not None else [])


def _aggregate_summary(rows: Iterable[LaneCycleKPI], controller_name: str):
    rows = list(rows)
    if not rows:
        return {
            "controller": controller_name,
            "queue_length_m": float("nan"),
            "avg_wait_time_s": float("nan"),
            "saturation_x": float("nan"),
            "observed_wait_time_s": float("nan"),
        }
    return {
        "controller": controller_name,
        "queue_length_m": float(np.mean([row.queue_length_meters for row in rows])),
        "avg_wait_time_s": float(np.mean([row.control_delay_seconds for row in rows])),
        "saturation_x": float(np.mean([row.degree_of_saturation for row in rows])),
        "observed_wait_time_s": float(np.mean([row.observed_avg_wait_seconds for row in rows])),
    }


def _write_cycle_rows(rows: Iterable[LaneCycleKPI], output_path: Path) -> None:
    rows = list(rows)
    fieldnames = list(LaneCycleKPI.__annotations__.keys())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_summary(rows, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "controller",
                "queue_length_m",
                "avg_wait_time_s",
                "saturation_x",
                "observed_wait_time_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(rows, output_path: Path) -> None:
    metrics = [
        ("Queue Length", "queue_length_m", "m"),
        ("Avg Wait Time", "avg_wait_time_s", "s"),
        ("Saturation", "saturation_x", ""),
    ]
    labels = [
        "Fixed time (Quang)" if row["controller"] == "fixed_time_quang" else "AI (RL)"
        for row in rows
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#d16b6b", "#4b9f70"]
    for ax, (title, key, unit) in zip(axes, metrics):
        values = [float(row[key]) for row in rows]
        bars = ax.bar(labels, values, color=colors[: len(values)])
        ax.set_title(title)
        ax.set_ylabel(f"{title} ({unit})" if unit else title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for bar, value in zip(bars, values):
            label = f"{value:.2f}s" if key == "avg_wait_time_s" else f"{value:.4f}"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("Crowded 2Intersection, 400s, one-cycle KPI")
    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sumocfg",
        default=str(
            repo_root
            / "SimulationData"
            / "Evaluate"
            / "Crowded"
            / "2Intersection"
            / "config.sumocfg"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            repo_root
            / "checkpoints"
            / "crowded_2intersection"
            / "crowded_2intersection_checkpoint.pt"
        ),
    )
    parser.add_argument("--seconds", type=int, default=400)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fixed-extension", type=float, default=30.0)
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "checkpoints" / "crowded_cycle_kpi_400s"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    kpi_config = load_kpi_config()

    fixed_rows = _run_fixed(
        InstrumentedTrafficEnvironment(
            sumocfg_path=args.sumocfg,
            use_gui=False,
            max_episode_seconds=args.seconds,
            collector=CycleKPICollector(kpi_config.constants, "fixed_time_quang"),
        ),
        max_seconds=args.seconds,
        fixed_extension=args.fixed_extension,
    )

    shape_env = InstrumentedTrafficEnvironment(
        sumocfg_path=args.sumocfg,
        use_gui=False,
        max_episode_seconds=args.seconds,
    )
    obs, _, _, _ = shape_env.reset()
    obs_dim = obs.shape[1] * obs.shape[2]
    tls_names = shape_env.tls_ids
    shape_env.close()

    trainer = initialize_acac(obs_dim=obs_dim + 2, action_dim=1, tls_names=tls_names, device=args.device)
    trainer.load_model(args.checkpoint)
    for actor in trainer.actors:
        actor.eval()
    for encoder in trainer.encoders:
        encoder.eval()
    trainer.critic.eval()

    rl_rows = _run_rl(
        InstrumentedTrafficEnvironment(
            sumocfg_path=args.sumocfg,
            use_gui=False,
            max_episode_seconds=args.seconds,
            collector=CycleKPICollector(kpi_config.constants, "ai_rl"),
        ),
        trainer=trainer,
        max_seconds=args.seconds,
    )

    summary_rows = [
        _aggregate_summary(fixed_rows, "fixed_time_quang"),
        _aggregate_summary(rl_rows, "ai_rl"),
    ]
    _write_cycle_rows(fixed_rows, output_dir / "fixed_time_quang_cycle_rows.csv")
    _write_cycle_rows(rl_rows, output_dir / "ai_rl_cycle_rows.csv")
    _write_summary(summary_rows, output_dir / "crowded_cycle_kpi_summary.csv")
    _plot_summary(summary_rows, output_dir / "crowded_cycle_kpi_compare.png")

    print("Crowded cycle KPI summary")
    for row in summary_rows:
        print(
            f"{row['controller']}: "
            f"queue={float(row['queue_length_m']):.4f} m, "
            f"wait={float(row['avg_wait_time_s']):.2f} s, "
            f"sat={float(row['saturation_x']):.4f}, "
            f"observed_wait={float(row['observed_wait_time_s']):.2f} s"
        )
    print(f"Saved summary CSV: {output_dir / 'crowded_cycle_kpi_summary.csv'}")
    print(f"Saved plot: {output_dir / 'crowded_cycle_kpi_compare.png'}")


if __name__ == "__main__":
    main()
