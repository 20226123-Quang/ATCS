from dataclasses import dataclass, field
from typing import Dict, Set

@dataclass
class LaneRuntimeStats:
    queue_count: float = 0.0
    cycle_inflow_pcu: float = 0.0
    green_seconds: float = 0.0
    previous_vehicle_ids: Set[str] = field(default_factory=set)

@dataclass(frozen=True)
class LaneKPI:
    control_delay_seconds: float
    degree_of_saturation: float
    queue_length_meters: float
    inflow_pcu_per_hour: float

class KPIEngine:
    def __init__(self):
        self._lane_stats: Dict[str, LaneRuntimeStats] = {}
        self.sat_headway = 1.8 # Khoảng cách giây giữa 2 xe khi bão hòa
        self.avg_veh_space = 7.0 # m/xe
        self.epsilon = 1e-6

    def update_lane(self, lane_id: str, current_veh_ids: Set[str], is_green: bool, step_len: float):
        if lane_id not in self._lane_stats:
            self._lane_stats[lane_id] = LaneRuntimeStats()
        stats = self._lane_stats[lane_id]
        
        # Đếm xe mới đi vào dựa trên ID
        new_vehs = current_veh_ids - stats.previous_vehicle_ids
        inflow = len(new_vehs)
        stats.cycle_inflow_pcu += inflow
        
        # Mô hình tích lũy hàng chờ: Q = Q_old + In - Out
        # Ở đây ta lấy đơn giản Out = 1 xe nếu đèn xanh và có xe chờ
        outflow = 1 if (is_green and stats.queue_count > 0) else 0
        stats.queue_count = max(0.0, stats.queue_count + inflow - (outflow * step_len))
        
        if is_green: stats.green_seconds += step_len
        stats.previous_vehicle_ids = current_veh_ids

    def compute_kpi(self, lane_id: str, cycle_len: float) -> LaneKPI:
        stats = self._lane_stats.get(lane_id, LaneRuntimeStats())
        cycle_len = max(cycle_len, self.epsilon)
        
        inflow_h = (stats.cycle_inflow_pcu * 3600.0) / cycle_len
        S = 3600.0 / self.sat_headway
        g_over_c = stats.green_seconds / cycle_len
        capacity = S * max(g_over_c, self.epsilon)
        v_over_c = inflow_h / max(capacity, self.epsilon)
        
        # Công thức tính hàng chờ và delay rút gọn từ TCCS 24:2018
        n_ge = (stats.green_seconds * S / 3600.0) * (v_over_c - 1.0) / 2.0 if v_over_c > 1 else v_over_c * 0.5
        n_ge = max(n_ge, stats.queue_count) # Lấy giá trị lớn nhất giữa lý thuyết và tích lũy
        
        t_w1 = (cycle_len * (1 - g_over_c)**2) / (2 * max(1 - (inflow_h/S), self.epsilon))
        t_w2 = (3600.0 * n_ge) / max(capacity, self.epsilon)
        
        return LaneKPI(min(t_w1 + t_w2, 300.0), v_over_c, n_ge * self.avg_veh_space, inflow_h)

    def reset_cycle(self, lane_id: str):
        if lane_id in self._lane_stats:
            self._lane_stats[lane_id].cycle_inflow_pcu = 0
            self._lane_stats[lane_id].green_seconds = 0