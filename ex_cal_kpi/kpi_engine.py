import traci
from dataclasses import dataclass, field
from typing import Dict, Set, List

@dataclass
class LaneRuntimeStats:
    # --- Dữ liệu tích lũy trong chu kỳ ---
    link_inflow_pcu: Dict[int, float] = field(default_factory=dict)
    lane_green_seconds: float = 0.0
    
    # --- Biến tồn dư (Quan trọng) ---
    n_ge_residual: float = 0.0 # Xe còn lại từ chu kỳ trước
    
    # --- Quản lý ID xe ---
    v_ids_in_cycle: Set[str] = field(default_factory=set)
    v_link_mapping: Dict[str, int] = field(default_factory=dict)
    last_step_v_ids: Set[str] = field(default_factory=set)

@dataclass
class LaneKPI:
    control_delay_seconds: float
    degree_of_saturation: float
    avg_queue_vehicles: float # N_uniform + N_GE (đơn vị: xe)
    inflow_pcu_per_hour: float # Lưu lượng mới vào
    total_demand_pcu_h: float  # Lưu lượng mới + tồn dư quy đổi ra h
    capacity_pcu_h: float
    n_ge: float # Xe tồn dư cuối chu kỳ này (để chuyển sang chu kỳ sau)

class KPIEngine:
    def __init__(self, constants: dict):
        self.constants = constants
        self.pcu_mapping = constants.get('pcu_mapping', {})
        self._lane_stats: Dict[str, LaneRuntimeStats] = {}
        self._veh_info_cache: Dict[str, dict] = {} 

    def _get_stats(self, lane_id: str) -> LaneRuntimeStats:
        if lane_id not in self._lane_stats:
            self._lane_stats[lane_id] = LaneRuntimeStats()
        return self._lane_stats[lane_id]

    def update_lane(self, lane_id: str, current_v_ids: List[str], v_movements: Dict[str, int], link_green_status: Dict[int, bool], step_len: float):
        stats = self._get_stats(lane_id)
        
        # 1. Cập nhật thời gian xanh của làn
        if any(link_green_status.values()):
            stats.lane_green_seconds += step_len

        # 2. Cập nhật Input Flow (Xe mới vào làn)
        for v_id in current_v_ids:
            if v_id not in stats.v_ids_in_cycle:
                if v_id not in self._veh_info_cache:
                    try:
                        v_type = traci.vehicle.getTypeID(v_id)
                        pcu = self.pcu_mapping.get(v_type, 1.0)
                        self._veh_info_cache[v_id] = {'pcu': pcu, 'type': v_type}
                    except: continue
                
                pcu = self._veh_info_cache[v_id]['pcu']
                link_idx = v_movements.get(v_id, -1)
                
                stats.link_inflow_pcu[link_idx] = stats.link_inflow_pcu.get(link_idx, 0.0) + pcu
                stats.v_ids_in_cycle.add(v_id)
                stats.v_link_mapping[v_id] = link_idx
        
        stats.last_step_v_ids = set(current_v_ids)

    def _calculate_nge_standard(self, x: float, m_max: float) -> float:
        m_tb = m_max
        if x <= 0.65: return 0.0
        elif x <= 0.9:
            return (x - 0.65) / 0.25 * (1.0 / (0.26 + m_tb / 150.0))
        elif x <= 1.0:
            val_09 = 1.0 / (0.26 + m_tb / 150.0)
            val_10 = 0.545 * (max(m_max, 0.0) ** 0.5)
            return val_09 + (x - 0.9) / 0.1 * (val_10 - val_09)
        elif x <= 1.2:
            val_10 = 0.545 * (max(m_max, 0.0) ** 0.5)
            val_12 = (m_max * 0.2 + 5.0) / 2.0
            return val_10 + (x - 1.0) / 0.2 * (val_12 - val_10)
        else: return m_max * (x - 1.0) / 2.0

    def compute_kpi(self, lane_id: str, cycle_len: float) -> LaneKPI:
        stats = self._get_stats(lane_id)
        eps = self.constants.get('epsilon', 1e-6)

        # 1. Tính S_hh dựa trên INPUT flow mới
        total_inflow = sum(stats.link_inflow_pcu.values())
        if total_inflow > 0:
            sum_inv_S = 0
            for link_idx, q_i in stats.link_inflow_pcu.items():
                fb, fr, fd = self.constants.get('f_b', 1.0), self.constants.get('f_r', 1.0), self.constants.get('f_d', 1.0)
                t_h = max(fb, fr, fd) * min(1.0, fd) * self.constants.get('t_h0', 1.8)
                S_i = (3600.0 / max(t_h, eps)) * self.constants.get('f_hv', 1.0)
                a_i = q_i / total_inflow
                sum_inv_S += a_i / max(S_i, eps)
            S_hh = 1.0 / max(sum_inv_S, eps)
        else:
            S_hh = (3600.0 / self.constants.get('t_h0', 1.8)) * self.constants.get('f_hv', 1.0)

        # 2. Thông số thời gian
        green_clamped = min(stats.lane_green_seconds, cycle_len)
        f_green = green_clamped / max(cycle_len, eps)
        capacity = S_hh * f_green
        m_max = (S_hh * green_clamped) / 3600.0
        
        # 3. TỔNG LƯU LƯỢNG (Input mới + Tồn dư chu kỳ trước)
        # q_total = q_inflow + n_ge_residual (quy đổi ra pcu/h để tính x và delay)
        total_demand_pcu = total_inflow + stats.n_ge_residual
        total_demand_h = (total_demand_pcu * 3600.0) / max(cycle_len, eps)
        inflow_h = (total_inflow * 3600.0) / max(cycle_len, eps)
        
        # 4. Mức độ bão hòa x và N_GE mới
        x = total_demand_h / max(capacity, eps)
        n_ge_new = self._calculate_nge_standard(x, m_max)
        n_ge_total = max(0.0, n_ge_new) # N_GE cho chu kỳ sau

        # 5. Thời gian trễ t_w
        red_time = max(0.0, cycle_len - green_clamped)
        q_over_S = total_demand_h / max(S_hh, eps)
        t_w1 = (cycle_len * (1 - f_green)**2) / (2 * max(1 - q_over_S, eps))
        t_w2 = (3600.0 * n_ge_total) / max(capacity, eps)
        
        # 6. Hàng chờ trung bình (xe)
        n_uniform = (total_demand_h * red_time) / 3600.0
        avg_queue = max(0.0, n_uniform + n_ge_total)
        
        return LaneKPI(
            control_delay_seconds=float(t_w1 + t_w2),
            degree_of_saturation=float(x),
            avg_queue_vehicles=float(avg_queue),
            inflow_pcu_per_hour=float(inflow_h),
            total_demand_pcu_h=float(total_demand_h),
            capacity_pcu_h=float(capacity),
            n_ge=float(n_ge_total)
        )

    def reset_cycle(self, lane_id: str, final_n_ge: float):
        if lane_id in self._lane_stats:
            s = self._lane_stats[lane_id]
            s.link_inflow_pcu.clear()
            s.lane_green_seconds = 0.0
            s.v_ids_in_cycle.clear()
            s.v_link_mapping.clear()
            s.n_ge_residual = final_n_ge # Lưu lại tồn dư cho chu kỳ sau
