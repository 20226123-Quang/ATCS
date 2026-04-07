import traci
from dataclasses import dataclass, field
from typing import Dict, Set, List

@dataclass
class LaneRuntimeStats:
    # --- Dữ liệu tích lũy trong chu kỳ ---
    # Inflow/Outflow tính theo PCU cho từng link_index (tlsid index)
    link_inflow_pcu: Dict[int, float] = field(default_factory=dict)
    link_outflow_pcu: Dict[int, float] = field(default_factory=dict)
    
    # Thời gian xanh tích lũy cho từng link_index
    link_green_seconds: Dict[int, float] = field(default_factory=dict)
    
    # Tổng thời gian xanh của làn (bất kỳ hướng nào xanh)
    lane_green_seconds: float = 0.0
    
    # --- Biến tồn dư ---
    n_ge_residual: float = 0.0 
    
    # --- Quản lý ID xe ---
    v_ids_in_cycle: Set[str] = field(default_factory=set) # Đã vào làn
    v_link_mapping: Dict[str, int] = field(default_factory=dict) # v_id -> link_index
    last_step_v_ids: Set[str] = field(default_factory=set)

@dataclass
class LaneKPI:
    control_delay_seconds: float
    degree_of_saturation: float
    queue_length_meters: float
    avg_queue_vehicles: float
    inflow_pcu_per_hour: float
    outflow_pcu_per_hour: float
    capacity_pcu_h: float
    n_ge: float

class KPIEngine:
    def __init__(self, constants: dict):
        self.constants = constants
        self.pcu_mapping = constants.get('pcu_mapping', {})
        self.avg_veh_space = constants.get('avg_veh_space', 7.5)
        self._lane_stats: Dict[str, LaneRuntimeStats] = {}
        self._veh_info_cache: Dict[str, dict] = {} 

    def _get_stats(self, lane_id: str) -> LaneRuntimeStats:
        if lane_id not in self._lane_stats:
            self._lane_stats[lane_id] = LaneRuntimeStats()
        return self._lane_stats[lane_id]

    def update_lane(self, lane_id: str, current_v_ids: List[str], v_movements: Dict[str, int], link_green_status: Dict[int, bool], step_len: float):
        """
        current_v_ids: Xe đang ở trên làn
        v_movements: v_id -> link_index (lấy từ getNextLinks trong main.py)
        link_green_status: link_index -> bool (trạng thái đèn xanh của từng kết nối)
        """
        stats = self._get_stats(lane_id)
        current_v_ids_set = set(current_v_ids)
        
        # 1. Cập nhật thời gian xanh
        any_lane_green = False
        for link_idx, is_green in link_green_status.items():
            if is_green:
                stats.link_green_seconds[link_idx] = stats.link_green_seconds.get(link_idx, 0.0) + step_len
                any_lane_green = True
        if any_lane_green:
            stats.lane_green_seconds += step_len

        # 2. Cập nhật Inflow (Xe mới vào làn)
        for v_id in current_v_ids:
            if v_id not in stats.v_ids_in_cycle:
                # Lấy PCU
                if v_id not in self._veh_info_cache:
                    try:
                        v_type = traci.vehicle.getTypeID(v_id)
                        pcu = self.pcu_mapping.get(v_type, 1.0)
                        self._veh_info_cache[v_id] = {'pcu': pcu, 'type': v_type}
                    except: continue
                
                pcu = self._veh_info_cache[v_id]['pcu']
                link_idx = v_movements.get(v_id, -1) # -1 nếu không xác định được (hiếm)
                
                stats.link_inflow_pcu[link_idx] = stats.link_inflow_pcu.get(link_idx, 0.0) + pcu
                stats.v_ids_in_cycle.add(v_id)
                stats.v_link_mapping[v_id] = link_idx

        # 3. Cập nhật Outflow (Xe rời khỏi làn - Thoát xe)
        exited_ids = stats.last_step_v_ids - current_v_ids_set
        for v_id in exited_ids:
            # Chỉ đếm là outflow nếu xe đó đã từng được ghi nhận inflow (tránh xe teleport/vừa sinh ra đã mất)
            if v_id in stats.v_link_mapping:
                pcu = self._veh_info_cache.get(v_id, {'pcu': 1.0})['pcu']
                link_idx = stats.v_link_mapping[v_id]
                stats.link_outflow_pcu[link_idx] = stats.link_outflow_pcu.get(link_idx, 0.0) + pcu
        
        stats.last_step_v_ids = current_v_ids_set

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

        # Tính S_hh (Lưu lượng bão hòa hỗn hợp) dựa trên lưu lượng THOÁT (Outflow) của từng link
        total_outflow = sum(stats.link_outflow_pcu.values())
        if total_outflow > 0:
            sum_inv_S = 0
            for link_idx, q_i in stats.link_outflow_pcu.items():
                # Giả định Si cho hướng này (có thể mở rộng mapping link_idx -> hướng rẽ)
                # Tạm thời dùng hệ số chung, bạn có thể truyền thêm mapping hướng vào đây
                fb, fr, fd = self.constants.get('f_b', 1.0), self.constants.get('f_r', 1.0), self.constants.get('f_d', 1.0)
                t_h = max(fb, fr, fd) * min(1.0, fd) * self.constants.get('t_h0', 1.8)
                S_i = (3600.0 / max(t_h, eps)) * self.constants.get('f_hv', 1.0)
                
                a_i = q_i / total_outflow
                sum_inv_S += a_i / max(S_i, eps)
            S_hh = 1.0 / max(sum_inv_S, eps)
        else:
            S_hh = (3600.0 / self.constants.get('t_h0', 1.8)) * self.constants.get('f_hv', 1.0)

        # Các thông số KPI
        f_green = stats.lane_green_seconds / max(cycle_len, eps)
        capacity = S_hh * f_green
        m_max = (S_hh * stats.lane_green_seconds) / 3600.0
        
        total_inflow = sum(stats.link_inflow_pcu.values())
        inflow_h = (total_inflow * 3600.0) / max(cycle_len, eps)
        outflow_h = (total_outflow * 3600.0) / max(cycle_len, eps)
        
        x = inflow_h / max(capacity, eps)
        n_ge_total = self._calculate_nge_standard(x, m_max) + stats.n_ge_residual

        # Delay và Queue
        t_w1 = (cycle_len * (1 - f_green)**2) / (2 * max(1 - (inflow_h/max(S_hh, eps)), eps))
        t_w2 = (3600.0 * n_ge_total) / max(capacity, eps)
        
        n_uniform = (inflow_h * (cycle_len - stats.lane_green_seconds)) / 3600.0
        avg_queue = n_uniform + n_ge_total
        
        return LaneKPI(
            control_delay_seconds=float(t_w1 + t_w2),
            degree_of_saturation=float(x),
            queue_length_meters=float(avg_queue * self.avg_veh_space),
            avg_queue_vehicles=float(avg_queue),
            inflow_pcu_per_hour=float(inflow_h),
            outflow_pcu_per_hour=float(outflow_h),
            capacity_pcu_h=float(capacity),
            n_ge=float(n_ge_total)
        )

    def reset_cycle(self, lane_id: str, final_n_ge: float):
        if lane_id in self._lane_stats:
            s = self._lane_stats[lane_id]
            s.link_inflow_pcu.clear()
            s.link_outflow_pcu.clear()
            s.link_green_seconds.clear()
            s.lane_green_seconds = 0.0
            s.v_ids_in_cycle.clear()
            s.v_link_mapping.clear()
            s.n_ge_residual = final_n_ge
