import traci
from dataclasses import dataclass, field
from typing import Dict, Set, List

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
    # Các biến bổ sung để khớp số dòng và logic
    waiting_time_total: float = 0.0
    v_ids_in_cycle: Set[str] = field(default_factory=set)

@dataclass
class LaneKPI:
    control_delay_seconds: float
    degree_of_saturation: float
    queue_length_meters: float
    inflow_pcu_per_hour: float
    outflow_pcu_per_hour: float
    avg_queue_vehicles: float
    # Thêm đầu ra cho chuẩn xác
    capacity_pcu_h: float
    green_ratio: float

class KPIEngine:
    def __init__(self, constants: dict):
        # Nạp tham số từ config_master.json
        self.constants = constants
        self.pcu_mapping = constants.get('pcu_mapping', {})
        self.t_h0 = constants.get('t_h0', 1.8)
        self.f_hv = constants.get('f_hv', 1.1)
        self.f_b = constants.get('f_b', 1.0)
        self.f_r = constants.get('f_r', 1.0)
        self.f_d = constants.get('f_d', 1.0)
        self.avg_veh_space = constants.get('avg_veh_space', 7.5)
        self.epsilon = constants.get('epsilon', 1e-6)
        
        self._lane_stats: Dict[str, LaneRuntimeStats] = {}

    def update_lane(self, lane_id: str, current_veh_ids: Set[str], is_green: bool, step_len: float):
        if lane_id not in self._lane_stats:
            self._lane_stats[lane_id] = LaneRuntimeStats()
            
        stats = self._lane_stats[lane_id]
        stats.cycle_steps += 1
        
        # A. Tích lũy thời gian xanh
        if is_green:
            stats.green_seconds += step_len
            
        # B. Cập nhật hàng chờ tức thời và tích phân
        halting_vehs = traci.lane.getLastStepHaltingNumber(lane_id)
        stats.queue_count = float(halting_vehs)
        stats.queue_integral += stats.queue_count * step_len
        
        # C. Logic Inflow PCU (Chỉ tính xe mới vào làn)
        new_vehs = current_veh_ids - stats.previous_vehicle_ids
        for v_id in new_vehs:
            if v_id not in stats.v_ids_in_cycle:
                v_type = traci.vehicle.getTypeID(v_id)
                pcu = self.pcu_mapping.get(v_type, 1.0)
                stats.cycle_inflow_pcu += pcu
                stats.v_ids_in_cycle.add(v_id)
                
        # D. Logic Outflow PCU (Xe rời khỏi làn)
        left_vehs = stats.previous_vehicle_ids - current_veh_ids
        for v_id in left_vehs:
            # Dùng try-except vì xe có thể đã biến mất khỏi SUMO
            try:
                v_type = traci.vehicle.getTypeID(v_id)
                stats.cycle_outflow_pcu += self.pcu_mapping.get(v_type, 1.0)
            except:
                pass
                
        stats.previous_vehicle_ids = current_veh_ids
        # Tích lũy Waiting Time (nếu cần cho các KPI phụ)
        stats.waiting_time_total += traci.lane.getWaitingTime(lane_id) * step_len

    def compute_kpi(self, lane_id: str, cycle_len: float) -> LaneKPI:
        stats = self._lane_stats.get(lane_id, LaneRuntimeStats())
        cycle_len = max(cycle_len, 1.0)
        eps = self.epsilon

        # 1. Saturation Flow (S) theo HBS
        f_adj = max(self.f_b, self.f_r, self.f_d)
        t_h = self.f_hv * f_adj * self.t_h0
        S = 3600.0 / max(t_h, eps)
        
        # 2. Flow Rates (PCU/h)
        inflow_h = (stats.cycle_inflow_pcu * 3600.0) / cycle_len
        outflow_h = (stats.cycle_outflow_pcu * 3600.0) / cycle_len
        
        # 3. Capacity & Saturation (v/c)
        g_over_c = stats.green_seconds / cycle_len
        capacity = S * max(g_over_c, eps)
        v_over_c = inflow_h / max(capacity, eps)

        # 4. Hàng chờ N_GE (Nội suy HBS 2001)
        m_max = (stats.green_seconds * S) / 3600.0
        m_tb = (inflow_h * cycle_len) / 3600.0
        x = v_over_c

        if x <= 0.65:
            n_ge = 0.0
        elif x <= 0.9:
            val_09 = 1.0 / (0.26 + m_tb / 150.0)
            n_ge = (x - 0.65) / 0.25 * val_09
        elif x <= 1.0:
            val_09 = 1.0 / (0.26 + m_tb / 150.0)
            val_10 = 0.3476 * (max(m_max, 0.0) ** 0.5)
            n_ge = val_09 + (x - 0.9) / 0.1 * (val_10 - val_09)
        elif x <= 1.2:
            val_10 = 0.3476 * (max(m_max, 0.0) ** 0.5)
            val_12 = (m_max * 0.2 + 5.0) / 2.0
            n_ge = val_10 + (x - 1.0) / 0.2 * (val_12 - val_10)
        else:
            n_ge = m_max * (x - 1.0) / 2.0

        # 5. Control Delay (t_w = t_w1 + t_w2)
        y = min(inflow_h / S, 0.99) # Tỷ lệ lưu lượng/bão hòa
        t_w1 = (cycle_len * (1 - g_over_c)**2) / (2 * (1 - y))
        t_w2 = (3600.0 * n_ge) / max(capacity, eps)
        
        delay = min(t_w1 + t_w2, 300.0)
        avg_q = stats.queue_integral / cycle_len

        return LaneKPI(
            control_delay_seconds=float(delay),
            degree_of_saturation=float(v_over_c),
            queue_length_meters=float(n_ge * self.avg_veh_space),
            inflow_pcu_per_hour=float(inflow_h),
            outflow_pcu_per_hour=float(outflow_h),
            avg_queue_vehicles=float(avg_q),
            capacity_pcu_h=float(capacity),
            green_ratio=float(g_over_c)
        )

    def reset_cycle(self, lane_id: str):
        if lane_id in self._lane_stats:
            stats = self._lane_stats[lane_id]
            stats.initial_cycle_queue = stats.queue_count
            stats.cycle_inflow_pcu = 0.0
            stats.cycle_outflow_pcu = 0.0
            stats.green_seconds = 0.0
            stats.cycle_steps = 0
            stats.queue_integral = 0.0
            stats.waiting_time_total = 0.0
            stats.v_ids_in_cycle.clear()
            # Lưu ý: previous_vehicle_ids KHÔNG reset để check xe đi qua ranh giới chu kỳ