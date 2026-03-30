from dataclasses import dataclass, field
from typing import Dict, Set, Optional

@dataclass
class LaneRuntimeStats:
    queue_count: float = 0.0
    cycle_inflow_pcu: float = 0.0
    green_seconds: float = 0.0
    previous_vehicle_ids: Set[str] = field(default_factory=set)
    # Các biến bổ sung để tính chuẩn TCCS
    cycle_steps: int = 0
    phase_queue_start: float = 0.0

@dataclass(frozen=True)
class LaneKPI:
    control_delay_seconds: float
    degree_of_saturation: float
    queue_length_meters: float
    inflow_pcu_per_hour: float

class KPIEngine:
    def __init__(self):
        self._lane_stats: Dict[str, LaneRuntimeStats] = {}
        # Hằng số chuẩn theo TCCS 24:2018 / HBS 2001
        self.t_h0 = 1.8           # Headway cơ bản (s)
        self.f_hv = 1.0           # Hệ số xe nặng (giả định 1.0 nếu đã đổi ra PCU)
        self.f_b = 1.0            # Hệ số bề rộng làn (mặc định)
        self.f_r = 1.0            # Hệ số rẽ
        self.f_d = 1.0            # Hệ số xe dừng đỗ
        self.avg_veh_space = 7.0  # Khoảng cách xe (m/xe)
        self.epsilon = 1e-6

    def update_lane(self, lane_id: str, current_veh_ids: Set[str], is_green: bool, step_len: float):
        if lane_id not in self._lane_stats:
            self._lane_stats[lane_id] = LaneRuntimeStats()
        stats = self._lane_stats[lane_id]
        
        # 1. Tính Inflow dựa trên sự xuất hiện của ID xe mới
        new_vehs = current_veh_ids - stats.previous_vehicle_ids
        inflow = len(new_vehs)
        stats.cycle_inflow_pcu += inflow
        
        # 2. Cập nhật hàng chờ tích lũy (Input-Output Model) 
        # Giả định năng suất thoát thực tế khi xanh là 1 xe / 1.8s
        outflow_rate = (1.0 / self.t_h0) if is_green else 0.0
        stats.queue_count = max(0.0, stats.queue_count + inflow - (outflow_rate * step_len))
        
        if is_green:
            stats.green_seconds += step_len
            
        stats.previous_vehicle_ids = current_veh_ids
        stats.cycle_steps += 1

    def compute_kpi(self, lane_id: str, cycle_len: float) -> LaneKPI:
        stats = self._lane_stats.get(lane_id, LaneRuntimeStats())
        cycle_len = max(cycle_len, 1.0)
        eps = self.epsilon # Thường là 1e-6 hoặc 0.001

        # A. Tính Công suất S (Saturation Flow) chuẩn
        f1 = max(self.f_b, self.f_r, self.f_d)
        f2 = min(1.0, self.f_d)
        t_h = self.f_hv * f1 * f2 * self.t_h0
        S = 3600.0 / max(t_h, eps)
        
        # B. Tính Inflow PCU/h và Capacity
        inflow_h = (stats.cycle_inflow_pcu * 3600.0) / cycle_len
        
        # --- SỬA TẠI ĐÂY: Đảm bảo tỷ lệ xanh không bằng 0 tuyệt đối để tránh lỗi chia ---
        g_over_c = stats.green_seconds / cycle_len
        g_over_c = max(min(g_over_c, 0.99), 0.01) # Giới hạn tối thiểu 0.01 để tránh capacity = 0
        
        capacity = S * g_over_c
        
        # --- SỬA TẠI ĐÂY: Kiểm soát v/c để không vượt quá ngưỡng thực tế ---
        v_over_c = inflow_h / max(capacity, 1.0) # Dùng 1.0 thay vì eps để kết quả ổn định hơn
        v_over_c = min(v_over_c, 2.0) # Saturation thực tế tối đa chỉ nên đạt 2.0 (kẹt cứng)

        # C. Tính hàng chờ N_GE (HBS 2001)
        m_max = (stats.green_seconds * S) / 3600.0
        m_tb = (inflow_h * cycle_len) / 3600.0
        g_val = v_over_c

        # Logic nội suy giữ nguyên nhưng bọc hàm an toàn
        if g_val <= 0.65:
            n_ge = 0.0
        elif g_val <= 0.9:
            denom = 0.26 + (m_tb / 150.0)
            n_ge = ((g_val - 0.65) / 0.25) * (1.0 / max(denom, 0.1))
        elif g_val <= 1.0:
            val_09 = 1.0 / max(0.26 + (m_tb / 150.0), 0.1)
            val_10 = 0.3476 * (max(m_max, 0.0) ** 0.5)
            n_ge = val_09 + (g_val - 0.9) / 0.1 * (val_10 - val_09)
        elif g_val <= 1.2:
            val_10 = 0.3476 * (max(m_max, 0.0) ** 0.5)
            val_12 = (m_max * 0.2 + 5.0) / 2.0
            n_ge = val_10 + (g_val - 1.0) / 0.2 * (val_12 - val_10)
        else:
            n_ge = m_max * (g_val - 1.0) / 2.0

        # D. Tính Control Delay (t_w = t_w1 + t_w2)
        q_over_S = min(inflow_h / max(S, 1.0), 0.99)
        
        # Tránh mẫu số bằng 0 ở t_w1
        denom_w1 = 2 * (1 - q_over_S)
        t_w1 = (cycle_len * (1 - g_over_c)**2) / max(denom_w1, 0.01)
        
        # t_w2: Delay do hàng chờ tích lũy
        t_w2 = (3600.0 * n_ge) / max(capacity, 1.0)
        
        delay = min(t_w1 + t_w2, 300.0) # Giới hạn delay tối đa 5 phút

        return LaneKPI(
            control_delay_seconds=round(float(delay), 2),
            degree_of_saturation=round(float(v_over_c), 2),
            queue_length_meters=round(float(n_ge * self.avg_veh_space), 2),
            inflow_pcu_per_hour=round(float(inflow_h), 2)
        )
    
    def reset_cycle(self, lane_id: str):
        if lane_id in self._lane_stats:
            stats = self._lane_stats[lane_id]
            stats.cycle_inflow_pcu = 0
            stats.green_seconds = 0
            stats.cycle_steps = 0