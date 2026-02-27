# ATCS – Adaptive Traffic Control System

Hệ thống điều khiển đèn giao thông thông minh sử dụng Reinforcement Learning (RL) kết hợp SUMO (Simulation of Urban Mobility).  
KPI chuẩn hóa theo **TCCS 24:2018/TCDBVN** và **Highway Capacity Manual (HCM)**.

---

## Cấu trúc thư mục

```
ATCS/
├── config/
│   └── kpi_config.json            # Cấu hình KPI, hằng số, công thức, LOS, reward
├── atcs/
│   ├── __init__.py
│   ├── config_loader.py           # Load cấu hình JSON → dataclass type-safe
│   ├── sumo_parser.py             # Parse .sumocfg / .net.xml → chương trình đèn
│   ├── kpi_engine.py              # Tích lũy hàng đợi + tính KPI theo HCM
│   └── environment.py             # TrafficEnvironment (reset/step) cho RL
├── examples/
│   └── smoke_env.py               # Script chạy nhanh để test môi trường
└── SimulationData/
    └── SampleData/
        └── Crowded/               # Kịch bản mật độ cao (mặc định)
            ├── config.sumocfg
            ├── network.net.xml
            └── random.rou.xml
```

---

## Mô tả chi tiết từng file

### 1. `config_loader.py` – Load cấu hình KPI

**Nhiệm vụ**: Đọc file `kpi_config.json` và chuyển sang các dataclass Python có kiểu dữ liệu rõ ràng.

| Class / Hàm | Mô tả |
|---|---|
| `SimulationSettings` | Chứa cài đặt mô phỏng: step_length, max_episode, green range, gui |
| `KPIConstants` | Chứa hằng số tính toán: saturation flow, PCU mapping, delay params |
| `KPIConfig` | Gộp tất cả: simulation + constants + formulas + LOS table + reward design |
| `load_kpi_config(path)` | Đọc JSON → trả về `KPIConfig` object |

**Được gọi bởi**: `environment.py` → `TrafficEnvironment.__init__()`.

---

### 2. `sumo_parser.py` – Parse mạng lưới SUMO

**Nhiệm vụ**: Đọc file `.sumocfg` → tìm file `.net.xml` → parse tất cả chương trình đèn tín hiệu.

| Class / Hàm | Mô tả |
|---|---|
| `PhaseDefinition` | Một pha đèn: index, duration, state string, loại (green/yellow/red) |
| `TLSProgram` | Chương trình đèn hoàn chỉnh: danh sách phases, chu kỳ, phase green đầu tiên |
| `ParsedSUMONetwork` | Kết quả parse: đường dẫn file + dict các TLSProgram |
| `_classify_phase_type(state)` | Phân loại state string → "green" / "yellow" / "red" |
| `_resolve_net_file(sumocfg_path)` | Đọc `.sumocfg` XML → tìm đường dẫn `.net.xml` |
| `parse_sumo_network(sumocfg_path)` | Entry point: parse toàn bộ mạng → trả `ParsedSUMONetwork` |

**Được gọi bởi**: `environment.py` → `TrafficEnvironment.__init__()`.

---

### 3. `kpi_engine.py` – Tính toán KPI

**Nhiệm vụ**: Tích lũy dữ liệu hàng đợi mỗi giây (Input-Output model) và tính KPI theo chuẩn HCM.

| Class / Hàm | Mô tả |
|---|---|
| `LaneRuntimeStats` | Trạng thái runtime từng lane: queue, inflow, outflow, green_seconds |
| `LaneKPI` | Kết quả KPI: control_delay, v/c, queue_length, capacity |
| `KPIEngine.__init__(constants)` | Khởi tạo engine với hằng số từ `KPIConstants` |
| `reset_lane_state(lane_id, ...)` | Reset trạng thái lane khi bắt đầu episode |
| `start_new_cycle(lane_id)` | Reset metrics tích lũy khi bắt đầu chu kỳ mới |
| `mark_lane_green_seconds(lane_id)` | Cộng thêm thời gian green cho lane |
| `update_lane(lane_id, inflow, outflow, ...)` | Cập nhật queue mỗi step: `Q(t) = max(0, Q(t-1) + in - out)` |
| `compute_lane_kpis(lane_id, cycle_length)` | Tính KPI: d1, d2, d3, v/c, queue_length → trả `LaneKPI` |

**Được gọi bởi**: `environment.py` – nhiều hàm gọi tới (xem sơ đồ bên dưới).

---

### 4. `environment.py` – Môi trường RL chính

**Nhiệm vụ**: Cung cấp interface `reset()` / `step(action)` cho RL agent.

| Hàm | Mô tả |
|---|---|
| `__init__(sumocfg_path, ...)` | Khởi tạo: load config, parse mạng, tạo KPI engine |
| `reset()` | Khởi động SUMO, build topology, trả observation/reward ban đầu |
| `step(action)` | Nhận action → apply → mô phỏng đến khi cần action tiếp → trả obs/reward |
| `close()` | Đóng kết nối SUMO |
| `_start_sumo()` | Khởi động tiến trình SUMO qua TraCI |
| `_build_lane_topology()` | Lấy danh sách lanes controlled bởi mỗi đèn từ TraCI |
| `_initialize_lane_runtime()` | Đọc trạng thái xe hiện tại trên mỗi lane → init KPIEngine |
| `_prepare_tls_runtime()` | Khởi tạo state runtime cho mỗi đèn (phase, remaining, cycle) |
| `_apply_pending_actions(action)` | Áp dụng action (extend green) vào runtime đèn |
| `_simulate_until_need_action()` | Chạy SUMO step-by-step cho đến khi cần agent quyết định |
| `_update_lane_accumulation()` | Mỗi step: tính inflow/outflow → gọi `kpi_engine.update_lane()` |
| `_mark_green_seconds_for_tls(tls_id, state)` | Đánh dấu lane nào đang green → gọi `kpi_engine.mark_lane_green_seconds()` |
| `_advance_to_next_phase(tls_id)` | Chuyển sang phase tiếp theo trong chương trình đèn |
| `_build_observation_reward()` | Gọi `kpi_engine.compute_lane_kpis()` → build tensor obs/reward |
| `_build_info(delta_t)` | Build dict thông tin bổ sung |
| `_vehicle_pcu(vehicle_id)` | Tra cứu PCU của xe theo vehicle type |
| `_check_done()` | Kiểm tra episode kết thúc chưa |

---

## Sơ đồ gọi hàm

### `reset()` flow
```
reset()
 ├── close()
 ├── _start_sumo()                      # Khởi động SUMO
 ├── _ensure_connection()
 ├── _build_lane_topology()             # Lấy lanes từ TraCI
 ├── traci.simulationStep()             # Chạy 1 step đầu
 ├── _initialize_lane_runtime()
 │    └── kpi_engine.reset_lane_state() # Reset KPI cho mỗi lane
 ├── _prepare_tls_runtime()
 │    ├── _reset_cycle_lane_metrics()
 │    │    └── kpi_engine.start_new_cycle()
 │    └── traci.trafficlight.setRedYellowGreenState()
 ├── _check_done()
 ├── _build_observation_reward()
 │    └── kpi_engine.compute_lane_kpis()  # Tính KPI → obs & reward
 └── _build_info()
```

### `step(action)` flow
```
step(action)
 ├── _ensure_connection()
 ├── _apply_pending_actions(action)
 │    ├── Clip extension vào [0, max_extension]
 │    ├── Cộng extension vào remaining_phase_seconds
 │    └── _advance_to_next_phase() (nếu remaining <= 0)
 │         ├── _reset_cycle_lane_metrics()
 │         │    └── kpi_engine.start_new_cycle()
 │         └── traci.trafficlight.setRedYellowGreenState()
 ├── _simulate_until_need_action()
 │    ├── traci.simulationStep()           # Loop cho đến khi cần action
 │    ├── _update_lane_accumulation()      # Mỗi step
 │    │    ├── _vehicle_pcu()              # Tra PCU mapping
 │    │    └── kpi_engine.update_lane()    # Q(t) = max(0, Q(t-1) + in - out)
 │    ├── _mark_green_seconds_for_tls()    # Mỗi step (nếu phase green)
 │    │    └── kpi_engine.mark_lane_green_seconds()
 │    ├── _advance_to_next_phase()         # Khi hết phase
 │    └── _check_done()
 ├── _build_observation_reward()
 │    └── kpi_engine.compute_lane_kpis()
 └── _build_info()
```

---

## Chi tiết `kpi_config.json`

### `simulation` – Cài đặt mô phỏng

| Tham số | Giá trị | Dùng trong hàm | Ý nghĩa |
|---|---|---|---|
| `default_step_length_seconds` | 1 | `__init__()` → `self.step_length_seconds` | Mỗi bước SUMO = 1 giây |
| `max_episode_seconds` | 3600 | `__init__()` → `self.max_episode_seconds` → `_check_done()` | Episode tối đa 3600s |
| `min_green_seconds` | 10 | `__init__()` → `self.min_green_seconds` → `_build_info()` | Green tối thiểu 10s |
| `max_green_seconds` | 60 | `__init__()` → `self.max_green_seconds` → `_apply_pending_actions()` | Green tối đa 60s |
| `yellow_fallback_seconds` | 3 | `__init__()` → `parse_sumo_network()` | Mặc định đèn vàng 3s nếu SUMO không định nghĩa |
| `use_gui` | false | `__init__()` → `_resolve_sumo_binary()` | Tắt/bật giao diện SUMO GUI |

### `constants` – Hằng số tính toán

| Tham số | Giá trị | Dùng trong hàm | Ý nghĩa |
|---|---|---|---|
| `saturation_flow_pcu_per_hour_per_lane` | 1900 | `compute_lane_kpis()` → tính `capacity` | Năng lực bão hòa: 1900 PCU/giờ/làn |
| `average_vehicle_space_meter` | 6.5 | `compute_lane_kpis()` → tính `queue_length_m` | Khoảng cách TB mỗi xe trong hàng đợi |
| `green_wave_pf_default` | 1.0 | `compute_lane_kpis()` → nhân với `d1` | Hệ số Progression Factor (1.0 = không phối hợp sóng xanh) |
| `incremental_delay_k` | 0.5 | `compute_lane_kpis()` → tính `d2` | Hệ số k trong công thức incremental delay |
| `incremental_delay_power` | 2.0 | `compute_lane_kpis()` → tính `d2` | Số mũ p: penalty tăng theo bình phương khi quá tải |
| `epsilon` | 1e-06 | `compute_lane_kpis()` → tránh chia 0 | Giá trị epsilon nhỏ |
| `max_control_delay_seconds` | 150 | `compute_lane_kpis()` → clip `control_delay` | Giới hạn trên delay (tránh giá trị vô lý) |
| `v_c_penalty_threshold` | 0.9 | `compute_lane_kpis()` → penalty trong `d2` | Bắt đầu phạt thêm khi v/c > 0.9 |
| `v_c_penalty_weight` | 12.0 | `compute_lane_kpis()` → weight penalty `d2` | Trọng số phạt khi v/c vượt ngưỡng |
| `default_pcu` | 1.0 | `_vehicle_pcu()` → fallback | PCU mặc định nếu không xác định loại xe |
| `pcu_mapping` | (bảng dưới) | `_vehicle_pcu()` → tra cứu | Bảng quy đổi xe tiêu chuẩn |

#### Bảng PCU Mapping

| Loại xe | PCU | Ý nghĩa |
|---|---|---|
| passenger / car | 1.0 | Xe con = 1 đơn vị chuẩn |
| truck | 2.0 | Xe tải = 2 xe con |
| bus | 2.5 | Xe buýt = 2.5 xe con |
| motorcycle | 0.5 | Xe máy = nửa xe con |
| bike / bicycle | 0.2 | Xe đạp = 0.2 xe con |

### `kpi_formulas` – Công thức KPI (tham khảo)

Các công thức được implement trong `kpi_engine.py` → `compute_lane_kpis()`:

| Key | Công thức | Implement |
|---|---|---|
| `average_control_delay` | `d = d1 * PF + d2 + d3` | Dòng cuối `compute_lane_kpis()` |
| `uniform_delay_d1` | `d1 = 0.5*C*(1-g/C)²/(1-min(1,v/c)*g/C)` | Biến `d1` trong `compute_lane_kpis()` |
| `incremental_delay_d2` | `d2 = k*max(0,v/c-1)^p*C + penalty` | Biến `d2` trong `compute_lane_kpis()` |
| `initial_queue_delay_d3` | `d3 = Q_initial / arrival_rate` | Biến `d3` trong `compute_lane_kpis()` |
| `degree_of_saturation` | `v/c = V / C_cap` | Biến `v_over_c` |
| `capacity` | `C_cap = s * (g/C)` | Biến `capacity` |
| `average_queue_length` | `L = Q_avg * S_L` | Biến `queue_length_m` |
| `queue_accumulation` | `Q(t) = max(0, Q(t-1) + V_in - V_out)` | `update_lane()` |

### `los_table` – Bảng Level of Service

Dùng để đánh giá chất lượng giao thông (tham khảo, chưa implement tự động trong code):

| LOS | Delay (s/xe) | v/c | Đánh giá |
|---|---|---|---|
| A | ≤ 10 | ≤ 0.6 | 🟢 Rất tốt |
| B | 10–20 | ≤ 0.7 | 🟢 Tốt |
| C | 20–35 | ≤ 0.8 | 🟡 Chấp nhận được |
| D | 35–55 | ≤ 0.9 | 🟠 Kém |
| E | 55–80 | ≈ 1.0 | 🔴 Rất kém |
| F | > 80 | > 1.0 | ⛔ Tắc nghẽn |

### `reward_design` – Thiết kế reward cho RL

| Key | Ý nghĩa | Dùng trong hàm |
|---|---|---|
| `shape: [n_intersections, n_lanes, 2]` | Tensor reward 3 chiều | `_build_observation_reward()` |
| `components[0]: control_delay` | Trễ điều khiển TB (s/xe) | `reward[i, j, 0]` |
| `components[1]: queue_length` | Chiều dài hàng đợi TB (m) | `reward[i, j, 1]` |

> **Lưu ý**: Cả 2 chỉ số đều **lower is better**. RL trainer cần đảo dấu (×-1) để tạo scalar reward.

---

## Output format

### Observation – shape `[n_intersections, max_lanes, 5]`

| Index | Feature | Nguồn |
|---|---|---|
| 0 | `control_delay_seconds` | `compute_lane_kpis()` |
| 1 | `degree_of_saturation` (v/c) | `compute_lane_kpis()` |
| 2 | `queue_length_meters` | `compute_lane_kpis()` |
| 3 | `remaining_cycle_time` | TLS runtime state |
| 4 | `current_phase_index` | TLS runtime state |

### Reward – shape `[n_intersections, max_lanes, 2]`

| Index | Feature | Nguồn |
|---|---|---|
| 0 | `control_delay_seconds` | `compute_lane_kpis()` |
| 1 | `queue_length_meters` | `compute_lane_kpis()` |

### Info dict

| Key | Ý nghĩa |
|---|---|
| `min_green` | Green tối thiểu (giây) |
| `max_green` | Green tối đa (giây) |
| `cycle_length` | Chu kỳ hiện tại từng nút |
| `delta_t` | Thời gian mô phỏng đã trôi trong step này |
| `intersection_require_action` | Danh sách nút cần agent ra quyết định |

---

## Cách chạy

### Smoke test (mặc định dùng Crowded scenario)
```bash
cd ATCS
python -m examples.smoke_env --steps 10
```

### Với GUI
```bash
python -m examples.smoke_env --steps 100 --gui
```

### Dùng scenario khác
```bash
python -m examples.smoke_env --sumocfg path/to/config.sumocfg
```

### Trong code RL
```python
from atcs.environment import TrafficEnvironment

env = TrafficEnvironment(sumocfg_path="SimulationData/SampleData/Crowded/config.sumocfg")
obs, reward, done, info = env.reset()

while not done:
    action = {tls_id: agent.predict(obs) for tls_id in info["intersection_require_action"]}
    obs, reward, done, info = env.step(action)

env.close()
```

---

## Yêu cầu môi trường

- **Python** 3.8+
- **SUMO** (cài đặt + biến môi trường `SUMO_HOME`)
- **Dependencies**: `traci`, `numpy`
