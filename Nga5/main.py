import traci
import socket
import threading
import time
import cal_crc
import json
import os
import csv
from lamp_mapping import LampMapper
from kpi_engine import KPIEngine

# ==========================================
# --- CẤU HÌNH NÚT GIAO (CHỈNH TẠI ĐÂY) ---
# ==========================================
# Đọc cấu hình Master để lấy hằng số KPI và Cycle Length kịch bản
with open('config_master.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)

TARGET_IP = cfg['cabinet_info']['ip']
TARGET_PORT_UDP = cfg['cabinet_info']['port_udp']
TLS_ID = cfg['cabinet_info']['tls_id_sumo']
CROSS_ID = cfg['cabinet_info']['cross_id'] # Ví dụ: 01
# Đổi lại lamp_range
LAMP_RANGE = range(16, 28) 
STEP_LEN = 0.5 
CYCLE_LEN_INPUT = float(cfg['cabinet_info']['cycle_length_default'])

mapper = LampMapper()
# Khởi tạo Engine với tham số constants từ file JSON Master
engine = KPIEngine(cfg['kpi_constants']) 

shared_lamp_status = {} 
status_lock = threading.Lock()
last_export_time = -1
# ==========================================

# --- 1. LOGIC GỬI DETECTOR (TCP) - GIỮ NGUYÊN CLASS CỦA BẠN ---
class FamaDetectorBridge:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.ip = data['cabinet_info']['tcp_ip_det']
        self.port = data['cabinet_info']['tcp_port_det']
        self.mapping = data['detectors']
        self.memory = {det_id: False for det_id in self.mapping.keys()}
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        try:
            self.client.connect((self.ip, self.port))
            print(f"[+] Detector connected TCP: {self.ip}")
            return True
        except: return False

    def send_signal(self, channel, status):
        # Giữ nguyên cấu trúc gói tin TCP có checksum của Quyên
        id_dev, ver, op, obj = 0x05, 0x10, 0x82, 0x08
        checksum = id_dev ^ ver ^ op ^ obj ^ channel ^ status
        packet = bytes([0x7E, id_dev, ver, op, obj, channel, status, checksum, 0x7E])
        try: 
            self.client.send(packet)
        except: pass

# --- 2. Logic truy vấn light group status 
def thread_sequential_query():
    global shared_lamp_status
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.3)
    
    # ĐÚNG VỊ TRÍ: CROSS_ID nằm sau byte 03
    base_payload = f"00 13 01 00 01 00 00 00 03 {CROSS_ID} 01 10 01 01 04 03 03 02"

    while True:
        for g_id in LAMP_RANGE:
            try:
                current_payload = f"{base_payload} {g_id:02x}"
                input_bin = ''.join(format(int(b, 16), '08b') for b in current_payload.split())
                crc = cal_crc.crc16_binary_simulation(input_bin)
                packet = bytes.fromhex(("7E " + current_payload + " " + f"{int(crc, 2):04X}" + " 7D").replace(" ", ""))
                sock.sendto(packet, (TARGET_IP, TARGET_PORT_UDP))
                data, _ = sock.recvfrom(1024)
                if data and len(data) >= 21 and data[12] == 0x20:
                    recv_id = data[19]
                    recv_val = data[20]
                    with status_lock:
                        shared_lamp_status[recv_id] = recv_val
            except: pass
            time.sleep(0.01)
        time.sleep(0.2)

# --- 3. CHẠY MÔ PHỎNG ---
def run():
    global last_export_time
    # Khởi tạo Detector Bridge từ file config_master.json
    det_bridge = FamaDetectorBridge('config_master.json')
    det_connected = det_bridge.connect()

    # Khởi động luồng truy vấn đèn
    threading.Thread(target=thread_sequential_query, daemon=True).start()
    
    traci.start(["sumo-gui", "-c", "nga5.sumocfg", "--start", "--step-length", str(STEP_LEN)])
    
    # Mở file CSV để ghi kết quả crowd_ad_ngabavoi.csv
    csv_f = open(f"few_in_nga5.csv", "w", newline="")
    writer = csv.writer(csv_f)
    # Header khớp hoàn toàn với LaneKPI trong kpi_engine.py của bạn
    writer.writerow(["Time", "Lane", "Delay_s", "Saturation", "Queue_m", "Inflow_PCU_h", "Outflow_PCU_h", "Avg_Queue"])

    # controlled_lanes = traci.trafficlight.getControlledLanes(TLS_ID)
    last_applied_state = ""

    #debug phần tính kpi
    all_lanes = traci.trafficlight.getControlledLanes(TLS_ID)
    unique_lanes = list(dict.fromkeys(all_lanes))
    lane_to_indices = {lane: [i for i, l in enumerate(all_lanes) if l == lane] for lane in unique_lanes}

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            curr_time = traci.simulation.getTime()

            # A. ĐỒNG BỘ ĐÈN (TỦ -> SUMO)
            with status_lock:
                local_status = shared_lamp_status.copy()
            if local_status:
                new_state = mapper.generate_state(TLS_ID, local_status)
                if new_state and new_state != last_applied_state:
                    traci.trafficlight.setRedYellowGreenState(TLS_ID, new_state)
                    last_applied_state = new_state

            # B. GỬI TÍN HIỆU DETECTOR (MAP THEO PHA TRONG TỦ)
            if det_connected:
                for det_id, channel in det_bridge.mapping.items():
                    try:
                        # Kiểm tra xe trên Lane Area Detector (E2)
                        is_occupied = traci.lanearea.getLastStepVehicleNumber(det_id) > 0
                        if is_occupied != det_bridge.memory[det_id]:
                            # Gửi trạng thái 0x01 (có xe) hoặc 0x00 (không xe)
                            det_bridge.send_signal(channel, 0x01 if is_occupied else 0x00)
                            det_bridge.memory[det_id] = is_occupied
                    except: pass

            # # C. CẬP NHẬT DỮ LIỆU KPI MỖI BƯỚC NHẢY (0.5s)
            # for lane in controlled_lanes:
            #     # Lấy tập hợp ID xe hiện tại trên làn
            #     v_ids = set(traci.lane.getLastStepVehicleIDs(lane))
            #     idx = controlled_lanes.index(lane)
            #     # Kiểm tra màu đèn từ chuỗi state hiện tại
            #     is_green = last_applied_state[idx].lower() in ['g', 'u'] if last_applied_state else False
                
            #     # Cập nhật vào engine (Bản chuẩn truyền set v_ids)
            #     engine.update_lane(lane, v_ids, is_green, STEP_LEN)

            #Debug tính KPI
            for lane in unique_lanes:
                # Lấy tập hợp ID xe trên làn thực tế (không bị đếm trùng)
                v_ids = set(traci.lane.getLastStepVehicleIDs(lane))
                
                # Xác định trạng thái đèn Xanh: Chỉ cần 1 trong các index của lane này Xanh là tính là Xanh
                is_green = False
                if last_applied_state:
                    for idx in lane_to_indices[lane]:
                        if last_applied_state[idx].lower() in ['g', 'u']:
                            is_green = True
                            break
                
                # Cập nhật vào engine
                engine.update_lane(lane, v_ids, is_green, STEP_LEN)

            # # D. XUẤT KPI THEO CHU KỲ (DÙNG CYCLE_LEN_INPUT)
            # curr_time_int = int(curr_time)
            # if curr_time_int % int(CYCLE_LEN_INPUT) == 0 and curr_time_int > 0 and curr_time_int != last_export_time:
            #     for lane in controlled_lanes:
            #         # Gọi compute_kpi với chu kỳ thực tế bạn nhập
            #         res = engine.compute_kpi(lane, CYCLE_LEN_INPUT)
                    
            #         # Ghi 8 cột dữ liệu chuẩn LaneKPI của bạn
            #         writer.writerow([
            #             curr_time_int, 
            #             lane, 
            #             round(res.control_delay_seconds, 2), 
            #             round(res.degree_of_saturation, 2),
            #             round(res.queue_length_meters, 2),
            #             round(res.inflow_pcu_per_hour, 2),
            #             round(res.outflow_pcu_per_hour, 2),
            #             round(res.avg_queue_vehicles, 2)
            #         ])
            #         # Reset dữ liệu để tính chu kỳ tiếp theo
            #         engine.reset_cycle(lane)
                
            #     last_export_time = curr_time_int
            #     print(f"[*] KPI Exported at {curr_time_int}s | Cycle: {CYCLE_LEN_INPUT}s")

            #Debug tính KPI
            curr_time_int = int(curr_time)
            if curr_time_int % int(CYCLE_LEN_INPUT) == 0 and curr_time_int > 0 and curr_time_int != last_export_time:
                for lane in unique_lanes:
                    # Tính toán KPI cho làn duy nhất
                    res = engine.compute_kpi(lane, CYCLE_LEN_INPUT)
                    
                    # Ghi dữ liệu chuẩn 8 cột
                    writer.writerow([
                        curr_time_int, 
                        lane, 
                        round(res.control_delay_seconds, 2), 
                        round(res.degree_of_saturation, 2),
                        round(res.queue_length_meters, 2),
                        round(res.inflow_pcu_per_hour, 2),
                        round(res.outflow_pcu_per_hour, 2),
                        round(res.avg_queue_vehicles, 2)
                    ])
                    # Reset dữ liệu cho chu kỳ mới
                    engine.reset_cycle(lane)
                
                last_export_time = curr_time_int
                print(f"[*] KPI Exported for {len(unique_lanes)} unique lanes at {curr_time_int}s")
            time.sleep(0.01)

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        csv_f.close()
        traci.close()

if __name__ == "__main__":
    run()