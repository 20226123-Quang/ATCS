import traci
import socket
import threading
import time
import cal_crc
import json
import os
import csv
from lamp_mapping import LampMapper
from ex_cal_kpi.kpi_engine import KPIEngine

# --- CẤU HÌNH HỆ THỐNG ---
TARGET_IP = "192.168.8.138"
TARGET_PORT_UDP = 4050
TLS_ID = "J7"  # ID nút Ngã Ba Voi trong SUMO

mapper = LampMapper()
engine = KPIEngine()
shared_lamp_status = {} 
status_lock = threading.Lock()

# --- 1. LOGIC GỬI DETECTOR (TCP) ---
class FamaDetectorBridge:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.ip = data['tcp_ip']
        self.port = data['tcp_port']
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
        # Frame format tủ FAMA
        id_dev, ver, op, obj = 0x05, 0x10, 0x82, 0x08
        checksum = id_dev ^ ver ^ op ^ obj ^ channel ^ status
        packet = bytes([0x7E, id_dev, ver, op, obj, channel, status, checksum, 0x7E])
        try: self.client.send(packet)
        except: pass

# --- 2. LOGIC TRUY VẤN ĐÈN TUẦN TỰ (UDP QUERY 1-10) ---
def thread_sequential_query():
    """Gửi lệnh hỏi từng nhóm đèn 1-10 và bóc tách theo thuật toán 7-byte"""
    global shared_lamp_status
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.3)
    
    # Payload cơ sở (chưa có Element ID cuối)
    base_payload = "00 13 01 00 01 00 00 00 03 01 01 10 01 01 04 03 03 02"

    while True:
        for g_id in range(1, 11): # Quét đích danh 10 nhóm đèn
            try:
                current_payload = f"{base_payload} {g_id:02x}"
                input_bin = ''.join(format(int(b, 16), '08b') for b in current_payload.split())
                crc = cal_crc.crc16_binary_simulation(input_bin)
                packet = bytes.fromhex(("7E " + current_payload + " " + f"{int(crc, 2):04X}" + " 7D").replace(" ", ""))
                
                sock.sendto(packet, (TARGET_IP, TARGET_PORT_UDP))
                data, _ = sock.recvfrom(1024)
                
                # Bóc tách khối 7-byte: [01 05 03 03 02 ID VAL] -> ID tại index 19, VAL tại 20
                if data and len(data) >= 21 and data[12] == 0x20:
                    recv_id = data[19]
                    recv_val = data[20]
                    if recv_id == g_id:
                        with status_lock:
                            shared_lamp_status[recv_id] = recv_val
            except: pass
            time.sleep(0.01)
        time.sleep(0.2)

# --- 3. CHẠY MÔ PHỎNG VÀ TÍNH KPI ---
def run():
    det_bridge = FamaDetectorBridge('detector_config.json')
    threading.Thread(target=thread_sequential_query, daemon=True).start()

    # Khởi động SUMO (Đồng bộ nhịp 0.5s để tính KPI chính xác hơn)
    traci.start(["sumo-gui", "-c", "ngabavoi.sumocfg", "--start", "--end", "500", "--step-length", "0.5"])
    det_bridge.connect()

    # Chuẩn bị file CSV báo cáo
    csv_f = open("demongabavoi.csv", "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["Time", "Lane", "Delay_s", "Queue_m", "Saturation", "LOS"])

    controlled_lanes = traci.trafficlight.getControlledLanes(TLS_ID)
    last_applied_state = ""
    step_len = 0.5 # Khớp với --step-length của SUMO

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            curr_time = traci.simulation.getTime()

            # A. ĐỒNG BỘ ĐÈN (MAPPER -> SUMO)
            with status_lock:
                local_status = shared_lamp_status.copy()

            if local_status:
                new_state = mapper.generate_state(TLS_ID, local_status)
                if len(new_state) == 19 and new_state != last_applied_state:
                    traci.trafficlight.setRedYellowGreenState(TLS_ID, new_state)
                    last_applied_state = new_state

            # B. GỬI DETECTOR LÊN TỦ (SUMO -> FAMA)
            for det_id, channel in det_bridge.mapping.items():
                try:
                    is_occupied = traci.lanearea.getLastStepVehicleNumber(det_id) > 0
                    if is_occupied != det_bridge.memory[det_id]:
                        det_bridge.send_signal(channel, 0x01 if is_occupied else 0x00)
                        det_bridge.memory[det_id] = is_occupied
                except: pass

            # C. CẬP NHẬT DỮ LIỆU KPI (MỖI BƯỚC NHẢY)
            for lane in controlled_lanes:
                v_ids = set(traci.lane.getLastStepVehicleIDs(lane))
                idx = controlled_lanes.index(lane)
                # Kiểm tra màu đèn tại lane để tính toán Delay dừng
                is_green = last_applied_state[idx].lower() in ['g', 'u'] if last_applied_state else False
                engine.update_lane(lane, v_ids, is_green, step_len)

            # D. XUẤT KPI MỖI 60S (DẠNG AGGREGATED)
            if int(curr_time) % 60 == 0 and int(curr_time) > 0:
                for lane in controlled_lanes:
                    res = engine.compute_kpi(lane, 60.0)
                    # Thêm LOS để báo cáo chuyên nghiệp hơn
                    writer.writerow([curr_time, lane, round(res.control_delay_seconds, 2), 
                                     round(res.queue_length_meters, 2), round(res.degree_of_saturation, 2)])
                    engine.reset_cycle(lane)
                print(f"[*] Integrated KPI Exported at {curr_time}s")

            if curr_time >= 500: break
            time.sleep(0.01)

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        csv_f.close()
        traci.close()
        print("[SYSTEM] Kết thúc mô phỏng và lưu KPI thành công.")

if __name__ == "__main__":
    run()