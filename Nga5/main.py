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
engine = KPIEngine(cfg['kpi_constants']) 

shared_lamp_status = {} 
status_lock = threading.Lock()

class CycleDetector:
    def __init__(self):
        self.first_state = None
        self.last_state = None
        self.start_time = 0
        self.has_changed = False

    def check_cycle(self, current_state, current_time):
        if not current_state:
            return None
        
        if self.first_state is None:
            self.first_state = current_state
            self.last_state = current_state
            self.start_time = current_time
            print(f"[*] Cycle Detection Started: First State captured at {current_time}s")
            return None
        
        if current_state != self.last_state:
            self.last_state = current_state
            self.has_changed = True
            
            # Kiểm tra quay lại trạng thái đầu tiên
            if current_state == self.first_state and self.has_changed:
                duration = current_time - self.start_time
                # Ràng buộc chu kỳ tối thiểu 10s để không nhầm với vàng/đỏ
                if duration >= 20.0:
                    self.start_time = current_time
                    self.has_changed = False
                    return duration
        return None

# --- 1. LOGIC GỬI DETECTOR (TCP) ---
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
        id_dev, ver, op, obj = 0x05, 0x10, 0x82, 0x08
        checksum = id_dev ^ ver ^ op ^ obj ^ channel ^ status
        packet = bytes([0x7E, id_dev, ver, op, obj, channel, status, checksum, 0x7E])
        try: 
            self.client.send(packet)
        except: pass

# --- 2. Logic truy vấn light group status ---
def thread_sequential_query():
    global shared_lamp_status
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.3)
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
    det_bridge = FamaDetectorBridge('config_master.json')
    det_connected = det_bridge.connect()

    threading.Thread(target=thread_sequential_query, daemon=True).start()
    
    traci.start(["sumo-gui", "-c", "nga5.sumocfg", "--start"])
    
    csv_f = open(f"nor_ft_nga5.csv", "w", newline="")
    writer = csv.writer(csv_f)
    # Header mới: Bỏ Queue_m, thêm TotalDemand_h
    writer.writerow(["Time", "Lane", "Delay_s", "Saturation", "Avg_Queue_Veh", "Inflow_PCU_h", "TotalDemand_h", "Capacity_h", "Residual_NGE"])

    last_applied_state = ""
    cycle_detector = CycleDetector()

    all_lanes = traci.trafficlight.getControlledLanes(TLS_ID)
    unique_lanes = list(dict.fromkeys(all_lanes))
    
    lane_to_indices = {}
    links = traci.trafficlight.getControlledLinks(TLS_ID)
    for i, link_list in enumerate(links):
        if not link_list: continue
        from_lane = link_list[0][0]
        if from_lane not in lane_to_indices:
            lane_to_indices[from_lane] = []
        lane_to_indices[from_lane].append(i)

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            curr_time = traci.simulation.getTime()
            delta_t = traci.simulation.getDeltaT()

            with status_lock:
                local_status = shared_lamp_status.copy()
            if local_status:
                new_state = mapper.generate_state(TLS_ID, local_status)
                if new_state and new_state != last_applied_state:
                    traci.trafficlight.setRedYellowGreenState(TLS_ID, new_state)
                    last_applied_state = new_state

            if det_connected:
                for det_id, channel in det_bridge.mapping.items():
                    try:
                        is_occupied = traci.lanearea.getLastStepVehicleNumber(det_id) > 0
                        if is_occupied != det_bridge.memory[det_id]:
                            det_bridge.send_signal(channel, 0x01 if is_occupied else 0x00)
                            det_bridge.memory[det_id] = is_occupied
                    except: pass

            for lane in unique_lanes:
                v_ids = traci.lane.getLastStepVehicleIDs(lane)
                
                v_movements = {} 
                for v_id in v_ids:
                    try:
                        next_links = traci.vehicle.getNextLinks(v_id)
                        for link in next_links:
                            if link[5] == TLS_ID:
                                v_movements[v_id] = link[6]
                                break
                    except: pass

                link_green_status = {} 
                if last_applied_state:
                    for idx in lane_to_indices.get(lane, []):
                        if idx < len(last_applied_state):
                            link_green_status[idx] = last_applied_state[idx].lower() in ['g', 'u']

                engine.update_lane(lane, v_ids, v_movements, link_green_status, delta_t)

            if last_applied_state:
                cycle_duration = cycle_detector.check_cycle(last_applied_state, curr_time)
                if cycle_duration:
                    print(f"[*] Cycle detected: {cycle_duration}s at {curr_time}s")
                    for lane in unique_lanes:
                        res = engine.compute_kpi(lane, cycle_duration)
                        writer.writerow([
                            int(curr_time), 
                            lane, 
                            round(res.control_delay_seconds, 2), 
                            round(res.degree_of_saturation, 2),
                            round(res.avg_queue_vehicles, 2),
                            round(res.inflow_pcu_per_hour, 2),
                            round(res.total_demand_pcu_h, 2),
                            round(res.capacity_pcu_h, 2),
                            round(res.n_ge, 2)
                        ])
                        engine.reset_cycle(lane, res.n_ge)
                    csv_f.flush()

            time.sleep(0.005)

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        csv_f.close()
        traci.close()

if __name__ == "__main__":
    run()
