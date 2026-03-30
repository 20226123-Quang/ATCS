import traci
import socket
import threading
import time
import cal_crc
import json
import os
import csv
from ngabavoi_mapping import NgaBaVoiMapping
from kpi_engine import KPIEngine

# --- CẤU HÌNH ---
TARGET_IP = "192.168.8.138"
TARGET_PORT_UDP = 4050
config = NgaBaVoiMapping()
current_fama_code = -1
engine = KPIEngine() # Khởi tạo bộ tính KPI

# --- 1. LOGIC GỬI DETECTOR (TCP) - KHÔI PHỤC ---
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
        id_dev, ver, op, obj = 0x05, 0x10, 0x82, 0x08
        checksum = id_dev ^ ver ^ op ^ obj ^ channel ^ status
        packet = bytes([0x7E, id_dev, ver, op, obj, channel, status, checksum, 0x7E])
        try: self.client.send(packet)
        except: pass

# --- 2. LOGIC NHẬN PHA (UDP) ---
def thread_listening():
    global current_fama_code
    input_hex = "00 13 01 00 01 00 00 00 03 01 01 10 01 01 04 0D 02 04 01"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.5)
    while True:
        try:
            input_bin = ''.join(format(int(b, 16), '08b') for b in input_hex.split())
            crc = cal_crc.crc16_binary_simulation(input_bin)
            packet = bytes.fromhex(("7E " + input_hex + " " + f"{int(crc, 2):04X}" + " 7D").replace(" ", ""))
            sock.sendto(packet, (TARGET_IP, TARGET_PORT_UDP))
            data, _ = sock.recvfrom(1024)
            if data: current_fama_code = int(data.hex(' ').split(' ')[-4], 16)
        except: pass
        time.sleep(1.0)

# --- 3. CHẠY MÔ PHỎNG VÀ TÍNH KPI ---
det_bridge = FamaDetectorBridge('detector_config.json')
threading.Thread(target=thread_listening, daemon=True).start()

traci.start(["sumo-gui", "-c", "ngabavoi.sumocfg", "--start", "--end", "500"])
det_bridge.connect()

# Chuẩn bị file CSV
csv_f = open("ad_500s_ngabavoi.csv", "w", newline="")
writer = csv.writer(csv_f)
writer.writerow(["Time", "Lane", "Delay_s", "Queue_m", "Saturation"])

controlled_lanes = traci.trafficlight.getControlledLanes(config.tls_id)
last_applied_state = ""
cycle_start_time = 0

try:
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        curr_time = traci.simulation.getTime()

        # A. ĐỒNG BỘ ĐÈN (TỦ -> SUMO)
        new_state = config.get_sumo_state(current_fama_code)
        if new_state and new_state != last_applied_state:
            traci.trafficlight.setRedYellowGreenState(config.tls_id, new_state)
            last_applied_state = new_state

        # B. GỬI DETECTOR (SUMO -> TỦ) - KHÔI PHỤC
        for det_id, channel in det_bridge.mapping.items():
            try:
                is_occupied = traci.lanearea.getLastStepVehicleNumber(det_id) > 0
                if is_occupied != det_bridge.memory[det_id]:
                    det_bridge.send_signal(channel, 0x01 if is_occupied else 0x00)
                    det_bridge.memory[det_id] = is_occupied
            except: pass

        # C. TÍNH TOÁN KPI (MỖI STEP)
        for lane in controlled_lanes:
            v_ids = set(traci.lane.getLastStepVehicleIDs(lane))
            idx = controlled_lanes.index(lane)
            is_green = last_applied_state[idx].lower() in ['g', 'u'] if last_applied_state else False
            engine.update_lane(lane, v_ids, is_green, 0.05) # Giả định bước nhảy 0.05s

        # D. XUẤT KPI MỖI 60S
        if int(curr_time) % 60 == 0 and int(curr_time) > 0:
            for lane in controlled_lanes:
                res = engine.compute_kpi(lane, 60.0)
                writer.writerow([curr_time, lane, round(res.control_delay_seconds, 2), 
                                 round(res.queue_length_meters, 2), round(res.degree_of_saturation, 2)])
                engine.reset_cycle(lane)
            print(f"[*] KPI Exported at {curr_time}s")

        time.sleep(0.01)
finally:
    csv_f.close()
    traci.close()