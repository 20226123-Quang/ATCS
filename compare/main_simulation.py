import traci
import socket
import threading
import time
import cal_crc
import json
import os
from nga5_mapping import Nga5Mapping

# --- CẤU HÌNH ---
TARGET_IP = "192.168.8.134"
TARGET_PORT_UDP = 4050
config = Nga5Mapping() # Sử dụng class Nga5Mapping từ file của bạn
current_fama_code = -1

# --- LOGIC GỬI DETECTOR (TCP) ---
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

    def update_detectors(self):
        for det_id, channel in self.mapping.items():
            try:
                # Đọc dữ liệu từ nga5.add.xml qua TraCI
                is_occupied = traci.lanearea.getLastStepVehicleNumber(det_id) > 0
                if is_occupied != self.memory[det_id]:
                    self.send_signal(channel, 0x01 if is_occupied else 0x00)
                    self.memory[det_id] = is_occupied
                    print(f"[DET] {det_id} -> {'XANH' if is_occupied else 'TRONG'}")
            except: continue

# --- LOGIC NHẬN PHA ĐÈN (UDP) ---
def thread_listening():
    global current_fama_code
    # # với cross1
    # input_hex = "00 13 01 00 01 00 00 00 03 01 01 10 01 01 04 0D 02 04 01"
    # với cross 2
    input_hex = "00 13 01 00 01 00 00 00 03 02 01 10 01 01 04 0D 02 04 02"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.5)
    
    while True:
        try:
            input_bin = ''.join(format(int(b, 16), '08b') for b in input_hex.split())
            crc = cal_crc.crc16_binary_simulation(input_bin)
            packet = bytes.fromhex(("7E " + input_hex + " " + f"{int(crc, 2):04X}" + " 7D").replace(" ", ""))
            
            sock.sendto(packet, (TARGET_IP, TARGET_PORT_UDP))
            data, _ = sock.recvfrom(1024)
            if data:
                res_list = data.hex(' ').split(' ')
                current_fama_code = int(res_list[-4], 16)
        except: pass
        time.sleep(1.0)

# --- CHẠY MÔ PHỎNG ---
det_bridge = FamaDetectorBridge('detector_config.json')
threading.Thread(target=thread_listening, daemon=True).start()

# Khởi động SUMO
traci.start(["sumo-gui", "-c", "../Nga5/nga5.sumocfg", "--start"])
det_bridge.connect()

last_applied_state = ""

try:
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        # 1. ĐỒNG BỘ ĐÈN (Sử dụng get_sumo_state từ file của bạn)
        if current_fama_code != -1:
            new_state = config.get_sumo_state(current_fama_code)
            if new_state and new_state != last_applied_state:
                traci.trafficlight.setRedYellowGreenState(config.tls_id, new_state)
                last_applied_state = new_state
                print(f"[*] Khop den: {hex(current_fama_code)} -> {config.tls_id}")

        # 2. GỬI DETECTOR LÊN TỦ
        det_bridge.update_detectors()
        
        time.sleep(0.05)
finally:
    traci.close()