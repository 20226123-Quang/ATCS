import socket
import threading
import time
import traci
import csv
import os
import cal_crc
from lamp_mapping import LampMapper
from kpi_engine import KPIEngine

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
            print(f"[*] {threading.current_thread().name} Cycle Detection Started at {current_time}s")
            return None
        
        if current_state != self.last_state:
            self.last_state = current_state
            self.has_changed = True
            
            if current_state == self.first_state and self.has_changed:
                duration = current_time - self.start_time
                if duration >= 20.0:
                    self.start_time = current_time
                    self.has_changed = False
                    return duration
        return None

class FamaJunctionControl:
    def __init__(self, config, kpi_constants):
        self.name = config['name']
        self.ip = config['ip']
        self.udp_port = config['udp_port']
        self.tcp_port = config['tcp_port']
        self.tls_id = config['tls_id_sumo']
        self.cross_id = config['cross_id']
        
        self.mapper = LampMapper()
        mapping = self.mapper.junction_indices.get(self.tls_id, {})
        default_groups = sorted(list(mapping.keys())) if mapping else []
        self.lamp_groups = config.get('lamp_groups', default_groups)
        
        self.engine = KPIEngine(kpi_constants)
        self.cycle_detector = CycleDetector()
        
        self.shared_lamp_status = {}
        self.status_lock = threading.Lock()
        self.last_applied_state = ""
        
        self.tcp_client = None
        self.detector_map = config.get('detectors', {})
        self.detector_memory = {det_id: False for det_id in self.detector_map.keys()}
        
        self.running = True
        self.has_received_data_flag = False
        
        safe_id = config.get('junction_id', self.tls_id)
        self.csv_filename = f"crow_in_{safe_id}.csv"
        self.csv_file = open(self.csv_filename, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Time", "Lane", "Delay_s", "Saturation", "Avg_Queue_Veh", "Inflow_PCU_h", "TotalDemand_h", "Capacity_h", "Residual_NGE"])

    def has_received_data(self):
        with self.status_lock:
            return self.has_received_data_flag

    def start_communication(self):
        """Khởi động các luồng kết nối tủ FAMA"""
        # 1. Luồng TCP cho Detector (Không chặn luồng chính)
        def _tcp_connect():
            while self.running:
                try:
                    self.tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.tcp_client.settimeout(5.0)
                    self.tcp_client.connect((self.ip, self.tcp_port))
                    print(f"[+] {self.name}: TCP Connected ({self.ip}:{self.tcp_port})")
                    break
                except:
                    time.sleep(5.0)
        
        threading.Thread(target=_tcp_connect, name=f"TCP_{self.name}", daemon=True).start()
        
        # 2. Luồng UDP cho Trạng thái đèn
        threading.Thread(target=self._udp_sequential_query_loop, name=f"UDP_{self.name}", daemon=True).start()

    def _udp_sequential_query_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.3)
        base_payload = f"00 13 01 00 01 00 00 00 03 {self.cross_id} 01 10 01 01 04 03 03 02"
        
        last_success_time = time.time()
        while self.running:
            success_count = 0
            for g_id in self.lamp_groups:
                try:
                    current_payload = f"{base_payload} {g_id:02x}"
                    input_bin = ''.join(format(int(b, 16), '08b') for b in current_payload.split())
                    crc_bin = cal_crc.crc16_binary_simulation(input_bin)
                    packet = bytes.fromhex(("7E " + current_payload + " " + f"{int(crc_bin, 2):04X}" + " 7D").replace(" ", ""))
                    
                    sock.sendto(packet, (self.ip, self.udp_port))
                    data, _ = sock.recvfrom(1024)
                    
                    if data and len(data) >= 21 and data[12] == 0x20:
                        recv_id = data[19]
                        recv_val = data[20]
                        with self.status_lock:
                            self.shared_lamp_status[recv_id] = recv_val
                            self.has_received_data_flag = True
                        success_count += 1
                        last_success_time = time.time()
                except:
                    pass
                time.sleep(0.01)
            
            if success_count == 0 and time.time() - last_success_time > 10:
                print(f"[!] {self.name}: UDP Timeout (10s) - No data from {self.ip}")
                last_success_time = time.time()
            
            time.sleep(0.2)

    def sync_to_sumo(self):
        with self.status_lock:
            local_status = self.shared_lamp_status.copy()
        if local_status:
            new_state = self.mapper.generate_state(self.tls_id, local_status)
            if new_state and new_state != self.last_applied_state:
                try:
                    traci.trafficlight.setRedYellowGreenState(self.tls_id, new_state)
                    self.last_applied_state = new_state
                except Exception as e:
                    print(f"[!] {self.name} TraCI error setting state: {e}")

    def sync_to_fama(self):
        if not self.tcp_client: return
        for det_id, channel in self.detector_map.items():
            try:
                is_occupied = traci.lanearea.getLastStepVehicleNumber(det_id) > 0
                if is_occupied != self.detector_memory[det_id]:
                    status = 0x01 if is_occupied else 0x00
                    id_dev, ver, op, obj = 0x05, 0x10, 0x82, 0x08
                    checksum = id_dev ^ ver ^ op ^ obj ^ channel ^ status
                    pkt = bytes([0x7E, id_dev, ver, op, obj, channel, status, checksum, 0x7E])
                    self.tcp_client.send(pkt)
                    self.detector_memory[det_id] = is_occupied
            except:
                pass

    def update_kpi_step(self, lane_id, v_ids, v_movements, link_green_status, step_len):
        self.engine.update_lane(lane_id, v_ids, v_movements, link_green_status, step_len)

    def check_and_process_cycle(self, current_time, unique_lanes):
        if not self.last_applied_state: return
        cycle_duration = self.cycle_detector.check_cycle(self.last_applied_state, current_time)
        if cycle_duration:
            print(f"[*] {self.name} Cycle ended: {cycle_duration}s at {current_time}s")
            for lane in unique_lanes:
                res = self.engine.compute_kpi(lane, cycle_duration)
                self.csv_writer.writerow([
                    int(current_time), lane, 
                    round(res.control_delay_seconds, 2), 
                    round(res.degree_of_saturation, 2),
                    round(res.avg_queue_vehicles, 2),
                    round(res.inflow_pcu_per_hour, 2),
                    round(res.total_demand_pcu_h, 2),
                    round(res.capacity_pcu_h, 2),
                    round(res.n_ge, 2)
                ])
                self.engine.reset_cycle(lane, res.n_ge)
            self.csv_file.flush()

    def close(self):
        self.running = False
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()
        if self.tcp_client:
            try: self.tcp_client.close()
            except: pass
