import socket
import threading
import time
import traci
import cal_crc
from mapping_manager import JunctionMapping

class FamaJunctionControl:
    def __init__(self, config):
        self.name = config['name']
        self.ip = config['ip']
        self.udp_port = config['udp_port']
        self.tcp_port = config['tcp_port']
        self.mapping = JunctionMapping(config['junction_type'])
        self.tls_id = self.mapping.tls_id
        self.current_fama_code = -1
        self.tcp_client = None # Không khởi tạo socket ở đây
        self.detector_map = config.get('detectors', {})
        self.detector_memory = {det_id: False for det_id in self.detector_map.keys()}
        self.running = True

    def start_communication(self):
        """Chỉ gọi hàm này SAU KHI traci.start() thành công"""
        def _tcp_connect():
            try:
                self.tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_client.settimeout(3.0)
                self.tcp_client.connect((self.ip, self.tcp_port))
                print(f"[+] {self.name}: TCP Connected")
            except Exception as e:
                print(f"[!] {self.name} TCP Error: {e}")

        # Chạy kết nối trong thread riêng để không làm đơ vòng lặp chính
        threading.Thread(target=_tcp_connect, daemon=True).start()
        threading.Thread(target=self._udp_pool_loop, daemon=True).start()

    def _udp_pool_loop(self):
        input_hex = "00 13 01 00 01 00 00 00 03 01 01 10 01 01 04 0D 02 04 01"
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.5)
        while self.running:
            try:
                input_bin = ''.join(format(int(b, 16), '08b') for b in input_hex.split())
                crc = cal_crc.crc16_binary_simulation(input_bin)
                packet = bytes.fromhex(("7E " + input_hex + " " + f"{int(crc, 2):04X}" + " 7D").replace(" ", ""))
                sock.sendto(packet, (self.ip, self.udp_port))
                data, _ = sock.recvfrom(1024)
                if data:
                    self.current_fama_code = int(data.hex(' ').split(' ')[-4], 16)
            except: pass
            time.sleep(1.0)

    def sync_to_sumo(self):
        """Ép pha từ tủ vào SUMO"""
        if self.current_fama_code != -1:
            state = self.mapping.get_sumo_state(self.current_fama_code)
            if state:
                traci.trafficlight.setRedYellowGreenState(self.tls_id, state)

    def sync_to_fama(self):
        """Gửi detector từ SUMO lên tủ"""
        if not self.tcp_client: return
        for det_id, channel in self.detector_map.items():
            try:
                occ = traci.lanearea.getLastStepVehicleNumber(det_id) > 0
                if occ != self.detector_memory[det_id]:
                    # Gửi bản tin Hex
                    status = 0x01 if occ else 0x00
                    chk = 0x05 ^ 0x10 ^ 0x82 ^ 0x08 ^ channel ^ status
                    pkt = bytes([0x7E, 0x05, 0x10, 0x82, 0x08, channel, status, chk, 0x7E])
                    self.tcp_client.send(pkt)
                    self.detector_memory[det_id] = occ
            except: continue