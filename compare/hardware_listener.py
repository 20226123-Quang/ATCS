import socket
import threading
import time
import fama_protocol

class FamaHardwareConnector:
    def __init__(self, tcp_ip, tcp_port, udp_ip, udp_port):
        self.tcp_ip, self.tcp_port = tcp_ip, tcp_port
        self.udp_ip, self.udp_port = udp_ip, udp_port
        self.running = False
        self.tcp_client = None
        self.udp_socket = None
        self.current_phase_code = None

    def connect(self):
        try:
            self.tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_client.connect((self.tcp_ip, self.tcp_port))
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.settimeout(1.5)
            self.running = True
            threading.Thread(target=self._udp_loop, daemon=True).start()
            print(f"Connected to Cabinet at {self.tcp_ip}")
        except Exception as e:
            print(f"Hardware Connection Error: {e}")

    def _udp_loop(self):
        heartbeat_packet = fama_protocol.build_fama_heartbeat()
        while self.running:
            try:
                self.udp_socket.sendto(heartbeat_packet, (self.udp_ip, self.udp_port))
                response, _ = self.udp_socket.recvfrom(1024)
                # Sửa lỗi UnboundLocalError: Kiểm tra ngay bên trong block try
                if response and len(response) >= 5:
                    self.current_phase_code = response[-4]
            except socket.timeout:
                pass 
            except Exception as e:
                print(f"UDP Error: {e}")
            time.sleep(1.0)

    def send_detector(self, channel, status):
        if self.tcp_client:
            packet = fama_protocol.build_detector_packet(channel, status)
            self.tcp_client.send(packet)