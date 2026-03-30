# nga5_mapping.py
class Nga5Mapping:
    def __init__(self):
        # ID ngã tư lấy từ file nga5.net.xml của bạn
        self.tls_id = "clusterJ1_J3_J5_J6_#1more"
        
        # BẢNG ÁNH XẠ: [Mã Hex từ tủ] -> [Chuỗi 30 ký tự SUMO]
        # Mình dựa vào cấu trúc 30 ký tự trong file nga5.net.xml bạn gửi
        self.phase_map = {
            0x03: "GGGGGGgrrrrrgrrrrrgrrrrrgrrrrr",
            0x04: "grrrrrGGGGGGgrrrrrgrrrrrgrrrrr",
            0x05: "grrrrrgrrrrrGGGGGGgrrrrrgrrrrr",
            0x06: "grrrrrgrrrrrgrrrrrGGGGGGgrrrrr",
            0x07: "grrrrrgrrrrrgrrrrrgrrrrrGGGGGG"
        }

    def get_sumo_state(self, fama_code):
        """Trả về chuỗi 30 ký tự tương ứng với mã nhận được"""
        return self.phase_map.get(fama_code)