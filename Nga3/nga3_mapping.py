# nga5_mapping.py
class Nga3Mapping:
    def __init__(self):
        # ID ngã tư lấy từ file nga5.net.xml của bạn
        self.tls_id = "clusterJ0_J2_J4"
        
        # BẢNG ÁNH XẠ: [Mã Hex từ tủ] -> [Chuỗi 30 ký tự SUMO]
        # Mình dựa vào cấu trúc 30 ký tự trong file nga3.net.xml bạn gửi
        self.phase_map = {
            0x03: "GGGGgrrrggrrrr",
            0x04: "grrrGGGGggrrrr",
            0x05: "grrrgrrrgGGgGG",
        }

    def get_sumo_state(self, fama_code):
        """Trả về chuỗi 30 ký tự tương ứng với mã nhận được"""
        return self.phase_map.get(fama_code)