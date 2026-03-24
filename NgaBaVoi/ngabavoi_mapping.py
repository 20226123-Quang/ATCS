# nga5_mapping.py
class NgaBaVoiMapping:
    def __init__(self):
        # ID ngã tư lấy từ file nga5.net.xml của bạn
        self.tls_id = "J7"
        
        # BẢNG ÁNH XẠ: [Mã Hex từ tủ] -> [Chuỗi 30 ký tự SUMO]
        # Mình dựa vào cấu trúc 30 ký tự trong file nga3.net.xml bạn gửi
        self.phase_map = {
            0x01: "GGGGgggrrrGGGGggrrr",
            0x02: "grrrrrGGgggrrrrGGGg",
        }

    def get_sumo_state(self, fama_code):
        """Trả về chuỗi 30 ký tự tương ứng với mã nhận được"""
        return self.phase_map.get(fama_code)