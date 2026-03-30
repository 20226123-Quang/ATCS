# mapping_manager.py

class JunctionMapping:
    def __init__(self, junction_type):
        if junction_type == "nga4qtllq":
            self.tls_id = "cluster5599998268_5599998291_7928757444_9323588912"
            self.phase_map = {
                0x01: "grrrrrGGGGggrrrrGGGGgg", # Pha 1
                0x02: "GGgGGggrrrrGGGGggrrrrr", # Pha 2
            }
        elif junction_type == "ngabavoi":
            self.tls_id = "J0"
            self.phase_map = {
                0x01: "GGGGgggrrrGGGGggrrr", # Hướng chính xanh
                0x02: "grrrrrGGgggrrrrGGGg", # Hướng phụ xanh
            }

    def get_sumo_state(self, fama_code):
        """Trả về chuỗi ký tự đèn tương ứng với mã Hex từ tủ FAMA"""
        return self.phase_map.get(fama_code, None)