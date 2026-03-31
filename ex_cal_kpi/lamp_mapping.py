class LampMapper:
    def __init__(self):
        self.color_map = {
            0x10: 'r', 0x20: 'G', 0x30: 'y',
            0x11: 'r', 0x21: 'g', 0x31: 'y', 0x01: 'r'
        }
        self.junction_indices = {
            "J7": {
                1: [4, 5], 2: [1, 2, 3], 3: [0, 15], 
                4: [8, 9], 5: [7], 6: [6], 7: [14], 
                8: [13, 12, 11], 9: [10], 10: [16, 17, 18]
            }
        }

    def generate_state(self, tls_id, status_dict):
        # Khởi tạo 19 ký tự J7
        state_list = list('rrrrrrrrrrrrrrrrrrr')
        
        # Mặc định index 0 và 15 luôn xanh dự phòng
        state_list[0] = 'G'
        state_list[15] = 'G'
        
        mapping = self.junction_indices.get(tls_id, {})
        for g_id, color_hex in status_dict.items():
            if g_id in mapping:
                char = self.color_map.get(color_hex, 'r')
                for idx in mapping[g_id]:
                    if idx < 19: state_list[idx] = char
        return "".join(state_list)