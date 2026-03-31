class LampMapper:
    def __init__(self):
        self.color_map = {
            0x10: 'r', 0x20: 'G', 0x30: 'y',
            0x11: 'r', 0x21: 'g', 0x31: 'y', 0x01: 'r'
        }
        self.junction_indices = {
            "clusterJ1_J3_J5_J6_#1more": {
                16: [24], 
                17: [4,5], 
                18: [1, 2, 3], 
                19: [0], 
                20: [7,8,9,10,11],
                21: [6],
                22: [13,14,15,16,17],
                23: [12],
                24: [19,20,21,22,23],
                25: [18],
                26: [25,26,27],
                27: [28,29]
            }
        }

    def generate_state(self, tls_id, status_dict):
        # Khởi tạo 30 ký tự J7
        state_list = list('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
        
        # Mặc định index 0 và 15 luôn xanh dự phòng
        state_list[24] = 'g'
        state_list[0] = 'g'
        state_list[6] = 'g'
        state_list[12] = 'g'
        state_list[18] = 'g'
        
        mapping = self.junction_indices.get(tls_id, {})
        for g_id, color_hex in status_dict.items():
            if g_id in mapping:
                char = self.color_map.get(color_hex, 'r')
                for idx in mapping[g_id]:
                    if idx < 30: state_list[idx] = char
        return "".join(state_list)