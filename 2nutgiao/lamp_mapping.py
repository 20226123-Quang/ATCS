class LampMapper:
    def __init__(self):
        self.color_map = {
            0x10: 'r', 0x20: 'G', 0x30: 'y',
            0x11: 'r', 0x21: 'g', 0x31: 'y', 0x01: 'r'
        }
        # Mapping từ Light Group ID (tủ FAMA) sang Link Index (SUMO)
        self.junction_indices = {
            # Nút Ngã 4 Quang Trung - Lê Lợi (22 link indices)
            "cluster5599998268_5599998291_7928757444_9323588912": {
                # Da khop với thực tế nút Ngã 4
                1: [19,20,21],    # Nhóm 1
                2: [17, 18],    # Nhóm 2
                3: [16],    # ...
                4: [4,5],
                5: [1,3],
                6: [0,2],
                7: [9,10],
                8: [8,7],
                9: [6],
                10: [15],
                11: [12,13,14],
                12: [11]
            },
            # Nút Ngã Ba Voi (Kế thừa mapping từ J7 Debug - 19 link indices)
            "J0": {
                1: [4, 5], 2: [1, 2, 3], 3: [0, 15], 
                4: [8, 9], 5: [7], 6: [6], 7: [14], 
                8: [13, 12, 11], 9: [10], 10: [16, 17, 18]
            }
        }

    def generate_state(self, tls_id, status_dict):
        mapping = self.junction_indices.get(tls_id, {})
        if not mapping:
            return None
            
        # Tự động tính độ dài chuỗi state dựa trên index lớn nhất trong mapping
        max_idx = -1
        for indices in mapping.values():
            if indices:
                max_idx = max(max_idx, max(indices))
        
        if max_idx == -1:
            return None
            
        state_len = max_idx + 1

        state_list = ['r'] * state_len
        
        for g_id, color_hex in status_dict.items():
            if g_id in mapping:
                char = self.color_map.get(color_hex, 'r')
                for idx in mapping[g_id]:
                    if idx < state_len:
                        state_list[idx] = char
        return "".join(state_list)
