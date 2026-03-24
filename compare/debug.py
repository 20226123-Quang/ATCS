import socket
import time
import cal_crc # Sử dụng file cal_crc.py có sẵn của bạn
from datetime import datetime

# --- Xem mã phase của tủ ---
TARGET_IP = "192.168.8.134"
TARGET_PORT_UDP = 4050

def get_fama_phase():
    # Nội dung bản tin Heartbeat từ file của Quyên
    # chỉnh theo cross của tủ
    input_hex = "00 13 01 00 01 00 00 00 03 02 01 10 01 01 04 0D 02 04 02"
    
    # Tính toán CRC (Dùng hàm từ file cal_crc.py của bạn)
    input_bin = ''.join(format(int(b, 16), '08b') for b in input_hex.split())
    crc = cal_crc.crc16_binary_simulation(input_bin)
    packet = bytes.fromhex(("7E " + input_hex + " " + f"{int(crc, 2):04X}" + " 7D").replace(" ", ""))
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        # Gửi Heartbeat lên tủ
        sock.sendto(packet, (TARGET_IP, TARGET_PORT_UDP))
        data, _ = sock.recvfrom(1024)
        
        if data:
            # Trích xuất byte thứ 4 từ dưới lên (Mã pha)
            res_list = data.hex(' ').split(' ')
            phase_code = int(res_list[-4], 16)
            return phase_code
    except Exception as e:
        return f"Lỗi: {e}"
    finally:
        sock.close()

if __name__ == "__main__":
    print(f"[*] Đang lắng nghe tín hiệu từ tủ FAMA ({TARGET_IP})...")
    print("-" * 50)
    print(f"{'Thời gian':<20} | {'Mã Hex':<10} | {'Mã Dec'}")
    print("-" * 50)
    
    last_code = None
    
    try:
        while True:
            current_code = get_fama_phase()
            
            # Chỉ in ra màn hình khi mã pha thay đổi
            if current_code != last_code:
                now = datetime.now().strftime("%H:%M:%S")
                if isinstance(current_code, int):
                    print(f"{now:<20} | {hex(current_code):<10} | {current_code}")
                else:
                    print(f"{now:<20} | {current_code}")
                
                last_code = current_code
                
            time.sleep(1.0) # Kiểm tra mỗi giây
    except KeyboardInterrupt:
        print("\n[*] Đã dừng lắng nghe.")