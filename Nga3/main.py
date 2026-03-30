import traci
import socket
import threading
import time
import cal_crc
import json
import os
import csv
from lamp_mapping import LampMapper
from ex_cal_kpi.kpi_engine import KPIEngine

# ==========================================
# --- CẤU HÌNH NÚT GIAO (CHỈNH TẠI ĐÂY) ---
# ==========================================
TARGET_IP = "192.168.8.138"
TARGET_PORT_UDP = 4050
TLS_ID = "J7"            # ID trong SUMO
CROSS_ID = "02"          # ID nút giao của tủ (Ví dụ: 01 cho ngã ba voi, 02 cho nút mới)
LAMP_RANGE = range(18, 25) # Dải đèn từ 18 đến 24
STEP_LEN = 0.5           # Bước nhảy thời gian mô phỏng (s)
# ==========================================

mapper = LampMapper()
engine = KPIEngine()
shared_lamp_status = {} 
status_lock = threading.Lock()

# --- 1. LOGIC TRUY VẤN ĐÈN TUẦN TỰ (LINH HOẠT) ---
def thread_sequential_query():
    global shared_lamp_status
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.3)
    
    # Cấu trúc lệnh: Class 3, Obj 3, Attr 2
    # Phần header giữ nguyên format 00 13...10 của Quyên
    base_payload = f"00 13 01 00 01 00 00 00 03 {CROSS_ID} 01 10 01 01 04 03 03 02"

    while True:
        for g_id in LAMP_RANGE: # Duyệt dải đèn đã cấu hình
            try:
                current_payload = f"{base_payload} {g_id:02x}"
                input_bin = ''.join(format(int(b, 16), '08b') for b in current_payload.split())
                crc = cal_crc.crc16_binary_simulation(input_bin)
                packet = bytes.fromhex(("7E " + current_payload + " " + f"{int(crc, 2):04X}" + " 7D").replace(" ", ""))
                
                sock.sendto(packet, (TARGET_IP, TARGET_PORT_UDP))
                data, _ = sock.recvfrom(1024)
                
                # Bóc tách theo thuật toán 7-byte chuẩn (ID tại index 19, VAL tại 20)
                if data and len(data) >= 21 and data[12] == 0x20:
                    recv_id = data[19]
                    recv_val = data[20]
                    if recv_id == g_id:
                        with status_lock:
                            shared_lamp_status[recv_id] = recv_val
            except: pass
            time.sleep(0.01)
        time.sleep(0.2)

# --- 2. CHẠY MÔ PHỎNG VÀ TÍNH KPI ---
def run():
    # Khởi động luồng Query
    threading.Thread(target=thread_sequential_query, daemon=True).start()
    time.sleep(1.0) 

    # Khởi động SUMO (Đồng bộ STEP_LEN)
    traci.start(["sumo-gui", "-c", "ngabavoi.sumocfg", "--start", "--end", "500", "--step-length", str(STEP_LEN)])
    
    # Chuẩn bị file CSV (Tự động đặt tên theo Cross)
    csv_f = open(f"kpi_cross_{CROSS_ID}_report.csv", "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["Time", "Lane", "Delay_s", "Queue_m", "Saturation"])

    controlled_lanes = traci.trafficlight.getControlledLanes(TLS_ID)
    last_applied_state = ""

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            curr_time = traci.simulation.getTime()

            # A. ĐỒNG BỘ ĐÈN
            with status_lock:
                local_status = shared_lamp_status.copy()

            if local_status:
                new_state = mapper.generate_state(TLS_ID, local_status)
                if new_state and new_state != last_applied_state:
                    traci.trafficlight.setRedYellowGreenState(TLS_ID, new_state)
                    last_applied_state = new_state

            # B. CẬP NHẬT DỮ LIỆU KPI MỖI BƯỚC NHẢY
            for lane in controlled_lanes:
                v_ids = set(traci.lane.getLastStepVehicleIDs(lane))
                idx = controlled_lanes.index(lane)
                # Kiểm tra màu đèn tại index tương ứng trong chuỗi state
                is_green = last_applied_state[idx].lower() in ['g', 'u'] if last_applied_state else False
                engine.update_lane(lane, v_ids, is_green, STEP_LEN) # Sử dụng STEP_LEN chuẩn

            # C. XUẤT KPI MỖI 60S
            if int(curr_time) % 60 == 0 and int(curr_time) > 0:
                for lane in controlled_lanes:
                    res = engine.compute_kpi(lane, 60.0)
                    writer.writerow([curr_time, lane, round(res.control_delay_seconds, 2), 
                                     round(res.queue_length_meters, 2), round(res.degree_of_saturation, 2)])
                    engine.reset_cycle(lane)
                print(f"[*] KPI Cross {CROSS_ID} Exported at {curr_time}s")

            if curr_time >= 500: break
            time.sleep(0.01)

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        csv_f.close()
        traci.close()

if __name__ == "__main__":
    run()