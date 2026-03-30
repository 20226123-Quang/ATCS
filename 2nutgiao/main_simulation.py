import os
import traci
import time
import json
from fama_junction_control import FamaJunctionControl

def run():
    with open('junctions_config.json', 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    controllers = [FamaJunctionControl(cfg) for cfg in config_data['junctions']]
    sumo_cfg = os.path.abspath("2nutgiao.sumocfg")

    # BƯỚC 1: Tắt sạch SUMO cũ để giải phóng Port
    os.system("taskkill /f /im sumo-gui.exe >nul 2>&1")
    time.sleep(1.0)

    try:
        # BƯỚC 2: Khởi động SUMO với nhãn riêng và đợi lâu hơn
        print(f"[*] Đang nạp mô hình 2 nút giao...")
        traci.start(["sumo-gui", "-c", sumo_cfg, "--start", "--real-time"], 
                    numRetries=500, label="multi_junction")
        
        # BƯỚC 3: Đợi SUMO GUI hiện hình hoàn toàn rồi mới kết nối mạng
        print("[*] Đợi 3 giây để hệ thống ổn định...")
        time.sleep(3.0)

        for ctrl in controllers:
            ctrl.start_communication()

        print("[+] Hệ thống đã thông suốt 2 tủ FAMA.")

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            for ctrl in controllers:
                ctrl.sync_to_sumo()
                ctrl.sync_to_fama()
            time.sleep(0.05)
            
    except Exception as e:
        print(f"[!] Lỗi kết nối TraCI: {e}")
    finally:
        traci.close()

if __name__ == "__main__":
    run()