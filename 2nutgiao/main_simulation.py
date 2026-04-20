import os
import sys
import time
import json
import threading
import traceback
import traci
from fama_junction_control import FamaJunctionControl

def run():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Nạp cấu hình
    config_path = os.path.join(script_dir, 'junctions_config.json')
    if not os.path.exists(config_path):
        print(f"[!] Khong tim thay file cau hinh: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    kpi_constants = config_data['kpi_constants']
    controllers = [FamaJunctionControl(cfg, kpi_constants) for cfg in config_data['junctions']]
    sumo_cfg = os.path.join(script_dir, "2nutgiao.sumocfg")

    # 2. Khoi dong ket noi tu (Truoc khi start SUMO - Theo Debug)
    print("[*] Dang khoi tao ket noi den cac tu dieu khien...")
    for ctrl in controllers:
        ctrl.start_communication()

    # 3. Dọn dẹp tiến trình cũ
    os.system("taskkill /f /im sumo-gui.exe >nul 2>&1")
    os.system("taskkill /f /im sumo.exe >nul 2>&1")
    time.sleep(1.0)

    try:
        # 4. Khoi dong SUMO (Don gian hoa - Theo Debug)
        print(f"[*] Dang khoi dong SUMO voi cau hinh: {sumo_cfg}")
        traci.start(["sumo-gui", "-c", sumo_cfg, "--start"])
        
        print("[*] SUMO da san sang. Dang thiet lap mapping...")

        # Mapping lane/link thong qua TraCI
        for ctrl in controllers:
            try:
                ctrl.all_lanes = traci.trafficlight.getControlledLanes(ctrl.tls_id)
                ctrl.unique_lanes = list(dict.fromkeys(ctrl.all_lanes))
                ctrl.lane_to_indices = {}
                links = traci.trafficlight.getControlledLinks(ctrl.tls_id)
                for i, link_list in enumerate(links):
                    if not link_list: continue
                    from_lane = link_list[0][0]
                    if from_lane not in ctrl.lane_to_indices:
                        ctrl.lane_to_indices[from_lane] = []
                    ctrl.lane_to_indices[from_lane].append(i)
                print(f"    [+] Mapping lane/link cho {ctrl.name} thanh cong.")
            except Exception as e:
                print(f"    [!] Loi khi mapping {ctrl.name} ({ctrl.tls_id}): {e}")

        # DOI DU LIEU TU TU (Addressing: "khong lay du du lieu tu tu de khoi tao")
        print("[*] Dang cho du lieu khoi tao tu cac tu (Toi da 10s)...")
        wait_start = time.time()
        while time.time() - wait_start < 10:
            if all(ctrl.has_received_data() for ctrl in controllers):
                print("[+] Tat ca cac tu da san sang.")
                break
            time.sleep(1.0)
        else:
            print("[!] Canh bao: Mot so tu chua gui du lieu. Van tiep tuc mo phong...")

        print("[+] Bat dau vong lap mo phong...")

        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            curr_time = traci.simulation.getTime()
            delta_t = traci.simulation.getDeltaT()

            for ctrl in controllers:
                # 1. Dong bo den tu vao SUMO
                ctrl.sync_to_sumo()
                # 2. Dong bo detector SUMO len tu
                ctrl.sync_to_fama()

                # 3. Cap nhat KPI theo tung buoc
                for lane in ctrl.unique_lanes:
                    v_ids = traci.lane.getLastStepVehicleIDs(lane)
                    v_movements = {}
                    for v_id in v_ids:
                        try:
                            next_links = traci.vehicle.getNextLinks(v_id)
                            for link in next_links:
                                if link[5] == ctrl.tls_id:
                                    v_movements[v_id] = link[6]
                                    break
                        except: pass

                    link_green_status = {}
                    if ctrl.last_applied_state:
                        for idx in ctrl.lane_to_indices.get(lane, []):
                            if idx < len(ctrl.last_applied_state):
                                link_green_status[idx] = ctrl.last_applied_state[idx].lower() in ['g', 'u']
                    
                    ctrl.update_kpi_step(lane, v_ids, v_movements, link_green_status, delta_t)

                # 4. Kiem tra ket thuc chu ky
                ctrl.check_and_process_cycle(curr_time, ctrl.unique_lanes)

            time.sleep(0.005)
            step += 1
            
    except Exception as e:
        print(f"[!] Loi: {e}")
        traceback.print_exc()
    finally:
        print("[*] Dang dong ket noi...")
        for ctrl in controllers:
            try: ctrl.close()
            except: pass
        try: traci.close()
        except: pass

if __name__ == "__main__":
    run()
