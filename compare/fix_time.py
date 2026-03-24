import traci
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

class TrafficObserver:
    def __init__(self, config_file, tls_id):
        self.config_file = config_file
        self.tls_id = tls_id
        self.report_data = []
        self.running = True

    def print_final_report(self):
        """Xuất bảng KPI khi kết thúc chương trình"""
        if not self.report_data: return
        print("\n" + "="*55 + "\nBÁO CÁO KPI GIAO THÔNG (OBSERVER MODE)\n" + "="*55)
        print(f"{'Thời gian(s)':<15} | {'Xe chờ':<8} | {'Hàng chờ(m)':<12} | {'Wait TB(s)':<10}")
        print("-" * 55)
        sum_veh, sum_len, sum_wait, n = 0, 0, 0, len(self.report_data)
        for r in self.report_data:
            print(f"{r['time']:<15.1f} | {r['veh']:<8} | {r['len']:<12.1f} | {r['wait']:<10.1f}")
            sum_veh += r['veh']; sum_len += r['len']; sum_wait += r['wait']
        if n > 0:
            print("="*55 + f"\n{'TRUNG BÌNH':<15} | {sum_veh/n:<8.1f} | {sum_len/n:<12.1f} | {sum_wait/n:<10.1f}")

    def run(self):
        # Chạy SUMO ở chế độ quan sát (không cần GUI cũng được, ở đây tôi để GUI cho bạn dễ theo dõi)
        traci.start(["sumo-gui", "-c", self.config_file])
        
        print(f"[*] Observer đã bắt đầu theo dõi ngã tư: {self.tls_id}")
        last_recorded_time = -1
        
        try:
            while True:
                traci.simulationStep()
                
                # Lấy mẫu KPI mỗi 5 giây mô phỏng
                current_time = int(traci.simulation.getTime())
                if current_time % 5 == 0 and current_time != last_recorded_time:
                    lanes = traci.trafficlight.getControlledLanes(self.tls_id)
                    q_veh, q_len, q_wait = 0, 0, 0
                    
                    for l in lanes:
                        q_veh += traci.lane.getLastStepHaltingNumber(l)
                        q_wait += traci.lane.getWaitingTime(l)
                        # Tính chiều dài hàng chờ chính xác (m)
                        for v in traci.lane.getLastStepVehicleIDs(l):
                            if traci.vehicle.getSpeed(v) < 0.1:
                                q_len += (traci.lane.getLength(l) - traci.vehicle.getLanePosition(v))
                    
                    self.report_data.append({'time': current_time, 'veh': q_veh, 'len': q_len, 'wait': q_wait})
                    last_recorded_time = current_time
                    print(f"[*] Đang ghi log giây {current_time}...")
                
        except KeyboardInterrupt:
            self.print_final_report()
        finally:
            traci.close()

if __name__ == "__main__":
    # Thay ID ngã tư vào đây
    TLS_ID = "cluster_12870044126_12870044128_12870044129_5204960748"
    observer = TrafficObserver("osm.sumocfg", TLS_ID)
    observer.run()