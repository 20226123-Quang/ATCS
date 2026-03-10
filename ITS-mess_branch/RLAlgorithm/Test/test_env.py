import traci
sumoBinary = "C:\\Phi\\Work\\Simulation\\SUMO\\bin\\sumo-gui.exe"
sumoCmd = [sumoBinary, "-c", "C:\\Phi\\Work\\SimulationData\\Intersect_Test\\config.sumocfg"]
traci.start(sumoCmd)
step = 0
tls_ids = traci.trafficlight.getIDList()
target_tls_ids = tls_ids[:2] if len(tls_ids) >= 2 else tls_ids

# Get current state to understand the format for each intersection
for idx, tls_id in enumerate(target_tls_ids):
    current_state = traci.trafficlight.getRedYellowGreenState(tls_id)
    print(f"Traffic light {idx}: {tls_id}")
    print(f"Current state: {current_state}")
    print(f"State length: {len(current_state)}")

print(f"Lane IDs: {traci.lane.getIDList()}")
switch = [0 for _ in target_tls_ids]
while step < 500:
    traci.simulationStep()
    for idx, tls_id in enumerate(target_tls_ids):
        # Đếm xe cho từng intersection (giả sử mỗi intersection có 4 hướng E0-E3, mỗi hướng 3 lane)
        total_E = []
        for d in range(4):
            total = 0
            for i in range(3):
                lane_id = f"-E{d}_{i}"
                if lane_id in traci.lane.getIDList():
                    total += len(traci.lane.getLastStepVehicleIDs(lane_id))
            total_E.append(total)
        print(f"Step {step} | TLS {tls_id}: E0={total_E[0]}, E1={total_E[1]}, E2={total_E[2]}, E3={total_E[3]}")
        try:
            if (step % 50 == 0 and idx == 0) or (step % 70 == 0 and idx == 1):
                switch[idx] = switch[idx] ^ 1
            state = "GGgGrrGGgGrr" if switch[idx] == 1 else "GrrGGgGrrGGg"
            state = list(state)
            state[1] = state[0] = 'o'
            state = ''.join(state)
            traci.trafficlight.setRedYellowGreenState(tls_id, state)
        except Exception as e:
            print(f"Error at TLS {tls_id}: {e}")
            break
    step += 1

traci.close()
