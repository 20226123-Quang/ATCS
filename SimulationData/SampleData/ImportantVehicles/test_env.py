import traci
sumoBinary = "C:\\Phi\\Work\\Simulation\\SUMO\\bin\\sumo-gui.exe"
sumoCmd = [sumoBinary, "-c", "./config.sumocfg"]
traci.start(sumoCmd)
step = 0
tls_ids = traci.trafficlight.getIDList()
target_tls_ids = tls_ids if tls_ids else None

# Get current state to understand the format
if target_tls_ids:
	current_state = traci.trafficlight.getRedYellowGreenState(target_tls_ids[0])
	print(f"Traffic light: {target_tls_ids[0]}")
	print(f"Current state: {current_state}")
	print(f"State length: {len(current_state)}")

print(f"Lane IDs: {traci.lane.getIDList()}")  
switch = 0
while step < 300:
	traci.simulationStep()
	total_E0 = 0;
	for i in range(3):
		total_E0 += len(traci.lane.getLastStepVehicleIDs("-E0_{}".format(i)))
	total_E1 = 0;
	for i in range(3):
		total_E1 += len(traci.lane.getLastStepVehicleIDs("-E1_{}".format(i)))
	total_E2 = 0;
	for i in range(3):
		total_E2 += len(traci.lane.getLastStepVehicleIDs("-E2_{}".format(i)))
	total_E3 = 0;
	for i in range(3):
		total_E3 += len(traci.lane.getLastStepVehicleIDs("-E3_{}".format(i)))
	print(f"Step {step}: E0={total_E0}, E1={total_E1}, E2={total_E2}, E3={total_E3}")
	if target_tls_ids:
		if step % 50 == 0:
			switch = switch ^ 1
		for target_tls_id in target_tls_ids:
			try:
				state = "GGgGrrGGgGrr" if switch == 1 else "GrrGGgGrrGGg"
				state = list(state)
				state = ''.join(state)
				traci.trafficlight.setRedYellowGreenState(target_tls_id, state)
			except Exception as e:
				print(f"Error: {e}")
				break
		step += 1

traci.close()
