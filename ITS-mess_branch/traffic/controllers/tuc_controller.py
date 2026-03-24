import traci
import numpy as np
import traceback
from scipy.linalg import solve_discrete_are, solve

from traffic.config import (
	FIXED_TIME_PLANS,
	TL_STATES,
	PCU_MAPPING,
	VEHICLE_STOP_THRESHOLD,
)
from traffic.controllers.signal_controller import SignalController


class TUCController(SignalController):
	"""TUC (Traffic responsive Urban Control) with LQ optimal control"""
		
	def __init__(self, min_phase=5, max_phase=80, min_cycle=60, max_cycle=120,
				 control_penalty=0.2, alpha_blend=0.3, rate_limit=10.0, gating_param=0.3):
		self.min_phase = min_phase
		self.max_phase = max_phase
		self.min_cycle = min_cycle
		self.max_cycle = max_cycle
		self.control_penalty = control_penalty
		self.alpha_blend = alpha_blend
		self.rate_limit = rate_limit
		self.gating_param = float(gating_param)
		
		self.control_period = 140
		self.cycle_time = max_cycle
		
		self.saturation_flow = {}
		self.turning_rates = {}
		self.link_capacity = {}
		
		self.cycle_start = {tls_id: 0 for tls_id in FIXED_TIME_PLANS}
		self.cycle_time_tls = {tls_id: sum(d for d, _ in FIXED_TIME_PLANS[tls_id]) for tls_id in FIXED_TIME_PLANS}
		self.current_plan = {tls_id: list(FIXED_TIME_PLANS[tls_id]) for tls_id in FIXED_TIME_PLANS}
		self.queue_state = {tls_id: {} for tls_id in FIXED_TIME_PLANS}
		self.control_input = {tls_id: None for tls_id in FIXED_TIME_PLANS}
		self.feedback_gain = {tls_id: {} for tls_id in FIXED_TIME_PLANS}
		self.phase_to_edges = {tls_id: {} for tls_id in FIXED_TIME_PLANS}
		self.nominal_green = {tls_id: {} for tls_id in FIXED_TIME_PLANS}
		self.phase_order_per_tls = {tls_id: [] for tls_id in FIXED_TIME_PLANS}
		self.loss_time = {tls_id: 12 for tls_id in FIXED_TIME_PLANS}
		self.global_edges = []
		self.edge_index_global = {}
		self.phase_mapping = {}
		self.total_phases = 0
		self.prev_global_green = None
		self.global_state_prev = None
		self.last_update_time = None
		
		self._initialize_network_parameters()
		
	def _initialize_network_parameters(self):
		try:
			# Bước 1: khởi tạo tham số từng nút giao
			self.incoming_edges_per_tls = {}
			for tls_id in FIXED_TIME_PLANS:
				self.saturation_flow[tls_id] = 0.5
				self.link_capacity[tls_id] = 30
				
				incoming_edges = set()
				controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
				for lane in controlled_lanes:
					edge = lane.rsplit("_", 1)[0] if "_" in lane else lane
					incoming_edges.add(edge)
				self.incoming_edges_per_tls[tls_id] = incoming_edges
				
				num_directions = len(incoming_edges)
				edge_list = list(incoming_edges)
				for from_edge in edge_list:
					for to_edge in edge_list:
						if from_edge != to_edge:
							self.turning_rates[f"{tls_id}:{from_edge}->{to_edge}"] = 1.0 / max(num_directions - 1, 1)
						else:
							self.turning_rates[f"{tls_id}:{from_edge}->{to_edge}"] = 0.0
					self.turning_rates[f"{tls_id}:{from_edge}->EXIT"] = 0.1
				
				plan = FIXED_TIME_PLANS[tls_id]
				phase_idx = 0
				for duration, state in plan:
					if self._is_green_phase(tls_id, state):
						phase_edges = self._get_edges_for_state(tls_id, state)
						if phase_edges:
							self.phase_to_edges[tls_id][phase_idx] = phase_edges
							self.nominal_green[tls_id][phase_idx] = duration
							phase_idx += 1
				# lưu thứ tự các pha xanh và thời gian mất mát cho từng nút giao
				self.phase_order_per_tls[tls_id] = sorted(self.nominal_green[tls_id].keys())
				total_green = sum(self.nominal_green[tls_id].values()) if self.nominal_green[tls_id] else 0.0
				self.loss_time[tls_id] = max(self.cycle_time - total_green, 0.0)

			print(f"[Init] Computing feedback gain L offline (shared across all intersections)...")
			
			# Bước 2: xây dựng danh sách global các incoming edge trên toàn mạng
			incoming_edges_global = set()
			self.edge_to_tls = {}
			for tls_id in FIXED_TIME_PLANS:
				controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
				for lane in controlled_lanes:
					edge = lane.rsplit("_", 1)[0] if "_" in lane else lane
					incoming_edges_global.add(edge)
					self.edge_to_tls[edge] = tls_id
			
			edges = sorted(list(incoming_edges_global))
			self.global_edges = edges
			if not edges:
				return
			
			# số link (incoming edge) toàn mạng
			n_links = len(edges)
			
			# Build A matrix với kích thước = tổng số incoming edge toàn mạng
			A = np.eye(n_links)
			
			# Build B matrix tương ứng với vector trạng thái global
			B = self._build_B_matrix(edges)
			
			# Lấy capacity đại diện (trung bình) cho các link
			capacity_values = list(self.link_capacity.values())
			if capacity_values:
				capacity = float(sum(capacity_values)) / float(len(capacity_values))
			else:
				capacity = 30.0
			
			# Solve Riccati equation to get L
			self.L = self._solve_riccati(A, B, capacity, self.control_penalty)
			
		except Exception as e:
			print(f"Warning: Could not initialize network parameters: {e}")
		
	def _build_B_matrix(self, edges):
		try:
			n_links = len(edges)
			# ánh xạ edge -> chỉ số hàng trong B
			edge_index = {edge: idx for idx, edge in enumerate(edges)}
			self.edge_index_global = edge_index
			# tổng số cột = tổng số giai đoạn xanh của tất cả nút giao
			total_phases = 0
			phase_mapping = {}  # (j, local_phase) -> global column index
			for intersection_id in FIXED_TIME_PLANS:
				plan = FIXED_TIME_PLANS[intersection_id]
				local_phase = 0
				for duration, state in plan:
					if self._is_green_phase(intersection_id, state):
						phase_mapping[(intersection_id, local_phase)] = total_phases
						total_phases += 1
						local_phase += 1
			B = np.zeros((n_links, total_phases))
			self.phase_mapping = phase_mapping
			self.total_phases = total_phases
			T = self.control_period
			C = self.cycle_time
			# Thuật toán duyệt theo từng nút giao j và từng giai đoạn i ∈ F_j
			for j in FIXED_TIME_PLANS:
				plan = FIXED_TIME_PLANS[j]
				# I_j: các đoạn đường đi vào nút j
				I_j = self.incoming_edges_per_tls.get(j, set())
				# O_j: các đoạn đường đi ra từ nút j (tồn tại t_{w,z} với w ∈ I_j)
				O_j = set()
				for key, val in self.turning_rates.items():
					if not key.startswith(f"{j}:"):
						continue
					_, rest = key.split(":", 1)
					from_edge, to_edge = rest.split("->")
					if to_edge != "EXIT":
						O_j.add(to_edge)
				local_phase = 0
				for duration, state in plan:
					# chỉ xét các giai đoạn có đèn xanh (stage i ∈ F_j với v_z/v_w khác rỗng)
					if not self._is_green_phase(j, state):
						continue
					col = phase_mapping.get((j, local_phase), -1)
					if col < 0:
						local_phase += 1
						continue
					# các đoạn đường được xanh trong giai đoạn i (v_z, v_w)
					phase_edges = self._get_edges_for_state(j, state)
					# --- Bước A: z ∈ I_j ---
					for z in I_j:
						if z not in edge_index:
							continue
						# nếu đoạn đường z được ưu tiên (i ∈ v_z)
						if z in phase_edges:
							row = edge_index[z]
							# S_z: dòng bão hòa của đoạn đường z (dùng theo nút j)
							S_z = self.saturation_flow.get(j, 0.5)
							# theo yêu cầu: giữ hệ số T/C
							B[row, col] -= (T / C) * S_z
					# --- Bước B: z ∈ O_j ---
					for z in O_j:
						if z not in edge_index:
							continue
						row = edge_index[z]
						# t_{z,0}: tỉ lệ xe rời mạng từ đoạn z
						t_z0 = self.turning_rates.get(f"{j}:{z}->EXIT", 0.1)
						# xét tất cả w ∈ I_j có luồng rẽ vào z
						for w in I_j:
							t_wz = self.turning_rates.get(f"{j}:{w}->{z}", 0.0)
							if t_wz <= 0:
								continue
							# nếu w được xanh trong giai đoạn i (i ∈ v_w)
							if w in phase_edges:
								S_w = self.saturation_flow.get(j, 0.5)
								# theo yêu cầu: giữ hệ số T/C
								B[row, col] += (T / C) * (1 - t_z0) * t_wz * S_w
					local_phase += 1
			print(B.shape)
			return B
		except Exception as e:
			print(f"Error building B matrix: {e}")
			return np.zeros((len(edges), 1))
		
	def _get_pcu_value(self, vehicle_id):
		try:
			vtype = traci.vehicle.getTypeID(vehicle_id)
			if vtype in PCU_MAPPING:
				return PCU_MAPPING[vtype]
			vtype_lower = vtype.lower()
			for vehicle_type, pcu in PCU_MAPPING.items():
				if vehicle_type in vtype_lower:
					return pcu
			return PCU_MAPPING['passenger']
		except:
			return PCU_MAPPING['passenger']
		
	def _solve_riccati(self, A, B, link_capacity, control_penalty):
		print(f"A={A}, B={B}, link_capacity={link_capacity}, control_penalty={control_penalty}")
		try:
			n_links, n_phases = B.shape
			if n_links == 0 or n_phases == 0:
				return None
			Q = np.diag(np.ones(n_links) / (link_capacity + 1e-6))
			R = control_penalty * np.eye(n_phases)
			try:
				P = solve_discrete_are(A, B, Q, R)
			except np.linalg.LinAlgError as e:
				print(f"[Riccati] DARE solver failed: {e}")
				print(f"  A shape: {A.shape}, B shape: {B.shape}")
				return None
			BtPB = B.T @ P @ B
			R_plus_BtPB = R + BtPB
			try:
				cond_number = np.linalg.cond(R_plus_BtPB)
				if cond_number > 1e10:
					print(f"[Riccati] Warning: (R + B^T P B) is ill-conditioned (cond={cond_number:.2e})")
			except:
				pass
			BtPA = B.T @ P @ A
			try:
				L = solve(R_plus_BtPB, BtPA)
			except np.linalg.LinAlgError as e:
				print(f"[Riccati] Failed to solve for L: {e}")
				return None
			if L.shape != (n_phases, n_links):
				print(f"[Riccati] ERROR: L has wrong shape {L.shape}, expected ({n_phases}, {n_links})")
				return None
			A_BL = A - B @ L
			_ = np.linalg.eigvals(A_BL)
			print(f"[Riccati] L computed, L={L}, A={A}, B={B}, Q={Q}, R={R}")
			return L
		except Exception as e:
			print(f"[Riccati] Unexpected error: {e}")
			traceback.print_exc()
			return None
		finally:
			# khởi tạo g_applied ban đầu theo thời gian xanh danh định nếu có thể
			try:
				if hasattr(self, "phase_mapping") and self.phase_mapping and self.total_phases > 0:
					g0 = np.zeros((self.total_phases, 1))
					for (tls_id, local_phase), col in self.phase_mapping.items():
						val = self.nominal_green.get(tls_id, {}).get(local_phase, float(self.min_phase))
						if 0 <= col < self.total_phases:
							g0[col, 0] = val
					self.prev_global_green = g0
			except Exception:
				pass
		
	def _get_edges_for_state(self, tls_id, state_string):
		edges = []
		if tls_id in TL_STATES:
			for edge, indices in TL_STATES[tls_id].items():
				for idx in indices[1:]:
					if idx < len(state_string) and state_string[idx] == 'G':
						edges.append(edge)
						break
		return edges

	def _is_green_phase(self, tls_id, state_string):
		if tls_id not in TL_STATES:
			return False
		for edge, indices in TL_STATES[tls_id].items():
			for idx in indices[1:]:
				if idx < len(state_string) and state_string[idx] == 'G':
					return True
		return False
		
	def _get_queue_state_vector(self, tls_id):
		try:
			queue_state = {}
			controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
			edge_vehicles = {}
			for lane in controlled_lanes:
				edge = lane.rsplit("_", 1)[0] if "_" in lane else lane
				if edge not in edge_vehicles:
					edge_vehicles[edge] = []
				vehicles_in_lane = traci.lane.getLastStepVehicleIDs(lane)
				for vehicle_id in vehicles_in_lane:
					try:
						speed = traci.vehicle.getSpeed(vehicle_id)
						if speed < VEHICLE_STOP_THRESHOLD:
							pcu = self._get_pcu_value(vehicle_id)
							edge_vehicles[edge].append(pcu)
					except:
						pass
			for edge, pcu_list in edge_vehicles.items():
				queue_state[edge] = sum(pcu_list)
			if tls_id in TL_STATES:
				for edge in TL_STATES[tls_id].keys():
					if edge not in queue_state:
						queue_state[edge] = 0.0
			return queue_state
		except Exception as e:
			print(f"Error getting queue state for {tls_id}: {e}")
			return {}
		
	def _map_phases_to_edges(self, tls_id, g_phases, edges):
		edge_to_green = {edge: 0.0 for edge in edges}
		n_phases = len(g_phases)
		for phase_idx in range(n_phases):
			phase_edges = self.phase_to_edges.get(tls_id, {}).get(phase_idx, [])
			if not phase_edges and phase_idx < len(edges):
				phase_edges = [edges[phase_idx]]
			for edge in phase_edges:
				if edge in edge_to_green:
					edge_to_green[edge] = max(edge_to_green[edge], g_phases[phase_idx])
		return edge_to_green

	def _build_global_state_vector(self):
		"""Thu thập trạng thái x(k) toàn mạng dựa trên hàng chờ (PCU) trên mỗi cạnh."""
		if not self.global_edges:
			return None
		try:
			edge_queue = {edge: 0.0 for edge in self.global_edges}
			for tls_id in FIXED_TIME_PLANS:
				q_state = self._get_queue_state_vector(tls_id)
				for edge, val in q_state.items():
					if edge in edge_queue:
						edge_queue[edge] = float(val)
			# Áp dụng hiệu ứng gating (tuỳ chọn) trên hàng chờ PCU
			b = getattr(self, "gating_param", 0.0)
			if b is not None and b > 0.0:
				for edge in list(edge_queue.keys()):
					x_z = edge_queue[edge]
					# ánh xạ edge -> nút giao để lấy capacity đại diện
					tls_id = getattr(self, "edge_to_tls", {}).get(edge, None)
					if tls_id is not None:
						x_z_max = float(self.link_capacity.get(tls_id, 30.0))
					else:
						x_z_max = 30.0
					if x_z_max <= 0:
						continue
					denom = 1.0 - (b * x_z / (x_z_max + 1e-6))
					if denom > 1e-6:
						x_z_prime = x_z / denom
					else:
						# nếu gần bão hoà hoặc dữ liệu nhiễu, giới hạn ở giá trị an toàn
						x_z_prime = x_z_max
					edge_queue[edge] = x_z_prime
			x_vec = np.array([edge_queue[edge] for edge in self.global_edges], dtype=float).reshape(-1, 1)
			return x_vec
		except Exception as e:
			print(f"Error building global state vector: {e}")
			return None

	def _project_green_times(self, g_raw_j, tls_id):
		"""Chiếu vector g_raw_j vào tập ràng buộc:
		   - tổng G_i = C - L_j
		   - g_min <= G_i <= g_max
		Mục tiêu: min ||G - g_raw||^2.
		"""
		g_raw_j = np.asarray(g_raw_j, dtype=float).flatten()
		n = g_raw_j.size
		if n == 0:
			return g_raw_j
		g_min = float(self.min_phase)
		g_max = float(self.max_phase)
		C = float(self.cycle_time)
		L_j = float(self.loss_time.get(tls_id, 0.0))
		# 1. Tổng thời gian xanh khả dụng tại nút j
		G_total = max(C - L_j, 0.0)
		G_min = n * g_min
		G_max = n * g_max
		# Đảm bảo tổng nằm trong miền khả thi
		if G_total < G_min:
			G_total = G_min
		elif G_total > G_max:
			G_total = G_max
		# 2. Khởi tạo tập pha còn lại và G ban đầu
		G = g_raw_j.copy()
		F_remaining = list(range(n))
		max_iter = max(1, 2 * n)
		for _ in range(max_iter):
			if not F_remaining:
				break
			# A. Chênh lệch so với tổng mong muốn
			current_sum = float(G.sum())
			Difference = G_total - current_sum
			# nếu đã rất gần mục tiêu thì dừng
			if abs(Difference) < 1e-3:
				break
			N_remaining = len(F_remaining)
			if N_remaining <= 0:
				break
			shift = Difference / N_remaining
			# B. Phân bổ chênh lệch cho các pha chưa bị chặn
			for i in F_remaining:
				G[i] += shift
			# C. Kiểm tra vi phạm và khoá pha chạm ngưỡng
			violation_new = False
			new_F = []
			for i in F_remaining:
				if G[i] < g_min:
					G[i] = g_min
					violation_new = True
				elif G[i] > g_max:
					G[i] = g_max
					violation_new = True
				else:
					new_F.append(i)
			F_remaining = new_F
			if not violation_new:
				# Không có vi phạm mới, coi như đã hội tụ
				break
		return G

	def _update_control_plans(self, current_time):
		"""Thực hiện 1 chu kỳ điều khiển on-line TUC cho toàn mạng."""
		if self.L is None or not self.global_edges or not self.phase_mapping:
			return
		x_k = self._build_global_state_vector()
		if x_k is None:
			return
		if self.global_state_prev is None or self.global_state_prev.shape != x_k.shape:
			delta_x = np.zeros_like(x_k)
		else:
			delta_x = x_k - self.global_state_prev
		# Bước 2: Luật tăng trưởng Δg = -L·[x(k) − x(k−1)]
		try:
			delta_g = - self.L @ delta_x
			# áp dụng giới hạn tốc độ thay đổi nếu cấu hình
			if self.rate_limit is not None and self.rate_limit > 0:
				delta_g = np.clip(delta_g, -self.rate_limit, self.rate_limit)
		except Exception as e:
			print(f"Error computing LQ feedback: {e}")
			return
		# g_raw = g_applied(k-1) + Δg
		if self.prev_global_green is None or self.prev_global_green.shape != delta_g.shape:
			g_prev = np.zeros_like(delta_g)
		else:
			g_prev = self.prev_global_green
		g_raw = g_prev + delta_g
		g_applied_global = np.zeros_like(g_raw)
		# Bước 3: với mỗi nút giao j, chiếu g_raw_j vào ràng buộc để tìm G_j
		for tls_id in FIXED_TIME_PLANS:
			phase_order = self.phase_order_per_tls.get(tls_id, [])
			if not phase_order:
				continue
			indices = []
			for local_phase in phase_order:
				col = self.phase_mapping.get((tls_id, local_phase), None)
				if col is not None:
					indices.append(col)
			if not indices:
				continue
			g_raw_j = np.array([g_raw[idx, 0] for idx in indices], dtype=float)
			G_j = self._project_green_times(g_raw_j, tls_id)
			for idx_local, col in enumerate(indices):
				if 0 <= col < g_applied_global.shape[0]:
					g_applied_global[col, 0] = float(G_j[idx_local])
			# xây dựng lại kế hoạch pha hiện tại cho nút giao này
			plan_template = FIXED_TIME_PLANS[tls_id]
			new_plan = []
			green_idx = 0
			for duration, state in plan_template:
				if self._is_green_phase(tls_id, state) and green_idx < len(G_j):
					#print(f"G_j[{green_idx}] = {G_j[green_idx]}, state={state}")
					g_phase = float(G_j[green_idx])
					new_plan.append((g_phase, state))
					green_idx += 1
				else:
					# giữ nguyên pha vàng/mất thời gian
					new_plan.append((duration, state))
			self.current_plan[tls_id] = new_plan
			self.cycle_time_tls[tls_id] = sum(d for d, _ in new_plan)
		# Bước 4: lưu trạng thái làm dữ liệu đầu vào cho chu kỳ kế tiếp
		self.global_state_prev = x_k
		#print(f"g_applied_global={g_applied_global}")
		#print(FIXED_TIME_PLANS)
		self.prev_global_green = g_applied_global
		self.last_update_time = current_time

	def apply_control(self, tls_id, current_time, **kwargs):
		"""Thực hiện điều khiển TUC on-line và đặt trạng thái đèn cho nút giao."""
		if tls_id not in FIXED_TIME_PLANS:
			return
		# nếu chưa có L hoặc tham số mạng, quay về điều khiển fixed-time
		if self.L is None or not self.current_plan.get(tls_id):
			try:
				plan = FIXED_TIME_PLANS[tls_id]
				cycle_duration = sum(d for d, _ in plan)
				elapsed = (current_time - self.cycle_start[tls_id]) % cycle_duration
				pos = 0.0
				for duration, state in plan:
					if pos + duration > elapsed:
						traci.trafficlight.setRedYellowGreenState(tls_id, state)
						return
					pos += duration
			except Exception as e:
				print(f"Error applying fixed-time fallback for {tls_id}: {e}")
			return
		# cập nhật kế hoạch điều khiển toàn mạng mỗi khoảng thời gian T
		if self.last_update_time is None or (current_time - self.last_update_time) >= self.control_period:
			self._update_control_plans(current_time)
		plan = self.current_plan.get(tls_id, FIXED_TIME_PLANS[tls_id])
		cycle_duration = self.cycle_time_tls.get(tls_id, sum(d for d, _ in plan))
		if cycle_duration <= 0:
			return
		elapsed = (current_time - self.cycle_start[tls_id]) % cycle_duration
		current_position = 0.0
		for duration, state in plan:
			if current_position + duration > elapsed:
				try:
					traci.trafficlight.setRedYellowGreenState(tls_id, state)
				except Exception as e:
					print(f"Error setting state for {tls_id}: {e}")
				return
			current_position += duration
		
	def reset(self, tls_id, current_time):
		self.cycle_start[tls_id] = current_time
