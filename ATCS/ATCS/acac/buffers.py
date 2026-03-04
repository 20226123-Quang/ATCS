"""Trajectory replay buffers for ACAC."""


class AsyncTrajectoryBuffer:
	def __init__(self, capacity, num_agents):
		self.capacity = capacity
		self.num_agents = num_agents
		self.buffers = {i: [] for i in range(num_agents)}

	def store(self, agent_id, entry):
		self.buffers[agent_id].append(entry)

	def get_agent_traj(self, agent_id):
		return self.buffers[agent_id]

	def clear(self):
		for i in range(self.num_agents):
			self.buffers[i].clear()


class SyncTrajectoryBuffer:
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffers = list()

	def store(self, entry):
		self.buffers.append(entry)

	def clear(self):
		self.buffers.clear()
