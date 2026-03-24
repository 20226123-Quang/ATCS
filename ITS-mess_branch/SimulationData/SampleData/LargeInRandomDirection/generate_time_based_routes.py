"""
Generate SUMO route files with time-based traffic density
Different routes will be crowded during different time intervals
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import random


class TimeBasedRouteGenerator:
	def __init__(self, network_file="network.net.xml"):
		"""
		Initialize the route generator
		
		Args:
			network_file: Path to the SUMO network file
		"""
		self.network_file = network_file
		self.edges = self.extract_edges_from_network()
		
	def extract_edges_from_network(self):
		"""Extract available edges from network file"""
		try:
			tree = ET.parse(self.network_file)
			root = tree.getroot()
			edges = []
			
			for edge in root.findall('.//edge'):
				edge_id = edge.get('id')
				# Skip internal edges (junctions)
				if edge_id and not edge_id.startswith(':'):
					edges.append(edge_id)
			
			print(f"Found {len(edges)} edges in network")
			return edges
		except Exception as e:
			print(f"Error reading network file: {e}")
			return []
		
	def get_valid_routes(self):
		"""Get valid routes from incoming edges to outgoing edges, avoiding right turns"""
		try:
			tree = ET.parse(self.network_file)
			root = tree.getroot()
			
			# Build connection map
			connections = {}
			
			for edge in root.findall('.//edge'):
				edge_id = edge.get('id')
				if edge_id and not edge_id.startswith(':'):
					if edge_id not in connections:
						connections[edge_id] = set()
			
			# Find connections between edges through junctions
			for edge in root.findall('.//edge'):
				edge_id = edge.get('id')
				if edge_id and not edge_id.startswith(':'):
					to_node = edge.get('to')
					
					# Find all edges starting from this node
					for other_edge in root.findall('.//edge'):
						other_id = other_edge.get('id')
						if other_id and not other_id.startswith(':'):
							from_node = other_edge.get('from')
							if from_node == to_node and edge_id != other_id:
								connections[edge_id].add(other_id)
			
			# Identify incoming edges (start with '-') and outgoing edges (don't start with '-')
			incoming_edges = [edge for edge in connections.keys() if edge.startswith('-')]
			outgoing_edges = [edge for edge in connections.keys() if not edge.startswith('-')]
			
			print(f"Found {len(incoming_edges)} incoming edges: {incoming_edges}")
			print(f"Found {len(outgoing_edges)} outgoing edges: {outgoing_edges}")
			
			# Find paths from incoming to outgoing edges using BFS
			valid_routes = []
			
			for start_edge in incoming_edges:
				# BFS to find paths to outgoing edges
				queue = [(start_edge, [start_edge])]
				visited = {start_edge}
				
				while queue:
					current_edge, path = queue.pop(0)
					
					# Check if we reached an outgoing edge
					if current_edge in outgoing_edges and len(path) > 1:
						# Use first incoming edge and last outgoing edge as route
						valid_routes.append((path[0], path[-1]))
						continue  # Found one path, continue searching
					
					# Don't go too deep (max 4 edges in path)
					if len(path) >= 4:
						continue
					
					# Explore neighbors
					if current_edge in connections:
						for next_edge in connections[current_edge]:
							# Avoid opposite direction edges and already visited
							if next_edge not in visited:
								if not (current_edge.startswith('-') and next_edge == current_edge[1:]) and \
								   not (next_edge.startswith('-') and current_edge == next_edge[1:]):
									visited.add(next_edge)
									queue.append((next_edge, path + [next_edge]))
			
			# Remove duplicates
			valid_routes = list(set(valid_routes))
			
			print(f"Found {len(valid_routes)} valid routes from incoming to outgoing edges")
			return valid_routes
		except Exception as e:
			print(f"Error finding valid routes: {e}")
			return []
		
	def generate_routes_with_time_periods(self, 
										 time_periods,
										 output_file="time_based_routes.trips.xml",
										 base_flow=0.5,
										 crowded_flow=3,
										 routes_per_period=3,
										 period_duration=100):
		"""
		Generate route file with time-based crowding
		Only ONE route is crowded per time period, others have base flow
		
		Args:
			time_periods: List of dictionaries with time period info
			output_file: Output XML file name
			base_flow: Base number of vehicles per time unit for non-crowded routes
			crowded_flow: Number of vehicles per time unit for crowded routes
		"""
		root = ET.Element('routes')
		root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
		root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/routes_file.xsd')
		
		# Determine total simulation time
		max_time = max(period['end'] for period in time_periods)
		
		# Get all valid routes
		valid_routes = self.get_valid_routes()
		
		# Create a schedule: for each route, track when it should be crowded
		crowded_schedule = {}  # route -> [(start, end, flow), ...]
		for period in time_periods:
			route = period['route']
			if route not in crowded_schedule:
				crowded_schedule[route] = []
			crowded_schedule[route].append((period['start'], period['end'], period.get('flow', crowded_flow)))
		
		# Generate vehicles based on time periods
		vehicles = []
		vehicle_id = 0
		
		# Generate for each period
		num_periods = len(time_periods) // routes_per_period
		
		for period_idx in range(num_periods):
			period_start = period_idx * period_duration
			period_end = (period_idx + 1) * period_duration
			
			# Find which routes are crowded in this period
			crowded_routes_in_period = set()
			crowded_flows = {}
			for period in time_periods:
				if period['start'] == period_start and period['end'] == period_end:
					crowded_routes_in_period.add(period['route'])
					crowded_flows[period['route']] = period.get('flow', crowded_flow)
			
			# Generate vehicles for ALL routes in this period
			for route in valid_routes:
				from_edge, to_edge = route
				
				# Determine flow rate: crowded if in this period's crowded list, else base
				if route in crowded_routes_in_period:
					flow = crowded_flows[route]
				else:
					flow = base_flow
				
				current_time = period_start
				interval = 1.0 / flow
				
				print(f"Generating vehicles for route {from_edge} -> {to_edge} from {period_start}s to {period_end}s with flow {flow} veh/s (interval {interval:.2f}s)")	
				while current_time < period_end:
					# Add Gaussian noise to departure time
					noise = random.gauss(0, interval * 0.7)  # 10% standard deviation of interval
					depart_time = max(period_start, min(period_end - 0.01, current_time + noise))
					
					vehicles.append({
						'id': vehicle_id,
						'depart': depart_time,
						'from': from_edge,
						'to': to_edge
					})
					#print(f"Generated vehicle {vehicle_id} on route {from_edge} -> {to_edge} at time {current_time:.2f}s")
					vehicle_id += 1
					current_time = current_time + interval
		
		# Sort vehicles by depart time to avoid explosion effect
		vehicles.sort(key=lambda v: v['depart'])
		
		# Debug: check vehicles distribution by period
		for period_idx in range(len(time_periods) // routes_per_period):
			period_start = period_idx * period_duration
			period_end = (period_idx + 1) * period_duration
			period_vehicles = [v for v in vehicles if period_start <= v['depart'] < period_end]
			print(f"Period {period_idx} ({period_start}-{period_end}s): {len(period_vehicles)} vehicles")
		
		# Add sorted vehicles to XML
		for v in vehicles:
			trip = ET.SubElement(root, 'trip')
			trip.set('id', str(v['id']))
			trip.set('depart', f"{v['depart']:.2f}")
			trip.set('from', v['from'])
			trip.set('to', v['to'])
		
		# Write to file with pretty formatting
		xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="	")
		
		with open(output_file, 'w', encoding='utf-8') as f:
			f.write(xml_str)
		
		print(f"Generated {vehicle_id} vehicles in {output_file}")
		print(f"Simulation time: 0 - {max_time}s")
		for period in time_periods:
			print(f"  {period['start']}-{period['end']}s: Route {period['route'][0]} -> {period['route'][1]} (flow: {period.get('flow', crowded_flow)} veh/s)")


def main():
	"""Example usage"""
	generator = TimeBasedRouteGenerator("network.net.xml")
		
	# Get valid routes from network
	valid_routes = generator.get_valid_routes()
		
	if len(valid_routes) < 8:
		print("Error: Not enough valid routes in network")
		return
		
	# Randomly select routes for different time periods
	time_periods = []
	num_periods = 4  # Number of time periods
	period_duration = 200  # Duration of each period in seconds
	routes_per_period = 2  # Number of crowded routes per period
		
	# Shuffle and divide routes into groups for each period
	shuffled_routes = random.sample(valid_routes, len(valid_routes))
		
	for period_idx in range(num_periods):
		# Get routes for this period
		start_idx = period_idx * routes_per_period
		end_idx = start_idx + routes_per_period
		period_routes = shuffled_routes[start_idx:end_idx]
		
		for route in period_routes:
			time_periods.append({
				'start': period_idx * period_duration,
				'end': (period_idx + 1) * period_duration,
				'route': route
				# flow will use default crowded_flow parameter
			})
		
	if not time_periods:
		print("Error: Could not generate any time periods")
		return
		
	generator.generate_routes_with_time_periods(
		time_periods=time_periods,
		output_file="time_based_routes.trips.xml",
		base_flow=0.02,  # Background traffic: 0.05 vehicles per second
		crowded_flow=0.4,  # Crowded traffic: 0.5 vehicles per second
		routes_per_period=routes_per_period,
		period_duration=period_duration
	)
		
	print("\nRoute file generated successfully!")
	print("To convert to route file, run:")
	print("duarouter -n network.net.xml -t time_based_routes.trips.xml -o time_based_routes.rou.xml")


if __name__ == "__main__":
	main()
