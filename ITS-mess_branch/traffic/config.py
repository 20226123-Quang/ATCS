# Shared configuration and signal plan constants

# SUMO configuration
SUMO_BINARY = r"C:\Phi\Work\Simulation\SUMO\bin\sumo.exe"
SUMO_CONFIG_PATH = r"C:\Phi\Work\SimulationData\SampleData\HighPriorityVehicles\config.sumocfg"
SUMO_CMD = [SUMO_BINARY, "-c", SUMO_CONFIG_PATH]

# Priority control parameters
LEADING_CAR_ID = "t_0"  # LEADING_CAR vehicle ID from routes file
PRIORITY_DURATION = 100  # Duration to keep priority (seconds)
DETECTION_DISTANCE = 50  # Distance to detect approaching vehicle (meters)

# Speed threshold for traffic metrics (when vehicle is considered stopped/queued)
VEHICLE_STOP_THRESHOLD = 0.1  # Speed threshold to detect vehicle stopping (m/s)

# Vehicle type to PCU mapping (Passenger Car Units)
PCU_MAPPING = {
    'passenger': 1.0,
    'truck': 2.0,
    'bus': 2.5,
    'motorcycle': 0.5,
    'bicycle': 0.2,
}

# Controlled area edges for travel time calculation


# Unified SUMO net file loader
import xml.etree.ElementTree as ET
import os

def load_sumonet_root(sumocfg_path=None):
    """
    Loads and returns the root of the SUMO net file as an ElementTree.Element.
    Returns None if not found or error.
    """
    if sumocfg_path is None:
        sumocfg_path = SUMO_CONFIG_PATH
    if not os.path.exists(sumocfg_path):
        return None
    try:
        tree = ET.parse(sumocfg_path)
        root = tree.getroot()
        net_file = None
        for input_elem in root.findall('input'):
            for net_elem in input_elem.findall('net-file'):
                net_file = net_elem.get('value')
        if not net_file:
            for net_elem in root.findall('.//net-file'):
                net_file = net_elem.get('value')
        if not net_file:
            return None
        net_file_path = os.path.join(os.path.dirname(sumocfg_path), net_file)
        if not os.path.exists(net_file_path):
            return None
        net_tree = ET.parse(net_file_path)
        return net_tree.getroot()
    except Exception:
        return None

def get_controlled_edges_from_sumocfg(sumocfg_path=None):
    """
    Parse SUMO net file to get all edge IDs.
    Returns a list of edge IDs, or None if not found.
    """
    net_root = load_sumonet_root(sumocfg_path)
    if net_root is None:
        return None
    edge_ids = []
    for edge in net_root.findall('edge'):
        eid = edge.get('id')
        if eid and not edge.get('function'):
            edge_ids.append(eid)
    return edge_ids

CONTROLLED_EDGES = get_controlled_edges_from_sumocfg()

# Adaptive TL_STATES loader
def get_tls_states_from_net(sumocfg_path=None):
    """
    Parse SUMO net file to find traffic light logic and controlled edges.
    Returns a dict: {tls_id: {edge_id: [indices]}}
    """
    net_root = load_sumonet_root(sumocfg_path)
    if net_root is None:
        return None
    tls_states = {}
    for tl in net_root.findall('tlLogic'):
        tls_id = tl.get('id')
        edge_map = {}
        for conn in net_root.findall('connection'):
            if conn.get('tl') == tls_id:
                from_edge = conn.get('from')
                idx = int(conn.get('linkIndex')) if conn.get('linkIndex') else None
                if from_edge:
                    if from_edge not in edge_map:
                        edge_map[from_edge] = []
                    if idx is not None:
                        edge_map[from_edge].append(idx)
        if edge_map:
            tls_states[tls_id] = edge_map
    return tls_states

TL_STATES = get_tls_states_from_net()

# Fixed-time signal plans for each intersection: 4 green phases, each followed by a yellow phase
FIXED_TIME_PLANS = {
    "J1": [
        (27.5, "grrgGrgrrgGr"),   # North-South through (green)
        (3,    "grrgyrgrrgyr"),   # Yellow after NS through
        (27.5, "grrgrGgrrgrG"),   # North-South left turn (green)
        (3,    "grrgrygrrgry"),   # Yellow after NS left
        (27.5, "gGrgrrgGrgrr"),   # East-West through (green)
        (3,    "gyrgrrgyrgrr"),   # Yellow after EW through
        (27.5, "grGgrrgrGgrr"),   # East-West left turn (green)
        (3,    "grygrrgrygrr"),   # Yellow after EW left
    ],
    "J3": [
        (27.5, "grrgGrgrrgGr"),
        (3,    "grrgyrgrrgyr"),
        (27.5, "grrgrGgrrgrG"),
        (3,    "grrgrygrrgry"),
        (27.5, "gGrgrrgGrgrr"),
        (3,    "gyrgrrgyrgrr"),
        (27.5, "grGgrrgrGgrr"),
        (3,    "grygrrgrygrr"),
    ]
}

def get_fixed_time_plans_from_net(sumocfg_path=None):
    """
    Parse SUMO net file to extract fixed-time signal plans for each traffic light.
    Returns a dict: {tls_id: [(duration, state_string), ...]}
    """
    net_root = load_sumonet_root(sumocfg_path)
    if net_root is None:
        return None
    plans = {}
    for tl in net_root.findall('tlLogic'):
        tls_id = tl.get('id')
        phases = []
        for phase in tl.findall('phase'):
            duration = float(phase.get('duration')) if phase.get('duration') else None
            state = phase.get('state')
            if duration is not None and state:
                phases.append((duration, state))
        if phases:
            plans[tls_id] = phases
    return plans
