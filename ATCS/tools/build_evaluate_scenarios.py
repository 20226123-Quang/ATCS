from __future__ import annotations

import math
import os
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATE_DIR = REPO_ROOT / "SimulationData" / "Evaluate"
SUMO_DURATION = 1800.0
LANE_WIDTH_OFFSET = 3.2

VEHICLE_TYPES = [
    (
        "car",
        {
            "vClass": "passenger",
            "length": "4.5",
            "maxSpeed": "16.67",
            "accel": "2.6",
            "decel": "4.5",
            "sigma": "0.5",
            "color": "0,120,255",
        },
    ),
    (
        "motorcycle",
        {
            "vClass": "motorcycle",
            "length": "2.0",
            "maxSpeed": "19.44",
            "accel": "3.1",
            "decel": "4.8",
            "sigma": "0.6",
            "color": "255,140,0",
        },
    ),
    (
        "truck",
        {
            "vClass": "truck",
            "length": "8.0",
            "maxSpeed": "11.11",
            "accel": "1.2",
            "decel": "4.0",
            "sigma": "0.45",
            "color": "150,150,150",
        },
    ),
    (
        "bus",
        {
            "vClass": "bus",
            "length": "11.0",
            "maxSpeed": "13.89",
            "accel": "1.1",
            "decel": "4.0",
            "sigma": "0.45",
            "color": "210,60,60",
        },
    ),
    (
        "car_van",
        {
            "vClass": "delivery",
            "length": "5.4",
            "maxSpeed": "15.28",
            "accel": "2.0",
            "decel": "4.2",
            "sigma": "0.5",
            "color": "80,80,210",
        },
    ),
]
VEHICLE_WEIGHTS = [0.44, 0.30, 0.12, 0.05, 0.09]
VEHICLE_TYPE_IDS = [vehicle_type_id for vehicle_type_id, _ in VEHICLE_TYPES]
TWO_INTERSECTION_COUNTS = {
    "Crowded": 2493,
    "Normal": 1232,
    "Few": 320,
}
DIRECTIONAL_BASE_COUNTS = {
    "3Direction": 720,
    "4Direction": 960,
    "5Direction": 1200,
}
DIRECTIONAL_DENSITY_MULTIPLIERS = {
    "Few": 0.25,
    "Normal": 1.0,
    "Crowded": 2.0,
}


def indent_and_write(tree: ET.ElementTree, path: Path) -> None:
    ET.indent(tree, space="    ")
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def add_vehicle_types(root: ET.Element) -> None:
    for elem in list(root.findall("vType")):
        root.remove(elem)

    insert_index = 0
    for vehicle_type_id, attrs in VEHICLE_TYPES:
        root.insert(insert_index, ET.Element("vType", {"id": vehicle_type_id, **attrs}))
        insert_index += 1


def assign_vehicle_types(route_path: Path, seed: int) -> None:
    tree = ET.parse(route_path)
    root = tree.getroot()
    add_vehicle_types(root)

    rng = random.Random(seed)
    for vehicle in root.findall("vehicle"):
        vehicle.set("type", rng.choices(VEHICLE_TYPE_IDS, weights=VEHICLE_WEIGHTS, k=1)[0])

    indent_and_write(tree, route_path)


def write_sumocfg(
    target_dir: Path,
    config_name: str = "config.sumocfg",
    additional_files: list[str] | None = None,
) -> None:
    root = ET.Element(
        "sumoConfiguration",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/sumoConfiguration.xsd",
        },
    )
    input_elem = ET.SubElement(root, "input")
    ET.SubElement(input_elem, "net-file", {"value": "network.net.xml"})
    ET.SubElement(input_elem, "route-files", {"value": "route.rou.xml"})
    if additional_files:
        ET.SubElement(input_elem, "additional-files", {"value": ",".join(additional_files)})

    time_elem = ET.SubElement(root, "time")
    ET.SubElement(time_elem, "begin", {"value": "0"})
    ET.SubElement(time_elem, "end", {"value": f"{SUMO_DURATION:.0f}"})

    report_elem = ET.SubElement(root, "report")
    ET.SubElement(report_elem, "verbose", {"value": "false"})
    ET.SubElement(report_elem, "no-step-log", {"value": "true"})

    indent_and_write(ET.ElementTree(root), target_dir / config_name)


def resolve_random_trips_script() -> str:
    random_trips_path = shutil.which("randomTrips.py")
    if random_trips_path:
        return random_trips_path

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate = Path(sumo_home) / "tools" / "randomTrips.py"
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError("randomTrips.py was not found in PATH or SUMO_HOME/tools.")


def collect_route_edges(route_path: Path) -> list[str]:
    root = ET.parse(route_path).getroot()
    route_edges: list[str] = []

    for vehicle in root.findall("vehicle"):
        route = vehicle.find("route")
        if route is not None and route.get("edges"):
            route_edges.append(route.get("edges"))

    if route_edges:
        return route_edges

    for route in root.findall("route"):
        if route.get("edges"):
            route_edges.append(route.get("edges"))

    if not route_edges:
        raise ValueError(f"No route edges were found in {route_path}.")
    return route_edges


def choose_route_subset(source_routes: list[str], vehicle_count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    if vehicle_count <= len(source_routes):
        return rng.sample(source_routes, k=vehicle_count)
    return [rng.choice(source_routes) for _ in range(vehicle_count)]


def build_route_file_from_edges(
    route_path: Path,
    route_edges_list: list[str],
    seed: int,
    id_prefix: str,
) -> None:
    rng = random.Random(seed)
    routes = list(route_edges_list)
    rng.shuffle(routes)

    root = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )
    headway = SUMO_DURATION / max(len(routes), 1)

    for index, route_edges in enumerate(routes):
        depart = min(index * headway + rng.uniform(0.0, 0.35 * headway), SUMO_DURATION - 0.1)
        vehicle = ET.SubElement(
            root,
            "vehicle",
            {
                "id": f"{id_prefix}_{index}",
                "depart": f"{depart:.2f}",
            },
        )
        ET.SubElement(vehicle, "route", {"edges": route_edges})

    indent_and_write(ET.ElementTree(root), route_path)


def generate_route_pool_from_network(
    network_path: Path,
    route_count: int,
    seed: int,
) -> list[str]:
    build_dir = network_path.parent / "_build_pool"
    build_dir.mkdir(parents=True, exist_ok=True)
    route_pool_path = build_dir / "seed_pool.rou.xml"
    period = SUMO_DURATION / max(route_count, 1)
    cmd = [
        sys.executable,
        resolve_random_trips_script(),
        "-n",
        str(network_path),
        "-r",
        str(route_pool_path),
        "--seed",
        str(seed),
        "--end",
        f"{SUMO_DURATION:.0f}",
        "--period",
        f"{period:.6f}",
        "--fringe-factor",
        "5",
        "--validate",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    route_edges = collect_route_edges(route_pool_path)
    shutil.rmtree(build_dir, ignore_errors=True)
    return route_edges


def build_nonconflict_phase_file(network_path: Path, output_path: Path) -> None:
    network_root = ET.parse(network_path).getroot()
    phase_root = ET.Element("additionals")

    for tl_logic in network_root.findall("tlLogic"):
        tls_id = tl_logic.get("id")
        if not tls_id:
            continue

        connections = []
        for connection in network_root.findall("connection"):
            if connection.get("tl") != tls_id:
                continue
            link_index = connection.get("linkIndex")
            from_edge = connection.get("from")
            if link_index is None or from_edge is None:
                continue
            connections.append((int(link_index), from_edge))

        if not connections:
            continue

        grouped_links: dict[str, list[int]] = {}
        for link_index, from_edge in connections:
            grouped_links.setdefault(from_edge, []).append(link_index)

        link_count = max(link_index for link_index, _ in connections) + 1
        phase_logic = ET.SubElement(
            phase_root,
            "tlLogic",
            {
                "id": tls_id,
                "type": "static",
                "programID": "fixed_nonconflict",
                "offset": "0",
            },
        )

        for _, link_indices in sorted(
            ((min(indices), indices) for indices in grouped_links.values()),
            key=lambda item: item[0],
        ):
            green_state = ["r"] * link_count
            yellow_state = ["r"] * link_count
            for link_index in sorted(set(link_indices)):
                green_state[link_index] = "G"
                yellow_state[link_index] = "y"

            ET.SubElement(phase_logic, "phase", {"state": "".join(green_state)})
            ET.SubElement(phase_logic, "phase", {"state": "".join(yellow_state)})

    indent_and_write(ET.ElementTree(phase_root), output_path)


def to_point(x: float, y: float) -> str:
    return f"{x:.2f},{y:.2f}"


def build_lane_shape(start: tuple[float, float], end: tuple[float, float], offset: float) -> tuple[str, float]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        raise ValueError("Zero-length edge is not allowed.")
    px = -dy / length
    py = dx / length
    sx = start[0] + px * offset
    sy = start[1] + py * offset
    ex = end[0] + px * offset
    ey = end[1] + py * offset
    return f"{to_point(sx, sy)} {to_point(ex, ey)}", length


def update_straight_edge(
    edge: ET.Element,
    positions: dict[str, tuple[float, float]],
    lane_offsets: list[float],
) -> None:
    edge_id = edge.get("id")
    from_node = edge.get("from")
    to_node = edge.get("to")
    if edge_id is None or from_node is None or to_node is None:
        return

    start = positions[from_node]
    end = positions[to_node]
    center_shape, edge_length = build_lane_shape(start, end, 0.0)
    edge.set("shape", center_shape)

    lanes = sorted(edge.findall("lane"), key=lambda lane: int(lane.get("index", "0")))
    for lane, offset in zip(lanes, lane_offsets):
        shape, length = build_lane_shape(start, end, offset)
        lane.set("shape", shape)
        lane.set("length", f"{length:.2f}")


def update_junction_positions(
    root: ET.Element,
    positions: dict[str, tuple[float, float]],
    conv_boundary: tuple[float, float, float, float],
) -> None:
    for junction in root.findall("junction"):
        junction_id = junction.get("id")
        if junction_id in positions:
            x, y = positions[junction_id]
            junction.set("x", f"{x:.2f}")
            junction.set("y", f"{y:.2f}")

    location = root.find("location")
    if location is not None:
        location.set(
            "convBoundary",
            f"{conv_boundary[0]:.2f},{conv_boundary[1]:.2f},{conv_boundary[2]:.2f},{conv_boundary[3]:.2f}",
        )


def restyle_two_intersection_network(network_path: Path) -> None:
    positions = {
        "J1": (0.0, 0.0),
        "J3": (1000.0, 0.0),
        "J0": (-500.0, 0.0),
        "J2": (0.0, 500.0),
        "J4": (0.0, -500.0),
        "J5": (1000.0, 500.0),
        "J6": (1500.0, 0.0),
        "J7": (1000.0, -500.0),
    }
    lane_offsets = {
        "E2": [-LANE_WIDTH_OFFSET, 0.0, LANE_WIDTH_OFFSET],
        "-E2": [LANE_WIDTH_OFFSET, 0.0, -LANE_WIDTH_OFFSET],
        "E0": [LANE_WIDTH_OFFSET, 0.0, -LANE_WIDTH_OFFSET],
        "-E0": [-LANE_WIDTH_OFFSET, 0.0, LANE_WIDTH_OFFSET],
        "E1": [LANE_WIDTH_OFFSET, 0.0, -LANE_WIDTH_OFFSET],
        "-E1": [-LANE_WIDTH_OFFSET, 0.0, LANE_WIDTH_OFFSET],
        "E3": [-LANE_WIDTH_OFFSET, 0.0, LANE_WIDTH_OFFSET],
        "-E3": [LANE_WIDTH_OFFSET, 0.0, -LANE_WIDTH_OFFSET],
        "E4": [LANE_WIDTH_OFFSET, 0.0, -LANE_WIDTH_OFFSET],
        "-E4": [-LANE_WIDTH_OFFSET, 0.0, LANE_WIDTH_OFFSET],
        "E5": [-LANE_WIDTH_OFFSET, 0.0, LANE_WIDTH_OFFSET],
        "-E5": [LANE_WIDTH_OFFSET, 0.0, -LANE_WIDTH_OFFSET],
        "E6": [-LANE_WIDTH_OFFSET, 0.0, LANE_WIDTH_OFFSET],
        "-E6": [LANE_WIDTH_OFFSET, 0.0, -LANE_WIDTH_OFFSET],
    }

    tree = ET.parse(network_path)
    root = tree.getroot()
    update_junction_positions(root, positions, (-500.0, -500.0, 1500.0, 500.0))

    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if edge.get("function") == "internal" or edge_id not in lane_offsets:
            continue
        update_straight_edge(edge, positions, lane_offsets[edge_id])

    indent_and_write(tree, network_path)


def restyle_three_intersection_network(network_path: Path) -> None:
    positions = {
        "J1": (0.0, 0.0),
        "J2": (450.0, 650.0),
        "J3": (900.0, 0.0),
        "J0": (-600.0, 0.0),
        "J4": (0.0, -600.0),
        "J6": (1500.0, 0.0),
        "J7": (900.0, -600.0),
        "J9": (1050.0, 1250.0),
        "J10": (-150.0, 1250.0),
    }
    lane_offsets = {
        "E0": [LANE_WIDTH_OFFSET, 0.0, -LANE_WIDTH_OFFSET],
        "-E0": [-LANE_WIDTH_OFFSET, 0.0, LANE_WIDTH_OFFSET],
        "E2": [-LANE_WIDTH_OFFSET, 0.0, LANE_WIDTH_OFFSET],
        "-E2": [LANE_WIDTH_OFFSET, 0.0, -LANE_WIDTH_OFFSET],
        "E3": [-LANE_WIDTH_OFFSET, 0.0, LANE_WIDTH_OFFSET],
        "-E3": [LANE_WIDTH_OFFSET, 0.0, -LANE_WIDTH_OFFSET],
        "E5": [-LANE_WIDTH_OFFSET, 0.0, LANE_WIDTH_OFFSET],
        "-E5": [LANE_WIDTH_OFFSET, 0.0, -LANE_WIDTH_OFFSET],
        "E6": [-LANE_WIDTH_OFFSET, 0.0, LANE_WIDTH_OFFSET],
        "-E6": [LANE_WIDTH_OFFSET, 0.0, -LANE_WIDTH_OFFSET],
        "E8": [LANE_WIDTH_OFFSET, 0.0, -LANE_WIDTH_OFFSET],
        "-E8": [-LANE_WIDTH_OFFSET, 0.0, LANE_WIDTH_OFFSET],
        "E9": [LANE_WIDTH_OFFSET, 0.0, -LANE_WIDTH_OFFSET],
        "-E9": [-LANE_WIDTH_OFFSET, 0.0, LANE_WIDTH_OFFSET],
        "E10": [LANE_WIDTH_OFFSET, 0.0, -LANE_WIDTH_OFFSET],
        "-E10": [-LANE_WIDTH_OFFSET, 0.0, LANE_WIDTH_OFFSET],
        "E11": [LANE_WIDTH_OFFSET, 0.0, -LANE_WIDTH_OFFSET],
        "-E11": [-LANE_WIDTH_OFFSET, 0.0, LANE_WIDTH_OFFSET],
    }

    tree = ET.parse(network_path)
    root = tree.getroot()
    update_junction_positions(root, positions, (-600.0, -600.0, 1500.0, 1250.0))

    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if edge.get("function") == "internal" or edge_id not in lane_offsets:
            continue
        update_straight_edge(edge, positions, lane_offsets[edge_id])

    indent_and_write(tree, network_path)


def run_netconvert(node_path: Path, edge_path: Path, output_path: Path) -> None:
    cmd = [
        "netconvert",
        "--node-files",
        str(node_path),
        "--edge-files",
        str(edge_path),
        "--output-file",
        str(output_path),
        "--offset.disable-normalization",
        "true",
        "--no-turnarounds",
        "true",
        "--junctions.corner-detail",
        "5",
        "--junctions.limit-turn-speed",
        "5.5",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def build_plain_one_intersection_network(
    target_dir: Path,
    scenario_name: str,
    angles_deg: list[float],
    approach_length: float,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    build_dir = target_dir / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    node_root = ET.Element("nodes")
    ET.SubElement(node_root, "node", {"id": "J1", "x": "0.0", "y": "0.0", "type": "traffic_light"})
    for index, angle_deg in enumerate(angles_deg):
        angle_rad = math.radians(angle_deg)
        x = math.cos(angle_rad) * approach_length
        y = math.sin(angle_rad) * approach_length
        ET.SubElement(
            node_root,
            "node",
            {"id": f"N{index}", "x": f"{x:.2f}", "y": f"{y:.2f}", "type": "priority"},
        )

    edge_root = ET.Element("edges")
    for index in range(len(angles_deg)):
        ET.SubElement(
            edge_root,
            "edge",
            {
                "id": f"in{index}",
                "from": f"N{index}",
                "to": "J1",
                "priority": "1",
                "numLanes": "3",
                "speed": "13.89",
            },
        )
        ET.SubElement(
            edge_root,
            "edge",
            {
                "id": f"out{index}",
                "from": "J1",
                "to": f"N{index}",
                "priority": "1",
                "numLanes": "3",
                "speed": "13.89",
            },
        )

    node_path = build_dir / f"{scenario_name}.nod.xml"
    edge_path = build_dir / f"{scenario_name}.edg.xml"
    indent_and_write(ET.ElementTree(node_root), node_path)
    indent_and_write(ET.ElementTree(edge_root), edge_path)

    run_netconvert(node_path, edge_path, target_dir / "network.net.xml")
    shutil.rmtree(build_dir, ignore_errors=True)


def build_route_file_for_single_intersection(
    route_path: Path,
    arm_count: int,
    vehicle_count: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    root = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )
    add_vehicle_types(root)

    headway = SUMO_DURATION / max(vehicle_count, 1)
    for index in range(vehicle_count):
        source = rng.randrange(arm_count)
        target = rng.randrange(arm_count - 1)
        if target >= source:
            target += 1

        base_depart = index * headway
        jitter = rng.uniform(-0.25 * headway, 0.25 * headway)
        depart = min(max(base_depart + jitter, 0.0), SUMO_DURATION - 0.1)

        vehicle = ET.SubElement(
            root,
            "vehicle",
            {
                "id": f"veh_{index}",
                "depart": f"{depart:.2f}",
                "type": rng.choices(VEHICLE_TYPE_IDS, weights=VEHICLE_WEIGHTS, k=1)[0],
            },
        )
        ET.SubElement(vehicle, "route", {"edges": f"in{source} out{target}"})

    indent_and_write(ET.ElementTree(root), route_path)


def build_route_file_for_existing_tls_network(
    network_path: Path,
    route_path: Path,
    vehicle_count: int,
    seed: int,
) -> None:
    network_root = ET.parse(network_path).getroot()
    route_options: dict[str, list[str]] = {}

    for connection in network_root.findall("connection"):
        if not connection.get("tl"):
            continue

        from_edge = connection.get("from")
        to_edge = connection.get("to")
        if not from_edge or not to_edge:
            continue
        if from_edge.startswith(":") or to_edge.startswith(":"):
            continue

        route_options.setdefault(from_edge, [])
        if to_edge not in route_options[from_edge]:
            route_options[from_edge].append(to_edge)

    if not route_options:
        raise ValueError(f"No valid TLS-controlled edge routes found in {network_path}")

    incoming_edges = sorted(route_options)
    rng = random.Random(seed)
    root = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )
    add_vehicle_types(root)

    headway = SUMO_DURATION / max(vehicle_count, 1)
    for index in range(vehicle_count):
        source_edge = rng.choice(incoming_edges)
        target_edge = rng.choice(route_options[source_edge])

        base_depart = index * headway
        jitter = rng.uniform(-0.25 * headway, 0.25 * headway)
        depart = min(max(base_depart + jitter, 0.0), SUMO_DURATION - 0.1)

        vehicle = ET.SubElement(
            root,
            "vehicle",
            {
                "id": f"veh_{index}",
                "depart": f"{depart:.2f}",
                "type": rng.choices(VEHICLE_TYPE_IDS, weights=VEHICLE_WEIGHTS, k=1)[0],
            },
        )
        ET.SubElement(vehicle, "route", {"edges": f"{source_edge} {target_edge}"})

    indent_and_write(ET.ElementTree(root), route_path)


def sample_existing_route_file(
    source_route_path: Path,
    target_route_path: Path,
    vehicle_count: int,
    seed: int,
) -> None:
    source_root = ET.parse(source_route_path).getroot()
    source_routes = []
    for vehicle in source_root.findall("vehicle"):
        route = vehicle.find("route")
        if route is not None and route.get("edges"):
            source_routes.append(route.get("edges"))

    rng = random.Random(seed)
    chosen_routes = rng.sample(source_routes, k=min(vehicle_count, len(source_routes)))
    chosen_routes.sort()

    root = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )
    add_vehicle_types(root)

    headway = SUMO_DURATION / max(len(chosen_routes), 1)
    for index, route_edges in enumerate(chosen_routes):
        depart = min(index * headway + rng.uniform(0.0, 0.35 * headway), SUMO_DURATION - 0.1)
        vehicle = ET.SubElement(
            root,
            "vehicle",
            {
                "id": f"few_{index}",
                "depart": f"{depart:.2f}",
                "type": rng.choices(VEHICLE_TYPE_IDS, weights=VEHICLE_WEIGHTS, k=1)[0],
            },
        )
        ET.SubElement(vehicle, "route", {"edges": route_edges})

    indent_and_write(ET.ElementTree(root), target_route_path)


def prepare_one_intersection_scenarios() -> None:
    one_dir = EVALUATE_DIR / "OneIntersection"
    scenarios = [
        ("3Direction", [180.0, 90.0, 0.0], 550.0, 720, 301),
        ("4Direction", [180.0, 90.0, 0.0, -90.0], 550.0, 960, 302),
        ("5Direction", [180.0, 108.0, 36.0, -36.0, -108.0], 580.0, 1200, 303),
    ]

    for name, angles_deg, approach_length, vehicle_count, seed in scenarios:
        scenario_dir = one_dir / name
        existing_osm_network = (scenario_dir / "map.osm").exists() and (scenario_dir / "network.net.xml").exists()
        if existing_osm_network:
            build_route_file_for_existing_tls_network(
                scenario_dir / "network.net.xml",
                scenario_dir / "route.rou.xml",
                vehicle_count=vehicle_count,
                seed=seed,
            )
        else:
            build_plain_one_intersection_network(scenario_dir, name.lower(), angles_deg, approach_length)
            build_route_file_for_single_intersection(
                scenario_dir / "route.rou.xml",
                arm_count=len(angles_deg),
                vehicle_count=vehicle_count,
                seed=seed,
            )
        write_sumocfg(scenario_dir)


def prepare_directional_density_scenarios() -> None:
    source_root = EVALUATE_DIR / "OneIntersection"
    density_seed_offsets = {"Few": 610, "Normal": 620, "Crowded": 630}

    for density_name, multiplier in DIRECTIONAL_DENSITY_MULTIPLIERS.items():
        target_root = EVALUATE_DIR / density_name
        for index, (direction_name, base_count) in enumerate(DIRECTIONAL_BASE_COUNTS.items()):
            source_dir = source_root / direction_name
            if not source_dir.exists():
                raise FileNotFoundError(f"Directional source scenario is missing: {source_dir}")

            target_dir = target_root / direction_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for source_path in source_dir.iterdir():
                if not source_path.is_file():
                    continue
                if source_path.name in {"route.rou.xml", "config.sumocfg"}:
                    continue
                shutil.copy2(source_path, target_dir / source_path.name)

            vehicle_count = max(1, int(round(base_count * multiplier)))
            build_route_file_for_existing_tls_network(
                target_dir / "network.net.xml",
                target_dir / "route.rou.xml",
                vehicle_count=vehicle_count,
                seed=density_seed_offsets[density_name] + index,
            )
            write_sumocfg(
                target_dir,
                additional_files=scenario_additional_files(target_dir),
            )


def prepare_two_intersection_scenarios_from_standard() -> bool:
    standard_dir = EVALUATE_DIR / "Crowded" / "2nut"
    standard_network = standard_dir / "2nutgiao.net.xml"
    standard_osm = standard_dir / "2nutgiao.osm"
    canonical_phase_template = (
        EVALUATE_DIR / "Crowded" / "2Intersection" / "2nutgiao_fixedtime(2).tll.xml"
    )
    if not standard_network.exists():
        return False

    scenario_dirs = {
        "Crowded": EVALUATE_DIR / "Crowded" / "2Intersection",
        "Normal": EVALUATE_DIR / "Normal" / "2Intersection",
        "Few": EVALUATE_DIR / "Few" / "2Intersection",
    }

    for scenario_dir in scenario_dirs.values():
        scenario_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(standard_network, scenario_dir / "network.net.xml")
        if standard_osm.exists():
            shutil.copy2(standard_osm, scenario_dir / "2nutgiao.osm")
        scenario_phase_template = scenario_dir / "2nutgiao_fixedtime(2).tll.xml"
        if canonical_phase_template.exists() and canonical_phase_template.resolve() != scenario_phase_template.resolve():
            shutil.copy2(canonical_phase_template, scenario_phase_template)
        phase_config_path = scenario_dir / "2nutgiao.fixedtime.ttl.xml"
        if not phase_config_path.exists():
            build_nonconflict_phase_file(
                scenario_dir / "network.net.xml",
                phase_config_path,
            )
        write_sumocfg(
            scenario_dir,
            additional_files=scenario_additional_files(scenario_dir),
        )

    pool_target = max(TWO_INTERSECTION_COUNTS.values()) + 900
    route_pool = generate_route_pool_from_network(standard_network, route_count=pool_target, seed=451)
    if len(route_pool) < TWO_INTERSECTION_COUNTS["Crowded"]:
        raise RuntimeError(
            "The generated canonical route pool is smaller than the crowded 2Intersection demand target."
        )

    crowded_routes = choose_route_subset(route_pool, TWO_INTERSECTION_COUNTS["Crowded"], seed=452)
    normal_routes = choose_route_subset(crowded_routes, TWO_INTERSECTION_COUNTS["Normal"], seed=453)
    few_routes = choose_route_subset(normal_routes, TWO_INTERSECTION_COUNTS["Few"], seed=454)

    build_route_file_from_edges(
        scenario_dirs["Crowded"] / "route.rou.xml",
        crowded_routes,
        seed=461,
        id_prefix="crowded2",
    )
    build_route_file_from_edges(
        scenario_dirs["Normal"] / "route.rou.xml",
        normal_routes,
        seed=462,
        id_prefix="normal2",
    )
    build_route_file_from_edges(
        scenario_dirs["Few"] / "route.rou.xml",
        few_routes,
        seed=463,
        id_prefix="few2",
    )
    return True


def prepare_few_scenarios(include_two_intersection: bool = True) -> None:
    few_dir = EVALUATE_DIR / "Few"
    one_dir = EVALUATE_DIR / "OneIntersection" / "4Direction"
    few_one_dir = few_dir / "1Intersection"

    few_one_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(one_dir / "network.net.xml", few_one_dir / "network.net.xml")
    write_sumocfg(few_one_dir)
    build_route_file_for_single_intersection(
        few_one_dir / "route.rou.xml",
        arm_count=4,
        vehicle_count=220,
        seed=401,
    )

    if include_two_intersection:
        few_two_dir = few_dir / "2Intersection"
        source_two_dir = EVALUATE_DIR / "Normal" / "2Intersection"
        few_two_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_two_dir / "network.net.xml", few_two_dir / "network.net.xml")
        write_sumocfg(few_two_dir)
        sample_existing_route_file(
            source_two_dir / "route.rou.xml",
            few_two_dir / "route.rou.xml",
            vehicle_count=TWO_INTERSECTION_COUNTS["Few"],
            seed=402,
        )


def normalize_all_route_files() -> None:
    route_files = sorted(EVALUATE_DIR.glob("**/route.rou.xml"))
    for index, route_path in enumerate(route_files):
        assign_vehicle_types(route_path, seed=700 + index)


def scenario_additional_files(target_dir: Path) -> list[str] | None:
    # Scenario-local *.fixedtime.ttl.xml is consumed by the ATCS environment
    # as a phase-state template, not by SUMO as an additional-file.
    return None


def main() -> None:
    prepared_from_standard = prepare_two_intersection_scenarios_from_standard()
    if not prepared_from_standard:
        for network_path in [
            EVALUATE_DIR / "Crowded" / "2Intersection" / "network.net.xml",
            EVALUATE_DIR / "Normal" / "2Intersection" / "network.net.xml",
        ]:
            restyle_two_intersection_network(network_path)

    for network_path in [
        EVALUATE_DIR / "Crowded" / "3Intersection" / "network.net.xml",
        EVALUATE_DIR / "Normal" / "3Intersection" / "network.net.xml",
    ]:
        restyle_three_intersection_network(network_path)

    prepare_one_intersection_scenarios()
    prepare_directional_density_scenarios()
    prepare_few_scenarios(include_two_intersection=not prepared_from_standard)

    for scenario_dir in sorted(EVALUATE_DIR.glob("**")):
        if scenario_dir.is_dir() and (scenario_dir / "network.net.xml").exists():
            write_sumocfg(
                scenario_dir,
                additional_files=scenario_additional_files(scenario_dir),
            )

    normalize_all_route_files()
    print("Evaluate scenarios rebuilt successfully.")


if __name__ == "__main__":
    main()
