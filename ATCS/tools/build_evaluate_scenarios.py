from __future__ import annotations

import math
import random
import shutil
import subprocess
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


def write_sumocfg(target_dir: Path) -> None:
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

    time_elem = ET.SubElement(root, "time")
    ET.SubElement(time_elem, "begin", {"value": "0"})
    ET.SubElement(time_elem, "end", {"value": f"{SUMO_DURATION:.0f}"})

    report_elem = ET.SubElement(root, "report")
    ET.SubElement(report_elem, "verbose", {"value": "false"})
    ET.SubElement(report_elem, "no-step-log", {"value": "true"})

    indent_and_write(ET.ElementTree(root), target_dir / "config.sumocfg")


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
        build_plain_one_intersection_network(scenario_dir, name.lower(), angles_deg, approach_length)
        build_route_file_for_single_intersection(
            scenario_dir / "route.rou.xml",
            arm_count=len(angles_deg),
            vehicle_count=vehicle_count,
            seed=seed,
        )
        write_sumocfg(scenario_dir)


def prepare_few_scenarios() -> None:
    few_dir = EVALUATE_DIR / "Few"
    one_dir = EVALUATE_DIR / "OneIntersection" / "4Direction"
    few_one_dir = few_dir / "1Intersection"
    few_two_dir = few_dir / "2Intersection"

    few_one_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(one_dir / "network.net.xml", few_one_dir / "network.net.xml")
    write_sumocfg(few_one_dir)
    build_route_file_for_single_intersection(
        few_one_dir / "route.rou.xml",
        arm_count=4,
        vehicle_count=220,
        seed=401,
    )

    source_two_dir = EVALUATE_DIR / "Normal" / "2Intersection"
    few_two_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_two_dir / "network.net.xml", few_two_dir / "network.net.xml")
    write_sumocfg(few_two_dir)
    sample_existing_route_file(
        source_two_dir / "route.rou.xml",
        few_two_dir / "route.rou.xml",
        vehicle_count=320,
        seed=402,
    )


def normalize_all_route_files() -> None:
    route_files = sorted(EVALUATE_DIR.glob("**/route.rou.xml"))
    for index, route_path in enumerate(route_files):
        assign_vehicle_types(route_path, seed=700 + index)


def main() -> None:
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
    prepare_few_scenarios()

    for scenario_dir in sorted(EVALUATE_DIR.glob("**")):
        if scenario_dir.is_dir() and (scenario_dir / "network.net.xml").exists():
            write_sumocfg(scenario_dir)

    normalize_all_route_files()
    print("Evaluate scenarios rebuilt successfully.")


if __name__ == "__main__":
    main()
