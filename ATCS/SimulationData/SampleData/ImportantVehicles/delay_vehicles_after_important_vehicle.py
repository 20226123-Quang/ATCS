#!/usr/bin/env python3
"""Shift SUMO depart times to leave a gap after an important vehicle.

The script reads a routes file, finds the chosen vehicle (by id or by type),
identifies its departure edge (first edge), and pushes the departures of
vehicles that start on the same edge so that the earliest departure after the
important vehicle starts no sooner than ``important_depart + clearance``.
IMPORTANT_CAR vehicles are never delayed. Ordering and headways between shifted
vehicles are preserved.

Example:
	python delay_vehicles_after_important_vehicle.py random.rou.xml \
		LEADING_CAR --clearance 25
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


@dataclass
class DelayResult:
	file: Path
	output: Path
	important_depart: float
	shift_applied: float
	shifted_count: int


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Delay departures after an important vehicle in SUMO routes/trips "
			"files."
		)
	)
	parser.add_argument("route_file", type=Path, help="Path to the .rou.xml file")
	parser.add_argument(
		"important_id",
		help="Vehicle id or type to protect (must exist in the routes file)",
	)
	parser.add_argument(
		"--clearance",
		type=float,
		default=70.0,
		help="Seconds to keep clear after the important vehicle departs",
	)
	parser.add_argument(
		"--route-out",
		dest="route_out",
		type=Path,
		help="Optional output path for the updated routes file",
	)
	return parser.parse_args()


def parse_depart(element: ET.Element) -> float:
	value = element.get("depart")
	if value is None:
		raise ValueError("Found element without a 'depart' attribute")
	try:
		return float(value)
	except ValueError as exc:
		raise ValueError(f"Depart value '{value}' is not numeric") from exc



def find_important(elements: list[ET.Element], important_id: str) -> ET.Element | None:
	for element in elements:
		if element.get("id") == important_id:
			return element
	for element in elements:
		if element.get("type") == important_id:
			return element
	return None


def format_depart(value: float) -> str:
	return f"{value:.2f}"


def get_vehicle_edges(element: ET.Element) -> list[str]:
	"""Extract ordered list of edges from a vehicle or trip element."""
	edges_str = ""
	# Check for <route> child
	route_child = element.find("route")
	if route_child is not None:
		edges_str = route_child.get("edges", "")
	else:
		# Check for direct edges attribute (in <trip>)
		edges_str = element.get("edges", "")
		# Also check from/to/via for trips
		if not edges_str:
			parts = []
			if element.get("from"):
				parts.append(element.get("from"))
			if element.get("via"):
				parts.extend(element.get("via").split())
			if element.get("to"):
				parts.append(element.get("to"))
			edges_str = " ".join(parts)
	
	if not edges_str:
		return []
	return edges_str.split()


def starts_on_same_edge(vehicle: ET.Element, important: ET.Element) -> bool:
	"""Check if vehicle starts on the same edge as the important vehicle."""
	# Get the first edge (departure edge) of both vehicles
	vehicle_from = vehicle.get("from")
	important_from = important.get("from")
	
	# For trip elements, check from attribute
	if vehicle_from and important_from:
		return vehicle_from == important_from
	
	# For vehicle elements with route child, check first edge
	vehicle_edges = get_vehicle_edges(vehicle)
	important_edges = get_vehicle_edges(important)
	
	if not vehicle_edges or not important_edges:
		return False
	
	# Compare only the first edge (departure edge)
	return vehicle_edges[0] == important_edges[0]


def is_important_car(element: ET.Element) -> bool:
	"""Check if element is an IMPORTANT_CAR by id or type."""
	if element.get("id") == "IMPORTANT_CAR":
		return True
	if element.get("type") == "IMPORTANT_CAR":
		return True
	return False


def delay_departures(
	file_path: Path,
	output_path: Path,
	element_tags: tuple[str, ...],
	important_id: str,
	clearance: float,
) -> DelayResult:
	tree = ET.parse(file_path)
	root = tree.getroot()

	elements: list[ET.Element] = []
	for tag in element_tags:
		elements.extend(root.findall(tag))

	important = find_important(elements, important_id)
	if important is None:
		raise ValueError(
			f"Vehicle with id or type '{important_id}' not found in provided elements"
		)
	important_depart = parse_depart(important)

	departures = []
	for element in elements:
		depart_time = parse_depart(element)
		departures.append((element, depart_time))

	# Identify vehicles on the same departure edge after the important vehicle.
	# Exclude the important vehicle itself and IMPORTANT_CAR vehicles.
	candidates = [
		d for element, d in departures
		if element is not important
		and not is_important_car(element)
		and d >= important_depart
		and starts_on_same_edge(element, important)
	]
	if not candidates:
		tree.write(output_path, encoding="UTF-8", xml_declaration=True)
		return DelayResult(
			file=file_path,
			output=output_path,
			important_depart=important_depart,
			shift_applied=0.0,
			shifted_count=0,
		)

	earliest_following = min(candidates)
	shift = max(0.0, (important_depart + clearance) - earliest_following)

	shifted_count = 0
	if shift > 0:
		for element, depart_time in departures:
			if element is important or is_important_car(element):
				continue
			if not starts_on_same_edge(element, important):
				continue
			if depart_time >= earliest_following:
				element.set("depart", format_depart(depart_time + shift))
				shifted_count += 1

	# Sort all vehicles and trips by depart time to maintain SUMO's required order
	all_elements = []
	for tag in element_tags:
		all_elements.extend(root.findall(tag))
	
	# Sort by depart time
	all_elements.sort(key=lambda el: parse_depart(el))
	
	# Remove all vehicles/trips from root
	for tag in element_tags:
		for el in root.findall(tag):
			root.remove(el)
	
	# Re-add in sorted order
	for el in all_elements:
		root.append(el)

	tree.write(output_path, encoding="UTF-8", xml_declaration=True)
	return DelayResult(
		file=file_path,
		output=output_path,
		important_depart=important_depart,
		shift_applied=shift,
		shifted_count=shifted_count,
	)


def derive_output(input_path: Path, override: Path | None) -> Path:
	if override:
		return override
	# Handle .rou.xml compound extension
	if input_path.name.endswith(".rou.xml"):
		base_name = input_path.name[:-8]  # Remove .rou.xml
		return input_path.with_name(f"{base_name}_delayed.rou.xml")
	suffix = input_path.suffix
	return input_path.with_name(f"{input_path.stem}_delayed{suffix}")


def main() -> int:
	args = parse_args()
	route_out = derive_output(args.route_file, args.route_out)

	try:
		route_result = delay_departures(
			file_path=args.route_file,
			output_path=route_out,
			element_tags=("vehicle", "trip"),
			important_id=args.important_id,
			clearance=args.clearance,
		)
	except Exception as exc:  # pragma: no cover - CLI convenience
		print(f"Error: {exc}", file=sys.stderr)
		return 1

	print(
		"Routes: delayed {count} vehicles by {shift:.2f}s (important at {depart:.2f}s) -> {out}".format(
			count=route_result.shifted_count,
			shift=route_result.shift_applied,
			depart=route_result.important_depart,
			out=route_result.output,
		)
	)
	return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
	sys.exit(main())
