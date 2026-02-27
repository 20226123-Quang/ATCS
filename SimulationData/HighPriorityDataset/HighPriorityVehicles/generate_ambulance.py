"""
Script để random generate xe cứu thương (ambulance) từ file route SUMO
"""
import xml.etree.ElementTree as ET
import random
import argparse


def generate_ambulance_routes(input_route_file, output_route_file, ambulance_ratio=0.1, min_ambulances=5):
    """
    Generate ambulance vehicles từ file route gốc
    
    Args:
        input_route_file: File route gốc (.rou.xml)
        output_route_file: File route output với ambulance
        ambulance_ratio: Tỷ lệ xe cứu thương (0.1 = 10%)
        min_ambulances: Số lượng xe cứu thương tối thiểu
    """
    # Parse file route gốc
    tree = ET.parse(input_route_file)
    root = tree.getroot()
    
    # Lấy tất cả vehicles
    vehicles = root.findall('vehicle')
    total_vehicles = len(vehicles)
    
    # Tính số xe cứu thương cần tạo
    num_ambulances = max(min_ambulances, int(total_vehicles * ambulance_ratio))
    
    print(f"Tổng số xe: {total_vehicles}")
    print(f"Số xe cứu thương sẽ tạo: {num_ambulances}")
    
    # Random chọn vehicles để convert thành ambulance
    ambulance_indices = random.sample(range(total_vehicles), min(num_ambulances, total_vehicles))
    
    # Tạo vtype cho ambulance nếu chưa có
    vtype_exists = False
    for vtype in root.findall('vType'):
        if vtype.get('id') == 'ambulance':
            vtype_exists = True
            break
    
    if not vtype_exists:
        ambulance_vtype = ET.Element('vType', {
            'id': 'ambulance',
            'vClass': 'emergency',
            'color': '1,0,0',  # Màu đỏ
            'speedFactor': '1.3',  # Nhanh hơn 30%
            'speedDev': '0.1',
            'sigma': '0',
            'minGap': '2.0',
            'accel': '3.0',
            'decel': '5.0',
            'length': '7.5',
            'width': '2.4'
        })
        # Insert vType vào đầu root
        root.insert(0, ambulance_vtype)
    
    # Convert selected vehicles thành ambulance
    ambulance_count = 0
    for idx in ambulance_indices:
        vehicle = vehicles[idx]
        original_id = vehicle.get('id')
        
        # Update vehicle properties
        vehicle.set('type', 'ambulance')
        vehicle.set('id', f'ambulance_{ambulance_count}')
        
        # Thêm color để dễ nhận diện
        vehicle.set('color', '1,0,0')
        
        ambulance_count += 1
        print(f"Converted vehicle {original_id} -> ambulance_{ambulance_count-1}")
    
    # Ghi file output
    tree.write(output_route_file, encoding='UTF-8', xml_declaration=True)
    print(f"\n✓ Đã tạo file {output_route_file} với {ambulance_count} xe cứu thương")


def generate_ambulance_from_scratch(output_route_file, network_file, num_ambulances=10, 
                                    begin_time=0, end_time=400, random_depart=True):
    """
    Generate ambulance vehicles hoàn toàn mới (không dựa vào route có sẵn)
    
    Args:
        output_route_file: File route output
        network_file: File network để lấy edges
        num_ambulances: Số xe cứu thương cần tạo
        begin_time: Thời gian bắt đầu
        end_time: Thời gian kết thúc
        random_depart: Random thời gian xuất phát
    """
    # Parse network file để lấy edges
    net_tree = ET.parse(network_file)
    net_root = net_tree.getroot()
    
    # Lấy tất cả edges (không lấy internal edges)
    edges = []
    for edge in net_root.findall('edge'):
        edge_id = edge.get('id')
        if edge_id and not edge_id.startswith(':'):
            edges.append(edge_id)
    
    if len(edges) < 2:
        print("Lỗi: Không đủ edges trong network")
        return
    
    print(f"Tìm thấy {len(edges)} edges trong network")
    
    # Tạo root element
    root = ET.Element('routes')
    root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/routes_file.xsd')
    
    # Tạo vType cho ambulance
    ambulance_vtype = ET.SubElement(root, 'vType', {
        'id': 'ambulance',
        'vClass': 'emergency',
        'color': '1,0,0',
        'speedFactor': '1.3',
        'speedDev': '0.1',
        'sigma': '0',
        'minGap': '2.0',
        'accel': '3.0',
        'decel': '5.0',
        'length': '7.5',
        'width': '2.4'
    })
    
    # Generate ambulances
    for i in range(num_ambulances):
        # Random depart time
        if random_depart:
            depart = round(random.uniform(begin_time, end_time), 2)
        else:
            depart = begin_time + (end_time - begin_time) * i / num_ambulances
        
        # Random chọn 2 edges khác nhau để làm origin và destination
        from_edge, to_edge = random.sample(edges, 2)
        
        # Tạo vehicle
        vehicle = ET.SubElement(root, 'vehicle', {
            'id': f'ambulance_{i}',
            'type': 'ambulance',
            'depart': f'{depart:.2f}',
            'color': '1,0,0'
        })
        
        # Tạo route
        route = ET.SubElement(vehicle, 'route', {
            'edges': f'{from_edge} {to_edge}'
        })
        
        print(f"Created ambulance_{i}: {from_edge} -> {to_edge} @ {depart:.2f}s")
    
    # Tạo tree và ghi file
    tree = ET.ElementTree(root)
    ET.indent(tree, space='    ')
    tree.write(output_route_file, encoding='UTF-8', xml_declaration=True)
    print(f"\n✓ Đã tạo file {output_route_file} với {num_ambulances} xe cứu thương mới")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ambulance vehicles for SUMO simulation')
    parser.add_argument('--mode', choices=['convert', 'generate'], default='convert',
                        help='convert: Chuyển vehicles có sẵn thành ambulance | generate: Tạo ambulance mới')
    parser.add_argument('--input', type=str, default='random.rou.xml',
                        help='Input route file (cho mode convert)')
    parser.add_argument('--output', type=str, default='ambulance.rou.xml',
                        help='Output route file')
    parser.add_argument('--network', type=str, default='network.net.xml',
                        help='Network file (cho mode generate)')
    parser.add_argument('--ratio', type=float, default=0.1,
                        help='Tỷ lệ xe cứu thương (0.1 = 10%) cho mode convert')
    parser.add_argument('--num', type=int, default=10,
                        help='Số xe cứu thương cho mode generate hoặc tối thiểu cho mode convert')
    parser.add_argument('--begin', type=float, default=0,
                        help='Thời gian bắt đầu (cho mode generate)')
    parser.add_argument('--end', type=float, default=400,
                        help='Thời gian kết thúc (cho mode generate)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed để tái tạo kết quả')
    
    args = parser.parse_args()
    
    # Set random seed nếu có
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed: {args.seed}")
    
    if args.mode == 'convert':
        generate_ambulance_routes(
            input_route_file=args.input,
            output_route_file=args.output,
            ambulance_ratio=args.ratio,
            min_ambulances=args.num
        )
    else:  # generate
        generate_ambulance_from_scratch(
            output_route_file=args.output,
            network_file=args.network,
            num_ambulances=args.num,
            begin_time=args.begin,
            end_time=args.end
        )
