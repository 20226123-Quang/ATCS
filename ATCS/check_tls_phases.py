import xml.etree.ElementTree as ET

net_root = ET.parse('SimulationData/SampleData/Crowded/network.net.xml').getroot()

def get_links(j_id):
    links = {}
    for c in net_root.findall('connection'):
        if c.get('tl') == j_id:
            links[int(c.get('linkIndex'))] = f"{c.get('from')} -> {c.get('to')} ({c.get('dir')})"
    return links
            
l1 = get_links('J1')
print('--- J1 ---')
for i in range(12):
    print(f"Link {i}: {l1.get(i, 'Unknown')}")

l3 = get_links('J3')
print('\n--- J3 ---')
for i in range(12):
    print(f"Link {i}: {l3.get(i, 'Unknown')}")
