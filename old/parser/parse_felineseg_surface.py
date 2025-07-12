
import pandas as pd
import re

def parse_felineseg_surface(filepath):
    """
    Parses a Tecplot ASCII file with ZONETYPE=FELINESEG and returns:
    - DataFrame of node data
    - List of connectivity elements (start_idx, end_idx)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # --- Step 1: Find VARIABLES and ZONE line ---
    var_line = next(i for i, line in enumerate(lines) if "VARIABLES" in line)
    var_names = re.findall(r'"(.*?)"', lines[var_line])

    zone_line = next(i for i, line in enumerate(lines) if "ZONE" in line and "FELINESEG" in line)
    zone_info = lines[zone_line]

    # --- Step 2: Extract counts from zone declaration ---
    n_nodes = int(re.search(r'NODES\s*=\s*(\d+)', zone_info).group(1))
    n_elems = int(re.search(r'ELEMENTS\s*=\s*(\d+)', zone_info).group(1))

    # --- Step 3: Read node data ---
    node_data_start = zone_line + 1
    node_data = [
        list(map(float, lines[i].strip().split()))
        for i in range(node_data_start, node_data_start + n_nodes)
    ]
    df = pd.DataFrame(node_data, columns=var_names)

    # --- Step 4: Read element connectivity ---
    element_data = [
        tuple(int(i) - 1 for i in lines[i].strip().split())  # convert to 0-based
        for i in range(node_data_start + n_nodes, node_data_start + n_nodes + n_elems)
    ]

    return df, element_data
