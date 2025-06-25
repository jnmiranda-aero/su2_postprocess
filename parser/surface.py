import pandas as pd
import re

def parse_surface_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find header line with VARIABLES =
    for i, line in enumerate(lines):
        if 'VARIABLES' in line:
            header_line = line
            start_idx = i + 1
            break

    header = re.findall(r'"(.*?)"', header_line)
    
    # Skip all non-numeric lines after header
    data_lines = []
    for line in lines[start_idx:]:
        if re.match(r'^[\s\d\.\-Ee+]+$', line):  # numeric line
            data_lines.append(line)

    data = [list(map(float, l.strip().split())) for l in data_lines if l.strip()]
    df = pd.DataFrame(data, columns=header)
    return df

