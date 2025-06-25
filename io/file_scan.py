
from pathlib import Path

def find_case_dirs(root):
    root = Path(root)
    return list(root.rglob('flow_surf_.dat'))
