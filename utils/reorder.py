#reorder.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.spatial import KDTree

def reorder_surface_nodes_from_elements(df, elements, zone_type='FELINESEG'):
    """
    Reorders surface nodes using element connectivity (FELINESEG).
    Arguments:
    ----------
    df : pd.DataFrame
        DataFrame with surface node coordinates (must have 'x', 'y')
    elements : list of (int, int)
        Connectivity list, each element is a 2-tuple of node indices
    zone_type : str
        Zone type (must be 'FELINESEG' for now)

    Returns:
    --------
    reordered_df : pd.DataFrame
        Node data reordered according to element connectivity
    """
    if len(elements) == 0 or zone_type != 'FELINESEG':
        return df.reset_index(drop=True)

    coords = df[['x', 'y']].to_numpy()
    start_node = elements[0][0]
    ordered_nodes = [start_node]
    current_node = start_node

    while len(ordered_nodes) < len(coords):
        found = False
        for elem in elements:
            if current_node in elem:
                next_node = elem[1] if elem[0] == current_node else elem[0]
                if next_node not in ordered_nodes:
                    ordered_nodes.append(next_node)
                    current_node = next_node
                    found = True
                    break
        if not found:
            break  # Stop early if stuck

    reordered_df = df.iloc[ordered_nodes].reset_index(drop=True)
    return reordered_df, ordered_nodes

def reorder_volume_nodes_to_surface(
    volume_nodes: np.ndarray,
    surface_xy_ordered: np.ndarray,
) -> np.ndarray:
    """
    Match each *ordered* surface node to the nearest (x, y) in the volume cloud
    and return the volume nodes in that same order.
    """
    kdtree = KDTree(volume_nodes[:, :2])
    _, idx = kdtree.query(surface_xy_ordered[:, :2])   # vectorised N×2 query
    return volume_nodes[idx]