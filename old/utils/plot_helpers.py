#plot_helpers.py
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def _warn_missing_overlay(labels, overlay_dict, name="Overlay"):
    if overlay_dict is None:
        return
    for label in labels:
        if label not in overlay_dict or overlay_dict[label] is None or (
            isinstance(overlay_dict[label], pd.DataFrame) and overlay_dict[label].empty
        ):
            print(f"[WARN] {name} files not found for: {label}")

def annotate_metadata_box(ax, forces_file):
    metadata = {}
    try:
        with open(forces_file, "r") as f:
            for line in f:
                if "Mach number" in line:
                    metadata['Mach'] = float(line.split()[-1])
                elif "Reynolds number" in line:
                    metadata['Re'] = float(line.split()[-1])
                elif "Angle of attack" in line:
                    metadata['AoA'] = float(line.split()[-1])
        textstr = f"M = {metadata['Mach']:.2f}\nRe = {metadata['Re']:.1e}\nAoA = {metadata['AoA']:.1f}°"
        props = dict(boxstyle='round', facecolor='white', alpha=0.75)
        ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
    except Exception as e:
        print(f"[WARN] Could not annotate metadata: {e}")
