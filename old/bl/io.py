"""
Very small I/O helpers for boundary-layer *.dat* tables.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

__all__ = [
    "write_bl_table",
    "read_bl_table",
    "scan_bl_dir_by_prefix",
]

_HEADER = 'VARIABLES = "x","value"\nZONE T="Boundary-Layer Data"\n'


# -----------------------------------------------------------------------------


def write_bl_table(path: Path, x, v):
    """Write two-column Tecplot file compatible with BL7 output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(_HEADER)
        for xv, vv in zip(np.asarray(x, float).ravel(),
                          np.asarray(v, float).ravel()):
            f.write(f"{float(xv):.6e} {float(vv):.6e}\n")


def read_bl_table(path: Path) -> pd.DataFrame:
    """Return *clean* dataframe or empty if unreadable."""
    try:
        df = pd.read_csv(path, delim_whitespace=True, comment="#",
                         names=["x", "y"], header=None, skiprows=2)
        return df.dropna()
    except Exception:
        return pd.DataFrame(columns=["x", "y"])


def scan_bl_dir_by_prefix(directory: Path):
    """
    Collect every BL-*.dat* file in *directory* and return a mapping:

    ``{"delta": {filename: DataFrame, …},
       "deltastar": { … }, … }``
    """
    kinds = ["delta", "deltastar", "theta", "h", "ue", "me"]
    out   = {k: {} for k in kinds}

    for f in Path(directory).glob("*.dat"):
        stem = f.stem.lower()
        for k in kinds:
            if stem.startswith(k):
                df = read_bl_table(f)
                if not df.empty:
                    out[k][stem] = df
    return out
