# su2_postprocess/bl/extract_new.py
from __future__ import annotations
from pathlib import Path
import time, logging
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import NearestNDInterpolator
from .data_classes import BLResult

# configure logging
target = Path("mesh_parser_debug.log")
logging.basicConfig(
    filename=target,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def march_bl7(
    profile_fn: callable,
    surface_pt: np.ndarray,
    normal: np.ndarray,
    threshold: float,
    ds: float,
    max_s: float
) -> tuple[float, float]:
    """
    BL7 marching: step by ds along normal from surface until vorticity
    drops below threshold fraction of reference.
    Returns (edge_thickness, edge_velocity).
    """
    # offset into flow to establish reference
    offset = surface_pt + normal * ds
    V0, w0 = profile_fn(offset)
    if np.isnan(w0) or abs(w0) < 1e-12:
        return np.nan, np.nan
    s = ds
    while s < max_s:
        pt = surface_pt + normal * s
        V, w = profile_fn(pt)
        if not np.isnan(w) and abs(w / w0) <= threshold:
            return s, V
        s += ds
    return np.nan, np.nan


def compute_bl_node(
    idx: int,
    surf_pt: np.ndarray,
    normal: np.ndarray,
    vel_interp: NearestNDInterpolator,
    vor_interp: NearestNDInterpolator,
    threshold: float,
    ds: float,
    max_s: float
) -> tuple[int, float, float, float, float]:
    """
    Compute BL at single node via BL7: returns
    (index, delta, delta_star, theta, edge_vel).
    """
    def profile(p: np.ndarray) -> tuple[float, float]:
        return vel_interp(p), vor_interp(p)

    s_edge, Ue = march_bl7(profile, surf_pt, normal, threshold, ds, max_s)
    if np.isnan(s_edge) or s_edge <= ds:
        return idx, np.nan, np.nan, np.nan, np.nan

    # integrate thicknesses
    s_vals = np.linspace(0.0, s_edge, 1000)
    pts = surf_pt + np.outer(s_vals, normal)
    Vc = vel_interp(pts)
    ur = Vc / Ue
    delta_star = np.trapz(1 - ur, s_vals)
    theta = np.trapz( ur * (1 - ur), s_vals)
    return idx, s_edge, delta_star, theta, Ue


def extract_boundary_layer(
    root:         Path,
    edge_method:  str,
    x_locs:       list[float],
    plot_lm_mach: bool,
    n_jobs:       int,
    verbose:      bool = False,
) -> BLResult:
    """
    Extract BL parameters using BL7 from SU2 files in `root`.
    """
    
    def parse(file: Path) -> tuple[list[str], np.ndarray]:
        names: list[str] = []
        rows: list[list[float]] = []
        with file.open() as f:
            for line in f:
                if line.startswith("VARIABLES"):
                    names = [v.strip(' \"') for v in line.split('=',1)[1].split(',')]
                if line.startswith("ZONE"):
                    break
            for line in f:
                s = line.strip()
                if not s or s.startswith("ZONE"): continue
                parts = s.split()
                if len(parts) != len(names): continue
                rows.append([float(p) for p in parts])
        return names, np.array(rows)

    # load data
    vol_vars, vol_data = parse(root / "flow_vol_.dat")
    surf_vars, surf_data = parse(root / "flow_surf_.dat")

    vmap = {n: i for i, n in enumerate(vol_vars)}
    smap = {n: i for i, n in enumerate(surf_vars)}

    # surface pts & normals
    x_s = surf_data[:, smap['x']]
    y_s = surf_data[:, smap['y']]
    normals = []
    for i in range(len(x_s)-1):
        dx, dy = x_s[i+1]-x_s[i], y_s[i+1]-y_s[i]
        n = np.array([-dy, dx])
        norm = np.linalg.norm(n)
        normals.append(n/norm if norm>0 else n)
    normals.append(normals[-1])
    normals = np.vstack(normals)

    # volume interpolation
    pts = np.column_stack((vol_data[:,vmap['x']], vol_data[:,vmap['y']]))
    Vmag = np.hypot(vol_data[:,vmap['Velocity_x']], vol_data[:,vmap['Velocity_y']])
    Vor  = vol_data[:,vmap['Vorticity']]
    vel_interp = NearestNDInterpolator(pts, Vmag)
    vor_interp = NearestNDInterpolator(pts, Vor)

    # BL7 search params
    threshold = 1e-4
    ds        = 1e-6
    max_s     = 1e-2

    args = [(
        i,
        np.array([x_s[i], y_s[i]]),
        normals[i],
        vel_interp,
        vor_interp,
        threshold,
        ds,
        max_s
    ) for i in range(len(x_s))]

    print(f"[BL] marching {len(args)} nodes on {n_jobs} cores...")
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_bl_node)(*a) for a in args
    )
    print(f"[BL] done in {time.time()-t0:.1f}s")

    # sort & unpack
    results.sort(key=lambda r: r[0])
    x = x_s.tolist()
    delta, delta_star, theta, Me = [], [], [], []
    for _, d, ds_val, th, ue in results:
        delta.append(d)
        delta_star.append(ds_val)
        theta.append(th)
        Me.append(ue)

    H = [ds_val/th if (not np.isnan(ds_val) and not np.isnan(th) and th!=0) else np.nan
         for ds_val, th in zip(delta_star, theta)]

    return BLResult(
        x=x,
        delta={edge_method: delta},
        delta_star={edge_method: delta_star},
        theta={edge_method: theta},
        H={edge_method: H},
        M_e={edge_method: Me},
        M_e_LM=None,
        velocity_profiles={},
    )