# su2_postprocess/bl/bl7_worker.py
"""
Original BL7 boundary-layer thickness computation logic with performance tweaks:
1) Reuse interpolation objects passed in arguments (no per-call rebuild).
2) Logging set to INFO by default, debug logs only when verbose.
"""
import numpy as np
import logging
from scipy.optimize import brentq

# Logging: default to INFO to avoid debug overhead
logging.basicConfig(
    filename='mesh_parser_debug.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def compute_bl_thickness_for_node(args):
    """
    Compute BL properties for one node.
    args: (i, surface_node, normal, vel_interp, vor_interp,
           methods, threshold, max_steps, step_size, verbose)
    """
    i, surface_node, normal, vel_interp, vor_interp, methods, threshold, max_steps, step_size, verbose = args
    try:
        # Surface values
        V_surface = vel_interp(surface_node)
        vor_surface = vor_interp(surface_node)
        if verbose:
            logging.debug(f"Node {i}: V_surface={V_surface}, vor_surface={vor_surface}")

        # Prepare output dicts
        bl_th = {}
        disp_th = {}
        mom_th = {}
        shape_f = {}
        edge_v = {}

        if np.isnan(V_surface):
            for m in methods:
                bl_th[m] = np.nan
                disp_th[m] = np.nan
                mom_th[m] = np.nan
                shape_f[m] = np.nan
                edge_v[m] = np.nan
            return (i, bl_th, disp_th, mom_th, shape_f, edge_v)

        for method in methods:
            # Edge detection
            if method == 'edge_velocity':
                def func(s):
                    p1 = surface_node + normal * s
                    p2 = surface_node + normal * (s + step_size)
                    v1 = vel_interp(p1)
                    v2 = vel_interp(p2)
                    if np.isnan(v1) or np.isnan(v2):
                        return 1e6
                    return v1 - threshold * v2

                try:
                    a, b = 0.0, 0.5
                    if func(a)*func(b) >= 0:
                        raise ValueError("Bracket invalid")
                    s_edge = brentq(func, a, b)
                    bl_th[method] = s_edge
                    edge_v[method] = vel_interp(surface_node + normal * s_edge)
                except Exception as e:
                    if verbose:
                        logging.info(f"Node {i} ({method}) failed bracket/root: {e}")
                    bl_th[method] = np.nan
                    edge_v[method] = np.nan

            elif method == 'vorticity_threshold':
                s = 0.0
                found = False
                for _ in range(int(max_steps)):
                    s += step_size
                    vor_val = vor_interp(surface_node + normal * s)
                    if np.isnan(vor_val) or np.isnan(vor_surface):
                        break
                    if abs(vor_val / vor_surface) <= 1e-4:
                        bl_th[method] = s
                        edge_v[method] = vel_interp(surface_node + normal * s)
                        found = True
                        break
                if not found or bl_th.get(method, 0) <= 1e-6:
                    bl_th[method] = np.nan
                    edge_v[method] = np.nan

            else:
                bl_th[method] = np.nan
                edge_v[method] = np.nan

            # Compute integrals
            s_edge = bl_th.get(method, np.nan)
            if not np.isnan(s_edge) and s_edge > step_size:
                s_vals = np.linspace(0, s_edge, 1000)
                disp, mom = 0.0, 0.0
                Ue = edge_v[method]
                for s in s_vals:
                    Vc = vel_interp(surface_node + normal * s)
                    if not np.isnan(Vc) and Ue != 0:
                        ur = Vc / Ue
                        disp += (1 - ur)
                        mom += (ur * (1 - ur))
                # Trapezoidal rule
                disp_th[method] = np.trapz(1 - (vel_interp(surface_node + normal * s_vals) / (Ue or 1)), s_vals)
                mom_th[method] = np.trapz((vel_interp(surface_node + normal * s_vals) / (Ue or 1)) * (1 - (vel_interp(surface_node + normal * s_vals) / (Ue or 1))), s_vals)
                shape_f[method] = disp_th[method] / mom_th[method] if mom_th[method] != 0 else np.nan
            else:
                disp_th[method] = np.nan
                mom_th[method] = np.nan
                shape_f[method] = np.nan

        return (i, bl_th, disp_th, mom_th, shape_f, edge_v)

    except Exception as e:
        logging.error(f"Error in node {i}: {e}")
        defaults = {m: np.nan for m in methods}
        return (i, defaults, defaults, defaults, defaults, defaults)
