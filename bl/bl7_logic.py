# su2_postprocess/bl/bl7_logic.py

import numpy as np
import logging
from scipy.optimize import brentq

logging.basicConfig(
    filename="bl7_debug.log",
    filemode="a",
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.DEBUG,
)

def compute_bl_thickness_for_node(args):
    """
    args = (
      i,
      surf_pt, normal,
      vel_interp, vort_interp,
      methods,
      threshold, max_steps, step_size, verbose
    )
    """
    (i, surf_pt, normal, vel_interp, vort_interp,
     methods, threshold, max_steps, step_size, verbose) = args

    # initialize
    bl_t = {m: np.nan for m in methods}
    disp = {m: np.nan for m in methods}
    mom  = {m: np.nan for m in methods}
    H    = {m: np.nan for m in methods}
    ue   = {m: np.nan for m in methods}

    V_s = vel_interp(surf_pt)
    ω_s = vort_interp(surf_pt)
    if np.isnan(V_s):
        logging.warning(f"[BL7] node {i}: surface vel NaN")
        return (i, bl_t, disp, mom, H, ue)

    for m in methods:
        if m == "vorticity_threshold":
            s = 0.0
            found = False
            for _ in range(int(max_steps)):
                s += step_size
                ω = vort_interp(surf_pt + normal*s)
                if np.isnan(ω) or np.isnan(ω_s):
                    break
                if abs(ω/ω_s) <= 1e-4:
                    bl_t[m] = s
                    ue[m]   = vel_interp(surf_pt + normal*s)
                    found = True
                    break
            if not found or ue[m] == 0:
                continue

            ss = np.linspace(0, bl_t[m], 500)
            uvals = vel_interp(surf_pt + np.outer(ss, normal))
            ur = uvals/ue[m]
            disp[m] = np.trapz(1 - ur, ss)
            mom [m] = np.trapz(   ur*(1-ur), ss)
            H   [m] = disp[m]/mom[m] if mom[m] != 0 else np.nan

        elif m == "edge_velocity":
            def f(s):
                return vel_interp(surf_pt + normal*s) - threshold*vel_interp(surf_pt + normal*(s+step_size))
            try:
                s0 = brentq(f, 0.0, 0.5)
                bl_t[m] = s0
                ue[m]   = vel_interp(surf_pt + normal*s0)
                ss = np.linspace(0, s0, 500)
                uvals = vel_interp(surf_pt + np.outer(ss, normal))
                ur = uvals/ue[m]
                disp[m] = np.trapz(1 - ur, ss)
                mom [m] = np.trapz(   ur*(1-ur), ss)
                H   [m] = disp[m]/mom[m] if mom[m] != 0 else np.nan
            except Exception as e:
                logging.warning(f"[BL7] node {i} edge_velocity brentq fail: {e}")

    return (i, bl_t, disp, mom, H, ue)
