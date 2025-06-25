# su2_postprocess/utils/bl_helpers.py
"""
Helper functions for boundary-layer analysis: smoothing, edge detection, and integration.
"""
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import NearestNDInterpolator
from scipy.optimize import brentq


def smooth_data(data: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """
    Smooth a 1D array using the Savitzky-Golay filter.

    Parameters:
    - data: input array
    - window_length: must be odd, <= len(data)
    - polyorder: polynomial order

    Returns:
    - smoothed array (same length)
    """
    if len(data) < window_length or window_length < polyorder + 2:
        return data
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(data, window_length, polyorder)


def compute_edge_velocity_thickness(
    surface_pt: np.ndarray,
    normal: np.ndarray,
    points: np.ndarray,
    V_mag: np.ndarray,
    threshold: float = 0.99,
    step_size: float = 1e-7,
    max_distance: float = 0.5
) -> tuple[float, float]:
    """
    Find BL thickness via edge-velocity method (u = threshold * u_e).
    Returns (delta, U_e) or (nan, nan).
    """
    interp = NearestNDInterpolator(points, V_mag)
    def f(s):
        u_s = interp(surface_pt + normal * s)
        u_next = interp(surface_pt + normal * (s + step_size))
        if np.isnan(u_s) or np.isnan(u_next):
            return np.nan
        return u_s - threshold * u_next
    try:
        # bracket search between 0 and max_distance
        a, b = 0.0, max_distance
        fa, fb = f(a), f(b)
        if np.isnan(fa) or np.isnan(fb) or fa * fb > 0:
            return np.nan, np.nan
        s_root = brentq(f, a, b)
        Ue = interp(surface_pt + normal * s_root)
        return s_root, Ue
    except Exception:
        return np.nan, np.nan


def compute_vorticity_thickness(
    surface_pt: np.ndarray,
    normal: np.ndarray,
    points: np.ndarray,
    vort: np.ndarray,
    v0: float,
    threshold_ratio: float = 1e-4,
    step_size: float = 1e-7,
    max_steps: int = 1000000
) -> tuple[float, float]:
    """
    Find BL thickness via vorticity decay method.
    Returns (delta, U_e) or (nan, nan).
    """
    interp_vort = NearestNDInterpolator(points, vort)
    interp_vel = NearestNDInterpolator(points, vort)  # reuse vort points for velocity? user replaces
    s = 0.0
    for i in range(int(max_steps)):
        s += step_size
        v_val = interp_vort(surface_pt + normal * s)
        if np.isnan(v_val) or np.isnan(v0):
            break
        if abs(v_val / v0) <= threshold_ratio:
            Ue = interp_vel(surface_pt + normal * s)
            return s, Ue
    return np.nan, np.nan


def integrate_thickness(
    surface_pt: np.ndarray,
    normal: np.ndarray,
    vel_interp: NearestNDInterpolator,
    Ue: float,
    delta: float,
    num_steps: int = 1000
) -> tuple[float, float]:
    """
    Compute displacement and momentum thicknesses over [0, delta].
    Returns (delta_star, theta).
    """
    s_vals = np.linspace(0, delta, num_steps)
    u_vals = vel_interp(surface_pt + np.outer(s_vals, normal))
    u_vals = np.nan_to_num(u_vals)
    u_ratio = u_vals / (Ue if Ue != 0 else 1)
    disp = np.trapz(1 - u_ratio, s_vals)
    mom = np.trapz(u_ratio * (1 - u_ratio), s_vals)
    return disp, mom