"""
Boundary-layer extraction core for *su2_postprocess*.

* 100 % functionality of the original **BL7.py**
* Clean API callable from CLI or notebooks
* Edge–finding methods: ``vorticity_threshold`` (default), ``edge_velocity``,
  and ``gradient``
* γ (ratio of specific heats) is passed in by the caller (parsed from the log)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import NearestNDInterpolator
from scipy.optimize import brentq

from su2_postprocess.utils.smoothing import savgol_1d_safe

# -----------------------------------------------------------------------------


EdgeMethod = Literal["vorticity_threshold", "edge_velocity", "gradient"]


@dataclass
class BLResult:
    """Container returned by :func:`extract_boundary_layer`."""

    x: np.ndarray
    y: np.ndarray

    delta: dict[str, np.ndarray]
    delta_star: dict[str, np.ndarray]
    theta: dict[str, np.ndarray]
    H: dict[str, np.ndarray]
    u_e: dict[str, np.ndarray]
    M_e: dict[str, np.ndarray]

    # optional extras filled in by CLI / caller
    M_e_LM: Optional[np.ndarray] = None          # SU2’s own edge-Mach if present
    velocity_profiles: dict | None = None        # filled only on demand

    @property
    def has_LM_Mach_e(self) -> bool:
        return self.M_e_LM is not None


# -----------------------------------------------------------------------------


def _integrate_profiles(u_norm: np.ndarray,
                        s_norm: np.ndarray) -> tuple[float, float]:
    """Return (δ*, θ) from normalised velocity/distance vectors."""
    disp = np.trapz(1.0 - u_norm, s_norm)
    mom  = np.trapz(u_norm * (1.0 - u_norm), s_norm)
    return disp, mom


def _edge_by_gradient(u_vals: np.ndarray,
                      s_vals: np.ndarray,
                      window: int = 5) -> float:
    """Edge = first point where |∂u/∂s| < 1 % of max(|∂u/∂s|)."""
    if len(u_vals) < window + 2:
        return np.nan
    du        = np.gradient(u_vals, s_vals)
    du_smooth = savgol_1d_safe(du, window_length=window, polyorder=2)
    thresh    = 0.01 * np.nanmax(np.abs(du_smooth))
    idx       = np.where(np.abs(du_smooth) < thresh)[0]
    return s_vals[idx[0]] if idx.size else np.nan


# -----------------------------------------------------------------------------


def _node_worker(
    idx: int,
    surf_pt: np.ndarray,
    normal: np.ndarray,
    points: np.ndarray,
    vel_mag: np.ndarray,
    mach_field: Optional[np.ndarray],
    vorticity: Optional[np.ndarray],
    methods: tuple[EdgeMethod, ...],
    gamma: float,
    *,
    u_ratio_edge: float = 0.99,
    w_ratio_edge: float = 1e-4,
    s_max: float = 5e-4,
    n_samples: int = 600,
    step_size: float = 1e-6,
):

    vi = NearestNDInterpolator(points, vel_mag)
    mi = (NearestNDInterpolator(points, mach_field)
          if mach_field is not None else None)

    # vorticity may be None → fill zeros (BL7 tolerance)
    if vorticity is None:
        vorticity = np.zeros_like(vel_mag)
    wi = NearestNDInterpolator(points, vorticity)

    out_delta, out_delta_star, out_theta, out_H, out_u_e, out_M_e = (
        defaultdict(lambda: np.nan) for _ in range(6)
    )

    # oversample normal ray once – reused by every method
    s_grid   = np.linspace(0.0, s_max, n_samples)
    pts_grid = surf_pt + np.outer(s_grid, normal)
    u_grid   = vi(pts_grid)
    w_grid   = wi(pts_grid)
    M_grid   = (mi(pts_grid) if mi is not None
                else np.full_like(u_grid, np.nan))

    for method in methods:
        s_edge = np.nan
        U_e    = np.nan

        # --------------- edge_velocity (BL7 root-finder) -----------------
        if method == "edge_velocity":

            def f(s):
                v_here = vi(surf_pt + normal * s)
                v_next = vi(surf_pt + normal * (s + step_size))
                if np.isnan(v_here) or np.isnan(v_next):
                    return 1e6          # keeps brentq inside bounds
                return v_here - u_ratio_edge * v_next

            try:
                s_edge = brentq(f, 0.0, s_max)
                U_e    = vi(surf_pt + normal * s_edge)
            except ValueError:
                pass

        # --------------- vorticity_threshold ----------------------------
        elif method == "vorticity_threshold":
            omega_wall = wi(surf_pt)
            if np.isfinite(omega_wall) and omega_wall != 0.0:
                ratio = np.abs(w_grid / omega_wall)
                hit   = np.where(ratio <= w_ratio_edge)[0]
                if hit.size:
                    s_edge = s_grid[hit[0]]
                    U_e    = u_grid[hit[0]]

        # --------------- gradient --------------------------------------
        elif method == "gradient":
            s_edge = _edge_by_gradient(u_grid, s_grid)
            if np.isfinite(s_edge):
                U_e = vi(surf_pt + normal * s_edge)

        # ---------------- store even if NaN -----------------------------
        out_delta[method] = s_edge
        out_u_e[method]   = U_e

        # need valid edge for integral metrics --------------------------
        if not np.isfinite(s_edge) or U_e <= 0.0:
            continue

        idx_e      = np.searchsorted(s_grid, s_edge)
        u_profile  = u_grid[: idx_e + 1] / U_e
        s_profile  = s_grid[: idx_e + 1] / s_edge
        d_star, th = _integrate_profiles(u_profile, s_profile)

        out_delta_star[method] = d_star * s_edge
        out_theta[method]      = th * s_edge
        out_H[method]          = d_star / th if th > 0 else np.nan
        out_M_e[method]        = M_grid[idx_e]

    return (idx, out_delta, out_delta_star, out_theta,
            out_H, out_u_e, out_M_e)


# -----------------------------------------------------------------------------


def extract_boundary_layer(
    *,
    surface_nodes: np.ndarray,
    surface_normals: Optional[np.ndarray],
    volume_points: np.ndarray,
    vel_mag: np.ndarray,
    vorticity: Optional[np.ndarray] = None,
    mach_field: Optional[np.ndarray] = None,
    gamma: float = 1.4,
    methods: Iterable[EdgeMethod] = ("vorticity_threshold",),
    n_jobs: int = -1,
) -> BLResult:
    """
    Compute δ, δ*, θ, H, uₑ, Mₑ for every surface node.

    All *arrays* must be **NumPy float64** and 2-D/1-D exactly as documented
    in the README (§ Developer API).
    """

    # ------------------------------------------------------------------
    # basic geometry sanity
    # ------------------------------------------------------------------
    methods = tuple(methods)

    if surface_normals is None:
        # simple polyline normals (same as BL7)
        tang   = np.diff(surface_nodes, axis=0, append=surface_nodes[-1:])
        normals= np.column_stack([-tang[:, 1], tang[:, 0]])
        nrm    = np.linalg.norm(normals, axis=1, keepdims=True)
        normals= np.divide(normals, np.where(nrm == 0, 1.0, nrm))
        surface_normals = normals

    # ------------------------------------------------------------------
    # parallel loop
    # ------------------------------------------------------------------
    args_iter = (
        (i, surface_nodes[i], surface_normals[i], volume_points,
         vel_mag, mach_field, vorticity, methods, gamma)
        for i in range(len(surface_nodes))
    )

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_node_worker)(*args) for args in args_iter
    )

    # keep only successful tuples, sort by index
    results = [r for r in results if isinstance(r[0], (int, np.integer))]
    if not results:
        raise RuntimeError("No boundary-layer points were computed; "
                           "check surface/volume data integrity.")
    results.sort(key=lambda t: t[0])

    def collect(pos: int) -> dict[str, np.ndarray]:
        return {m: np.array([r[pos].get(m, np.nan) for r in results])
                for m in methods}

    return BLResult(
        x   = surface_nodes[:, 0],
        y   = surface_nodes[:, 1],
        delta      = collect(1),
        delta_star = collect(2),
        theta      = collect(3),
        H          = collect(4),
        u_e        = collect(5),
        M_e        = collect(6),
    )
