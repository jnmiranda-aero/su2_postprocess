"""mesh_comparator.py – clean dual-method BL extractor (velocity 0.99 and vorticity 1e-4)
--------------------------------------------------------------------------
Exports
    run_vel(surface, volume, out)   # δ_99 velocity edge
    run_vort(surface, volume, out)  # ω/ω₀ ≤ 1e-4 vorticity edge
    compare_bl_datasets([...], parameter)  # quick overlay helper
The CLI wires bl-single / bl-compare to these.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.interpolate import NearestNDInterpolator
from scipy.optimize import brentq
from joblib import Parallel, delayed
from collections import defaultdict

# ------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------------
# Worker – single node BL metrics
# ------------------------------------------------------------------------

def _bl_node(args):
    (
        idx, p, n, pts, Vmag, Vort,
        mode, vel_thr, vort_ratio_thr,
        step, s_max,
    ) = args

    vel_interp  = NearestNDInterpolator(pts, Vmag)
    vort_interp = NearestNDInterpolator(pts, Vort) if mode == "vort" else None

    # ------------------------------------------------ velocity (δ_99) -----
    if mode == "vel":
        def f(s):
            return vel_interp(p + n*s) - vel_thr * vel_interp(p + n*(s+step))
        try:
            delta = brentq(f, 0.0, 0.1)
        except ValueError:
            return idx, *([np.nan]*5)
        Ue = vel_interp(p + n*delta)

    # ------------------------------------------------ vorticity (ω decay) --
    else:  # "vort"
        ω0 = abs(vort_interp(p))
        if not np.isfinite(ω0) or ω0 == 0.0:
            return idx, *([np.nan]*5)
        s_scan = np.linspace(0.0, s_max, 300)
        ω_scan = abs(vort_interp(p[None] + np.outer(s_scan, n)))
        target = ω0 * vort_ratio_thr
        hit = np.where(ω_scan <= target)[0]
        if len(hit) == 0:
            return idx, *([np.nan]*5)
        k = hit[0]
        if k == 0:
            delta = s_scan[0]
        else:
            s1, s2 = s_scan[k-1:k+1]
            w1, w2 = ω_scan[k-1:k+1]
            frac   = (target - w1) / (w2 - w1)
            delta  = s1 + frac*(s2 - s1)
        Ue = vel_interp(p + n*delta)

    # ------------------------------------------------ integrals ----------
    if not np.isfinite(Ue) or Ue == 0.0:
        return idx, *([np.nan]*5)
    s_vec = np.linspace(0.0, delta, 400)
    u_vec = vel_interp(p[None] + np.outer(s_vec, n))
    disp  = np.trapz(1 - u_vec/Ue, s_vec)
    mom   = np.trapz((u_vec/Ue)*(1 - u_vec/Ue), s_vec)
    H     = disp/mom if mom else np.nan
    return idx, delta, disp, mom, H, Ue

# ------------------------------------------------------------------------
# MeshParser – minimal reader + BL computation
# ------------------------------------------------------------------------

class MeshParser:
    """Reads SU2 *.dat (surface & volume) and extracts BL metrics."""

    def __init__(self, surf_file: str, vol_file: str, out_dir: str = "BL"):
        self.surf_file = surf_file
        self.vol_file  = vol_file
        self.out_dir   = out_dir
        os.makedirs(out_dir, exist_ok=True)

    # ----------------------- utilities ----------------------------------
    @staticmethod
    def _read_dat(path: str) -> np.ndarray:
        data, ncol = [], None
        with open(path) as fp:
            for ln in fp:
                ln = ln.strip()
                if not ln or ln.startswith(("VARIABLES", "ZONE")):
                    continue
                try:
                    nums = [float(x) for x in ln.split()]
                except ValueError:
                    continue
                if ncol is None:
                    ncol = len(nums)
                if len(nums) == ncol:
                    data.append(nums)
        return np.asarray(data)

    def load(self):
        surf = self._read_dat(self.surf_file)
        vol  = self._read_dat(self.vol_file)
        self.surf_pts = surf[:, :2]
        self.vol_pts  = vol[:, :2]
        self.Vmag     = np.hypot(vol[:, 2], vol[:, 3])
        self.Vort     = vol[:, 4]
        log.info("Loaded %d surface pts, %d volume pts", len(self.surf_pts), len(self.vol_pts))

    # ----------------------- core BL engine -----------------------------
    def compute_bl(self, *, edge_method="vort", step=1e-6, s_max=0.5, jobs=-1):
        tang = np.diff(self.surf_pts, axis=0, append=self.surf_pts[-1:])
        norms = np.column_stack([-tang[:, 1], tang[:, 0]])
        lens  = np.linalg.norm(norms, axis=1, keepdims=True)
        norms = np.divide(norms, lens, out=np.zeros_like(norms), where=lens != 0)

        mask = np.isfinite(self.Vmag) & np.isfinite(self.Vort)
        pts  = self.vol_pts[mask]
        vmag = self.Vmag[mask]
        vort = self.Vort[mask]

        args = [
            (
                i, p, n, pts, vmag, vort,
                "vort" if edge_method == "vort" else "vel",
                0.99, 1e-4, step, s_max,
            )
            for i, (p, n) in enumerate(zip(self.surf_pts, norms))
        ]
        out = Parallel(n_jobs=jobs)(delayed(_bl_node)(a) for a in args)
        out.sort(key=lambda r: r[0])
        self.delta, self.delta_star, self.theta, self.H, self.Ue = map(list, zip(*[t[1:] for t in out]))

    # ----------------------- save helpers -------------------------------
    def _save(self, tag: str, arr):
        fn = os.path.join(self.out_dir, f"{tag}.dat")
        with open(fn, "w") as f:
            f.write('VARIABLES = "x", "value"\nZONE T="BL"\n')
            for x, v in zip(self.surf_pts[:, 0], arr):
                f.write(f"{x} {v}\n")
        log.info("Wrote %s", fn)

    def save_all(self, tag: str):
        self._save(f"delta_{tag}",       self.delta)
        self._save(f"delta_star_{tag}",  self.delta_star)
        self._save(f"theta_{tag}",       self.theta)
        self._save(f"H_{tag}",           self.H)
        self._save(f"Ue_{tag}",          self.Ue)

# ------------------------------------------------------------------------
# Convenience wrappers
# ------------------------------------------------------------------------

def run_vel(surf, vol, out="BL"):
    mp = MeshParser(surf, vol, out); mp.load(); mp.compute_bl(edge_method="vel");  mp.save_all("bl7");  return mp

def run_vort(surf, vol, out="BL"):
    mp = MeshParser(surf, vol, out); mp.load(); mp.compute_bl(edge_method="vort"); mp.save_all("vort"); return mp

# ------------------------------------------------------------------------
# Comparison helper
# ------------------------------------------------------------------------

def compare_bl_datasets(dirs, *, parameter="delta_bl7", labels=None):
    plt.figure()
    for i, d in enumerate(dirs):
        path = os.path.join(d, f"{parameter}.dat")
        if not os.path.isfile(path):
            log.warning("%s missing", path)
            continue
        x, v = np.loadtxt(path, skiprows=2).T
        lbl = labels[i] if labels else os.path.basename(os.path.abspath(d))
        plt.plot(x, v, label=lbl)
    plt.xlabel("x/c"); plt.ylabel(parameter); plt.legend(); plt.grid(False); plt.tight_layout(); return plt
