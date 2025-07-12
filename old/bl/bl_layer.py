# su2_postprocess/bl_layer.py
"""
Boundary-layer analysis module for SU2 postprocessing.
Provides thickness, displacement, momentum, shape-factor, edge-velocity
and velocity-profile extraction at specified x-locations.
"""
import os
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import NearestNDInterpolator
from joblib import Parallel, delayed

from .utils.bl_helpers import (
    smooth_data,
    compute_edge_velocity_thickness,
    compute_vorticity_thickness,
    integrate_thickness
)

class BoundaryLayerAnalyzer:
    def __init__(
        self,
        flow_data: dict,
        surface_nodes: np.ndarray,
        surface_normals: np.ndarray,
        x_locations: list[float],
        methods: list[str] = ['edge_velocity'],
        output_dir: str = 'bl_output',
        n_jobs: int = -1
    ):
        self.flow_data = flow_data
        self.surface_nodes = surface_nodes[:, :2]
        self.surface_normals = surface_normals
        self.x_locations = x_locations
        self.methods = methods
        self.output_dir = output_dir
        self.n_jobs = n_jobs

        os.makedirs(self.output_dir, exist_ok=True)

        # results
        self.bl = {}  # {method: np.ndarray}
        self.disp = {}
        self.mom = {}
        self.H = {}
        self.Ue = {}
        self.velocity_profiles = {x: [] for x in x_locations}

    def compute_thicknesses(self):
        pts = np.column_stack((self.flow_data['x'], self.flow_data['y']))
        Ux, Uy = self.flow_data['Velocity_x'], self.flow_data['Velocity_y']
        Vmag = np.hypot(Ux, Uy)
        vort = self.flow_data.get('Vorticity', None)

        valid = ~np.isnan(Vmag)
        pts, Vmag = pts[valid], Vmag[valid]
        if vort is not None:
            vort = vort[valid]

        # parallel compute for each node & method
        def process_node(i_pt, pt, normal):
            out = {}
            for m in self.methods:
                if m == 'edge_velocity':
                    delta, Ue = compute_edge_velocity_thickness(pt, normal, pts, Vmag)
                elif m == 'vorticity_threshold' and vort is not None:
                    v0 = vort[i_pt]
                    delta, Ue = compute_vorticity_thickness(pt, normal, pts, vort, v0)
                else:
                    delta, Ue = np.nan, np.nan
                if np.isnan(delta):
                    out[m] = (np.nan, np.nan, np.nan, np.nan, np.nan)
                else:
                    interp = NearestNDInterpolator(pts, Vmag)
                    d, t = integrate_thickness(pt, normal, interp, Ue, delta)
                    H = (d/t if t != 0 else np.nan)
                    out[m] = (delta, d, t, H, Ue)
            return i_pt, out

        args = [(i, pt, n) for i, (pt, n) in enumerate(zip(self.surface_nodes, self.surface_normals))]
        results = Parallel(n_jobs=self.n_jobs)(delayed(process_node)(*a) for a in args)
        # collect
        for m in self.methods:
            self.bl[m] = np.full(len(self.surface_nodes), np.nan)
            self.disp[m] = np.full(len(self.surface_nodes), np.nan)
            self.mom[m] = np.full(len(self.surface_nodes), np.nan)
            self.H[m] = np.full(len(self.surface_nodes), np.nan)
            self.Ue[m] = np.full(len(self.surface_nodes), np.nan)
        for i, out in results:
            for m, vals in out.items():
                self.bl[m][i], self.disp[m][i], self.mom[m][i], self.H[m][i], self.Ue[m][i] = vals

    def extract_velocity_profiles(self, num_points: int = 100, smoothing: bool = True, window_length: int = 11, polyorder: int = 3):
        pts = np.column_stack((self.flow_data['x'], self.flow_data['y']))
        Ux, Uy = self.flow_data['Velocity_x'], self.flow_data['Velocity_y']
        Vmag = np.hypot(Ux, Uy)
        interp = NearestNDInterpolator(pts, Vmag)

        for x_loc in self.x_locations:
            # find closest node
            idx = np.abs(self.surface_nodes[:, 0] - x_loc).argmin()
            delta = self.bl['edge_velocity'][idx]
            Ue = self.Ue['edge_velocity'][idx]
            if np.isnan(delta) or np.isnan(Ue) or Ue == 0:
                continue
            normal = self.surface_normals[idx]
            s_vals = np.linspace(0, delta, num_points)
            pos = self.surface_nodes[idx] + np.outer(s_vals, normal)
            vel = interp(pos)
            valid = ~np.isnan(vel)
            s_vals, vel = s_vals[valid], vel[valid]
            u_norm = vel / Ue
            s_norm = s_vals / delta
            if smoothing:
                u_norm = smooth_data(u_norm, window_length, polyorder)
            self.velocity_profiles[x_loc].append((s_norm, u_norm))

    def save_bl_data(self):
        """Save BL quantities to .dat files"""
        xcoords = self.surface_nodes[:, 0]
        for m in self.methods:
            for name, arr in [('delta', self.bl[m]),
                              ('deltaStar', self.disp[m]),
                              ('theta', self.mom[m]),
                              ('H', self.H[m]),
                              ('ue', self.Ue[m])]:
                fname = f"bl_{name}_{m}.dat"
                with open(os.path.join(self.output_dir, fname), 'w') as f:
                    f.write('VARIABLES = "x", "value"\n')
                    f.write(f'ZONE T="BL {name} ({m})"\n')
                    for x, v in zip(xcoords, arr):
                        f.write(f'{x:.6f} {v:.6e}\n')

    # plotting methods would mirror your existing compare/single plotting infrastructure

    def run(self):
        self.compute_thicknesses()
        self.extract_velocity_profiles()
        self.save_bl_data()
        # integrate with existing plotting routines via CLI command
