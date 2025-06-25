#!/usr/bin/env python3
"""
compare_bl.py

Modular boundary-layer analysis & comparison tool.
- Either compute BL + velocity profiles from raw SU2 surface/volume files
  OR load existing BL .dat + velocity_profile .dat files.
- Then plot & compare multiple cases on the same axes.
"""
import os, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from collections import defaultdict

# -------------------------------------------------------------------
# 1. A thin wrapper to parse precomputed .dat files
# -------------------------------------------------------------------
def read_xy_dat(path, skip=2):
    """Read a two-column .dat with `x y` pairs, skipping header lines."""
    data = np.loadtxt(path, skiprows=skip)
    return data[:,0], data[:,1]

class PrecomputedBL:
    def __init__(self, bl_dir):
        self.bl_dir = bl_dir
        # Expect BL files with these fixed names:
        self.files = {
            'delta'     : os.path.join(bl_dir,'bl_thickness_delta.dat'),
            'deltaStar': os.path.join(bl_dir,'bl_displacement_thickness_deltaStar.dat'),
            'theta'    : os.path.join(bl_dir,'bl_momentum_thickness_theta.dat'),
            'H'        : os.path.join(bl_dir,'bl_shape_factor_H.dat'),
            'ue'       : os.path.join(bl_dir,'bl_edge_velocity_ue.dat')
        }
    def load(self):
        self.x, self.delta      = read_xy_dat(self.files['delta'])
        _,    self.deltaStar  = read_xy_dat(self.files['deltaStar'])
        _,    self.theta      = read_xy_dat(self.files['theta'])
        _,    self.H          = read_xy_dat(self.files['H'])
        _,    self.ue         = read_xy_dat(self.files['ue'])
    def load_velocity_profiles(self):
        """Loads all velocity_profile_x_*.dat under bl_dir."""
        self.profiles = defaultdict(list)
        for fn in sorted(glob.glob(os.path.join(self.bl_dir,"velocity_profile_x_*.dat"))):
            # filename: velocity_profile_x_{xloc:.4f}_node_{idx}.dat
            parts = os.path.basename(fn).split('_')
            xloc = float(parts[3])
            y, u = read_xy_dat(fn, skip=2)  # y -> s_normalized, u -> u_normalized
            self.profiles[xloc].append((u,y))
        return self.profiles

# -------------------------------------------------------------------
# 2. Raw SU2 → BL & velocity profiles (MeshParser)
#    (you can drop your existing MeshParser class here, 
#     unchanged except for moving it into a module or inlining it)
# -------------------------------------------------------------------
# ... Paste your fully-debugged MeshParser class here, with methods:
#     compute_boundary_layer_thickness(...)
#     compute_velocity_profiles(...)
#     save_boundary_layer_to_file(...)
#     save_velocity_profiles_to_files(...)
# -------------------------------------------------------------------

# For brevity:
from meshparser import MeshParser

# -------------------------------------------------------------------
# 3. High-level "Case" container
# -------------------------------------------------------------------
class Case:
    def __init__(self, name, surf_file, vol_file, bl_dir, x_locs, methods, jobs, compute_bl, outdir):
        self.name      = name
        self.surf_file = surf_file
        self.vol_file  = vol_file
        self.bl_dir    = bl_dir
        self.x_locs    = x_locs
        self.methods   = methods
        self.jobs      = jobs
        self.compute_bl= compute_bl
        self.outdir    = outdir
        self.bl        = None

    def prepare(self):
        os.makedirs(self.outdir, exist_ok=True)
        if self.compute_bl:
            mp = MeshParser(self.surf_file, self.vol_file, output_dir=self.bl_dir, x_locations=self.x_locs)
            mp.run(methods=self.methods, n_jobs=self.jobs, verbose=False)
            self.bl = PrecomputedBL(self.bl_dir)
            self.bl.load()
            self.bl.load_velocity_profiles()
        else:
            self.bl = PrecomputedBL(self.bl_dir)
            self.bl.load()
            self.bl.load_velocity_profiles()

# -------------------------------------------------------------------
# 4. Plotting & comparison routines
# -------------------------------------------------------------------
def compare_bl_thickness(cases, outdir):
    plt.figure()
    for case in cases:
        x, δ = case.bl.x, case.bl.delta
        plt.plot(x, δ/np.nanmax(δ), label=case.name) 
    plt.xlabel('x/c')
    plt.ylabel(r'$\delta/\delta_{max}$')
    plt.title('BL thickness comparison')
    plt.legend()
    plt.grid(False)
    plt.savefig(os.path.join(outdir,'compare_delta.png'))

def compare_velocity_profiles(cases, outdir):
    for xloc in cases[0].x_locs:
        plt.figure()
        for case in cases:
            for u,y in case.bl.profiles.get(xloc,[]):
                plt.plot(u,y,label=case.name)
        plt.xlabel('u/u_e'); plt.ylabel('y/δ')
        plt.title(f'Velocity profiles @ x={xloc:.3f}')
        plt.legend(); plt.grid(False)
        plt.savefig(os.path.join(outdir,f'compare_velprof_x_{xloc:.3f}.png'))

# -------------------------------------------------------------------
# 5. CLI
# -------------------------------------------------------------------
if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cases', nargs='+', required=True,
                   help="For each case: NAME:surface_su2:volume_su2 or NAME:BL_DAT_DIR (if not computing)")
    p.add_argument('--x-locs', type=float, nargs='+', default=[0.1,0.4,0.9])
    p.add_argument('--methods', nargs='+', default=['edge_velocity'])
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--compute-bl', action='store_true',
                   help="Compute BL from raw SU2 instead of loading precomputed .dat")
    p.add_argument('--outdir', default='comparison_plots')
    args = p.parse_args()

    # build cases
    cases = []
    for spec in args.cases:
        name, a, b = spec.split(':')
        if args.compute_bl:
            surf, vol = a, b
            bl_dir = os.path.join('BL', name)
        else:
            surf = vol = None
            bl_dir = a
        c = Case(name, surf, vol, bl_dir,
                 x_locs   = args.x_locs,
                 methods  = args.methods,
                 jobs     = args.jobs,
                 compute_bl = args.compute_bl,
                 outdir   = args.outdir)
        c.prepare()
        cases.append(c)

    # make comparison plots
    os.makedirs(args.outdir, exist_ok=True)
    compare_bl_thickness(cases, args.outdir)
    compare_velocity_profiles(cases, args.outdir)

    print("Done—plots saved in", args.outdir)
