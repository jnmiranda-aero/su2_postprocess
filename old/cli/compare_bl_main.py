from pathlib import Path
import sys

from su2_postprocess.bl.extract_new import extract_boundary_layer
from su2_postprocess.bl.plots       import plot_bl_params_multi, plot_velocity_profiles_multi

def write_bl_table(path: Path, x_vals, y_vals):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write('VARIABLES = "x", "value"\n')
        f.write('ZONE T="Boundary-Layer Data"\n')
        for x, y in zip(x_vals, y_vals):
            try:
                f.write(f"{x:.6e} {float(y):.6e}\n")
            except:
                f.write(f"{x:.6e} {y}\n")

def compare_bl_main(args) -> int:
    cases = args.cases
    results = []
    for case in cases:
        res = extract_boundary_layer(
            root         = Path(case),
            edge_method  = args.edge_method,
            x_locs       = [],
            plot_lm_mach = False,
            n_jobs       = args.n_jobs
        )
        results.append(res)

    # Overlay plots
    plot_bl_params_multi(results, [args.edge_method])
    plot_velocity_profiles_multi(results, [args.edge_method])

    # Write out each case’s BL tables
    for case, res in zip(cases, results):
        out = Path(case) / "BL"
        out.mkdir(exist_ok=True)
        m = args.edge_method
        write_bl_table(out / "bl_thickness_delta.dat",         res.x,   res.delta[m])
        write_bl_table(out / "bl_displacement_thickness_deltaStar.dat", res.x,   res.delta_star[m])
        write_bl_table(out / "bl_momentum_thickness_theta.dat",        res.x,   res.theta[m])
        write_bl_table(out / "bl_shape_factor_H.dat",                  res.x,   res.H[m])
        write_bl_table(out / "bl_edge_velocity_ue.dat",                res.x,   res.M_e[m])

    return 0
