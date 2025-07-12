# su2_postprocess/cli/single_bl_main.py

import time
from pathlib import Path

from su2_postprocess.bl.extract_new import extract_boundary_layer

def single_bl_main(args):
    root = Path(args.root)
    print(f"[BL] single-case mode → {root}")
    start = time.time()

    # Inline write_bl_table (no common.py)
    def write_bl_table(path: Path, x: list[float], y: list[float]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write('VARIABLES = "x", "value"\n')
            f.write('ZONE T="Boundary-Layer Data"\n')
            for xi, yi in zip(x, y):
                yi_val = float(yi)
                f.write(f"{xi:.6e} {yi_val:.6e}\n")

    # 1) extract
    result = extract_boundary_layer(
        root         = root,
        edge_method  = args.edge_method,
        x_locs       = args.x_locs,
        plot_lm_mach = args.plot_lm_mach,
        n_jobs       = args.n_jobs,
        verbose      = args.verbose
    )

    elapsed = time.time() - start
    print(f"  → done in {elapsed:.1f}s")

    # 2) write out tables under BL/
    bl_dir = root / "BL"
    method = args.edge_method

    write_bl_table(bl_dir/"bl_thickness_delta.dat",       result.x, result.delta[method])
    write_bl_table(bl_dir/"bl_displacement_deltaStar.dat", result.x, result.delta_star[method])
    write_bl_table(bl_dir/"bl_momentum_theta.dat",         result.x, result.theta[method])
    write_bl_table(bl_dir/"bl_shape_H.dat",                result.x, result.H[method])
    write_bl_table(bl_dir/"bl_edge_velocity_ue.dat",       result.x, result.M_e[method])

    return 0
