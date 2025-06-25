import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from su2_postprocess.utils.mesh_comparator import MeshParser, compare_bl_datasets
from su2_postprocess.cli.single_main         import single_main
from su2_postprocess.cli.compare_surface_main import compare_surface_main
from su2_postprocess.utils.mesh_comparator import (MeshParser, run_vort, compare_bl_datasets)
# from su2_postprocess.utils.bl7_meshparser import run_bl7


def main():
    parser = argparse.ArgumentParser(description="SU2 Postprocessing CLI")
    subparsers = parser.add_subparsers(dest="command")

    # SINGLE CASE MODE
    single = subparsers.add_parser("single", help="Process a single SU2 case")
    single.add_argument('--root',        required=True)
    single.add_argument('--xfoil-dir',   type=str)
    single.add_argument('--xfoil-file',  type=str)
    single.add_argument('--mses-dir', action='append', metavar='DIR', help='MSES overlay directory (repeatable)')
    single.add_argument('--mses-file',   type=str)    
    single.add_argument('--exp-dir',     type=str)
    single.add_argument('--format',      type=str, default='svg')
    single.add_argument('--dpi',         type=int, default=300)
    single.add_argument('--it',          dest='it',
                        action='store_true', help='Include i_t subplot')
    single.add_argument('--map',  dest='map',
                        action='store_true', help='Include transition map subplot')
    single.add_argument('--show-airfoil', dest='show_airfoil',
                        action='store_true', help='Include x–y airfoil panel')
    single.set_defaults(it=False, map=True, show_airfoil=False)
    single.add_argument('--label-style',
                        choices=['short','full','metadata','auto','sense'],
                        default='metadata',
                        help="Legend label style")
    single.add_argument('--case-colors', nargs='+',
                        help='Colors for each SU2 case')
    single.add_argument('--case-styles', nargs='+',
                        help='Line styles for each SU2 case')
    single.add_argument('--xfoil-colors', nargs='+',
                        help='Colors for each XFOIL overlay')
    single.add_argument('--exp-colors', nargs='+',
                        help='Colors for each experimental overlay')   
    single.add_argument('--xfoil-styles', nargs='+',
                    help='Line styles for each XFOIL overlay')
    single.add_argument('--exp-styles', nargs='+',
                    help='Line styles for each experimental overlay')  
    single.add_argument('--exp-markers', nargs='+',
                    help='Markers  for each experimental overlay')                                                               

    # MULTI-CASE COMPARISON MODE
    compare = subparsers.add_parser("compare-surface",
                                    help="Compare Cp/Cf/Transition across cases")
    compare.add_argument('--cases',      nargs='+', required=True)
    compare.add_argument('--xfoil-dir',  type=str)
    compare.add_argument('--xfoil-file', type=str)
    compare.add_argument('--mses-dir',   type=str)
    compare.add_argument('--mses-file',  type=str)
    compare.add_argument('--exp-dir',    type=str)
    compare.add_argument('--save',       type=str, default='A')
    compare.add_argument('--format',     type=str, default='svg')
    compare.add_argument('--dpi',        type=int, default=300)
    compare.add_argument('--it',   dest='it',
                         action='store_true', help='Include i_t subplot')
    compare.add_argument('--map',  dest='map',
                         action='store_true', help='Include transition map subplot')
    compare.add_argument('--show-airfoil', dest='show_airfoil',
                         action='store_true', help='Include x–y airfoil panel')
    # default show_airfoil = True here:
    compare.set_defaults(it=True, map=False, show_airfoil=True)
    compare.add_argument('--label-style',
                         choices=['short','full','metadata','auto','sense'],
                         default='auto')
    compare.add_argument('--case-colors', nargs='+',
                         help='Colors for each SU2 case')
    compare.add_argument('--case-styles', nargs='+',
                         help='Line styles for each SU2 case')
    compare.add_argument('--legends-cp', dest='legends_cp',
                         action='store_false')
    compare.add_argument('--no-legends-cp', dest='legends_cp',
                         action='store_true')
    compare.set_defaults(legends_cp=None)
    compare.add_argument('--legends-cf', dest='legends_cf',
                         action='store_false')
    compare.add_argument('--no-legends-cf', dest='legends_cf',
                         action='store_false')
    compare.set_defaults(legends_cf=False)
    compare.add_argument('--legends-it', dest='legends_it',
                         action='store_true')
    compare.add_argument('--no-legends-it', dest='legends_it',
                         action='store_false')
    compare.set_defaults(legends_it=False)

    # BOUNDARY-LAYER SINGLE
    bl_single = subparsers.add_parser("bl-single", help="Boundary-layer (single)")
    bl_single.add_argument("--root",        required=True)
    # bl_single.add_argument("--edge-method",
    #                        choices=["edge_velocity","vorticity_threshold","gradient"],
    #                        default="vorticity_threshold")
    bl_single.add_argument("--edge-method",
                           choices=["vort"],       # only BL-7 is supported now
                           default="vort")                           
    bl_single.add_argument("--plot-lm-mach", action="store_true")
    bl_single.add_argument("--x-locs",     nargs="+", type=float, default=[])
    bl_single.add_argument("--n-jobs",     type=int, default=-1)
    bl_single.add_argument("--format",     default="svg")
    bl_single.add_argument("--dpi",        type=int, default=300)
    bl_single.add_argument("--verbose",    action="store_true")
    bl_single.set_defaults(verbose=False)

    # BOUNDARY-LAYER MULTI
    bl_compare = subparsers.add_parser("bl-compare", help="Boundary-layer (multi)")
    bl_compare.add_argument("--cases",      nargs="+", required=True)
    # bl_compare.add_argument("--edge-method",
    #                         choices=["edge_velocity","vorticity_threshold","gradient"],
    #                         default="vorticity_threshold")
    bl_compare.add_argument("--edge-method",
                            choices=["vort"],
                            default="vort")                            
    bl_compare.add_argument("--plot-lm-mach", action="store_true")
    bl_compare.add_argument("--n-jobs",     type=int, default=-1)
    bl_compare.add_argument("--save",       default="A")
    bl_compare.add_argument("--format",     default="svg")
    bl_compare.add_argument("--dpi",        type=int, default=300)

    args = parser.parse_args()
    if args.command == "single":
        return single_main(args)
    elif args.command == "compare-surface":
        return compare_surface_main(args)
    elif args.command == "bl-single":
        root  = Path(args.root)
        surf  = root / "flow_surf_.dat"
        vol   = root / "flow_vol_.dat"
        out   = root / "BL"
        run_bl7(
            surface_file=str(surf),
            flow_file=str(vol),
            output_dir=str(out),
            x_locations=args.x_locs,
            methods=["vorticity_threshold"],
            threshold=0.99,
            max_steps=1e6,
            step_size=1e-7,
            tolerance=1e-3,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
        )
        print(f"✔ BL-7 data written to {out}")
        return 0
    elif args.command == "bl-compare":
        # overlay across existing BL folders
        plt = compare_bl_datasets(
            dataset_paths=args.cases,
            parameter=f"delta_{args.edge_method}",
            labels=None
        )
        outname = f"bl_compare_{args.edge_method}.{args.format}"
        plt.savefig(outname, dpi=args.dpi, bbox_inches="tight")
        print(f"✔ Saved comparison plot: {outname}")
        return 0

if __name__ == "__main__":
    sys.exit(main() or 0)



# # su2_postprocess/cli/cli_main.py

# import argparse
# from su2_postprocess.cli.single_main import single_main
# from su2_postprocess.cli.compare_surface_main import compare_surface_main

# def main():
#     parser = argparse.ArgumentParser(description="SU2 Postprocessing CLI")
#     subparsers = parser.add_subparsers(dest="command")

#     # SINGLE CASE MODE
#     single = subparsers.add_parser("single", help="Process a single SU2 case")
#     single.add_argument('--root',        required=True)
#     single.add_argument('--xfoil-dir',  type=str)
#     single.add_argument('--xfoil-file', type=str)
#     single.add_argument('--exp-dir',     type=str)
#     single.add_argument('--format',      type=str, default='svg')
#     single.add_argument('--dpi',         type=int, default=300)
#     single.add_argument('--no-it',       action='store_true')
#     single.add_argument('--no-map',      action='store_true')
#     single.add_argument('--show-airfoil', action='store_true',
#                         help='Include x–y airfoil shape panel')
#     single.add_argument('--label-style',
#                         choices=['short', 'full', 'metadata', 'auto'],
#                         default='metadata',
#                         help="Legend label style")
#     # New color/style options:
#     single.add_argument('--case-colors', nargs='+',
#                         help='Colors for each SU2 case (e.g. red blue)')
#     single.add_argument('--case-styles', nargs='+',
#                         help='Line styles for each SU2 case (e.g. - -- : )')

#     # MULTI-CASE COMPARISON MODE
#     compare = subparsers.add_parser("compare-surface",
#                                     help="Compare Cp/Cf/Transition plots across cases")
#     compare.add_argument('--cases',       nargs='+', required=True)
#     compare.add_argument('--xfoil-dir',   type=str)
#     compare.add_argument('--exp-dir',     type=str)
#     compare.add_argument('--save',        type=str, default='A')
#     compare.add_argument('--format',      type=str, default='svg')
#     compare.add_argument('--dpi',         type=int, default=300)
#     compare.add_argument('--no-it',       action='store_false')
#     compare.add_argument('--no-map',      action='store_false')
#     compare.add_argument('--show-airfoil', action='store_true',
#                          help='Include x–y airfoil shape panel')
#     compare.add_argument('--label-style',
#                          choices=['short', 'full', 'metadata', 'auto'],
#                          default='short')
#     # New color/style options:
#     compare.add_argument('--case-colors', nargs='+',
#                          help='Colors for each SU2 case (e.g. red blue)')
#     compare.add_argument('--case-styles', nargs='+',
#                          help='Line styles for each SU2 case (e.g. - -- : )')

#     # Optional legend control
#     compare.add_argument('--legends-cp', dest='legends_cp',
#                          action='store_true')
#     compare.add_argument('--no-legends-cp', dest='legends_cp',
#                          action='store_false')
#     compare.set_defaults(legends_cp=None)

#     compare.add_argument('--legends-cf', dest='legends_cf',
#                          action='store_false')
#     compare.add_argument('--no-legends-cf', dest='legends_cf',
#                          action='store_false')
#     compare.set_defaults(legends_cf=None)

#     compare.add_argument('--legends-it', dest='legends_it',
#                          action='store_true')
#     compare.add_argument('--no-legends-it', dest='legends_it',
#                          action='store_false')
#     compare.set_defaults(legends_it=None)

#     args = parser.parse_args()

#     if args.command == "single":
#         single_main(args)
#     elif args.command == "compare-surface":
#         compare_surface_main(args)
#     else:
#         parser.print_help()

# if __name__ == "__main__":
#     main()


# # su2_postprocess/cli/cli_main.py

# import argparse
# from su2_postprocess.cli.single_main import single_main
# from su2_postprocess.cli.compare_surface_main import compare_surface_main

# def main():
#     parser = argparse.ArgumentParser(description="SU2 Postprocessing CLI")
#     subparsers = parser.add_subparsers(dest="command")

#     # SINGLE CASE MODE
#     single = subparsers.add_parser("single", help="Process a single SU2 case")
#     single.add_argument('--root', required=True)
#     single.add_argument('--xfoil-dir', type=str)
#     single.add_argument('--xfoil-file', type=str)
#     single.add_argument('--exp-dir', type=str)
#     single.add_argument('--format', type=str, default='png')
#     single.add_argument('--dpi', type=int, default=300)
#     single.add_argument('--no-it', action='store_true')
#     single.add_argument('--no-map', action='store_true')
#     single.add_argument('--show-airfoil', action='store_true', help='Include x–y airfoil shape panel')
#     single.add_argument('--label-style',
#                         choices=['short', 'full', 'metadata', 'auto'],
#                         default='metadata',
#                         help="Legend label style")

#     # MULTI-CASE COMPARISON MODE
#     compare = subparsers.add_parser("compare-surface", help="Compare Cp/Cf/Transition plots across cases")
#     compare.add_argument('--cases', nargs='+', required=True)
#     compare.add_argument('--xfoil-dir', type=str)
#     compare.add_argument('--exp-dir', type=str)
#     compare.add_argument('--save', type=str, default='A')
#     compare.add_argument('--format', type=str, default='png')
#     compare.add_argument('--dpi', type=int, default=300)
#     compare.add_argument('--no-it', action='store_true')
#     compare.add_argument('--no-map', action='store_true')
#     compare.add_argument('--show-airfoil', action='store_true', help='Include x–y airfoil shape panel')
#     compare.add_argument('--label-style',
#                          choices=['short', 'full', 'metadata', 'auto'],
#                          default='metadata')

#     # Optional legend control (defaults to auto if omitted)
#     compare.add_argument('--legends-cp', dest='legends_cp', action='store_true')
#     compare.add_argument('--no-legends-cp', dest='legends_cp', action='store_false')
#     compare.set_defaults(legends_cp=None)

#     compare.add_argument('--legends-cf', dest='legends_cf', action='store_true')
#     compare.add_argument('--no-legends-cf', dest='legends_cf', action='store_false')
#     compare.set_defaults(legends_cf=None)

#     compare.add_argument('--legends-it', dest='legends_it', action='store_true')
#     compare.add_argument('--no-legends-it', dest='legends_it', action='store_false')
#     compare.set_defaults(legends_it=None)

#     args = parser.parse_args()

#     if args.command == "single":
#         single_main(args)
#     elif args.command == "compare-surface":
#         compare_surface_main(args)
#     else:
#         parser.print_help()

# if __name__ == "__main__":
#     main()



# # su2_postprocess/cli/cli_main.py

# import argparse
# from su2_postprocess.cli.single_main import single_main
# from su2_postprocess.cli.compare_surface_main import compare_surface_main


# def main():
#     parser = argparse.ArgumentParser(description="SU2 Postprocessing CLI")
#     subparsers = parser.add_subparsers(dest="command")

#     # SINGLE CASE MODE
#     single = subparsers.add_parser("single", help="Process a single SU2 case")
#     single.add_argument('--root', required=True)
#     single.add_argument('--xfoil-dir', type=str)
#     single.add_argument('--xfoil-file', type=str)
#     single.add_argument('--exp-dir', type=str)
#     single.add_argument('--format', type=str, default='png')
#     single.add_argument('--dpi', type=int, default=300)
#     single.add_argument('--no-it', action='store_true')
#     single.add_argument('--no-map', action='store_true')
#     single.add_argument('--label-style',
#                         choices=['short', 'full', 'metadata', 'auto'],
#                         default='metadata',
#                         help="Legend label style")

#     # MULTI-CASE COMPARISON MODE
#     compare = subparsers.add_parser("compare-surface", help="Compare Cp/Cf/Transition plots across cases")
#     compare.add_argument('--cases', nargs='+', required=True)
#     compare.add_argument('--xfoil-dir', type=str)
#     compare.add_argument('--exp-dir', type=str)
#     compare.add_argument('--save', type=str, default='A',
#     help="Save mode: A (common ancestor), B (cwd), C (home), D (first case)")
#     compare.add_argument('--format', type=str, default='png')
#     compare.add_argument('--dpi', type=int, default=300)
#     compare.add_argument('--no-it', action='store_true')
#     compare.add_argument('--no-map', action='store_true')
#     compare.add_argument('--label-style',
#                          choices=['short', 'full', 'metadata', 'auto'],
#                          default='short')

#     # Optional legend control (defaults to auto if omitted)
#     compare.add_argument('--legends-cp', dest='legends_cp', action='store_true')
#     compare.add_argument('--no-legends-cp', dest='legends_cp', action='store_false')
#     compare.set_defaults(legends_cp=None)

#     compare.add_argument('--legends-cf', dest='legends_cf', action='store_true')
#     compare.add_argument('--no-legends-cf', dest='legends_cf', action='store_false')
#     compare.set_defaults(legends_cf=None)

#     compare.add_argument('--legends-it', dest='legends_it', action='store_true')
#     compare.add_argument('--no-legends-it', dest='legends_it', action='store_false')
#     compare.set_defaults(legends_it=None)

#     args = parser.parse_args()

#     if args.command == "single":
#         single_main(args)
#     elif args.command == "compare-surface":
#         compare_surface_main(args)
#     else:
#         parser.print_help()

# if __name__ == "__main__":
#     main()