# # compare_surface_py
# import argparse
# from pathlib import Path
# import pandas as pd
# from su2_postprocess.io.file_scan import find_case_dirs
# from su2_postprocess.io.output import save_plot
# from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
# from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback
# from su2_postprocess.utils.path_helpers import get_comparison_save_path
# from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi
# from su2_postprocess.utils.overlay_loader import load_overlay_dir_by_prefix
# from su2_postprocess.utils.plot_helpers import _warn_missing_overlay


# def compare_surface_main(args):
#     case_dirs = [Path(d) for d in args.cases]
#     all_dfs, all_labels, all_forces, all_case_paths = [], [], [], []

#     xfoil_data = load_overlay_dir_by_prefix(args.xfoil_dir) if args.xfoil_dir else None
#     exp_data = load_overlay_dir_by_prefix(args.exp_dir) if args.exp_dir else None

#     for case_path in case_dirs:
#         surface_files = find_case_dirs(case_path)
#         if not surface_files:
#             print(f"[WARN] No surface file found in {case_path}")
#             continue

#         surface_path = surface_files[0]
#         forces_files = list(case_path.glob("forces_bdwn_*.dat"))
#         if not forces_files:
#             print(f"[WARN] No forces_bdwn_*.dat file in {case_path}")
#             continue
#         forces_path = forces_files[0]

#         try:
#             df, elems = parse_felineseg_surface(surface_path)
#             df = reorder_surface_nodes_from_elements(df, elems)
#             label, metadata = extract_case_metadata_fallback(forces_path, return_metadata=True)

#             if args.label_style == 'short':
#                 label_str = f"{metadata['turbulence_model']}-{metadata['transition_model']}"
#             elif args.label_style == 'full':
#                 label_str = f"{metadata['turbulence_model']}-{metadata['transition_model']}_M{metadata['mach']:.2f}_Re{int(metadata['reynolds']/1e6)}e6_AoA{int(round(metadata['alpha']))}"
#             elif args.label_style == 'metadata':
#                 label_str = None  # suppress legend
#             else:
#                 label_str = case_path.name

#             all_dfs.append(df)
#             all_labels.append(label_str)
#             all_forces.append(forces_path)
#             all_case_paths.append(case_path)

#         except Exception as e:
#             print(f"[ERROR] Failed to parse case {case_path}: {e}")

#     save_path = get_comparison_save_path(all_case_paths, mode=args.save or 'A')
#     label_set = [lbl if lbl else "case" for lbl in all_labels]

#     fig = plot_cp_cf_it_multi(
#         all_dfs,
#         label_set,
#         show_cp=True,
#         show_cf=True,
#         show_it=not args.no_it,
#         show_map=not args.no_map,
#         forces_files=all_forces,
#         xfoil_data=xfoil_data,
#         exp_data=exp_data,
#         show_legends=args.legends
#     )

#     save_plot(fig, save_path, "Cp_Cf_TurbIndex_Compare", format=args.format, dpi=args.dpi)


# def register_compare_surface_parser(subparsers):
#     compare = subparsers.add_parser("compare-surface")
#     compare.add_argument('--cases', nargs='+', required=True)
#     compare.add_argument('--xfoil-dir', type=str)
#     compare.add_argument('--exp-dir', type=str)
#     compare.add_argument('--save', type=str)
#     compare.add_argument('--format', type=str, default='png')
#     compare.add_argument('--dpi', type=int, default=300)
#     compare.add_argument('--no-it', action='store_true')
#     compare.add_argument('--no-map', action='store_true')
#     compare.add_argument('--label-style', choices=['short', 'full', 'metadata', 'auto'], default='metadata')
#     compare.add_argument('--legends', action='store_true')



# # import matplotlib.pyplot as plt
# # import numpy as np
# # from pathlib import Path
# # from matplotlib.collections import LineCollection
# # from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
# # from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
# # from su2_postprocess.utils.parse_forces import extract_case_metadata

# # def plot_comparison(case_dirs, log_cf=False, save_path=None, fmt='png'):
# #     colors = plt.get_cmap('tab10').colors
# #     fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True, height_ratios=[1, 1, 1])
# #     ax_cp, ax_cf, ax_it = axes

# #     for idx, case in enumerate(case_dirs):
# #         surface_path = Path(case) / "flow_surf_.dat"
# #         forces_path = Path(case) / "forces_bdwn_.dat"
# #         if not surface_path.exists():
# #             print(f"[Skipped] Missing surface file in {case}")
# #             continue

# #         df, elements = parse_felineseg_surface(surface_path)
# #         df = reorder_surface_nodes_from_elements(df, elements)

# #         x = df["x"].to_numpy()
# #         cp = df["Pressure_Coefficient"].to_numpy()
# #         cf = df["Skin_Friction_Coefficient_x"].to_numpy()
# #         it = df["Turb_index"].clip(0, 1).to_numpy() if "Turb_index" in df.columns else None
# #         label = extract_case_metadata(forces_path)

# #         color = colors[idx % len(colors)]
# #         ax_cp.plot(x, cp, lw=1, label=label, color=color)
# #         ax_cf.plot(x, cf, lw=1, color=color)
# #         if it is not None:
# #             ax_it.plot(x, it, lw=1, color=color)

# #     # Cp formatting
# #     ax_cp.set_ylabel(r"$C_p$")
# #     ax_cp.invert_yaxis()
# #     ax_cp.grid(False)
# #     ax_cp.legend(
# #         fontsize=9,
# #         loc='center left',
# #         bbox_to_anchor=(1.01, 0.5),
# #         frameon=True
# #     )
# #     fig.subplots_adjust(right=0.75)


# # 	# Cf formatting
# #     ax_cf.set_ylabel(r"$C_f$")
# #     ax_cf.grid(False)
# #     # if log_cf:
# #     #     ax_cf.set_yscale("log")
# #     # else:
# #     #     ax_cf.set_yscale("linear")

# #     # i_t formatting
# #     ax_it.set_ylabel(r"$i_t$")
# #     ax_it.set_ylim(0, 1.05)
# #     ax_it.set_xlabel(r"$x/c$")
# #     ax_it.grid(False)

# #     for ax in axes:
# #         ax.spines['top'].set_visible(False)
# #         ax.spines['right'].set_visible(False)
# #         ax.tick_params(axis='both', direction='in', width=0.8)

# #     fig.tight_layout()

# #     if save_path:
# #         save_path = Path(save_path).with_suffix(f".{fmt}")
# #         fig.savefig(save_path, dpi=300, bbox_inches='tight')
# #         print(f"[Saved] Comparison plot saved to: {save_path}")
# #     else:
# #         plt.show()
