from su2_postprocess.cli.cli_main import main

if __name__ == "__main__":
    main()


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
# from su2_postprocess.plots.cp_cf import plot_cp_cf
# from su2_postprocess.plots.transition import plot_transition_map
# from su2_postprocess.utils.overlay_loader import (
#     load_overlay_file,
#     load_overlay_dir_by_prefix,
# )
# from su2_postprocess.utils.plot_helpers import _warn_missing_overlay


# def main():
#     parser = argparse.ArgumentParser(description="SU2 Postprocessing CLI")
#     subparsers = parser.add_subparsers(dest="command")

#     single = subparsers.add_parser("single", help="Process a single SU2 case")
#     single.add_argument('--root', required=True)
#     single.add_argument('--xfoil-dir', type=str)
#     single.add_argument('--xfoil-file', type=str)
#     single.add_argument('--exp-dir', type=str)
#     single.add_argument('--format', type=str, default='png')
#     single.add_argument('--dpi', type=int, default=300)
#     single.add_argument('--no-it', action='store_true')
#     single.add_argument('--no-map', action='store_true')
#     single.add_argument(
#         '--label-style',
#         choices=['short', 'full', 'metadata', 'auto'],
#         default='metadata',
#         help="Legend label style: 'short' (turb+trans), 'full' (turb+trans+M+Re+AoA), 'metadata' (none, box only), 'auto' (fallback to folder name)"
#     )


#     compare = subparsers.add_parser("compare-surface")
#     compare.add_argument('--cases', nargs='+', required=True)
#     compare.add_argument('--xfoil-dir', type=str)
#     compare.add_argument('--exp-dir', type=str)
#     compare.add_argument('--save', type=str)
#     compare.add_argument('--format', type=str, default='png')

#     args = parser.parse_args()

#     if args.command == "single":
#         print(f"Entered 'single' mode for root: {args.root}")
#         case_files = find_case_dirs(args.root)
#         print(f"Found {len(case_files)} surface file(s)")

#         exp_df = load_overlay_dir_by_prefix(args.exp_dir) if args.exp_dir else None
#         xfoil_input = None

#         if args.xfoil_file:
#             xfoil_input = load_overlay_file(Path(args.xfoil_file))
#             print(f"[INFO] Loaded XFOIL file: {args.xfoil_file}")
#         elif args.xfoil_dir:
#             xf_path = Path(args.xfoil_dir)
#             if xf_path.is_file():
#                 xfoil_input = load_overlay_file(xf_path)
#                 print(f"[INFO] Loaded XFOIL file: {xf_path}")
#             elif xf_path.is_dir():
#                 xfoil_input = load_overlay_dir_by_prefix(xf_path)
#                 if isinstance(xfoil_input, tuple):
#                     cp_dict, cf_dict = xfoil_input
#                     print(f"[INFO] Loaded XFOIL directory with Cp entries: {list(cp_dict.keys())}")
#                     print(f"[INFO] Loaded XFOIL directory with Cf entries: {list(cf_dict.keys())}")
#                 else:
#                     print(f"[INFO] Loaded XFOIL directory with entries: {list(xfoil_input.keys()) if xfoil_input else 'None'}")

#         for surface_path in case_files:
#             print(f"Processing: {surface_path}")
#             case = surface_path.parent
#             forces_files = list(case.glob("forces_bdwn_*.dat"))

#             if not forces_files:
#                 print(f"[WARN] No forces_bdwn_*.dat file found in: {case}")
#                 continue

#             forces_path = forces_files[0]

#             try:
#                 df, elems = parse_felineseg_surface(surface_path)
#                 df = reorder_surface_nodes_from_elements(df, elems)
#                 label, metadata = extract_case_metadata_fallback(forces_path, return_metadata=True)

#                 fallback_label = case.name

#                 try:
#                     turb = metadata['turbulence_model']
#                     trans = metadata['transition_model']
#                     mach = metadata['mach']
#                     re = metadata['reynolds']
#                     aoa = metadata['alpha']

#                     if args.label_style == 'short':
#                         label_str = f"{turb}-{trans}"
#                     elif args.label_style == 'full':
#                         label_str = f"{turb}-{trans}_M{mach:.2f}_Re{int(re/1e6)}e6_AoA{int(round(aoa))}"
#                     elif args.label_style == 'metadata':
#                         label_str = None  # suppress legend label
#                     else:
#                         label_str = fallback_label

#                 except Exception:
#                     print("[WARN] Incomplete metadata, falling back to directory name.")
#                     label_str = fallback_label

#                 # label_str = f"M{metadata['mach']:.2f}_Re{int(metadata['reynolds']/1e6)}e6_AoA{int(round(metadata['alpha']))}" if all(v is not None for v in (metadata['mach'], metadata['reynolds'], metadata['alpha'])) else case.name

#                 xfoil_label_input = None
#                 if isinstance(xfoil_input, tuple):
#                     cp_dict, cf_dict = xfoil_input
#                     xfoil_matches = {k: v for k, v in cp_dict.items() if k.startswith(label_str)}
#                     if xfoil_matches:
#                         print(f"[INFO] Matched multiple XFOIL Cp overlays for: {label_str} -> {list(xfoil_matches.keys())}")
#                         xfoil_label_input = xfoil_matches
#                     else:
#                         print(f"[WARN] No XFOIL Cp overlay found for: {label_str}")
#                 elif isinstance(xfoil_input, dict):
#                     xfoil_matches = {k: v for k, v in xfoil_input.items() if k.startswith(label_str)}
#                     if xfoil_matches:
#                         print(f"[INFO] Matched multiple XFOIL overlays for: {label_str} -> {list(xfoil_matches.keys())}")
#                         xfoil_label_input = xfoil_matches
#                     else:
#                         print(f"[WARN] No XFOIL overlay found or empty for: {label_str}")
#                 elif isinstance(xfoil_input, pd.DataFrame):
#                     xfoil_label_input = xfoil_input
#                     print("[INFO] Using directly provided XFOIL overlay from file")

#                 if isinstance(xfoil_label_input, pd.DataFrame) and xfoil_label_input.empty:
#                     xfoil_label_input = None

#                 exp_input = exp_df.get(label_str) if isinstance(exp_df, dict) else exp_df

#                 if xfoil_label_input is None:
#                     print(f"[WARN] No XFOIL overlay found or empty for: {label_str}")
#                 else:
#                     _warn_missing_overlay([label_str], {label_str: xfoil_label_input} if isinstance(xfoil_label_input, pd.DataFrame) else xfoil_label_input, name="XFOIL")

#                 if exp_input is not None:
#                     _warn_missing_overlay([label_str], exp_df, name="EXP")

#                 fig = plot_cp_cf_it_multi(
#                     [df], [label_str if label_str else fallback_label],
#                     show_cp=True, show_cf=True,
#                     show_it=False,
#                     show_map=not args.no_map,
#                     forces_files=[forces_path],
#                     xfoil_data=xfoil_label_input,
#                     exp_data=exp_input,
#                     label_style=args.label_style,
#                     show_legends_cp=True,
#                     show_legends_cf=False,
#                     show_legends_it=False,
#                 )

#                 fig_cp, fig_cf = plot_cp_cf(df)
#                 fig_ti = plot_transition_map(df)

#                 save_plot(fig, case, "Cp_Cf_TurbIndex", format=args.format, dpi=args.dpi)
#                 save_plot(fig_cp, case, "Cp_vs_x", format=args.format, dpi=args.dpi)
#                 save_plot(fig_cf, case, "Cf_vs_x", format=args.format, dpi=args.dpi)
#                 if fig_ti is not None:
#                     save_plot(fig_ti, case, "Transition_Map", format=args.format, dpi=args.dpi)

#             except Exception as e:
#                 print(f"[ERROR] Failed on {case}: {e}")

#     elif args.command == "compare-surface":
#         print("[ERROR] compare-surface not yet updated for new metadata structure.")
#     else:
#         parser.print_help()


# if __name__ == "__main__":
#     main()





# import argparse
# from pathlib import Path
# import pandas as pd
# from su2_postprocess.io.file_scan import find_case_dirs
# from su2_postprocess.io.output import save_plot
# from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
# from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_from_log
# from su2_postprocess.utils.path_helpers import get_comparison_save_path
# from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi
# from su2_postprocess.plots.cp_cf import plot_cp_cf
# from su2_postprocess.plots.transition import plot_transition_map
# from su2_postprocess.utils.overlay_loader import (
#     load_overlay_file,
#     load_overlay_dir_by_prefix,
#     find_xfoil_by_metadata,
# )
# from su2_postprocess.utils.plot_helpers import _warn_missing_overlay


# def main():
#     parser = argparse.ArgumentParser(description="SU2 Postprocessing CLI")
#     subparsers = parser.add_subparsers(dest="command")

#     single = subparsers.add_parser("single", help="Process a single SU2 case")
#     single.add_argument('--root', required=True)
#     single.add_argument('--xfoil-dir', type=str)
#     single.add_argument('--xfoil-file', type=str)
#     single.add_argument('--exp-dir', type=str)

#     compare = subparsers.add_parser("compare-surface")
#     compare.add_argument('--cases', nargs='+', required=True)
#     compare.add_argument('--xfoil-dir', type=str)
#     compare.add_argument('--exp-dir', type=str)
#     compare.add_argument('--save', type=str)
#     compare.add_argument('--format', type=str, default='png')

#     args = parser.parse_args()

#     if args.command == "single":
#         print(f"Entered 'single' mode for root: {args.root}")
#         case_files = find_case_dirs(args.root)
#         print(f"Found {len(case_files)} surface file(s)")

#         exp_df = load_overlay_dir_by_prefix(args.exp_dir) if args.exp_dir else None
#         xfoil_input = None

#         if args.xfoil_file:
#             xfoil_input = load_overlay_file(Path(args.xfoil_file))
#             print(f"[INFO] Loaded XFOIL file: {args.xfoil_file}")
#         elif args.xfoil_dir:
#             xf_path = Path(args.xfoil_dir)
#             if xf_path.is_file():
#                 xfoil_input = load_overlay_file(xf_path)
#                 print(f"[INFO] Loaded XFOIL file: {xf_path}")
#             elif xf_path.is_dir():
#                 xfoil_input = load_overlay_dir_by_prefix(xf_path)
#                 if isinstance(xfoil_input, tuple):
#                     cp_dict, cf_dict = xfoil_input
#                     print(f"[INFO] Loaded XFOIL directory with Cp entries: {list(cp_dict.keys())}")
#                     print(f"[INFO] Loaded XFOIL directory with Cf entries: {list(cf_dict.keys())}")
#                 else:
#                     print(f"[INFO] Loaded XFOIL directory with entries: {list(xfoil_input.keys()) if xfoil_input else 'None'}")

#         for surface_path in case_files:
#             print(f"Processing: {surface_path}")
#             case = surface_path.parent
#             forces_path = case / "forces_bdwn_.dat"

#             try:
#                 df, elems = parse_felineseg_surface(surface_path)
#                 df = reorder_surface_nodes_from_elements(df, elems)
#                 label, metadata = extract_case_metadata_from_log(forces_path, return_metadata=True)
#                 label_str = f"M{metadata['mach']:.2f}_Re{int(metadata['reynolds']/1e6)}e6_AoA{int(round(metadata['alpha']))}"

#                 xfoil_label_input = None
#                 if isinstance(xfoil_input, tuple):
#                     cp_dict, cf_dict = xfoil_input
#                     xfoil_matches = {k: v for k, v in cp_dict.items() if k.startswith(label_str)}
#                     if xfoil_matches:
#                         print(f"[INFO] Matched multiple XFOIL Cp overlays for: {label_str} -> {list(xfoil_matches.keys())}")
#                         xfoil_label_input = xfoil_matches
#                     else:
#                         print(f"[WARN] No XFOIL Cp overlay found for: {label_str}")
#                 elif isinstance(xfoil_input, dict):
#                     xfoil_matches = {k: v for k, v in xfoil_input.items() if k.startswith(label_str)}
#                     if xfoil_matches:
#                         print(f"[INFO] Matched multiple XFOIL overlays for: {label_str} -> {list(xfoil_matches.keys())}")
#                         xfoil_label_input = xfoil_matches
#                     else:
#                         print(f"[WARN] No XFOIL overlay found or empty for: {label_str}")
#                 elif isinstance(xfoil_input, pd.DataFrame):
#                     xfoil_label_input = xfoil_input
#                     print("[INFO] Using directly provided XFOIL overlay from file")

#                 if isinstance(xfoil_label_input, pd.DataFrame) and xfoil_label_input.empty:
#                     xfoil_label_input = None

#                 exp_input = exp_df.get(label_str) if isinstance(exp_df, dict) else exp_df

#                 if xfoil_label_input is None:
#                     print(f"[WARN] No XFOIL overlay found or empty for: {label_str}")
#                 else:
#                     _warn_missing_overlay([label_str], {label_str: xfoil_label_input} if isinstance(xfoil_label_input, pd.DataFrame) else xfoil_label_input, name="XFOIL")

#                 if exp_input is not None:
#                     _warn_missing_overlay([label_str], exp_df, name="EXP")

#                 fig = plot_cp_cf_it_multi(
#                     [df], [label_str],
#                     show_cp=True, show_cf=True,
#                     show_it=False, show_map=True,
#                     forces_files=[forces_path],
#                     xfoil_data=xfoil_label_input,
#                     exp_data=exp_input,
#                 )

#                 fig_cp, fig_cf = plot_cp_cf(df)
#                 fig_ti = plot_transition_map(df)

#                 save_plot(fig, case, "Cp_Cf_TurbIndex")
#                 save_plot(fig_cp, case, "Cp_vs_x")
#                 save_plot(fig_cf, case, "Cf_vs_x")
#                 if fig_ti is not None:
#                     save_plot(fig_ti, case, "Transition_Map")

#             except Exception as e:
#                 print(f"[ERROR] Failed on {case}: {e}")

#     elif args.command == "compare-surface":
#         print("[ERROR] compare-surface not fully updated in this context.")
#     else:
#         parser.print_help()


# if __name__ == "__main__":
#     main()


# import argparse
# from pathlib import Path
# from su2_postprocess.io.file_scan import find_case_dirs
# from su2_postprocess.io.output import save_plot
# from su2_postprocess.parser.surface import parse_surface_file
# from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
# from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
# from su2_postprocess.utils.parse_forces import extract_case_metadata_from_log
# from su2_postprocess.utils.path_helpers import get_comparison_save_path
# from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi
# from su2_postprocess.plots.cp_cf import plot_cp_cf
# from su2_postprocess.plots.transition import plot_transition_map
# from su2_postprocess.utils.overlay_loader import load_overlay_dir_by_prefix
# from su2_postprocess.utils.plot_helpers import _warn_missing_overlay

# def main():
#     parser = argparse.ArgumentParser(description="SU2 Postprocessing CLI")
#     subparsers = parser.add_subparsers(dest="command")

#     # --- Single case processing ---
#     single = subparsers.add_parser("single", help="Process all cases under --root")
#     single.add_argument('--root', required=True, help='Root directory of SU2 case')
#     single.add_argument('--xfoil-dir', type=str, help='Directory containing XFOIL overlay files')
#     single.add_argument('--exp-dir', type=str, help='Directory containing EXP overlay files')

#     # --- Multi-case comparison ---
#     compare = subparsers.add_parser("compare-surface", help="Compare Cp, Cf, i_t across multiple cases")
#     compare.add_argument('--cases', nargs='+', type=str, required=True, help='List of case directories')
#     compare.add_argument('--xfoil-dir', type=str, help='Directory containing XFOIL overlay files')
#     compare.add_argument('--exp-dir', type=str, help='Directory containing EXP overlay files')
#     compare.add_argument('--save', type=str, help='Path to save output plot (omit to show only)')
#     compare.add_argument('--format', type=str, default='png', choices=['png', 'svg', 'eps'], help='Output file format')

#     args = parser.parse_args()

#     if args.command == "compare-surface":
#         dfs, labels, forces_files = [], [], []
#         case_paths = [Path(c) for c in args.cases]

#         for path in case_paths:
#             surface = path / "flow_surf_.dat"
#             df, elems = parse_felineseg_surface(surface)
#             df = reorder_surface_nodes_from_elements(df, elems)
#             dfs.append(df)
#             labels.append(path.name)
#             forces_files.append(path / "forces_bdwn_.dat")

#         xfoil_df = load_overlay_dir_by_prefix(args.xfoil_dir) if args.xfoil_dir else None
#         exp_df = load_overlay_dir_by_prefix(args.exp_dir) if args.exp_dir else None

#         _warn_missing_overlay(labels, xfoil_df, name="XFOIL")
#         _warn_missing_overlay(labels, exp_df, name="EXP")

#         fig = plot_cp_cf_it_multi(
#             dfs, labels,
#             forces_files=forces_files,
#             xfoil_data=xfoil_df, exp_data=exp_df)

#         save_dir = Path(args.save) if args.save else get_comparison_save_path(case_paths)
#         save_plot(fig, save_dir, "Cp_Cf_iT_Compare", fmt=args.format)

#     elif args.command == "single":
#         print(f"Entered 'single' mode for root: {args.root}")
#         case_files = find_case_dirs(args.root)
#         print(f"Found {len(case_files)} surface file(s)")

#         xfoil_df = load_overlay_dir_by_prefix(args.xfoil_dir) if args.xfoil_dir else None
#         exp_df = load_overlay_dir_by_prefix(args.exp_dir) if args.exp_dir else None

#         for surface_path in case_files:
#             print(f"Processing: {surface_path}")
#             case = surface_path.parent
#             forces_path = case / "forces_bdwn_.dat"

#             try:
#                 df, elements = parse_felineseg_surface(surface_path)
#                 df = reorder_surface_nodes_from_elements(df, elements)

#                 label = case.name
#                 _warn_missing_overlay([label], xfoil_df, name="XFOIL")
#                 _warn_missing_overlay([label], exp_df, name="EXP")

#                 fig = plot_cp_cf_it_multi(
#                     [df], [label],
#                     show_cp=True, show_cf=True,
#                     show_it=False, show_map=True,
#                     forces_files=[forces_path],
#                     xfoil_data=xfoil_df.get(label) if isinstance(xfoil_df, dict) else xfoil_df,
#                     exp_data=exp_df.get(label) if isinstance(exp_df, dict) else exp_df)

#                 fig_cp, fig_cf = plot_cp_cf(df)
#                 fig_ti = plot_transition_map(df)

#                 save_plot(fig, case, "Cp_Cf_TurbIndex")
#                 save_plot(fig_cp, case, "Cp_vs_x")
#                 save_plot(fig_cf, case, "Cf_vs_x")
#                 if fig_ti is not None:
#                     save_plot(fig_ti, case, "Transition_Map")

#             except Exception as e:
#                 print(f"[ERROR] Failed on {case}: {e}")


#     else:
#         parser.print_help()

# if __name__ == "__main__":
#     main()


# import argparse
# from pathlib import Path
# from su2_postprocess.io.file_scan import find_case_dirs
# from su2_postprocess.io.output import save_plot
# from su2_postprocess.parser.surface import parse_surface_file
# from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
# from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
# from su2_postprocess.utils.parse_forces import extract_case_metadata_from_log
# from su2_postprocess.utils.path_helpers import get_comparison_save_path
# from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi
# from su2_postprocess.plots.cp_cf import plot_cp_cf
# from su2_postprocess.plots.transition import plot_transition_map
# from su2_postprocess.utils.overlay_loader import load_overlay_dir


# def main():
#     parser = argparse.ArgumentParser(description="SU2 Postprocessing CLI")
#     subparsers = parser.add_subparsers(dest="command")

#     # --- Single case processing ---
#     single = subparsers.add_parser("single", help="Process all cases under --root")
#     single.add_argument('--root', required=True, help='Root directory of SU2 case')
#     single.add_argument('--xfoil-dir', type=str, help='Directory containing XFOIL overlay files')
#     single.add_argument('--exp-dir', type=str, help='Directory containing EXP overlay files')

#     # --- Multi-case comparison ---
#     compare = subparsers.add_parser("compare-surface", help="Compare Cp, Cf, i_t across multiple cases")
#     compare.add_argument('--cases', nargs='+', type=str, required=True, help='List of case directories')
#     compare.add_argument('--xfoil-dir', type=str, help='Directory containing XFOIL overlay files')
#     compare.add_argument('--exp-dir', type=str, help='Directory containing EXP overlay files')
#     compare.add_argument('--save', type=str, help='Path to save output plot (omit to show only)')
#     compare.add_argument('--format', type=str, default='png', choices=['png', 'svg', 'eps'], help='Output file format')

#     args = parser.parse_args()

#     if args.command == "compare-surface":
#         dfs, labels, forces_files = [], [], []
#         case_paths = [Path(c) for c in args.cases]

#         for path in case_paths:
#             surface = path / "flow_surf_.dat"
#             df, elems = parse_felineseg_surface(surface)
#             df = reorder_surface_nodes_from_elements(df, elems)
#             dfs.append(df)
#             labels.append(path.name)
#             forces_files.append(path / "forces_bdwn_.dat")

#         xfoil_df = load_overlay_dir(args.xfoil_dir) if args.xfoil_dir else None
#         exp_df = load_overlay_dir(args.exp_dir) if args.exp_dir else None

#         fig = plot_cp_cf_it_multi(dfs, labels, forces_files=forces_files,
#                                   xfoil_data=xfoil_df, exp_data=exp_df)

#         save_dir = Path(args.save) if args.save else get_comparison_save_path(case_paths)
#         save_plot(fig, save_dir, "Cp_Cf_iT_Compare", fmt=args.format)

#     elif args.command == "single":
#         print(f"Entered 'single' mode for root: {args.root}")
#         case_files = find_case_dirs(args.root)
#         print(f"Found {len(case_files)} surface file(s)")

#         xfoil_df = load_overlay_dir(args.xfoil_dir) if args.xfoil_dir else None
#         exp_df = load_overlay_dir(args.exp_dir) if args.exp_dir else None

#         for surface_path in case_files:
#             print(f"Processing: {surface_path}")
#             case = surface_path.parent
#             forces_path = case / "forces_bdwn_.dat"

#             try:
#                 df, elements = parse_felineseg_surface(surface_path)
#                 df = reorder_surface_nodes_from_elements(df, elements)

#                 fig = plot_cp_cf_it_multi([df], [case.name],
#                                           show_cp=True, show_cf=True,
#                                           show_it=False, show_map=True,
#                                           forces_files=[forces_path],
#                                           xfoil_data=xfoil_df.get(case.name) if isinstance(xfoil_df, dict) else xfoil_df,
#                                           exp_data=exp_df.get(case.name) if isinstance(exp_df, dict) else exp_df)

#                 fig_cp, fig_cf = plot_cp_cf(df)
#                 fig_ti = plot_transition_map(df)

#                 save_plot(fig, case, "Cp_Cf_TurbIndex")
#                 save_plot(fig_cp, case, "Cp_vs_x")
#                 save_plot(fig_cf, case, "Cf_vs_x")
#                 if fig_ti is not None:
#                     save_plot(fig_ti, case, "Transition_Map")

#             except Exception as e:
#                 print(f"[ERROR] Failed on {case}: {e}")

#     else:
#         parser.print_help()


# if __name__ == "__main__":
#     main()




# import argparse
# from pathlib import Path
# from su2_postprocess.io.file_scan import find_case_dirs
# from su2_postprocess.io.output import save_plot
# from su2_postprocess.parser.surface import parse_surface_file
# from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
# from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
# from su2_postprocess.utils.parse_forces import extract_case_metadata_from_log
# from su2_postprocess.plots.cp_cf import plot_cp_cf
# from su2_postprocess.plots.transition import plot_transition_map
# from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi
# from su2_postprocess.utils.path_helpers import get_comparison_save_path


# def main():
#     parser = argparse.ArgumentParser(description="SU2 Postprocessing CLI")
#     subparsers = parser.add_subparsers(dest="command")

#     # --- Subcommand: Standard processing for batch case directory
#     single = subparsers.add_parser("single", help="Process all cases under --root")
#     single.add_argument('--root', required=True, help='Root directory of SU2 cases')

#     # --- Subcommand: Compare Cp/Cf/i_t across multiple cases
#     compare = subparsers.add_parser("compare-surface", help="Compare Cp, Cf, i_t across multiple cases")
#     compare.add_argument('--cases', nargs='+', type=str, required=True, help='List of case directories')
#     compare.add_argument('--log-cf', action='store_true', help='Use log scale for Cf axis')
#     compare.add_argument('--save', type=str, help='Path to save output plot (omit to show only)')
#     compare.add_argument('--format', type=str, default='png', choices=['png', 'svg', 'eps'], help='Output file format')

#     args = parser.parse_args()

#     if args.command == "compare-surface":
#         dfs, labels, forces_files = [], [], []
#         case_paths = [Path(c) for c in args.cases]

#         for path in case_paths:
#             surface = path / "flow_surf_.dat"
#             df, elems = parse_felineseg_surface(surface)
#             df = reorder_surface_nodes_from_elements(df, elems)
#             dfs.append(df)
#             labels.append(path.name)
#             forces_files.append(path / "forces_bdwn_.dat")

#         fig = plot_cp_cf_it_multi(dfs, labels, forces_files=forces_files)

#         save_dir = get_comparison_save_path(case_paths)
#         save_plot(fig, save_dir, "Cp_Cf_iT_Compare", fmt=args.format)

#     elif args.command == "single":
#         print(f"Entered 'single' mode for root: {args.root}")
#         case_files = find_case_dirs(args.root)
#         print(f"Found {len(case_files)} surface file(s)")
        
#         for surface_path in case_files:    
#             print(f"Processing: {surface_path}")
#             case = surface_path.parent
#             forces_path = case / "forces_bdwn_.dat"
#             try:
#                 df, elements = parse_felineseg_surface(surface_path)
#                 df = reorder_surface_nodes_from_elements(df, elements)
#                 fig_cp, fig_cf = plot_cp_cf(df)
#                 fig_ti = plot_transition_map(df)
#                 fig_combined = plot_cp_cf_it_multi(
#                     [df], ["SU2"],
#                     show_cp=True,
#                     show_cf=True,
#                     show_it=False,
#                     show_map=True,
#                     forces_files=[forces_path]
#                 )
#                 save_plot(fig_combined, case, "Cp_Cf_TurbIndex")
#                 save_plot(fig_cp, case, "Cp_vs_x")
#                 save_plot(fig_cf, case, "Cf_vs_x")
#                 if fig_ti is not None:
#                     save_plot(fig_ti, case, "Transition_Map")
#             except Exception as e:
#                 print(f"[ERROR] Failed on {case}: {e}")
#     else:
#         parser.print_help()

# if __name__ == "__main__":
#     main()        



'''
def main():
    parser = argparse.ArgumentParser(description="SU2 Postprocessing Pipeline")
    parser.add_argument('--root', required=True, help='Root directory of SU2 cases')
    args = parser.parse_args()

    compare = subparsers.add_parser("compare-surface", help="Compare Cp, Cf, i_t across multiple cases")
    compare.add_argument('--cases', nargs='+', type=str, required=True, help='List of case directories')
    compare.add_argument('--log-cf', action='store_true', help='Use log scale for Cf axis')
    compare.add_argument('--save', type=str, help='Path to save output plot (omit to show only)')
    compare.add_argument('--format', type=str, default='png', choices=['png', 'svg'], help='Output file format')

    elif args.command == "compare-surface":
        from su2_postprocess.plots.compare_surface import plot_comparison
        case_paths = [Path(c) for c in args.cases]
        plot_comparison(case_paths, log_cf=args.log_cf, save_path=args.save, fmt=args.format)


    case_files = find_case_dirs(args.root)
    for surface_path in case_files:
        case = surface_path.parent
        print(f"Processing case: {case}")
        forces_path = case / "forces_bdwn_.dat"
        df, elements = parse_felineseg_surface(surface_path)
        df = reorder_surface_nodes_from_elements(df, elements)
        fig_cp, fig_cf = plot_cp_cf(df)
        fig_ti = plot_transition_map(df)
        fig_combined = plot_cp_cf_it(df, show_cp=True, show_cf=True, show_it=False, show_map=True, forces_file=forces_path)
        save_plot(fig_combined, case, "Cp_Cf_TurbIndex")
        save_plot(fig_cp, case, "Cp_vs_x")
        save_plot(fig_cf, case, "Cf_vs_x")
        if fig_ti is not None:
        	save_plot(fig_ti, case, "Transition_Map")

if __name__ == "__main__":
    main()
'''    
