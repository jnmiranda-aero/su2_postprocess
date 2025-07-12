from pathlib import Path
from su2_postprocess.io.file_scan import find_case_dirs
from su2_postprocess.io.output import save_plot
from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback
from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi
from su2_postprocess.plots.cp_cf import plot_cp_cf
from su2_postprocess.plots.transition import plot_transition_map
from su2_postprocess.utils.label_helpers import flow_metadata_text, legend_label
from su2_postprocess.utils.overlay_loader import (load_overlay_file, load_overlay_dir_by_prefix)

def single_main(args):
    print(f"Entered 'single' mode for root: {args.root}")
    surf_files = find_case_dirs(args.root)
    print(f"Found {len(surf_files)} surface file(s)")
    if not surf_files:
        print("[ERROR] No surface files found.")
        return

    # ---------- overlay dirs/files -------------------------------------------
    exp_cp, exp_cf = {}, {}
    xcp, xcf       = {}, {}
    msescp, msescf = {}, {}

    if args.mses_dir:
        for d in args.mses_dir:
            cp_d, cf_d = load_overlay_dir_by_prefix(Path(d))
            # use the directory’s basename as a prefix
            tag = Path(d).name
            # rename each key so they’re unique
            cp_d = {f"{tag}_{k}": v for k, v in cp_d.items()}
            cf_d = {f"{tag}_{k}": v for k, v in cf_d.items()}
            msescp.update(cp_d)
            msescf.update(cf_d)

    # 2) EXP overlays
    if args.exp_dir:
        exp_cp, exp_cf = load_overlay_dir_by_prefix(Path(args.exp_dir))

    # 3) Single-file MSES override
    if args.mses_file:
        p = Path(args.mses_file)
        target = msescp if p.stem.lower().startswith("cp_") else msescf
        target[p.stem.lower()] = load_overlay_file(p)

    # 4) Finally load XFOIL overlays
    if args.xfoil_file:
        p = Path(args.xfoil_file)
        target = xcp if p.stem.lower().startswith("cp_") else xcf
        target[p.stem.lower()] = load_overlay_file(p)
    elif args.xfoil_dir:
        xcp, xcf = load_overlay_dir_by_prefix(Path(args.xfoil_dir))

    # ---------- process ALL cases --------------------------------------------
    for surf_path in surf_files:
        try:
            case_dir   = surf_path.parent
            force_file = next(case_dir.glob("forces_bdwn_*.dat"), None)
            if force_file is None:
                print(f"[WARN] No forces_bdwn_*.dat found in {case_dir}")
                continue

            df, elems = parse_felineseg_surface(surf_path)
            df = reorder_surface_nodes_from_elements(df, elems)
            label_full, meta = extract_case_metadata_fallback(force_file, return_metadata=True, fields=["turbulence_model", "transition_model", "reynolds", "tu", "mach", "alpha"])
            slug = f"m{meta['mach']:.2f}_re{int(meta['reynolds']/1e6)}e6_aoa{int(round(meta['alpha']))}".lower()
            
            def pick(dct, prefix):
                # exact match first
                key = f"{prefix}_{slug}"
                if key in dct:
                    return dct[key]
                for k, v in dct.items():
                    if slug in k:
                        return v
                # only fallback if prefix == "cp" *and* exactly one cp exists
                if prefix == "cp" and len(dct) == 1:
                    return next(iter(dct.values()))
                    return None

            all_xfoil_cp = list(xcp.values())
            all_xfoil_cf = list(xcf.values())
            all_mses_cp  = list(msescp.values())
            all_mses_cf  = list(msescf.values())
            all_exp_cp   = list(exp_cp.values())
            all_exp_cf   = list(exp_cf.values())                

            print("Loaded MSES CP overlays:", len(all_mses_cp))
            print("Loaded MSES CF overlays:", len(all_mses_cf))

            fig = plot_cp_cf_it_multi(
                [df], [legend_label(meta, args.label_style, "", 1)],
                meta_box_text=label_full,
                show_meta_box=True,
                show_cp=True, show_cf=True,
                show_it=args.it, show_map=args.map,
                show_airfoil=args.show_airfoil,
                forces_files=[force_file],
                xfoil_cp_list=all_xfoil_cp,
                xfoil_cf_list=all_xfoil_cf,
                mses_cp_list= all_mses_cp,
                mses_cf_list= all_mses_cf,
                exp_cp_list= all_exp_cp,
                exp_cf_list= all_exp_cf,
                label_style=args.label_style,
                show_legends_cp=False,
                show_legends_cf=False,
                show_legends_it=False,
                case_colors=args.case_colors,
                case_styles=args.case_styles,
            )

            save_plot(fig, case_dir, "Cp_Cf_TurbIndex", format=args.format, dpi=args.dpi)


            # Optional specialized plots
            fig_cp, fig_cf = plot_cp_cf(df)
            fig_ti         = plot_transition_map(df)
            save_plot(fig_cp, case_dir, "Cp_vs_x", format=args.format, dpi=args.dpi)
            save_plot(fig_cf, case_dir, "Cf_vs_x", format=args.format, dpi=args.dpi)
            if fig_ti:
                save_plot(fig_ti, case_dir, "Transition_Map", format=args.format, dpi=args.dpi)

        except Exception as e:
            print(f"[ERROR] Failed on {surf_path.parent}: {e}")



# # su2_postprocess/cli/single_main.py

# import pandas as pd
# from pathlib import Path
# from su2_postprocess.io.file_scan import find_case_dirs
# from su2_postprocess.io.output import save_plot
# from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
# from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback
# from su2_postprocess.utils.overlay_loader import load_overlay_file, load_overlay_dir_by_prefix
# from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi
# from su2_postprocess.plots.cp_cf import plot_cp_cf
# from su2_postprocess.plots.transition import plot_transition_map

# def single_main(args):
#     print(f"Entered 'single' mode for root: {args.root}")
#     case_files = find_case_dirs(args.root)
#     print(f"Found {len(case_files)} surface file(s)")

#     # EXP overlays
#     exp_cp_dict, exp_cf_dict = {}, {}
#     if args.exp_dir:
#         exp_cp_dict, exp_cf_dict = load_overlay_dir_by_prefix(Path(args.exp_dir))

#     # XFOIL overlays
#     xfoil_cp_dict, xfoil_cf_dict = {}, {}
#     if args.xfoil_file:
#         p = Path(args.xfoil_file); stem = p.stem.lower()
#         if stem.startswith("cp_"):
#             xfoil_cp_dict[stem] = load_overlay_file(p)
#         elif stem.startswith("cf_"):
#             xfoil_cf_dict[stem] = load_overlay_file(p)
#     elif args.xfoil_dir:
#         xfoil_cp_dict, xfoil_cf_dict = load_overlay_dir_by_prefix(Path(args.xfoil_dir))
#     if args.xfoil_dir or args.xfoil_file:
#         src = args.xfoil_file or args.xfoil_dir
#         xfoil_cp_dict, xfoil_cf_dict = load_overlay_dir_by_prefix(Path(src))

#     for surface_path in case_files:
#         case = surface_path.parent
#         print(f"Processing: {surface_path}")
#         forces_files = list(case.glob("forces_bdwn_*.dat"))
#         if not forces_files:
#             print(f"[WARN] No forces_bdwn_*.dat file found in: {case}")
#             continue
#         forces_path = forces_files[0]

#         try:
#             # parse surface + reorder
#             df, elems = parse_felineseg_surface(surface_path)
#             df = reorder_surface_nodes_from_elements(df, elems)

#             # metadata
#             _, meta = extract_case_metadata_fallback(forces_path, return_metadata=True)
#             slug = f"m{meta['mach']:.2f}_re{int(meta['reynolds']/1e6)}e6_aoa{int(round(meta['alpha']))}".lower()

#             # safe pick() avoids ambiguous DataFrame truth-tests
#             def pick(dct, prefix):
#                 val = dct.get(f"{prefix}_{slug}")
#                 if val is not None:
#                     return val
#                 for k,v in dct.items():
#                     if slug in k:
#                         return v
#                 return None

#             xcp    = pick(xfoil_cp_dict, "cp")
#             xcf    = pick(xfoil_cf_dict, "cf")
#             exp_cp = pick(exp_cp_dict,   "cp")
#             exp_cf = pick(exp_cf_dict,   "cf")

#             if xcp is None and xcf is None:
#                 print(f"[WARN] No XFOIL overlay for: {slug}")
#             if exp_cp is None and exp_cf is None:
#                 print(f"[WARN] No EXP overlay for: {slug}")

#             # plot (show_legends_it is now hardcoded False)
#             fig = plot_cp_cf_it_multi(
#                 [df], [slug],
#                 show_cp=True,
#                 show_cf=True,
#                 show_it=args.it,
#                 show_map=args.map,
#                 show_airfoil=args.show_airfoil,
#                 forces_files=[forces_path],
#                 xfoil_cp_list=[xcp],
#                 xfoil_cf_list=[xcf],
#                 exp_cp_list=[exp_cp],
#                 exp_cf_list=[exp_cf],
#                 label_style=args.label_style,
#                 show_legends_cp=True,
#                 show_legends_cf=False,
#                 show_legends_it=False,    # <— no more args.legends_it
#                 case_colors=args.case_colors,
#                 case_styles=args.case_styles,
#             )

#             # save plots
#             fig_cp, fig_cf = plot_cp_cf(df)
#             fig_ti = plot_transition_map(df)
#             save_plot(fig,   case, "Cp_Cf_TurbIndex", format=args.format, dpi=args.dpi)
#             save_plot(fig_cp, case, "Cp_vs_x",        format=args.format, dpi=args.dpi)
#             save_plot(fig_cf, case, "Cf_vs_x",        format=args.format, dpi=args.dpi)
#             if fig_ti:
#                 save_plot(fig_ti, case, "Transition_Map", format=args.format, dpi=args.dpi)

#         except Exception as e:
#             print(f"[ERROR] Failed on {case}: {e}")



# # su2_postprocess/cli/single_main.py

# import pandas as pd
# from pathlib import Path
# from su2_postprocess.io.file_scan import find_case_dirs
# from su2_postprocess.io.output import save_plot
# from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
# from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback
# from su2_postprocess.utils.overlay_loader import load_overlay_file, load_overlay_dir_by_prefix
# from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi
# from su2_postprocess.plots.cp_cf import plot_cp_cf
# from su2_postprocess.plots.transition import plot_transition_map

# def single_main(args):
#     print(f"Entered 'single' mode for root: {args.root}")
#     case_files = find_case_dirs(args.root)
#     print(f"Found {len(case_files)} surface file(s)")

#     # EXP overlays
#     if args.exp_dir:
#         exp_cp_dict, exp_cf_dict = load_overlay_dir_by_prefix(Path(args.exp_dir))
#     else:
#         exp_cp_dict, exp_cf_dict = {}, {}

#     # XFOIL overlays
#     xfoil_cp_dict, xfoil_cf_dict = {}, {}
#     if args.xfoil_file:
#         p = Path(args.xfoil_file)
#         stem = p.stem.lower()
#         if stem.startswith("cp_"):
#             xfoil_cp_dict[stem] = load_overlay_file(p)
#         elif stem.startswith("cf_"):
#             xfoil_cf_dict[stem] = load_overlay_file(p)
#         else:
#             print(f"[WARN] xfoil-file '{stem}' does not start with cp_/cf_; ignoring")
#     elif args.xfoil_dir:
#         xfoil_cp_dict, xfoil_cf_dict = load_overlay_dir_by_prefix(Path(args.xfoil_dir))

#     for surface_path in case_files:
#         case = surface_path.parent
#         print(f"Processing: {surface_path}")
#         forces_files = list(case.glob("forces_bdwn_*.dat"))
#         if not forces_files:
#             print(f"[WARN] No forces_bdwn_*.dat file found in: {case}")
#             continue
#         forces_path = forces_files[0]

#         try:
#             # surface + connectivity
#             df, elems = parse_felineseg_surface(surface_path)
#             df = reorder_surface_nodes_from_elements(df, elems)

#             # metadata
#             _, meta = extract_case_metadata_fallback(forces_path, return_metadata=True)
#             mach = meta['mach']
#             re_val = meta['reynolds']
#             aoa = meta['alpha']

#             # build a lower-case slug
#             slug = f"m{mach:.2f}_re{int(re_val/1e6)}e6_aoa{int(round(aoa))}".lower()

#             # XFOIL CP
#             xcp = xfoil_cp_dict.get(f"cp_{slug}")
#             if xcp is None:
#                 xcp = next((d for s,d in xfoil_cp_dict.items() if slug in s), None)

#             # XFOIL CF
#             xcf = xfoil_cf_dict.get(f"cf_{slug}")
#             if xcf is None:
#                 xcf = next((d for s,d in xfoil_cf_dict.items() if slug in s), None)

#             # EXP CP
#             exp_cp = exp_cp_dict.get(f"cp_{slug}")
#             if exp_cp is None:
#                 exp_cp = next((d for s,d in exp_cp_dict.items() if slug in s), None)

#             # EXP CF
#             exp_cf = exp_cf_dict.get(f"cf_{slug}")
#             if exp_cf is None:
#                 exp_cf = next((d for s,d in exp_cf_dict.items() if slug in s), None)

#             # warn
#             if xcp is None and xcf is None:
#                 print(f"[WARN] No XFOIL overlay for: {slug}")
#             if exp_cp is None and exp_cf is None:
#                 print(f"[WARN] No EXP overlay for: {slug}")

#             # plot
#             fig = plot_cp_cf_it_multi(
#                 [df], [slug],
#                 show_cp=True,
#                 show_cf=True,
#                 show_it=not args.no_it,
#                 show_map=not args.no_map,
#                 show_airfoil=args.show_airfoil,
#                 forces_files=[forces_path],
#                 xfoil_cp_list=[xcp],
#                 xfoil_cf_list=[xcf],
#                 exp_cp_list=[exp_cp],
#                 exp_cf_list=[exp_cf],
#                 label_style=args.label_style,
#                 show_legends_cp=True,
#                 show_legends_cf=False,
#                 show_legends_it=False,
#                 case_colors=args.case_colors,
#                 case_styles=args.case_styles,
#             )

#             # save
#             fig_cp, fig_cf = plot_cp_cf(df)
#             fig_ti = plot_transition_map(df)
#             save_plot(fig,   case, "Cp_Cf_TurbIndex", format=args.format, dpi=args.dpi)
#             save_plot(fig_cp, case, "Cp_vs_x",        format=args.format, dpi=args.dpi)
#             save_plot(fig_cf, case, "Cf_vs_x",        format=args.format, dpi=args.dpi)
#             if fig_ti is not None:
#                 save_plot(fig_ti, case, "Transition_Map", format=args.format, dpi=args.dpi)

#         except Exception as e:
#             print(f"[ERROR] Failed on {case}: {e}")


# # su2_postprocess/cli/single_main.py

# import pandas as pd
# from pathlib import Path
# from su2_postprocess.io.file_scan import find_case_dirs
# from su2_postprocess.io.output import save_plot
# from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
# from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback
# from su2_postprocess.utils.overlay_loader import load_overlay_file, load_overlay_dir_by_prefix
# from su2_postprocess.utils.plot_helpers import _warn_missing_overlay
# from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi
# from su2_postprocess.plots.cp_cf import plot_cp_cf
# from su2_postprocess.plots.transition import plot_transition_map

# def single_main(args):
#     print(f"Entered 'single' mode for root: {args.root}")
#     case_files = find_case_dirs(args.root)
#     print(f"Found {len(case_files)} surface file(s)")

#     # Load EXP overlays
#     if args.exp_dir:
#         exp_cp_dict, exp_cf_dict = load_overlay_dir_by_prefix(Path(args.exp_dir))
#     else:
#         exp_cp_dict, exp_cf_dict = {}, {}

#     # Load XFOIL overlays
#     xfoil_cp_dict, xfoil_cf_dict = {}, {}
#     if args.xfoil_file:
#         df_overlay = load_overlay_file(Path(args.xfoil_file))
#         stem = Path(args.xfoil_file).stem.lower()
#         if "cf" in stem:
#             xfoil_cf_dict[stem] = df_overlay
#         else:
#             xfoil_cp_dict[stem] = df_overlay
#     elif args.xfoil_dir:
#         xfoil_cp_dict, xfoil_cf_dict = load_overlay_dir_by_prefix(Path(args.xfoil_dir))

#     for surface_path in case_files:
#         case = surface_path.parent
#         print(f"Processing: {surface_path}")
#         forces_files = list(case.glob("forces_bdwn_*.dat"))
#         if not forces_files:
#             print(f"[WARN] No forces_bdwn_*.dat file found in: {case}")
#             continue
#         forces_path = forces_files[0]

#         try:
#             df, elems = parse_felineseg_surface(surface_path)
#             df = reorder_surface_nodes_from_elements(df, elems)
#             _, metadata = extract_case_metadata_fallback(forces_path, return_metadata=True)

#             # Build label_str
#             turb, trans = metadata['turbulence_model'], metadata['transition_model']
#             mach, re, aoa = metadata['mach'], metadata['reynolds'], metadata['alpha']

#             if args.label_style == 'short':
#                 label_str = f"{turb}-{trans}"
#             elif args.label_style == 'full':
#                 label_str = f"{turb}-{trans}_M{mach:.2f}_Re{int(re/1e6)}e6_AoA{int(round(aoa))}"
#             elif args.label_style == 'metadata':
#                 parts = [
#                     rf"Re={int(re/1e6)}e6",
#                     rf"M={mach:.2f}",
#                     rf"AoA={int(round(aoa))}\u00b0"
#                 ]
#                 label_str = ", ".join(parts)
#             else:
#                 label_str = case.name

#             # Matching key
#             key = f"M{mach:.2f}_Re{int(re/1e6)}e6_AoA{int(round(aoa))}"

#             # Find overlays by key
#             xfoil_cp_input = next((df for nm,df in xfoil_cp_dict.items() if key in nm), None)
#             xfoil_cf_input = next((df for nm,df in xfoil_cf_dict.items() if key in nm), None)
#             exp_cp_input   = next((df for nm,df in exp_cp_dict.items()   if key in nm), None)
#             exp_cf_input   = next((df for nm,df in exp_cf_dict.items()   if key in nm), None)

#             # Corrected boolean checks
#             if xfoil_cp_input is None and xfoil_cf_input is None:
#                 print(f"[WARN] No XFOIL overlay for: {label_str}")
#             if exp_cp_input is None and exp_cf_input is None:
#                 print(f"[WARN] No EXP overlay for: {label_str}")

#             fig = plot_cp_cf_it_multi(
#                 [df], [label_str],
#                 show_cp=True,
#                 show_cf=True,
#                 show_it=False,
#                 show_map=not args.no_map,
#                 show_airfoil=args.show_airfoil,
#                 forces_files=[forces_path],
#                 xfoil_cp_list=[xfoil_cp_input],
#                 xfoil_cf_list=[xfoil_cf_input],
#                 exp_cp_list=[exp_cp_input],
#                 exp_cf_list=[exp_cf_input],
#                 label_style=args.label_style,
#                 show_legends_cp=True,
#                 show_legends_cf=False,
#                 show_legends_it=False,
#                 case_colors=args.case_colors,
#                 case_styles=args.case_styles,
#             )

#             # Save individual plots
#             fig_cp, fig_cf = plot_cp_cf(df)
#             fig_ti = plot_transition_map(df)
#             save_plot(fig,   case, "Cp_Cf_TurbIndex", format=args.format, dpi=args.dpi)
#             save_plot(fig_cp, case, "Cp_vs_x",        format=args.format, dpi=args.dpi)
#             save_plot(fig_cf, case, "Cf_vs_x",        format=args.format, dpi=args.dpi)
#             if fig_ti is not None:
#                 save_plot(fig_ti, case, "Transition_Map", format=args.format, dpi=args.dpi)

#         except Exception as e:
#             print(f"[ERROR] Failed on {case}: {e}")



# # su2_postprocess/cli/single_main.py

# import pandas as pd
# from pathlib import Path
# from su2_postprocess.io.file_scan import find_case_dirs
# from su2_postprocess.io.output import save_plot
# from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
# from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback
# from su2_postprocess.utils.overlay_loader import load_overlay_file, load_overlay_dir_by_prefix
# from su2_postprocess.utils.plot_helpers import _warn_missing_overlay
# from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi
# from su2_postprocess.plots.cp_cf import plot_cp_cf
# from su2_postprocess.plots.transition import plot_transition_map

# def single_main(args):
#     print(f"Entered 'single' mode for root: {args.root}")
#     case_files = find_case_dirs(args.root)
#     print(f"Found {len(case_files)} surface file(s)")

#     # Load EXP overlays
#     exp_cp_dict, exp_cf_dict = ({}, {})
#     if args.exp_dir:
#         exp_cp_dict, exp_cf_dict = load_overlay_dir_by_prefix(Path(args.exp_dir))

#     # Load XFOIL overlays
#     xfoil_cp_dict, xfoil_cf_dict = {}, {}
#     if args.xfoil_file:
#         df = load_overlay_file(Path(args.xfoil_file))
#         stem = Path(args.xfoil_file).stem.lower()
#         if "cf" in stem:
#             xfoil_cf_dict[stem] = df
#         else:
#             xfoil_cp_dict[stem] = df
#     elif args.xfoil_dir:
#         xfoil_cp_dict, xfoil_cf_dict = load_overlay_dir_by_prefix(Path(args.xfoil_dir))

#     for surface_path in case_files:
#         case = surface_path.parent
#         print(f"Processing: {surface_path}")
#         forces_files = list(case.glob("forces_bdwn_*.dat"))
#         if not forces_files:
#             print(f"[WARN] No forces_bdwn_*.dat file found in: {case}")
#             continue
#         forces_path = forces_files[0]

#         try:
#             df, elems = parse_felineseg_surface(surface_path)
#             df = reorder_surface_nodes_from_elements(df, elems)
#             _, metadata = extract_case_metadata_fallback(forces_path, return_metadata=True)

#             # Build label_str for legend
#             turb, trans = metadata['turbulence_model'], metadata['transition_model']
#             mach, re, aoa = metadata['mach'], metadata['reynolds'], metadata['alpha']

#             if args.label_style == 'short':
#                 label_str = f"{turb}-{trans}"
#             elif args.label_style == 'full':
#                 label_str = f"{turb}-{trans}_M{mach:.2f}_Re{int(re/1e6)}e6_AoA{int(round(aoa))}"
#             elif args.label_style == 'metadata':
#                 parts = [
#                     rf"Re={int(re/1e6)}e6",
#                     rf"M={mach:.2f}",
#                     rf"AoA={int(round(aoa))}\u00b0"
#                 ]
#                 label_str = ", ".join(parts)
#             else:
#                 label_str = case.name

#             # Construct matching key for overlays
#             key = f"M{mach:.2f}_Re{int(re/1e6)}e6_AoA{int(round(aoa))}"

#             # Find per-case overlays by key substring
#             xfoil_cp_input = next((df for nm,df in xfoil_cp_dict.items() if key in nm), None)
#             xfoil_cf_input = next((df for nm,df in xfoil_cf_dict.items() if key in nm), None)
#             exp_cp_input   = next((df for nm,df in exp_cp_dict.items()   if key in nm), None)
#             exp_cf_input   = next((df for nm,df in exp_cf_dict.items()   if key in nm), None)

#             if not (xfoil_cp_input or xfoil_cf_input):
#                 print(f"[WARN] No XFOIL overlay for: {label_str}")
#             if not (exp_cp_input or exp_cf_input):
#                 print(f"[WARN] No EXP overlay for: {label_str}")

#             # Plot, passing in the four overlay lists and optional color/style
#             fig = plot_cp_cf_it_multi(
#                 [df], [label_str],
#                 show_cp=True, show_cf=True,
#                 show_it=False,
#                 show_map=not args.no_map,
#                 show_airfoil=args.show_airfoil,
#                 forces_files=[forces_path],
#                 xfoil_cp_list=[xfoil_cp_input],
#                 xfoil_cf_list=[xfoil_cf_input],
#                 exp_cp_list=[exp_cp_input],
#                 exp_cf_list=[exp_cf_input],
#                 label_style=args.label_style,
#                 show_legends_cp=True,
#                 show_legends_cf=False,
#                 show_legends_it=False,
#                 case_colors=args.case_colors,
#                 case_styles=args.case_styles,
#             )

#             # Also save the individual Cp, Cf, and Transition plots
#             fig_cp, fig_cf = plot_cp_cf(df)
#             fig_ti = plot_transition_map(df)
#             save_plot(fig,   case, "Cp_Cf_TurbIndex", format=args.format, dpi=args.dpi)
#             save_plot(fig_cp, case, "Cp_vs_x",        format=args.format, dpi=args.dpi)
#             save_plot(fig_cf, case, "Cf_vs_x",        format=args.format, dpi=args.dpi)
#             if fig_ti is not None:
#                 save_plot(fig_ti, case, "Transition_Map", format=args.format, dpi=args.dpi)

#         except Exception as e:
#             print(f"[ERROR] Failed on {case}: {e}")


# # su2_postprocess/cli/single_main.py

# import pandas as pd
# from pathlib import Path
# from su2_postprocess.io.file_scan import find_case_dirs
# from su2_postprocess.io.output import save_plot
# from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
# from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback
# from su2_postprocess.utils.overlay_loader import load_overlay_file, load_overlay_dir_by_prefix
# from su2_postprocess.utils.plot_helpers import _warn_missing_overlay
# from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi
# from su2_postprocess.plots.cp_cf import plot_cp_cf
# from su2_postprocess.plots.transition import plot_transition_map

# def single_main(args):
#     print(f"Entered 'single' mode for root: {args.root}")
#     case_files = find_case_dirs(args.root)
#     print(f"Found {len(case_files)} surface file(s)")

#     # Unpack EXP overlays
#     exp_cp_dict, exp_cf_dict = load_overlay_dir_by_prefix(Path(args.exp_dir)) if args.exp_dir else ({}, {})
#     # Unpack XFOIL overlays (if any)
#     xfoil_cp_dict, xfoil_cf_dict = {}, {}
#     if args.xfoil_file:
#         df = load_overlay_file(Path(args.xfoil_file))
#         # assume single DataFrame means both CP & CF in one
#         xfoil_cp_dict = {args.xfoil_file: df}
#         xfoil_cf_dict = {}
#     elif args.xfoil_dir:
#         xfoil_cp_dict, xfoil_cf_dict = load_overlay_dir_by_prefix(Path(args.xfoil_dir))

#     for surface_path in case_files:
#         case = surface_path.parent
#         print(f"Processing: {surface_path}")
#         forces_files = list(case.glob("forces_bdwn_*.dat"))
#         if not forces_files:
#             print(f"[WARN] No forces_bdwn_*.dat file found in: {case}")
#             continue
#         forces_path = forces_files[0]

#         try:
#             df, elems = parse_felineseg_surface(surface_path)
#             df = reorder_surface_nodes_from_elements(df, elems)
#             _, metadata = extract_case_metadata_fallback(forces_path, return_metadata=True)

#             # Build label_str per style
#             turb = metadata['turbulence_model']
#             trans = metadata['transition_model']
#             mach = metadata['mach']
#             re   = metadata['reynolds']
#             aoa  = metadata['alpha']

#             if args.label_style == 'short':
#                 label_str = f"{turb}-{trans}"
#             elif args.label_style == 'full':
#                 label_str = f"{turb}-{trans}_M{mach:.2f}_Re{int(re/1e6)}e6_AoA{int(round(aoa))}"
#             elif args.label_style == 'metadata':
#                 parts = []
#                 parts.append(rf"Re={int(re/1e6)}e6")
#                 parts.append(rf"M={mach:.2f}")
#                 parts.append(rf"AoA={int(round(aoa))}\u00b0")
#                 label_str = ", ".join(parts)
#             else:
#                 label_str = case.name

#             # Pull out per-case overlays
#             xfoil_cp_input = xfoil_cp_dict.get(label_str)
#             xfoil_cf_input = xfoil_cf_dict.get(label_str)
#             exp_cp_input   = exp_cp_dict.get(label_str)
#             exp_cf_input   = exp_cf_dict.get(label_str)

#             if xfoil_cp_input is None and xfoil_cf_input is None:
#                 print(f"[WARN] No XFOIL overlay for: {label_str}")
#             if exp_cp_input is None and exp_cf_input is None:
#                 print(f"[WARN] No EXP overlay for: {label_str}")

#             fig = plot_cp_cf_it_multi(
#                 [df],
#                 [label_str or fallback_label],
#                 show_cp=True,
#                 show_cf=True,
#                 show_it=False,
#                 show_map=not args.no_map,
#                 show_airfoil=args.show_airfoil,
#                 forces_files=[forces_path],
#                 xfoil_cp_list=[xfoil_cp_input],
#                 xfoil_cf_list=[xfoil_cf_input],
#                 exp_cp_list=[exp_cp_input],
#                 exp_cf_list=[exp_cf_input],
#                 label_style=args.label_style,
#                 show_legends_cp=True,
#                 show_legends_cf=False,
#                 show_legends_it=False,
#                 # pass new options:
#                 case_colors=args.case_colors,
#                 case_styles=args.case_styles,
#             )

#             # fig = plot_cp_cf_it_multi(
#             #     [df], [label_str],
#             #     show_cp=True, show_cf=True,
#             #     show_it=False, show_map=not args.no_map,
#             #     show_airfoil=args.show_airfoil,
#             #     forces_files=[forces_path],
#             #     xfoil_cp_list=[xfoil_cp_input],
#             #     xfoil_cf_list=[xfoil_cf_input],
#             #     exp_cp_list=[exp_cp_input],
#             #     exp_cf_list=[exp_cf_input],
#             #     label_style=args.label_style,
#             #     show_legends_cp=True,
#             #     show_legends_cf=False,
#             #     show_legends_it=False,
#             # )

#             # also save the individual Cp, Cf, Transition plots
#             fig_cp, fig_cf = plot_cp_cf(df)
#             fig_ti = plot_transition_map(df)
#             save_plot(fig,  case, "Cp_Cf_TurbIndex", format=args.format, dpi=args.dpi)
#             save_plot(fig_cp, case, "Cp_vs_x",        format=args.format, dpi=args.dpi)
#             save_plot(fig_cf, case, "Cf_vs_x",        format=args.format, dpi=args.dpi)
#             if fig_ti is not None:
#                 save_plot(fig_ti, case, "Transition_Map", format=args.format, dpi=args.dpi)

#         except Exception as e:
#             print(f"[ERROR] Failed on {case}: {e}")




# # su2_postprocess/cli/single_main.py

# import pandas as pd
# from pathlib import Path
# from su2_postprocess.io.file_scan import find_case_dirs
# from su2_postprocess.io.output import save_plot
# from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
# from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback
# from su2_postprocess.utils.overlay_loader import load_overlay_file, load_overlay_dir_by_prefix
# from su2_postprocess.utils.plot_helpers import _warn_missing_overlay
# from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi
# from su2_postprocess.plots.cp_cf import plot_cp_cf
# from su2_postprocess.plots.transition import plot_transition_map

# def single_main(args):
#     print(f"Entered 'single' mode for root: {args.root}")
#     case_files = find_case_dirs(args.root)
#     print(f"Found {len(case_files)} surface file(s)")

#     exp_df = load_overlay_dir_by_prefix(args.exp_dir) if args.exp_dir else None
#     xfoil_input = None

#     if args.xfoil_file:
#         xfoil_input = load_overlay_file(Path(args.xfoil_file))
#         print(f"[INFO] Loaded XFOIL file: {args.xfoil_file}")
#     elif args.xfoil_dir:
#         xf_path = Path(args.xfoil_dir)
#         if xf_path.is_file():
#             xfoil_input = load_overlay_file(xf_path)
#             print(f"[INFO] Loaded XFOIL file: {xf_path}")
#         elif xf_path.is_dir():
#             xfoil_input = load_overlay_dir_by_prefix(xf_path)
#             print(f"[INFO] Loaded XFOIL directory")

#     for surface_path in case_files:
#         print(f"Processing: {surface_path}")
#         case = surface_path.parent
#         forces_files = list(case.glob("forces_bdwn_*.dat"))

#         if not forces_files:
#             print(f"[WARN] No forces_bdwn_*.dat file found in: {case}")
#             continue

#         forces_path = forces_files[0]

#         try:
#             df, elems = parse_felineseg_surface(surface_path)
#             df = reorder_surface_nodes_from_elements(df, elems)
#             label, metadata = extract_case_metadata_fallback(forces_path, return_metadata=True)

#             fallback_label = case.name
#             try:
#                 turb = metadata['turbulence_model']
#                 trans = metadata['transition_model']
#                 mach = metadata['mach']
#                 re = metadata['reynolds']
#                 aoa = metadata['alpha']

#                 if args.label_style == 'short':
#                     label_str = f"{turb}-{trans}"
#                 elif args.label_style == 'full':
#                     label_str = f"{turb}-{trans}_M{mach:.2f}_Re{int(re/1e6)}e6_AoA{int(round(aoa))}"
#                 elif args.label_style == 'metadata':
#                     label_parts = []
#                     if re: label_parts.append(rf"Re={int(re/1e6)}e6")
#                     if mach: label_parts.append(rf"M={mach:.2f}")
#                     if aoa is not None: label_parts.append(rf"AoA={int(round(aoa))}\u00b0")
#                     label_str = ", ".join(label_parts)
#                 else:
#                     label_str = fallback_label
#             except Exception:
#                 print("[WARN] Incomplete metadata, falling back to directory name.")
#                 label_str = fallback_label

#             xfoil_label_input = None
#             if isinstance(xfoil_input, tuple):
#                 cp_dict, cf_dict = xfoil_input
#                 xfoil_label_input = {
#                     k: v for k, v in cp_dict.items() if k.startswith(label_str)
#                 } or None
#             elif isinstance(xfoil_input, dict):
#                 xfoil_label_input = {
#                     k: v for k, v in xfoil_input.items() if k.startswith(label_str)
#                 } or None
#             elif isinstance(xfoil_input, pd.DataFrame):
#                 xfoil_label_input = xfoil_input

#             if xfoil_label_input is None:
#                 print(f"[WARN] No XFOIL overlay found or empty for: {label_str}")
#             else:
#                 _warn_missing_overlay([label_str], xfoil_label_input, name="XFOIL")

#             exp_input = exp_df.get(label_str) if isinstance(exp_df, dict) else exp_df
#             if exp_input is not None:
#                 _warn_missing_overlay([label_str], exp_df, name="EXP")

#             fig = plot_cp_cf_it_multi(
#                 [df], [label_str or fallback_label],
#                 show_cp=True, show_cf=True,
#                 show_it=False,
#                 show_map=not args.no_map,
#                 forces_files=[forces_path],
#                 show_airfoil=args.show_airfoil,
#                 xfoil_data=xfoil_label_input,
#                 exp_data=exp_input,
#                 label_style=args.label_style,
#                 show_legends_cp=True,
#                 show_legends_cf=False,
#                 show_legends_it=False,
#             )

#             fig_cp, fig_cf = plot_cp_cf(df)
#             fig_ti = plot_transition_map(df)

#             save_plot(fig, case, "Cp_Cf_TurbIndex", format=args.format, dpi=args.dpi)
#             save_plot(fig_cp, case, "Cp_vs_x", format=args.format, dpi=args.dpi)
#             save_plot(fig_cf, case, "Cf_vs_x", format=args.format, dpi=args.dpi)
#             if fig_ti is not None:
#                 save_plot(fig_ti, case, "Transition_Map", format=args.format, dpi=args.dpi)

#         except Exception as e:
#             print(f"[ERROR] Failed on {case}: {e}")



# # su2_postprocess/cli/single_main.py

# import pandas as pd
# from pathlib import Path
# from su2_postprocess.io.file_scan import find_case_dirs
# from su2_postprocess.io.output import save_plot
# from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
# from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback
# from su2_postprocess.utils.overlay_loader import load_overlay_file, load_overlay_dir_by_prefix
# from su2_postprocess.utils.plot_helpers import _warn_missing_overlay
# from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi
# from su2_postprocess.plots.cp_cf import plot_cp_cf
# from su2_postprocess.plots.transition import plot_transition_map


# def single_main(args):
#     print(f"Entered 'single' mode for root: {args.root}")
#     case_files = find_case_dirs(args.root)
#     print(f"Found {len(case_files)} surface file(s)")

#     exp_df = load_overlay_dir_by_prefix(args.exp_dir) if args.exp_dir else None
#     xfoil_input = None

#     if args.xfoil_file:
#         xfoil_input = load_overlay_file(Path(args.xfoil_file))
#         print(f"[INFO] Loaded XFOIL file: {args.xfoil_file}")
#     elif args.xfoil_dir:
#         xf_path = Path(args.xfoil_dir)
#         if xf_path.is_file():
#             xfoil_input = load_overlay_file(xf_path)
#             print(f"[INFO] Loaded XFOIL file: {xf_path}")
#         elif xf_path.is_dir():
#             xfoil_input = load_overlay_dir_by_prefix(xf_path)
#             print(f"[INFO] Loaded XFOIL directory")

#     for surface_path in case_files:
#         print(f"Processing: {surface_path}")
#         case = surface_path.parent
#         forces_files = list(case.glob("forces_bdwn_*.dat"))

#         if not forces_files:
#             print(f"[WARN] No forces_bdwn_*.dat file found in: {case}")
#             continue

#         forces_path = forces_files[0]

#         try:
#             df, elems = parse_felineseg_surface(surface_path)
#             df = reorder_surface_nodes_from_elements(df, elems)
#             label, metadata = extract_case_metadata_fallback(forces_path, return_metadata=True)

#             fallback_label = case.name
#             try:
#                 turb = metadata['turbulence_model']
#                 trans = metadata['transition_model']
#                 mach = metadata['mach']
#                 re = metadata['reynolds']
#                 aoa = metadata['alpha']

#                 if args.label_style == 'short':
#                     label_str = f"{turb}-{trans}"
#                 elif args.label_style == 'full':
#                     label_str = f"{turb}-{trans}_M{mach:.2f}_Re{int(re/1e6)}e6_AoA{int(round(aoa))}"
#                 elif args.label_style == 'metadata':
#                     label_str = None
#                 else:
#                     label_str = fallback_label

#             except Exception:
#                 print("[WARN] Incomplete metadata, falling back to directory name.")
#                 label_str = fallback_label

#             xfoil_label_input = None
#             if isinstance(xfoil_input, tuple):
#                 cp_dict, cf_dict = xfoil_input
#                 xfoil_label_input = {
#                     k: v for k, v in cp_dict.items() if k.startswith(label_str)
#                 } or None
#             elif isinstance(xfoil_input, dict):
#                 xfoil_label_input = {
#                     k: v for k, v in xfoil_input.items() if k.startswith(label_str)
#                 } or None
#             elif isinstance(xfoil_input, pd.DataFrame):
#                 xfoil_label_input = xfoil_input

#             if xfoil_label_input is None:
#                 print(f"[WARN] No XFOIL overlay found or empty for: {label_str}")
#             else:
#                 _warn_missing_overlay([label_str], xfoil_label_input, name="XFOIL")

#             exp_input = exp_df.get(label_str) if isinstance(exp_df, dict) else exp_df
#             if exp_input is not None:
#                 _warn_missing_overlay([label_str], exp_df, name="EXP")

#             fig = plot_cp_cf_it_multi(
#                 [df], [label_str or fallback_label],
#                 show_cp=True, show_cf=True,
#                 show_it=False,
#                 show_map=not args.no_map,
#                 forces_files=[forces_path],
#                 xfoil_data=xfoil_label_input,
#                 exp_data=exp_input,
#                 label_style=args.label_style,
#                 show_legends_cp=True,
#                 show_legends_cf=False,
#                 show_legends_it=False,
#             )

#             fig_cp, fig_cf = plot_cp_cf(df)
#             fig_ti = plot_transition_map(df)

#             save_plot(fig, case, "Cp_Cf_TurbIndex", format=args.format, dpi=args.dpi)
#             save_plot(fig_cp, case, "Cp_vs_x", format=args.format, dpi=args.dpi)
#             save_plot(fig_cf, case, "Cf_vs_x", format=args.format, dpi=args.dpi)
#             if fig_ti is not None:
#                 save_plot(fig_ti, case, "Transition_Map", format=args.format, dpi=args.dpi)

#         except Exception as e:
#             print(f"[ERROR] Failed on {case}: {e}")
