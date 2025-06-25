from pathlib import Path
from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback
from su2_postprocess.utils.path_helpers import get_comparison_save_path
from su2_postprocess.io.output import save_plot
from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi
from su2_postprocess.utils.label_helpers import legend_label, flow_metadata_text
from su2_postprocess.utils.overlay_loader import (load_overlay_file, load_overlay_dir_by_prefix)


def compare_surface_main(args):
    print("Entered 'compare-surface' mode")
    cases = [Path(p) for p in args.cases]

    # ------------ overlay files ------------------------------------------------
    # exp_cp, exp_cf   = {}, {}
    # xcp, xcf         = {}, {}
    # msescp, msescf   = {}, {}

    # if args.exp_dir:
    #     exp_cp, exp_cf = load_overlay_dir_by_prefix(Path(args.exp_dir))

    # if args.mses_file:
    #     p = Path(args.mses_file)
    #     (msescp if p.stem.lower().startswith("cp_") else msescf)[p.stem.lower()] = load_overlay_file(p)
    # elif args.mses_dir:
    #     msescp, msescf = load_overlay_dir_by_prefix(Path(args.mses_dir))
    
    # if args.xfoil_dir:
    #     xcp, xcf = load_overlay_dir_by_prefix(Path(args.xfoil_dir))

    dfs, labels, forces_paths = [], [], []
    xcp_list, xcf_list, ecp_list, ecf_list, msescp_list, msescf_list = [], [], [], [], [], []
    meta_texts = []

    # def pick(dct, prefix, slug):
    #     if slug in dct:
    #         return dct[slug]
    #     for k, v in dct.items():
    #         if slug in k:
    #             return v
    #     return None

    # def uniq_append(lst, item):
    #     """Append item but replace duplicates with None so they don't get drawn."""
    #     if any(id(item) == id(prev) for prev in lst):
    #         lst.append(None)
    #     else:
    #         lst.append(item)

    exp_cp, exp_cf   = {}, {}
    xcp, xcf         = {}, {}
    msescp, msescf   = {}, {}

    if args.exp_dir:
        exp_cp, exp_cf = load_overlay_dir_by_prefix(Path(args.exp_dir))

    if args.xfoil_file:
        p = Path(args.xfoil_file)
        (xcp if p.stem.lower().startswith("cp_") else xcf)[p.stem.lower()] = load_overlay_file(p)
    elif args.xfoil_dir:
        xcp, xcf = load_overlay_dir_by_prefix(Path(args.xfoil_dir))

    if args.mses_file:
        p = Path(args.mses_file)
        (msescp if p.stem.lower().startswith("cp_") else msescf)[p.stem.lower()] = load_overlay_file(p)
    elif args.mses_dir:
        msescp, msescf = load_overlay_dir_by_prefix(Path(args.mses_dir))


    def lookup(dct: dict, slug: str):
        """Return the overlay whose *stem* contains *slug*.
        If exactly one file exists and nothing matches, use it as global."""
        for k, v in dct.items():
            if slug in k:
                return v
        if len(dct) == 1:
            return next(iter(dct.values()))
        return None

    # ------------ iterate cases ------------------------------------------------
    for case in cases:
        print(f"[INFO] Processing {case}")
        surf  = case / "flow_surf_.dat"
        force = next(case.glob("forces_bdwn_*.dat"), None)
        if not (surf.exists() and force):
            print(f"[WARN] Missing surface or forces in {case}")
            continue

        df, elems = parse_felineseg_surface(surf)
        df = reorder_surface_nodes_from_elements(df, elems)

        label_full, meta = extract_case_metadata_fallback(force, return_metadata=True)
        slug = f"m{meta['mach']:.2f}_re{int(meta['reynolds']/1e6)}e6_aoa{int(round(meta['alpha']))}".lower()
        dfs.append(df)
        forces_paths.append(force)
        labels.append(legend_label(meta, args.label_style, case.name, len(labels)+1))
        meta_texts.append(flow_metadata_text(meta))

        # pick out the matching overlays for this slug
        xcp_list.append(lookup(xcp, slug))
        xcf_list.append(lookup(xcf, slug))
        msescp_list.append(lookup(msescp, slug))
        msescf_list.append(lookup(msescf, slug))
        ecp_list.append(lookup(exp_cp, slug))
        ecf_list.append(lookup(exp_cf, slug))
        labels.append(legend_label(meta, args.label_style, case.name, len(labels)+1))
        meta_texts.append(flow_metadata_text(meta))

        dfs.append(df)
        forces_paths.append(force)

        # xcp_list.append(pick(xcp,  "cp", slug))
        # xcf_list.append(pick(xcf,  "cf", slug))
        # ecp_list.append(pick(exp_cp, "cp", slug))
        # ecf_list.append(pick(exp_cf, "cf", slug))

        # uniq_append(xcp_list, pick(xcp,       "cp", slug))
        # uniq_append(xcf_list, pick(xcf,       "cf", slug))
        # uniq_append(msescp_list, pick(msescp, "cp", slug))
        # uniq_append(msescf_list, pick(msescf, "cf", slug))
        # uniq_append(ecp_list, pick(exp_cp,    "cp", slug))
        # uniq_append(ecf_list, pick(exp_cf,    "cf", slug))


    # ------------ figure options ----------------------------------------------
    n          = len(labels)
    common_txt = meta_texts[0] if n > 0 and len(set(meta_texts)) == 1 else None
    fig = plot_cp_cf_it_multi(
        dfs, labels,
        meta_box_text=common_txt,
        show_meta_box=(common_txt is not None),
        show_cp=True, show_cf=True,
        show_it=args.it, show_map=args.map,
        show_airfoil=args.show_airfoil,
        forces_files=forces_paths,
        xfoil_cp_list=xcp_list, xfoil_cf_list=xcf_list,
        mses_cp_list=msescp_list, mses_cf_list=msescf_list,
        exp_cp_list=ecp_list,  exp_cf_list=ecf_list,
        label_style=args.label_style,
        show_legends_cp=(args.legends_cp if args.legends_cp is not None else n > 1),
        show_legends_cf=(args.legends_cf if args.legends_cf is not None else n > 1),
        show_legends_it=(args.legends_it if args.legends_it is not None else n > 1),
        case_colors=args.case_colors,
        case_styles=args.case_styles,
    )

    save_plot(fig,
              get_comparison_save_path(cases, args.save),
              "Cp_Cf_TurbIndex_Compare",
              format=args.format, dpi=args.dpi)


# from pathlib import Path
# from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
# from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
# from su2_postprocess.utils.parse_metadata import *
# from su2_postprocess.utils.overlay_loader import load_overlay_dir_by_prefix
# from su2_postprocess.io.output import save_plot
# from su2_postprocess.utils.path_helpers import get_comparison_save_path
# from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi
# from su2_postprocess.utils.label_helpers import legend_label, flow_metadata_text


# def compare_surface_main(args):
#     print("Entered 'compare-surface' mode")
#     cases = [Path(p) for p in args.cases]

#     # load overlays
#     exp_cp_dict, exp_cf_dict     = load_overlay_dir_by_prefix(Path(args.exp_dir))   if args.exp_dir   else ({}, {})
#     xfoil_cp_dict, xfoil_cf_dict = load_overlay_dir_by_prefix(Path(args.xfoil_dir)) if args.xfoil_dir else ({}, {})

#     dfs, labels, forces_paths = [], [], []
#     xcp_list, xcf_list = [], []
#     ecp_list, ecf_list = [], []
#     meta_texts = []  

#     def match(dct, prefix, slug):
#         val = dct.get(f"{prefix}_{slug}")
#         if val is not None:
#             return val
#         for k,v in dct.items():
#             if slug in k:
#                 return v
#         return None

#     for case_path in cases:
#         print(f"[INFO] Processing case: {case_path}")
#         surf = case_path / "flow_surf_.dat"
#         forces = list(case_path.glob("forces_bdwn_*.dat"))
#         if not surf.exists() or not forces:
#             print(f"[WARN] Missing data in: {case_path}")
#             continue

#         df, elems = parse_felineseg_surface(surf)
#         df = reorder_surface_nodes_from_elements(df, elems)

#         # metadata + slug
#         _, meta = extract_case_metadata_fallback(forces[0], return_metadata=True)
#         slug = f"m{meta['mach']:.2f}_re{int(meta['reynolds']/1e6)}e6_aoa{int(round(meta['alpha']))}".lower()

#         # # legend
#         # turb  = meta['turbulence_model']
#         # trans = meta['transition_model']
#         # if args.label_style=='short':
#         #     label = f"{turb}-{trans}"
#         # elif args.label_style=='full':
#         #     label = f"{turb}-{trans}"#_M{meta['mach']:.2f}_Re{int(meta['reynolds']/1e6)}e6_AoA{int(round(meta['alpha']))}"
#         # elif args.label_style=='metadata':
#         #     label = "\n".join([
#         #     rf"Re={int(meta['reynolds']/1e6)}e6",
#         #     rf"M={meta['mach']:.2f}",
#         #     rf"AoA={int(round(meta['alpha']))}°"
#         #     ])
#         # else:
#         #     label = case_path.name

#         label = legend_label(meta, args.label_style, case_path.name, len(labels)+1)
#         labels.append(label)
#         meta_text = flow_metadata_text(meta)
#         # labels.append(legend_label(meta, args.label_style))
#         meta_texts.append(meta_text) 

#         dfs.append(df)
#         print(labels)
#         forces_paths.append(forces[0])

#         # pick overlays
#         xcp_list.append(match(xfoil_cp_dict, "cp", slug))
#         xcf_list.append(match(xfoil_cf_dict, "cf", slug))
#         ecp_list.append(match(exp_cp_dict,    "cp", slug))
#         ecf_list.append(match(exp_cf_dict,    "cf", slug))

#         if xcp_list[-1] is None and xcf_list[-1] is None:
#             print(f"[WARN] No XFOIL overlay for: {label}")
#         if ecp_list[-1] is None and ecf_list[-1] is None:
#             print(f"[WARN] No EXP overlay for: {label}")

#     print(f"[DEBUG compare_surface_main] final labels = {labels!r}")
#     print(f"[DEBUG compare_surface_main] n cases = {len(labels)}")

#     n = len(labels)
#     show_leg_cp = args.legends_cp if args.legends_cp is not None else (n>1)
#     show_leg_cf = args.legends_cf if args.legends_cf is not None else (n>1)
#     show_leg_it = args.legends_it if args.legends_it is not None else (n>1)

#     common_meta = meta_texts[0] if len(set(meta_texts)) == 1 else None
#     show_meta   = common_meta is not None

#     fig = plot_cp_cf_it_multi(
#         dfs, labels,
#         meta_box_text = common_meta,   # NEW kwarg
#         show_meta_box = show_meta, # NEW kwarg
#         show_cp=True,
#         show_cf=True,
#         show_it=args.it,            # controlled by --it
#         show_map=args.map,          # controlled by --map
#         show_airfoil=args.show_airfoil,  # now defaults to True
#         forces_files=forces_paths,
#         xfoil_cp_list=xcp_list,
#         xfoil_cf_list=xcf_list,
#         exp_cp_list=ecp_list,
#         exp_cf_list=ecf_list,
#         label_style=args.label_style,
#         show_legends_cp=show_leg_cp,
#         show_legends_cf=show_leg_cf,
#         show_legends_it=show_leg_it,
#         case_colors=args.case_colors,
#         case_styles=args.case_styles,
#     )

#     save_path = get_comparison_save_path(cases, args.save)
#     save_plot(fig, save_path, "Cp_Cf_TurbIndex_Compare",
#               format=args.format, dpi=args.dpi)



# # su2_postprocess/cli/compare_surface_main.py

# from pathlib import Path
# from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
# from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback
# from su2_postprocess.utils.overlay_loader import load_overlay_dir_by_prefix
# from su2_postprocess.utils.plot_helpers import _warn_missing_overlay
# from su2_postprocess.io.output import save_plot
# from su2_postprocess.utils.path_helpers import get_comparison_save_path
# from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi

# def compare_surface_main(args):
#     print("Entered 'compare-surface' mode")
#     cases = [Path(p) for p in args.cases]

#     # Unpack both EXP and XFOIL directories
#     exp_cp_dict, exp_cf_dict       = load_overlay_dir_by_prefix(Path(args.exp_dir))   if args.exp_dir   else ({}, {})
#     xfoil_cp_dict, xfoil_cf_dict   = load_overlay_dir_by_prefix(Path(args.xfoil_dir)) if args.xfoil_dir else ({}, {})

#     dfs, labels, forces_paths = [], [], []
#     xfoil_cp_list, xfoil_cf_list = [], []
#     exp_cp_list,   exp_cf_list   = [], []

#     for case_path in cases:
#         print(f"[INFO] Processing case: {case_path}")
#         surf = case_path / "flow_surf_.dat"
#         forces_files = list(case_path.glob("forces_bdwn_*.dat"))
#         if not surf.exists() or not forces_files:
#             print(f"[WARN] Missing data in: {case_path}")
#             continue
#         df, elems = parse_felineseg_surface(surf)
#         df = reorder_surface_nodes_from_elements(df, elems)
#         label_str, metadata = extract_case_metadata_fallback(forces_files[0], return_metadata=True)

#         # rebuild label_str for consistency
#         turb = metadata['turbulence_model']
#         trans= metadata['transition_model']
#         mach = metadata['mach']
#         re   = metadata['reynolds']
#         aoa  = metadata['alpha']
#         tu   = metadata.get('freestream_tu', None)

#         if args.label_style=='short':
#             label = f"{turb}-{trans}"
#         elif args.label_style=='full':
#             label = f"{turb}-{trans}_M{mach:.2f}_Re{int(re/1e6)}e6_AoA{int(round(aoa))}"
#         elif args.label_style=='metadata':
#             parts = [rf"Re={int(re/1e6)}e6", rf"M={mach:.2f}", rf"AoA={int(round(aoa))}\u00b0"]
#             if tu is not None:
#                 parts.append(rf"Tu={tu:.2f}\%")
#             label = ", ".join(parts)
#         else:
#             label = case_path.name

#         dfs.append(df)
#         labels.append(label)
#         forces_paths.append(forces_files[0])

#         # per-case overlays
#         xfoil_cp_list.append(xfoil_cp_dict.get(label))
#         xfoil_cf_list.append(xfoil_cf_dict.get(label))
#         exp_cp_list.append(exp_cp_dict.get(label))
#         exp_cf_list.append(exp_cf_dict.get(label))

#     # warn for any missing overlays
#     _warn_missing_overlay(labels, xfoil_cp_dict, name="XFOIL CP")
#     _warn_missing_overlay(labels, xfoil_cf_dict, name="XFOIL CF")
#     _warn_missing_overlay(labels, exp_cp_dict,   name="EXP CP")
#     _warn_missing_overlay(labels, exp_cf_dict,   name="EXP CF")

#     # auto-legend logic
#     n = len(labels)
#     show_legends_cp = args.legends_cp if args.legends_cp is not None else (n>1)
#     show_legends_cf = args.legends_cf if args.legends_cf is not None else (n>1)
#     show_legends_it = args.legends_it if args.legends_it is not None else (n>1)

#     fig = plot_cp_cf_it_multi(
#         dfs,
#         labels,
#         show_cp=True,
#         show_cf=True,
#         show_it=not args.no_it,
#         show_map=not args.no_map,
#         show_airfoil=not args.show_airfoil,
#         forces_files=forces_paths,
#         xfoil_cp_list=xfoil_cp_list,
#         xfoil_cf_list=xfoil_cf_list,
#         exp_cp_list=exp_cp_list,
#         exp_cf_list=exp_cf_list,
#         label_style=args.label_style,
#         show_legends_cp=show_legends_cp,
#         show_legends_cf=show_legends_cf,
#         show_legends_it=show_legends_it,
#         # pass new options:
#         case_colors=args.case_colors,
#         case_styles=args.case_styles,
#     )

#     # fig = plot_cp_cf_it_multi(
#     #     dfs, labels,
#     #     show_cp=True, show_cf=True,
#     #     show_it=not args.no_it,
#     #     show_map=not args.no_map,
#     #     show_airfoil=args.show_airfoil,
#     #     forces_files=forces_paths,
#     #     xfoil_cp_list=xfoil_cp_list,
#     #     xfoil_cf_list=xfoil_cf_list,
#     #     exp_cp_list=exp_cp_list,
#     #     exp_cf_list=exp_cf_list,
#     #     label_style=args.label_style,
#     #     show_legends_cp=show_legends_cp,
#     #     show_legends_cf=show_legends_cf,
#     #     show_legends_it=show_legends_it,
#     # )

#     save_path = get_comparison_save_path(cases, args.save)
#     save_plot(fig, save_path, "Cp_Cf_TurbIndex_Compare", format=args.format, dpi=args.dpi)




# # su2_postprocess/cli/compare_surface_main.py

# from pathlib import Path
# from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
# from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback
# from su2_postprocess.utils.overlay_loader import load_overlay_dir_by_prefix
# from su2_postprocess.utils.plot_helpers import _warn_missing_overlay
# from su2_postprocess.io.output import save_plot
# from su2_postprocess.utils.path_helpers import get_comparison_save_path
# from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi

# def compare_surface_main(args):
#     print("Entered 'compare-surface' mode")
#     cases = [Path(p) for p in args.cases]

#     exp_df = load_overlay_dir_by_prefix(args.exp_dir) if args.exp_dir else None
#     xfoil_input = load_overlay_dir_by_prefix(args.xfoil_dir) if args.xfoil_dir else None

#     dfs, labels, forces_paths, exp_data_list, xfoil_data_list = [], [], [], [], []

#     for case_path in cases:
#         print(f"[INFO] Processing case: {case_path}")
#         surface_file = case_path / "flow_surf_.dat"
#         forces_files = list(case_path.glob("forces_bdwn_*.dat"))

#         if not surface_file.exists() or not forces_files:
#             print(f"[WARN] Missing data in: {case_path}")
#             continue

#         try:
#             df, elems = parse_felineseg_surface(surface_file)
#             df = reorder_surface_nodes_from_elements(df, elems)
#             label, metadata = extract_case_metadata_fallback(forces_files[0], return_metadata=True)

#             fallback_label = case_path.name
#             try:
#                 turb = metadata['turbulence_model']
#                 trans = metadata['transition_model']
#                 mach = metadata['mach']
#                 re = metadata['reynolds']
#                 aoa = metadata['alpha']
#                 tu = metadata.get('freestream_tu', None)

#                 if args.label_style == 'short':
#                     label_str = f"{turb}-{trans}"
#                 elif args.label_style == 'full':
#                     label_str = f"{turb}-{trans}_M{mach:.2f}_Re{int(re/1e6)}e6_AoA{int(round(aoa))}"
#                 elif args.label_style == 'metadata':
#                     label_parts = []
#                     if re: label_parts.append(rf"Re={int(re/1e6)}e6")
#                     if mach: label_parts.append(rf"M={mach:.2f}")
#                     if aoa is not None: label_parts.append(rf"AoA={int(round(aoa))}°")
#                     if tu: label_parts.append(rf"Tu={tu:.2f}%")
#                     label_str = ", ".join(label_parts)
#                 else:
#                     label_str = fallback_label

#             except Exception:
#                 label_str = fallback_label

#             # Try to match overlay data by prefix
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

#             exp_input = exp_df.get(label_str) if isinstance(exp_df, dict) else exp_df

#             dfs.append(df)
#             labels.append(label_str or fallback_label)
#             forces_paths.append(forces_files[0])
#             exp_data_list.append(exp_input)
#             xfoil_data_list.append(xfoil_label_input)

#         except Exception as e:
#             print(f"[ERROR] Failed to process {case_path}: {e}")

#     _warn_missing_overlay(labels, xfoil_input, name="XFOIL")
#     _warn_missing_overlay(labels, exp_df, name="EXP")

#     # Auto-legend detection if not explicitly set
#     n_cases = len(labels)
#     show_legends_cp = args.legends_cp if args.legends_cp is not None else (n_cases > 1)
#     show_legends_cf = args.legends_cf if args.legends_cf is not None else (n_cases > 1)
#     show_legends_it = args.legends_it if args.legends_it is not None else (n_cases > 1)

#     fig = plot_cp_cf_it_multi(
#         dfs, labels,
#         show_cp=True,
#         show_cf=True,
#         show_it=not args.no_it,
#         show_map=not args.no_map,
#         show_airfoil=True,
#         forces_files=forces_paths,
#         xfoil_data=xfoil_input,
#         exp_data=exp_df,
#         label_style=args.label_style,
#         show_legends_cp=show_legends_cp,
#         show_legends_cf=show_legends_cf,
#         show_legends_it=show_legends_it,
#     )

#     save_path = get_comparison_save_path([Path(p) for p in cases], args.save)
#     save_plot(fig, save_path, "Cp_Cf_TurbIndex_compare", format=args.format, dpi=args.dpi)


# # su2_postprocess/cli/compare_surface_main.py

# from pathlib import Path
# from su2_postprocess.parser.parse_felineseg_surface import parse_felineseg_surface
# from su2_postprocess.utils.reorder import reorder_surface_nodes_from_elements
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback
# from su2_postprocess.utils.overlay_loader import load_overlay_dir_by_prefix
# from su2_postprocess.utils.plot_helpers import _warn_missing_overlay
# from su2_postprocess.io.output import save_plot
# from su2_postprocess.utils.path_helpers import get_comparison_save_path
# from su2_postprocess.plots.plot_cp_cf_it_multi import plot_cp_cf_it_multi

# def compare_surface_main(args):
#     print("Entered 'compare-surface' mode")
#     cases = [Path(p) for p in args.cases]

#     exp_df = load_overlay_dir_by_prefix(args.exp_dir) if args.exp_dir else None
#     xfoil_input = load_overlay_dir_by_prefix(args.xfoil_dir) if args.xfoil_dir else None

#     dfs, labels, forces_paths, exp_data_list, xfoil_data_list = [], [], [], [], []

#     for case_path in cases:
#         print(f"[INFO] Processing case: {case_path}")
#         surface_file = case_path / "flow_surf_.dat"
#         forces_files = list(case_path.glob("forces_bdwn_*.dat"))

#         if not surface_file.exists() or not forces_files:
#             print(f"[WARN] Missing data in: {case_path}")
#             continue

#         try:
#             df, elems = parse_felineseg_surface(surface_file)
#             df = reorder_surface_nodes_from_elements(df, elems)
#             label, metadata = extract_case_metadata_fallback(forces_files[0], return_metadata=True)

#             fallback_label = case_path.name
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
#                 label_str = fallback_label

#             # Try to match overlay data by prefix
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

#             exp_input = exp_df.get(label_str) if isinstance(exp_df, dict) else exp_df

#             dfs.append(df)
#             labels.append(label_str or fallback_label)
#             forces_paths.append(forces_files[0])
#             exp_data_list.append(exp_input)
#             xfoil_data_list.append(xfoil_label_input)

#         except Exception as e:
#             print(f"[ERROR] Failed to process {case_path}: {e}")

#     _warn_missing_overlay(labels, xfoil_input, name="XFOIL")
#     _warn_missing_overlay(labels, exp_df, name="EXP")

#     # Auto-legend detection if not explicitly set
#     n_cases = len(labels)
#     show_legends_cp = args.legends_cp if args.legends_cp is not None else (n_cases > 1)
#     show_legends_cf = args.legends_cf if args.legends_cf is not None else (n_cases > 1)
#     show_legends_it = args.legends_it if args.legends_it is not None else (n_cases > 1)

#     fig = plot_cp_cf_it_multi(
#         dfs, labels,
#         show_cp=True,
#         show_cf=True,
#         show_it=False,
#         show_map=not args.no_map,
#         forces_files=forces_paths,
#         xfoil_data=xfoil_input,
#         exp_data=exp_df,
#         label_style=args.label_style,
#         show_legends_cp=show_legends_cp,
#         show_legends_cf=show_legends_cf,
#         show_legends_it=show_legends_it,
#     )

#     save_path = get_comparison_save_path(cases, args.save)
#     save_plot(fig, save_path, "Cp_Cf_TurbIndex_compare", format=args.format, dpi=args.dpi)
