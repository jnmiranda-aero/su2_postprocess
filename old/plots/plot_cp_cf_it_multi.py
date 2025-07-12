import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ---------------------------------------------------------------------
# Helper – make sure overlay x / y are numeric
# ---------------------------------------------------------------------
def _sanitize(df):
    if df is None or not hasattr(df, "columns"):
        return df
    for col in ("x", "y"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["x", "y"])

# ---------------------------------------------------------------------
# Main figure routine
# ---------------------------------------------------------------------
def plot_cp_cf_it_multi(
    dfs, labels,
    *,                       # all following must be passed by keyword
    meta_box_text=None,
    show_meta_box=False,
    merge_slider=0.05,
    show_cp=True,
    show_cf=True,
    show_it=False,
    show_map=True,
    show_airfoil=True,
    forces_files=None,
    xfoil_cp_list=None,
    xfoil_cf_list=None,
    exp_cp_list=None,
    exp_cf_list=None,
    mses_cp_list=None,
    mses_cf_list=None,    
    label_style=None,
    show_legends_cp=True,
    show_legends_cf=False,
    show_legends_it=False,
    case_colors=None,
    case_styles=None,
):
    # ------------------------------------------------------------
    # Safety & layout
    # ------------------------------------------------------------
    n = len(dfs)
    if len(labels) != n:
        raise ValueError("dfs and labels length mismatch.")
    if forces_files and len(forces_files) != n:
        raise ValueError("forces_files length mismatch.")
    merge_slider = np.clip(merge_slider, 0.0, 1.0)

    panels = []
    if show_cp:      panels.append("cp")
    if show_cf:      panels.append("cf")
    if show_it and any("Turb_index" in df.columns for df in dfs):
        panels.append("it")
    if show_airfoil: panels.append("airfoil")
    if show_map and "Turb_index" in dfs[0].columns:
        panels.append("map")

    section_h = {"cp": .28, "cf": .22, "it": .18, "airfoil": .20, "map": .25}
    spacing   = 0.1 * merge_slider
    bottoms   = {}
    cur = 1.0
    for p in panels:
        bottoms[p] = cur - section_h[p]
        cur = bottoms[p] - spacing
    fig_h = sum(section_h[p] for p in panels) + len(panels) * spacing
    fig = plt.figure(figsize=(5, fig_h * 9))

    def stylize(ax, *, show_xlabel=False, show_xticks=False, show_bottom=False):
        ax.set_facecolor("none")
        ax.grid(True, ls=":", lw=.5, alpha=.1)
        ax.tick_params(axis="x", bottom=show_xticks, labelbottom=show_xticks)
        ax.tick_params(axis="y", direction="out", width=.8)
        ax.spines["top"].set_visible(False)
        for sp in ("left", "right", "bottom"):
            ax.spines[sp].set_linewidth(1.0)
        ax.spines["bottom"].set_visible(show_bottom)
        if show_xlabel:
            ax.set_xlabel(r"$x/c$", fontsize=13)

    last_panel = panels[-3] if "map" in panels and "airfoil" in panels else (
                 panels[-2] if ("map" in panels or "airfoil" in panels) else
                 panels[-1])

    # -----------------------------------------------------------------
    # CP panel
    # -----------------------------------------------------------------
    if "cp" in panels:
        ax = fig.add_axes([.12, bottoms["cp"], .75, section_h["cp"]])
        cp_vals = []

        for i, (df, lbl) in enumerate(zip(dfs, labels)):
            kw = {"lw": .6}
            if case_colors and i < len(case_colors): kw["color"] = case_colors[i]
            if case_styles and i < len(case_styles): kw["linestyle"] = case_styles[i]
            ax.plot(df["x"], df["Pressure_Coefficient"], label=lbl, **kw)
            cp_vals.append(df["Pressure_Coefficient"].to_numpy())

            for src, style, color, lab in (
                (xfoil_cp_list, "-.", "black", "XFOIL"),
                (mses_cp_list,  ":",  "gray",  "MSES" ),
                (exp_cp_list,   "x",  "red",   "EXP"  ),
            ):
                odf = _sanitize(src[i]) if (src and i < len(src)) else None
                if odf is None or odf.empty:
                    continue

                # plot in the original file order:
                ax.plot(
                    odf["x"], odf["y"],
                    style, lw=1, ms=3.5,
                    color=color,
                    label=lab if i == 0 else None,
                    zorder=12
                )

        if cp_vals:
            flat = np.concatenate(cp_vals)
            pad  = .05 * (flat.max() - flat.min())
            ax.set_ylim(flat.max() + pad, flat.min() - pad)

        ax.set_ylabel(r"$C_p$", fontsize=13)
        ax.set_xlim(0, dfs[0]["x"].max())
        stylize(ax, show_xticks=False)

        if show_meta_box and meta_box_text:
            ax.text(1.2, 1.12, meta_box_text,
                    transform=ax.transAxes, fontsize=10,
                    ha="right", va="top",
                    bbox=dict(facecolor="white", edgecolor="black",
                              boxstyle="round,pad=0.3"))

        if show_legends_cp and ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=9, loc="lower right")

    # -----------------------------------------------------------------
    # CF panel
    # -----------------------------------------------------------------
    if "cf" in panels:
        ax = fig.add_axes([.12, bottoms["cf"], .75, section_h["cf"]])
        cf_vals = []

        # 1) plot the SU2 CF curve
        for i, (df, lbl) in enumerate(zip(dfs, labels)):
            kw = {"lw": .6}
            if case_colors and i < len(case_colors): kw["color"] = case_colors[i]
            if case_styles and i < len(case_styles): kw["linestyle"] = case_styles[i]
            ax.plot(df["x"], df["Skin_Friction_Coefficient_x"], label=lbl, **kw)
            cf_vals.append(df["Skin_Friction_Coefficient_x"].to_numpy())

            for src, style, color, lab in (
                (xfoil_cf_list, "-.", "black", "XFOIL"),
                (mses_cf_list,  ":",  "gray",  "MSES" ),
                (exp_cf_list,   "x",  "red",   "EXP"  )
            ):
                odf = _sanitize(src[i]) if (src and i < len(src)) else None
                if odf is None or odf.empty:
                    continue

                # 1) filter to [0,1], preserve file order
                odf = odf[(0.0 <= odf.x) & (odf.x <= 1.0)].reset_index(drop=True)

                # 2) find the biggest x‐jump (where the curve jumps back)
                x_arr = odf["x"].to_numpy()
                jumps = np.abs(np.diff(x_arr))
                split_i = int(np.argmax(jumps)) + 1

                # 3) plot each continuous segment separately
                for seg in (odf.iloc[:split_i], odf.iloc[split_i:]):
                    ax.plot(
                        seg["x"], seg["y"],
                        style, lw=0.8, ms=3.5,
                        color=color,
                        label=lab if (i == 0 and seg is odf.iloc[:split_i]) else None,
                        zorder=12
                    )
                    cf_vals.append(seg["y"].to_numpy())


        # 3) autoscale Y to include both SU2 and overlay
        if cf_vals:
            flat = np.concatenate(cf_vals)
            pad  = .05 * (flat.max() - flat.min())
            ax.set_ylim(flat.min() - pad, flat.max() + pad)

        ax.set_ylabel(r"$C_f$", fontsize=13)
        ax.set_xlim(0, dfs[0]["x"].max())
        stylize(ax,
                show_xlabel=(last_panel == "cf"),
                show_xticks=(last_panel == "cf"),
                show_bottom=(last_panel == "cf"))
        if show_legends_cf and ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=9, loc="upper right")

    # -----------------------------------------------------------------
    # i_t panel
    # -----------------------------------------------------------------
    if "it" in panels:
        ax = fig.add_axes([.12, bottoms["it"], .75, section_h["it"]])
        for i, (df, lbl) in enumerate(zip(dfs, labels)):
            if "Turb_index" not in df.columns:
                continue
            kw = {"lw": .6}
            if case_colors and i < len(case_colors): kw["color"] = case_colors[i]
            if case_styles and i < len(case_styles): kw["linestyle"] = case_styles[i]
            ax.plot(df["x"], df["Turb_index"].clip(0, 1), label=lbl, **kw)

        ax.set_ylabel(r"$i_t$", fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        stylize(ax,
                show_xlabel=(last_panel == "it"),
                show_xticks=(last_panel == "it"),
                show_bottom=(last_panel == "it"))
        if show_legends_it and ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=9, loc="upper right")

    # -----------------------------------------------------------------
    # Airfoil outline
    # -----------------------------------------------------------------
    if "airfoil" in panels:
        ax = fig.add_axes([.12, bottoms["airfoil"], .75, section_h["airfoil"]])
        for df in dfs:
            ax.plot(df["x"], df["y"], "-", lw=.6, color="black")
        ax.set_xlim(-.005, 1)
        ymin = min(d["y"].min() for d in dfs) - .01
        ymax = max(d["y"].max() for d in dfs) + .01
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.axis("off")

    # -----------------------------------------------------------------
    # Transition map (only first case)
    # -----------------------------------------------------------------
    if "map" in panels and "Turb_index" in dfs[0].columns:
        df0 = dfs[0]
        pts = np.array([df0["x"], df0["y"]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis_r", norm=plt.Normalize(0, 1))
        lc.set_array(df0["Turb_index"].clip(0, 1))
        lc.set_linewidth(1.2)

        ax = fig.add_axes([.12, bottoms["map"], .75, section_h["map"]])
        ax.add_collection(lc)
        ax.set_xlim(-.005, 1)
        ax.set_ylim(df0["y"].min() - .01, df0["y"].max() + .01)
        ax.set_aspect("equal")
        ax.axis("off")

        cax = fig.add_axes([.9, bottoms["map"] + .06, .02, .16])
        cbar = plt.colorbar(lc, cax=cax)
        cbar.set_label(r"Turbulence Index, $i_t$", fontsize=12,
                       rotation=90, va="center")
        cbar.ax.yaxis.set_label_coords(5, .605)
        cbar.set_ticks([0, .5, 1])
        cbar.ax.tick_params(labelsize=9)

    return fig




# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection

# # ---------------------------------------------------------------------
# # Helper – make sure overlay x / y are numeric
# # ---------------------------------------------------------------------
# def _sanitize(df):
#     if df is None or not hasattr(df, "columns"):
#         return df
#     for col in ("x", "y"):
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors="coerce")
#     return df.dropna(subset=["x", "y"])

# # ---------------------------------------------------------------------
# # Main figure routine
# # ---------------------------------------------------------------------
# def plot_cp_cf_it_multi(
#     dfs, labels,
#     *,                       # all following must be passed by keyword
#     meta_box_text=None,
#     show_meta_box=False,
#     merge_slider=0.03,
#     show_cp=True,
#     show_cf=True,
#     show_it=False,
#     show_map=True,
#     show_airfoil=True,
#     forces_files=None,
#     xfoil_cp_list=None,
#     xfoil_cf_list=None,
#     exp_cp_list=None,
#     exp_cf_list=None,
#     mses_cp_list=None,
#     mses_cf_list=None,    
#     label_style=None,
#     show_legends_cp=True,
#     show_legends_cf=False,
#     show_legends_it=False,
#     case_colors=None,
#     case_styles=None,
# ):
#     # ------------------------------------------------------------
#     # Safety & layout
#     # ------------------------------------------------------------
#     n = len(dfs)
#     if len(labels) != n:
#         raise ValueError("dfs and labels length mismatch.")
#     if forces_files and len(forces_files) != n:
#         raise ValueError("forces_files length mismatch.")
#     merge_slider = np.clip(merge_slider, 0.0, 1.0)

#     panels = []
#     if show_cp:      panels.append("cp")
#     if show_cf:      panels.append("cf")
#     if show_it and any("Turb_index" in df.columns for df in dfs):
#         panels.append("it")
#     if show_airfoil: panels.append("airfoil")
#     if show_map and "Turb_index" in dfs[0].columns:
#         panels.append("map")

#     section_h = {"cp": .28, "cf": .22, "it": .18, "airfoil": .20, "map": .25}
#     spacing   = 0.1 * merge_slider
#     bottoms   = {}
#     cur = 1.0
#     for p in panels:
#         bottoms[p] = cur - section_h[p]
#         cur = bottoms[p] - spacing
#     fig_h = sum(section_h[p] for p in panels) + len(panels) * spacing
#     fig = plt.figure(figsize=(5, fig_h * 9))

#     def stylize(ax, *, show_xlabel=False, show_xticks=False, show_bottom=False):
#         ax.set_facecolor("none")
#         ax.grid(True, ls=":", lw=.5, alpha=.1)
#         ax.tick_params(axis="x", bottom=show_xticks, labelbottom=show_xticks)
#         ax.tick_params(axis="y", direction="out", width=.8)
#         ax.spines["top"].set_visible(False)
#         for sp in ("left", "right", "bottom"):
#             ax.spines[sp].set_linewidth(1.0)
#         ax.spines["bottom"].set_visible(show_bottom)
#         if show_xlabel:
#             ax.set_xlabel(r"$x/c$", fontsize=13)

#     last_panel = panels[-3] if "map" in panels and "airfoil" in panels else (
#                  panels[-2] if ("map" in panels or "airfoil" in panels) else
#                  panels[-1])

#     # -----------------------------------------------------------------
#     # CP panel
#     # -----------------------------------------------------------------
#     if "cp" in panels:
#         ax = fig.add_axes([.12, bottoms["cp"], .75, section_h["cp"]])
#         cp_vals = []

#         for i, (df, lbl) in enumerate(zip(dfs, labels)):
#             kw = {"lw": .6}
#             if case_colors and i < len(case_colors): kw["color"] = case_colors[i]
#             if case_styles and i < len(case_styles): kw["linestyle"] = case_styles[i]
#             ax.plot(df["x"], df["Pressure_Coefficient"], label=lbl, **kw)
#             cp_vals.append(df["Pressure_Coefficient"].to_numpy())

#             for src, style, lab in (
#                 (xfoil_cp_list,  "-.",   f"XFOIL {i}"),
#                 (mses_cp_list,   ":",    f"MSES  {i}"),
#                 (exp_cp_list,    "x",    "EXP"      ),
#             ):
#                 odf = _sanitize(src[i]) if (src and i < len(src)) else None
#                 if odf is not None:
#                     odf = odf[odf.x <= 1.0]
#                     ax.plot(odf["x"], odf["y"], style, lw=.6, ms=3.5,
#                             color="black" if "XFOIL" in lab else "red",
#                             label=lab)
#                     cp_vals.append(odf["y"].to_numpy())

#         if cp_vals:
#             flat = np.concatenate(cp_vals)
#             pad  = .05 * (flat.max() - flat.min())
#             ax.set_ylim(flat.max() + pad, flat.min() - pad)

#         ax.set_ylabel(r"$C_p$", fontsize=13)
#         ax.set_xlim(0, dfs[0]["x"].max())
#         stylize(ax, show_xticks=False)

#         if show_meta_box and meta_box_text:
#             ax.text(1.2, 1.12, meta_box_text,
#                     transform=ax.transAxes, fontsize=10,
#                     ha="right", va="top",
#                     bbox=dict(facecolor="white", edgecolor="black",
#                               boxstyle="round,pad=0.3"))

#         if show_legends_cp and ax.get_legend_handles_labels()[1]:
#             ax.legend(fontsize=9, loc="lower right")

#     # -----------------------------------------------------------------
#     # CF panel
#     # -----------------------------------------------------------------
#     if "cf" in panels:
#         ax = fig.add_axes([.12, bottoms["cf"], .75, section_h["cf"]])
#         cf_vals = []

#         for i, (df, lbl) in enumerate(zip(dfs, labels)):
#             kw = {"lw": .6}
#             if case_colors and i < len(case_colors): kw["color"] = case_colors[i]
#             if case_styles and i < len(case_styles): kw["linestyle"] = case_styles[i]
#             ax.plot(df["x"], df["Skin_Friction_Coefficient_x"], label=lbl, **kw)
#             cf_vals.append(df["Skin_Friction_Coefficient_x"].to_numpy())

#             for src, style, lab in (
#                 (xfoil_cp_list,  "-.",   f"XFOIL {i}"),
#                 (mses_cp_list,   ":",    f"MSES  {i}"),
#                 (exp_cp_list,    "x",    "EXP"      ),
#             ):
#                 odf = _sanitize(src[i]) if (src and i < len(src)) else None
#                 if odf is None:
#                     continue
#                 odf = odf[odf.x <= 1.0]
#                 xs  = odf["x"].to_numpy()
#                 idx = int(np.nanargmin(np.diff(xs))) + 1
#                 up, lo = odf.iloc[:idx], odf.iloc[idx:][::-1]
#                 for seg in (up, lo):
#                     ax.plot(seg["x"], seg["y"], style, lw=.6,
#                             color="black" if "XFOIL" in lab else "red",
#                             label=lab if seg is up else None)
#                     cf_vals.append(seg["y"].to_numpy())

#         if cf_vals:
#             flat = np.concatenate(cf_vals)
#             pad  = .05 * (flat.max() - flat.min())
#             ax.set_ylim(flat.min() - pad, flat.max() + pad)

#         ax.set_ylabel(r"$C_f$", fontsize=13)
#         ax.set_xlim(0, dfs[0]["x"].max())
#         stylize(ax,
#                 show_xlabel=(last_panel == "cf"),
#                 show_xticks=(last_panel == "cf"),
#                 show_bottom=(last_panel == "cf"))
#         if show_legends_cf and ax.get_legend_handles_labels()[1]:
#             ax.legend(fontsize=9, loc="upper right")

#     # -----------------------------------------------------------------
#     # i_t panel
#     # -----------------------------------------------------------------
#     if "it" in panels:
#         ax = fig.add_axes([.12, bottoms["it"], .75, section_h["it"]])
#         for i, (df, lbl) in enumerate(zip(dfs, labels)):
#             if "Turb_index" not in df.columns:
#                 continue
#             kw = {"lw": .6}
#             if case_colors and i < len(case_colors): kw["color"] = case_colors[i]
#             if case_styles and i < len(case_styles): kw["linestyle"] = case_styles[i]
#             ax.plot(df["x"], df["Turb_index"].clip(0, 1), label=lbl, **kw)

#         ax.set_ylabel(r"$i_t$", fontsize=13)
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1.05)
#         stylize(ax,
#                 show_xlabel=(last_panel == "it"),
#                 show_xticks=(last_panel == "it"),
#                 show_bottom=(last_panel == "it"))
#         if show_legends_it and ax.get_legend_handles_labels()[1]:
#             ax.legend(fontsize=9, loc="upper right")

#     # -----------------------------------------------------------------
#     # Airfoil outline
#     # -----------------------------------------------------------------
#     if "airfoil" in panels:
#         ax = fig.add_axes([.12, bottoms["airfoil"], .75, section_h["airfoil"]])
#         for df in dfs:
#             ax.plot(df["x"], df["y"], "-", lw=.6, color="black")
#         ax.set_xlim(-.005, 1)
#         ymin = min(d["y"].min() for d in dfs) - .01
#         ymax = max(d["y"].max() for d in dfs) + .01
#         ax.set_ylim(ymin, ymax)
#         ax.set_aspect("equal")
#         ax.axis("off")

#     # -----------------------------------------------------------------
#     # Transition map (only first case)
#     # -----------------------------------------------------------------
#     if "map" in panels and "Turb_index" in dfs[0].columns:
#         df0 = dfs[0]
#         pts = np.array([df0["x"], df0["y"]]).T.reshape(-1, 1, 2)
#         segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
#         lc = LineCollection(segs, cmap="viridis_r", norm=plt.Normalize(0, 1))
#         lc.set_array(df0["Turb_index"].clip(0, 1))
#         lc.set_linewidth(1.2)

#         ax = fig.add_axes([.12, bottoms["map"], .75, section_h["map"]])
#         ax.add_collection(lc)
#         ax.set_xlim(-.005, 1)
#         ax.set_ylim(df0["y"].min() - .01, df0["y"].max() + .01)
#         ax.set_aspect("equal")
#         ax.axis("off")

#         cax = fig.add_axes([.9, bottoms["map"] + .06, .02, .16])
#         cbar = plt.colorbar(lc, cax=cax)
#         cbar.set_label(r"Turbulence Index, $i_t$", fontsize=12,
#                        rotation=90, va="center")
#         cbar.ax.yaxis.set_label_coords(5, .605)
#         cbar.set_ticks([0, .5, 1])
#         cbar.ax.tick_params(labelsize=9)

#     return fig

# # su2_postprocess/plots/plot_cp_cf_it_multi.py

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback
# from su2_postprocess.utils.label_helpers import legend_label, flow_metadata_text

# def plot_cp_cf_it_multi(
#     dfs,
#     labels,
#     meta_box_text=None,  # NEW kwarg
#     show_meta_box=False, # NEW kwarg
#     merge_slider=0.03,
#     show_cp=True,
#     show_cf=True,
#     show_it=False,
#     show_map=True,
#     show_airfoil=True,
#     forces_files=None,
#     xfoil_cp_list=None,
#     xfoil_cf_list=None,
#     exp_cp_list=None,
#     exp_cf_list=None,
#     label_style=None,
#     show_legends_cp=True,
#     show_legends_cf=False,
#     show_legends_it=False,
#     case_colors=None,
#     case_styles=None,
# ):

#     print(f"[DEBUG plot_cp_cf_it_multi] received labels = {labels!r}")

#     # ----------------------------------------
#     # Sanity checks
#     # ----------------------------------------
#     n = len(dfs)
#     if len(labels) != n:
#         raise ValueError("dfs and labels must match in length.")
#     if forces_files and len(forces_files) != n:
#         raise ValueError("forces_files must match dfs length.")
#     merge_slider = np.clip(merge_slider, 0.0, 1.0)

#     # ----------------------------------------
#     # Determine panels
#     # ----------------------------------------
#     panels = []
#     if show_cp:      panels.append("cp")
#     if show_cf:      panels.append("cf")
#     if show_it and any("Turb_index" in df.columns for df in dfs):
#         panels.append("it")
#     if show_airfoil: panels.append("airfoil")
#     if show_map and "Turb_index" in dfs[0].columns:
#         panels.append("map")

#     # ----------------------------------------
#     # Layout: heights & spacing
#     # ----------------------------------------
#     section_heights = {"cp":0.28, "cf":0.22, "it":0.18, "airfoil":0.20, "map":0.25}
#     spacing = 0.1 * merge_slider
#     bottoms = {}
#     cur = 1.0
#     for sec in panels:
#         h = section_heights[sec]
#         bottoms[sec] = cur - h
#         cur = bottoms[sec] - spacing

#     fig_height = sum(section_heights[p] for p in panels) + len(panels)*spacing
#     fig = plt.figure(figsize=(5, fig_height*9))

#     # ----------------------------------------
#     # Shared stylize()
#     # ----------------------------------------
#     def stylize(ax, show_xlabel=False, show_xticks=False, show_bottom_spine=False):
#         ax.set_facecolor("none")
#         ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.1)
#         ax.tick_params(axis="x", bottom=show_xticks, labelbottom=show_xticks)
#         ax.tick_params(axis="y", direction="out", width=0.8)
#         ax.spines["top"].set_visible(False)
#         for sp in ["left","right","bottom"]:
#             ax.spines[sp].set_visible(True)
#             ax.spines[sp].set_linewidth(1.0)
#         ax.spines["bottom"].set_visible(show_bottom_spine)
#         ax.spines["bottom"].set_linewidth(1.0 if show_bottom_spine else 0.0)
#         if show_xlabel:
#             ax.set_xlabel(r"$x/c$", fontsize=13)

#     if "map" in panels and "airfoil" in panels:
#         # both present
#         bottom_most = panels[-3]
#     elif "map" in panels:
#         # only map present
#         bottom_most = panels[-2]
#     elif "airfoil" in panels:
#         # only airfoil present
#         bottom_most = panels[-2]
#     else:
#         # neither present
#         bottom_most = panels[-1]

#     # ============================================================================
#     # CP PANEL
#     # ============================================================================
#     if "cp" in panels:
#         ax = fig.add_axes([0.12, bottoms["cp"], 0.75, section_heights["cp"]])
#         cp_vals = []

#         for i,(df,lbl) in enumerate(zip(dfs, labels)):
#             # SU2 curve
#             kw = {'lw':0.6}#,'color':'black','linestyle':'-'}
#             if case_colors and i < len(case_colors):
#                 kw['color'] = case_colors[i]
#             if case_styles and i < len(case_styles):
#                 kw['linestyle'] = case_styles[i]
#             labelarg = lbl if label_style!="metadata" else None
#             ax.plot(df["x"], df["Pressure_Coefficient"], label=lbl, **kw)
#             cp_vals.append(df["Pressure_Coefficient"].to_numpy())

#             # XFOIL CP overlay: raw file order
#             odf_cp = xfoil_cp_list[i] if xfoil_cp_list else None
#             if hasattr(odf_cp,'columns') and {"x","y"}.issubset(odf_cp.columns):
#                 o = odf_cp[odf_cp.x<=1.0].copy()
#                 ax.plot(o["x"], o["y"], '-.', lw=0.6, color='black', label=f"XFOIL {i}")
#                 cp_vals.append(o["y"].to_numpy())

#             # EXP CP overlay
#             odf_ecp = exp_cp_list[i] if exp_cp_list else None
#             if hasattr(odf_ecp,'columns') and {"x","y"}.issubset(odf_ecp.columns):
#                 e = odf_ecp[odf_ecp.x<=1.0].copy()
#                 # ax.plot(e["x"], e["y"], 'x', lw=0.6, color='red', label=f"EXP {i}")
#                 ax.plot(e["x"], e["y"], 'x', markersize=3.5, lw=0.6, color='red', label=f"EXP")
#                 cp_vals.append(e["y"].to_numpy())

#         # autoscale & invert CP
#         if cp_vals:
#             flat = np.concatenate(cp_vals)
#             m = 0.05*(flat.max() - flat.min())
#             ax.set_ylim(flat.max()+m, flat.min()-m)

#         ax.set_ylabel(r"C$_p$", fontsize=13)
#         ax.set_xlim(0, dfs[0]["x"].max())
#         stylize(ax, show_xticks=False)

#         # metadata box
#         # if forces_files and label_style in ("full","metadata"):
#         #     txt,_ = extract_case_metadata_fallback(forces_files[0], return_metadata=True)
#         #     ax.text(
#         #         1.2, 1.15, txt,
#         #         transform=ax.transAxes, fontsize=10,
#         #         ha="right", va="top",
#         #         bbox=dict(
#         #             facecolor="white",
#         #             edgecolor="black",
#         #             linewidth=0.8,
#         #             boxstyle="round,pad=0.3",
#         #             alpha=1
#         #         )
#         #     )

#         if show_meta_box and meta_box_text:
#             ax.text(1.2, 1.12, meta_box_text,
#                 transform=ax.transAxes,
#                 fontsize=10, ha="right", va="top",
#                 bbox=dict(facecolor="white",
#                 edgecolor="black",
#                 boxstyle="round,pad=0.3"))

#         if show_legends_cp and ax.get_legend_handles_labels()[1]:
#             ax.legend(fontsize=9, loc="lower right")

#     # ============================================================================
#     # CF PANEL
#     # ============================================================================
#     if "cf" in panels:
#         ax = fig.add_axes([0.12, bottoms["cf"], 0.75, section_heights["cf"]])
#         cf_vals = []

#         for i,(df,lbl) in enumerate(zip(dfs, labels)):
#             # SU2 CF
#             kw = {'lw':0.6}#,'color':'black','linestyle':'-'}
#             if case_colors and i < len(case_colors):
#                 kw['color'] = case_colors[i]
#             if case_styles and i < len(case_styles):
#                 kw['linestyle'] = case_styles[i]
#             labelarg = lbl if label_style!="metadata" else None
#             ax.plot(df["x"], df["Skin_Friction_Coefficient_x"], label=lbl, **kw)
#             cf_vals.append(df["Skin_Friction_Coefficient_x"].to_numpy())

#             # XFOIL CF: split at largest drop (true TE→LE)
#             odf_cf = xfoil_cf_list[i] if xfoil_cf_list else None
#             if hasattr(odf_cf,'columns') and {"x","y"}.issubset(odf_cf.columns):
#                 of = odf_cf[odf_cf.x<=1.0].copy()
#                 xs = of["x"].to_numpy()
#                 idx = int(np.nanargmin(np.diff(xs))) + 1
#                 up = of.iloc[:idx]
#                 lo = of.iloc[idx:].iloc[::-1]   # reverse lower
#                 ax.plot(up["x"], up["y"], '-.', lw=0.6, color='black', label=f"XFOIL {i}")
#                 ax.plot(lo["x"], lo["y"], '-.', lw=0.6, color='black')
#                 cf_vals.append(up["y"].to_numpy())
#                 cf_vals.append(lo["y"].to_numpy())

#             # EXP CF: same split
#             odf_ecf = exp_cf_list[i] if exp_cf_list else None
#             if hasattr(odf_ecf,'columns') and {"x","y"}.issubset(odf_ecf.columns):
#                 of2 = odf_ecf[odf_ecf.x<=1.0].copy()
#                 xs2 = of2["x"].to_numpy()
#                 idx2 = int(np.nanargmin(np.diff(xs2))) + 1
#                 up2 = of2.iloc[:idx2]
#                 lo2 = of2.iloc[idx2:].iloc[::-1]
#                 ax.plot(up2["x"], up2["y"], 'x-', lw=0.6, color='red', label=f"EXP {i}")
#                 ax.plot(lo2["x"], lo2["y"], 'x-', lw=0.6, color='red')
#                 cf_vals.append(up2["y"].to_numpy())
#                 cf_vals.append(lo2["y"].to_numpy())

#         # autoscale CF
#         if cf_vals:
#             flat = np.concatenate(cf_vals)
#             m = 0.05*(flat.max() - flat.min())
#             ax.set_ylim(flat.min() - m, flat.max() + m)

#         ax.set_ylabel(r"C$_f$", fontsize=13)
#         ax.set_xlim(0, dfs[0]["x"].max())
#         stylize(ax,
#                 show_xlabel=(bottom_most=="cf"),
#                 show_xticks=(bottom_most=="cf"),
#                 show_bottom_spine=(bottom_most=="cf"))
#         if show_legends_cf and ax.get_legend_handles_labels()[1]:
#             ax.legend(fontsize=9, loc="upper right")

#     # ============================================================================
#     # i_t PANEL
#     # ============================================================================
#     if "it" in panels:
#         ax = fig.add_axes([0.12, bottoms["it"], 0.75, section_heights["it"]])
#         for i,(df,lbl) in enumerate(zip(dfs, labels)):
#             if "Turb_index" in df.columns:
#                 kw = {'lw':0.6}#,'color':'black','linestyle':'-'}
#                 if case_colors and i < len(case_colors):
#                     kw['color'] = case_colors[i]
#                 if case_styles and i < len(case_styles):
#                     kw['linestyle'] = case_styles[i]
#                 ax.plot(df["x"], df["Turb_index"].clip(0,1),
#                         label=(lbl if label_style!="metadata" else ""), **kw)

#         ax.set_ylabel(r"i$_t$", fontsize=13)
#         ax.set_xlim(0,1)
#         ax.set_ylim(0,1.05)
#         stylize(ax,
#                 show_xlabel=(bottom_most=="it"),
#                 show_xticks=(bottom_most=="it"),
#                 show_bottom_spine=(bottom_most=="it"))
#         if show_legends_it and ax.get_legend_handles_labels()[1]:
#             ax.legend(fontsize=9, loc="upper right")

#     # ============================================================================
#     # AIRFOIL PANEL
#     # ============================================================================
#     if "airfoil" in panels:
#         ax = fig.add_axes([0.12, bottoms["airfoil"], 0.75, section_heights["airfoil"]])
#         for df in dfs:
#             ax.plot(df["x"], df["y"], "-", lw=0.6, color="black")
#         ax.set_xlim(0,1)
#         ax.set_xlim(-0.005,1)
#         ymin = min(d["y"].min() for d in dfs) - 0.01
#         ymax = max(d["y"].max() for d in dfs) + 0.01
#         ax.set_ylim(ymin, ymax)
#         ax.set_aspect("equal")
#         ax.axis("off")

#     # ============================================================================
#     # MAP PANEL
#     # ============================================================================
#     if "map" in panels and "Turb_index" in dfs[0].columns:
#         df0 = dfs[0]
#         pts = np.array([df0["x"], df0["y"]]).T.reshape(-1,1,2)
#         segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
#         lc = LineCollection(segs, cmap="viridis_r", norm=plt.Normalize(0,1))
#         lc.set_array(df0["Turb_index"].clip(0,1))
#         lc.set_linewidth(1.2)

#         ax = fig.add_axes([0.12, bottoms["map"], 0.75, section_heights["map"]])
#         ax.add_collection(lc)
#         ax.set_xlim(-0.005,1)
#         ax.set_ylim(df0["y"].min()-0.01, df0["y"].max()+0.01)
#         ax.set_aspect("equal")
#         ax.axis("off")

#         cax = fig.add_axes([0.9, bottoms["map"]+0.06, 0.02, 0.16])
#         cbar = plt.colorbar(lc, cax=cax)
#         cbar.set_label(r"Turbulence Index, i$_t$", fontsize=12, rotation=90, va="center")
#         cbar.ax.yaxis.set_label_coords(5, 0.605)

#         cbar.set_ticks([0,0.5,1])
#         cbar.ax.tick_params(labelsize=9)

#     return fig




# # su2_postprocess/plots/plot_cp_cf_it_multi.py

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback

# def plot_cp_cf_it_multi(
#     dfs,
#     labels,
#     merge_slider=0.03,
#     show_cp=True,
#     show_cf=True,
#     show_it=False,        # iₜ off by default
#     show_map=True,
#     show_airfoil=True,
#     forces_files=None,
#     xfoil_cp_list=None,
#     xfoil_cf_list=None,
#     exp_cp_list=None,
#     exp_cf_list=None,
#     label_style="short",
#     show_legends_cp=True,
#     show_legends_cf=False,
#     show_legends_it=False,
#     case_colors=None,
#     case_styles=None,
# ):
#     # Sanity checks
#     n = len(dfs)
#     if len(labels) != n:
#         raise ValueError("dfs and labels must match length")
#     if forces_files and len(forces_files) != n:
#         raise ValueError("forces_files must match dfs length")
#     merge_slider = np.clip(merge_slider, 0.0, 1.0)

#     # Determine panels
#     section_order = []
#     if show_cp:
#         section_order.append("cp")
#     if show_cf:
#         section_order.append("cf")
#     if show_it and any("Turb_index" in df.columns for df in dfs):
#         section_order.append("it")
#     if show_airfoil:
#         section_order.append("airfoil")
#     if show_map and "Turb_index" in dfs[0].columns:
#         section_order.append("map")

#     # Layout: heights & spacing
#     heights = {"cp":0.28, "cf":0.22, "it":0.18, "airfoil":0.16, "map":0.25}
#     spacing = 0.1 * merge_slider
#     bottoms = {}
#     cur = 1.0
#     for sec in section_order:
#         h = heights[sec]
#         bottoms[sec] = cur - h
#         cur = bottoms[sec] - spacing

#     fig_height = sum(heights[s] for s in section_order) + len(section_order)*spacing
#     fig = plt.figure(figsize=(5, fig_height*9))

#     # Shared stylize()
#     def stylize(ax, show_xlabel=False, show_xticks=False, show_bottom_spine=False):
#         ax.set_facecolor("none")
#         ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.1)
#         ax.tick_params(axis="x", bottom=show_xticks, labelbottom=show_xticks)
#         ax.tick_params(axis="y", direction="out", width=0.8)
#         ax.spines["top"].set_visible(False)
#         for sp in ["left","right","bottom"]:
#             ax.spines[sp].set_visible(True)
#             ax.spines[sp].set_linewidth(0.8)
#         ax.spines["bottom"].set_visible(show_bottom_spine)
#         ax.spines["bottom"].set_linewidth(0.8 if show_bottom_spine else 0.0)
#         if show_xlabel:
#             ax.set_xlabel(r"$x/c$", fontsize=12)

#     bottom_most = section_order[-2] if "map" in section_order else section_order[-1]

#     # CP PANEL
#     if "cp" in section_order:
#         ax = fig.add_axes([0.12, bottoms["cp"], 0.75, heights["cp"]])
#         cp_vals = []
#         for i,(df,lbl) in enumerate(zip(dfs, labels)):
#             kw = {'lw':0.5,'color':'black','linestyle':'-'}
#             if case_colors and i<len(case_colors): kw['color']=case_colors[i]
#             if case_styles and i<len(case_styles): kw['linestyle']=case_styles[i]
#             lab = lbl if label_style!="metadata" else None
#             ax.plot(df["x"], df["Pressure_Coefficient"], label=lab, **kw)
#             cp_vals.append(df["Pressure_Coefficient"].to_numpy())

#             # XFOIL CP overlay (raw order)
#             odf_cp = xfoil_cp_list[i] if xfoil_cp_list else None
#             if hasattr(odf_cp,'columns') and {"x","y"}.issubset(odf_cp.columns):
#                 o = odf_cp[odf_cp.x<=1.0].copy()
#                 ax.plot(o["x"], o["y"], '-.', lw=1.0, color='black', label=f"XFOIL {i}")
#                 cp_vals.append(o["y"].to_numpy())

#             # EXP CP overlay
#             odf_ecp = exp_cp_list[i] if exp_cp_list else None
#             if hasattr(odf_ecp,'columns') and {"x","y"}.issubset(odf_ecp.columns):
#                 e = odf_ecp[odf_ecp.x<=1.0].copy()
#                 ax.plot(e["x"], e["y"], '-.', lw=1.0, color='gray', label=f"EXP {i}")
#                 cp_vals.append(e["y"].to_numpy())

#         if cp_vals:
#             flat = np.concatenate(cp_vals)
#             m = 0.05*(flat.max()-flat.min())
#             ax.set_ylim(flat.max()+m, flat.min()-m)

#         ax.set_ylabel(r"C$_p$", fontsize=12)
#         ax.set_xlim(0, dfs[0]["x"].max())
#         stylize(ax, show_xticks=False)

#         if forces_files and label_style in ("full","metadata"):
#             txt,_ = extract_case_metadata_fallback(forces_files[0], return_metadata=True)
#             ax.text(
#                 1.2,1.15,txt,
#                 transform=ax.transAxes, fontsize=10,
#                 ha="right", va="top",
#                 bbox=dict(facecolor="white", edgecolor="black",
#                           linewidth=0.8, boxstyle="round,pad=0.3", alpha=1)
#             )

#         if show_legends_cp and ax.get_legend_handles_labels()[1]:
#             ax.legend(fontsize=9, loc="lower right")

#     # CF PANEL
#     if "cf" in section_order:
#         ax = fig.add_axes([0.12, bottoms["cf"], 0.75, heights["cf"]])
#         cf_vals = []
#         for i,(df,lbl) in enumerate(zip(dfs, labels)):
#             kw = {'lw':0.5,'color':'black','linestyle':'-'}
#             if case_colors and i<len(case_colors): kw['color']=case_colors[i]
#             if case_styles and i<len(case_styles): kw['linestyle']=case_styles[i]
#             lab = lbl if label_style!="metadata" else None
#             ax.plot(df["x"], df["Skin_Friction_Coefficient_x"], label=lab, **kw)
#             cf_vals.append(df["Skin_Friction_Coefficient_x"].to_numpy())

#             # XFOIL CF: split at largest drop (TE)
#             odf_cf = xfoil_cf_list[i] if xfoil_cf_list else None
#             if hasattr(odf_cf,'columns') and {"x","y"}.issubset(odf_cf.columns):
#                 of = odf_cf[odf_cf.x<=1.0].copy()
#                 xs = of["x"].to_numpy()
#                 split = int(np.nanargmin(np.diff(xs))) + 1
#                 upper = of.iloc[:split]
#                 lower = of.iloc[split:]  # no reverse
#                 ax.plot(upper["x"], upper["y"], '--', lw=1.0, color='black', label=f"XFOIL {i}")
#                 ax.plot(lower["x"], lower["y"], '--', lw=1.0, color='black')
#                 cf_vals.append(upper["y"].to_numpy()); cf_vals.append(lower["y"].to_numpy())

#             # EXP CF: same split
#             odf_ecf = exp_cf_list[i] if exp_cf_list else None
#             if hasattr(odf_ecf,'columns') and {"x","y"}.issubset(odf_ecf.columns):
#                 of2 = odf_ecf[odf_ecf.x<=1.0].copy()
#                 xs2 = of2["x"].to_numpy()
#                 split2 = int(np.nanargmin(np.diff(xs2))) + 1
#                 up2 = of2.iloc[:split2]
#                 lo2 = of2.iloc[split2:]  # no reverse
#                 ax.plot(up2["x"], up2["y"], '--', lw=1.0, color='gray', label=f"EXP {i}")
#                 ax.plot(lo2["x"], lo2["y"], '--', lw=1.0, color='gray')
#                 cf_vals.append(up2["y"].to_numpy()); cf_vals.append(lo2["y"].to_numpy())

#         if cf_vals:
#             flat = np.concatenate(cf_vals)
#             m = 0.05*(flat.max()-flat.min())
#             ax.set_ylim(flat.min()-m, flat.max()+m)

#         ax.set_ylabel(r"C$_f$", fontsize=12)
#         ax.set_xlim(0, dfs[0]["x"].max())
#         stylize(
#             ax,
#             show_xlabel=(bottom_most=="cf"),
#             show_xticks=(bottom_most=="cf"),
#             show_bottom_spine=(bottom_most=="cf")
#         )
#         if show_legends_cf and ax.get_legend_handles_labels()[1]:
#             ax.legend(fontsize=9, loc="upper right")

#     # i_t PANEL
#     if "it" in section_order:
#         ax = fig.add_axes([0.12, bottoms["it"], 0.75, heights["it"]])
#         for i,(df,lbl) in enumerate(zip(dfs, labels)):
#             if "Turb_index" in df.columns:
#                 kw = {'lw':0.5,'color':'black','linestyle':'-'}
#                 if case_colors and i<len(case_colors): kw['color']=case_colors[i]
#                 if case_styles and i<len(case_styles): kw['linestyle']=case_styles[i]
#                 ax.plot(df["x"], df["Turb_index"].clip(0,1),
#                         label=(lbl if label_style!="metadata" else ""), **kw)

#         ax.set_ylabel(r"i$_t$", fontsize=12)
#         ax.set_xlim(0,1)
#         ax.set_ylim(0,1.05)
#         stylize(
#             ax,
#             show_xlabel=(bottom_most=="it"),
#             show_xticks=(bottom_most=="it"),
#             show_bottom_spine=(bottom_most=="it")
#         )
#         if show_legends_it and ax.get_legend_handles_labels()[1]:
#             ax.legend(fontsize=9, loc="upper right")

#     # AIRFOIL PANEL
#     if "airfoil" in section_order:
#         ax = fig.add_axes([0.12, bottoms["airfoil"], 0.75, heights["airfoil"]])
#         for df in dfs:
#             ax.plot(df["x"], df["y"], "-", lw=0.6, color="black")
#         ax.set_xlim(0,1)
#         ymin = min(d["y"].min() for d in dfs) - 0.01
#         ymax = max(d["y"].max() for d in dfs) + 0.01
#         ax.set_ylim(ymin, ymax)
#         ax.set_aspect("equal")
#         ax.axis("off")

#     # MAP PANEL
#     if "map" in section_order and "Turb_index" in dfs[0].columns:
#         df0 = dfs[0]
#         pts = np.array([df0["x"], df0["y"]]).T.reshape(-1, 1, 2)
#         segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
#         lc = LineCollection(segs, cmap="viridis_r", norm=plt.Normalize(0, 1))
#         lc.set_array(df0["Turb_index"].clip(0, 1))
#         lc.set_linewidth(1.2)

#         ax = fig.add_axes([0.12, bottoms["map"], 0.75, section_heights["map"]])
#         ax.add_collection(lc)
#         ax.set_xlim(-0.005, 1)
#         ax.set_ylim(df0["y"].min() - 0.01, df0["y"].max() + 0.01)
#         ax.set_aspect("equal")
#         ax.axis("off")

#         cax = fig.add_axes([0.9, bottoms["map"] + 0.06, 0.02, 0.16])
#         cbar = plt.colorbar(lc, cax=cax)
#         cbar.set_label(r"Turbulence Index, i$_t$", fontsize=10)
#         cbar.set_ticks([0, 0.5, 1])
#         cbar.ax.tick_params(labelsize=9)

#     return fig









# # su2_postprocess/plots/plot_cp_cf_it_multi.py

# import pandas as pd
# import numpy as np
# from pathlib import Path
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback

# def plot_cp_cf_it_multi(
#     dfs,
#     labels,
#     merge_slider=0.03,
#     show_cp=True,
#     show_cf=True,
#     show_it=False,
#     show_map=True,
#     show_airfoil=True,
#     forces_files=None,
#     xfoil_cp_list=None,
#     xfoil_cf_list=None,
#     exp_cp_list=None,
#     exp_cf_list=None,
#     label_style="short",
#     show_legends_cp=True,
#     show_legends_cf=False,
#     show_legends_it=False,
#     case_colors=None,
#     case_styles=None,
# ):
#     # Sanity checks
#     n = len(dfs)
#     if n != len(labels):
#         raise ValueError("dfs and labels must match in length.")
#     if forces_files and len(forces_files) != n:
#         raise ValueError("forces_files must match dfs length.")

#     single_case = (n == 1)
#     merge_slider = np.clip(merge_slider, 0.0, 1.0)

#     # Determine panels
#     section_order = []
#     if show_cp:
#         section_order.append("cp")
#     if show_cf:
#         section_order.append("cf")
#     if show_it and any("Turb_index" in df.columns for df in dfs):
#         section_order.append("it")
#     if show_airfoil:
#         section_order.append("airfoil")
#     if show_map and "Turb_index" in dfs[0].columns:
#         section_order.append("map")

#     # Heights and spacing
#     section_heights = {
#         "cp":      0.28,
#         "cf":      0.22,
#         "it":      0.18,
#         "airfoil": 0.16,
#         "map":     0.25,
#     }
#     spacing = 0.1 * merge_slider

#     bottoms = {}
#     current_bottom = 1.0
#     for sec in section_order:
#         h = section_heights[sec]
#         bottoms[sec] = current_bottom - h
#         current_bottom = bottoms[sec] - spacing

#     fig_height = sum(section_heights[s] for s in section_order) + len(section_order) * spacing
#     fig = plt.figure(figsize=(5, fig_height * 9))

#     def stylize(ax, show_xlabel=False, show_xticks=False, show_bottom_spine=False):
#         ax.set_facecolor("none")
#         ax.grid(False)
#         ax.tick_params(axis='x', direction='out', width=0.8,
#                        bottom=show_xticks, labelbottom=show_xticks)
#         ax.tick_params(axis='y', direction='out', width=0.8)
#         for spine in ['top', 'right']:
#             ax.spines[spine].set_visible(False)
#         ax.spines['left'].set_visible(True)
#         ax.spines['left'].set_linewidth(0.8)
#         ax.spines['right'].set_visible(True)
#         ax.spines['right'].set_linewidth(0.8)
#         ax.spines['bottom'].set_visible(show_bottom_spine)
#         ax.spines['bottom'].set_linewidth(0.8 if show_bottom_spine else 0.0)
#         if show_xlabel:
#             ax.set_xlabel(r'x/c', fontsize=12)

#     # Determine which panel is bottom-most for x-axis
#     bottom_most = section_order[-2] if "map" in section_order else section_order[-1]

#     # ------------- CP PANEL -------------
#     if "cp" in section_order:
#         ax = fig.add_axes([0.12, bottoms["cp"], 0.75, section_heights["cp"]])
#         cp_vals = []

#         # SU2 curves
#         for i, (df, lbl) in enumerate(zip(dfs, labels)):
#             kwargs = {'lw': 0.5}
#             if case_colors and i < len(case_colors):
#                 kwargs['color'] = case_colors[i]
#             if case_styles and i < len(case_styles):
#                 kwargs['linestyle'] = case_styles[i]
#             labelarg = lbl if (label_style != "metadata" and lbl) else None
#             ax.plot(df["x"], df["Pressure_Coefficient"], label=labelarg, **kwargs)
#             cp_vals.append(df["Pressure_Coefficient"].to_numpy())

#             # XFOIL CP overlay
#             if xfoil_cp_list:
#                 odf = xfoil_cp_list[i]
#                 if isinstance(odf, pd.DataFrame):
#                     odf = odf.copy()
#                     odf.columns = [str(c).strip() for c in odf.columns]
#                     if "x" in odf.columns and "Cp" in odf.columns:
#                         ax.plot(odf["x"], odf["Cp"], '-.', lw=1.0, color='black', label=f"XFOIL: {i}")
#                         cp_vals.append(odf["Cp"].to_numpy())

#             # EXP CP overlay
#             if exp_cp_list:
#                 odf = exp_cp_list[i]
#                 if isinstance(odf, pd.DataFrame):
#                     odf = odf.copy()
#                     odf.columns = [str(c).strip() for c in odf.columns]
#                     if "x" in odf.columns and "Cp" in odf.columns:
#                         ax.plot(odf["x"], odf["Cp"], '-.', lw=1.0, color='gray', label=f"EXP: {i}")
#                         cp_vals.append(odf["Cp"].to_numpy())

#         # Autoscale and invert
#         if cp_vals:
#             flat = np.concatenate(cp_vals)
#             m = 0.05 * (flat.max() - flat.min())
#             ax.set_ylim(flat.max() + m, flat.min() - m)

#         ax.set_ylabel(r'C$_p$', fontsize=12)
#         ax.set_xlim(0, dfs[0]["x"].max())
#         stylize(ax, show_xticks=False)

#         # Metadata box for full/metadata style
#         if forces_files and label_style in ("full", "metadata"):
#             txt, _ = extract_case_metadata_fallback(forces_files[0], return_metadata=True)
#             ax.text(
#                 1.2, 1.15, txt,
#                 transform=ax.transAxes,
#                 fontsize=10, ha='right', va='top',
#                 bbox=dict(facecolor='white', edgecolor='black', linewidth=0.3,
#                           boxstyle='round,pad=.2', alpha=1)
#             )

#         if show_legends_cp and label_style != "metadata" and ax.get_legend_handles_labels()[1]:
#             ax.legend(fontsize=9, loc="lower right", handlelength=2, borderpad=0.3, labelspacing=0.3)
#         ax.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.1)

#     # ------------- CF PANEL -------------
#     if "cf" in section_order:
#         ax = fig.add_axes([0.12, bottoms["cf"], 0.75, section_heights["cf"]])
#         cf_vals = []

#         for i, (df, lbl) in enumerate(zip(dfs, labels)):
#             kwargs = {'lw': 0.5}
#             if case_colors and i < len(case_colors):
#                 kwargs['color'] = case_colors[i]
#             if case_styles and i < len(case_styles):
#                 kwargs['linestyle'] = case_styles[i]
#             labelarg = lbl if (label_style != "metadata" and lbl) else None
#             ax.plot(df["x"], df["Skin_Friction_Coefficient_x"], label=labelarg, **kwargs)
#             cf_vals.append(df["Skin_Friction_Coefficient_x"].to_numpy())

#             # XFOIL CF overlay
#             if xfoil_cf_list:
#                 odf = xfoil_cf_list[i]
#                 if isinstance(odf, pd.DataFrame):
#                     odf = odf.copy()
#                     odf.columns = [str(c).strip() for c in odf.columns]
#                     if "x" in odf.columns and "cf" in odf.columns:
#                         ax.plot(odf["x"], odf["cf"], '--', lw=1.0, color='black', label=f"XFOIL: {i}")
#                         cf_vals.append(odf["cf"].to_numpy())

#             # EXP CF overlay
#             if exp_cf_list:
#                 odf = exp_cf_list[i]
#                 if isinstance(odf, pd.DataFrame):
#                     odf = odf.copy()
#                     odf.columns = [str(c).strip() for c in odf.columns]
#                     if "x" in odf.columns and "cf" in odf.columns:
#                         ax.plot(odf["x"], odf["cf"], '--', lw=1.0, color='gray', label=f"EXP: {i}")
#                         cf_vals.append(odf["cf"].to_numpy())

#         if cf_vals:
#             flat = np.concatenate(cf_vals)
#             m = 0.05 * (flat.max() - flat.min())
#             ax.set_ylim(flat.min() - m, flat.max() + m)

#         ax.set_ylabel(r'C$_f$', fontsize=12)
#         ax.set_xlim(0, dfs[0]["x"].max())
#         stylize(
#             ax,
#             show_xlabel=(bottom_most == "cf"),
#             show_xticks=(bottom_most == "cf"),
#             show_bottom_spine=(bottom_most == "cf")
#         )
#         if show_legends_cf and ax.get_legend_handles_labels()[1]:
#             ax.legend(fontsize=9, loc="upper right", handlelength=2, borderpad=0.3, labelspacing=0.3)
#         ax.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.1)


#     # ------------- IT PANEL -------------
#     if "it" in section_order:
#         ax = fig.add_axes([0.12, bottoms["it"], 0.75, section_heights["it"]])
#         for df, lbl in zip(dfs, labels):
#             if "Turb_index" in df.columns:
#                 lab = lbl if label_style != "metadata" else ""
#                 ax.plot(df["x"], df["Turb_index"].clip(0, 1), 'k-', lw=0.5, label=lab)
#         ax.set_ylabel(r'i$_t$', fontsize=12)
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1.05)
#         stylize(
#             ax,
#             show_xlabel=(bottom_most == "it"),
#             show_xticks=(bottom_most == "it"),
#             show_bottom_spine=(bottom_most == "it")
#         )
#         if show_legends_it and ax.get_legend_handles_labels()[1]:
#             ax.legend(fontsize=9, loc="best")

#     # -------- AIRFOIL PANEL --------
#     if "airfoil" in section_order:
#         ax = fig.add_axes([0.12, bottoms["airfoil"], 0.75, section_heights["airfoil"]])
#         for df in dfs:
#             ax.plot(df["x"], df["y"], '-', lw=0.6, color="black")
#         ax.set_xlim(0, 1)
#         ax.set_ylim(min(d["y"].min() for d in dfs) - 0.01,
#                     max(d["y"].max() for d in dfs) + 0.01)
#         ax.set_aspect("equal")
#         ax.axis("off")

#     # -------- MAP PANEL --------
#     if "map" in section_order and "Turb_index" in dfs[0].columns:
#         df0 = dfs[0]
#         x, y = df0["x"], df0["y"]
#         it = df0["Turb_index"].clip(0, 1)
#         pts = np.array([x, y]).T.reshape(-1, 1, 2)
#         segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
#         lc = LineCollection(segs, cmap="viridis_r", norm=plt.Normalize(0, 1))
#         lc.set_array(it)
#         lc.set_linewidth(1.2)

#         ax = fig.add_axes([0.12, bottoms["map"], 0.75, section_heights["map"]])
#         ax.add_collection(lc)
#         ax.set_xlim(-0.005, 1)
#         ax.set_ylim(y.min() - 0.01, y.max() + 0.01)
#         ax.set_aspect("equal")
#         ax.axis("off")

#         cax = fig.add_axes([0.9, bottoms["map"] + 0.06, 0.02, 0.16])
#         cbar = plt.colorbar(lc, cax=cax)
#         cbar.set_label(r'Turbulence Index, i$_t$', fontsize=10)
#         cbar.set_ticks([0, 0.5, 1])
#         cbar.ax.tick_params(labelsize=9)

#     return fig



# import pandas as pd
# import numpy as np
# from pathlib import Path
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback

# def plot_cp_cf_it_multi(
#     dfs,
#     labels,
#     merge_slider=0.03,
#     show_cp=True,
#     show_cf=True,
#     show_it=True,
#     show_map=True,
#     show_airfoil=True,
#     forces_files=None,
#     xfoil_data=None,
#     exp_data=None,
#     label_style="short",
#     show_legends_cp=True,
#     show_legends_cf=False,
#     show_legends_it=False,
# ):
#     # Sanity checks
#     if len(dfs) != len(labels):
#         raise ValueError("dfs and labels must match in length.")
#     if forces_files and len(forces_files) != len(dfs):
#         raise ValueError("forces_files must be None or match dfs length.")

#     single_case = (len(dfs) == 1)
#     merge_slider = np.clip(merge_slider, 0.0, 1.0)

#     # Determine which panels to draw
#     section_order = []
#     if show_cp:     section_order.append("cp")
#     if show_cf:     section_order.append("cf")
#     if show_it and any("Turb_index" in df.columns for df in dfs):
#         section_order.append("it")
#     if show_airfoil:
#         section_order.append("airfoil")
#     if show_map and "Turb_index" in dfs[0].columns:
#         section_order.append("map")

#     # Heights and spacing
#     section_heights = {
#         "cp":      0.28,
#         "cf":      0.22,
#         "it":      0.18,
#         "airfoil": 0.16,
#         "map":     0.25,
#     }
#     spacing = 0.1 * merge_slider

#     bottoms = {}
#     current_bottom = 1.0
#     for name in section_order:
#         h = section_heights[name]
#         bottoms[name] = current_bottom - h
#         current_bottom = bottoms[name] - spacing

#     fig_height = sum(section_heights[n] for n in section_order) + len(section_order)*spacing
#     fig = plt.figure(figsize=(5, fig_height * 9))

#     # Styling helper (no grid here; Cp panel will turn on its grid)
#     def stylize(ax, show_xlabel=False, show_xticks=False, show_bottom_spine=False):
#         ax.set_facecolor("none")
#         ax.grid(False)
#         ax.tick_params(axis='x', direction='out', width=0.8,
#                        bottom=show_xticks, labelbottom=show_xticks)
#         ax.tick_params(axis='y', direction='out', width=0.8)
#         for spine in ['top','right']:
#             ax.spines[spine].set_visible(False)
#         ax.spines['left'].set_visible(True)
#         ax.spines['left'].set_linewidth(0.8)
#         ax.spines['right'].set_visible(True)
#         ax.spines['right'].set_linewidth(0.8)
#         ax.spines['bottom'].set_visible(show_bottom_spine)
#         ax.spines['bottom'].set_linewidth(0.8 if show_bottom_spine else 0.0)
#         if show_xlabel:
#             ax.set_xlabel(r'x/c', fontsize=12)

#     # Which panel sits just above the map?  (for x-axis)
#     bottom_most_panel = section_order[-2] if "map" in section_order else section_order[-1]

#     # -------------------------
#     # CP PANEL
#     if "cp" in section_order:
#         ax_cp = fig.add_axes([0.12, bottoms["cp"], 0.75, section_heights["cp"]])
#         cp_all = []

#         # Plot SU2 Cp
#         for df, lbl in zip(dfs, labels):
#             if label_style!="metadata" and lbl:
#                 ax_cp.plot(df["x"], df["Pressure_Coefficient"], 'k-', lw=0.5, label=lbl)
#             else:
#                 ax_cp.plot(df["x"], df["Pressure_Coefficient"], 'k-', lw=0.5)
#             cp_all.append(df["Pressure_Coefficient"].to_numpy())

#         # Plot overlays
#         for name, src in [("XFOIL", xfoil_data), ("EXP", exp_data)]:
#             if isinstance(src, dict):
#                 for lab, odf in src.items():
#                     odf.columns = [str(c).strip() for c in odf.columns]
#                     if "x" in odf.columns and "Cp" in odf.columns:
#                         ax_cp.plot(
#                             odf["x"], odf["Cp"],
#                             '-.', lw=1.0,
#                             color='black' if name=="XFOIL" else 'gray',
#                             label=f"{name}: {lab}"
#                         )
#                         cp_all.append(odf["Cp"].to_numpy())
#             elif isinstance(src, pd.DataFrame):
#                 odf = src.copy()
#                 odf.columns = [str(c).strip() for c in odf.columns]
#                 if "x" in odf.columns and "Cp" in odf.columns:
#                     ax_cp.plot(
#                         odf["x"], odf["Cp"],
#                         '-.', lw=1.0,
#                         color='black' if name=="XFOIL" else 'gray',
#                         label=name
#                     )
#                     cp_all.append(odf["Cp"].to_numpy())

#         # Rescale and invert
#         if cp_all:
#             flat = np.concatenate(cp_all)
#             m = 0.05*(flat.max() - flat.min())
#             ax_cp.set_ylim(flat.max()+m, flat.min()-m)

#         # Axes & styling
#         ax_cp.set_ylabel(r'C$_p$', fontsize=12)
#         ax_cp.set_xlim(0, dfs[0]["x"].max())
#         stylize(ax_cp, show_xticks=False)

#         # Metadata box
#         if single_case and forces_files and label_style in ("full","metadata"):
#             force_path = Path(forces_files[0])
#             if force_path.exists():
#                 label_box, _ = extract_case_metadata_fallback(force_path, return_metadata=True)
#                 ax_cp.text(
#                     1.2, 1.15, label_box,
#                     fontsize=10, ha='right', va='top',
#                     transform=ax_cp.transAxes,
#                     bbox=dict(
#                         facecolor='white',
#                         edgecolor='black',
#                         linewidth=0.3,
#                         boxstyle='round,pad=.2',
#                         alpha=1
#                     )
#                 )

#         # Legend (only when not metadata style)
#         if show_legends_cp and label_style!="metadata" and ax_cp.get_legend_handles_labels()[1]:
#             ax_cp.legend(
#                 fontsize=9, loc="lower right",
#                 handlelength=2, borderpad=0.3, labelspacing=0.3
#             )

#         # Grid
#         ax_cp.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.1)

#     # -------------------------
#     # CF PANEL
#     if "cf" in section_order:
#         ax_cf = fig.add_axes([0.12, bottoms["cf"], 0.75, section_heights["cf"]])
#         cf_all = []

#         for df, lbl in zip(dfs, labels):
#             if label_style!="metadata" and lbl:
#                 ax_cf.plot(df["x"], df["Skin_Friction_Coefficient_x"],
#                            'k-', lw=0.5, label=lbl)
#             else:
#                 ax_cf.plot(df["x"], df["Skin_Friction_Coefficient_x"],
#                            'k-', lw=0.5)
#             cf_all.append(df["Skin_Friction_Coefficient_x"].to_numpy())

#         for name, src in [("XFOIL", xfoil_data), ("EXP", exp_data)]:
#             if isinstance(src, dict):
#                 for lab, odf in src.items():
#                     odf.columns = [str(c).strip() for c in odf.columns]
#                     if "x" in odf.columns and "cf" in odf.columns:
#                         ax_cf.plot(
#                             odf["x"], odf["cf"],
#                             '--', lw=1.0,
#                             color='black' if name=="XFOIL" else 'gray',
#                             label=f"{name}: {lab}"
#                         )
#                         cf_all.append(odf["cf"].to_numpy())
#             elif isinstance(src, pd.DataFrame):
#                 odf = src.copy()
#                 odf.columns = [str(c).strip() for c in odf.columns]
#                 if "x" in odf.columns and "cf" in odf.columns:
#                     ax_cf.plot(
#                         odf["x"], odf["cf"],
#                         '--', lw=1.0,
#                         color='black' if name=="XFOIL" else 'gray',
#                         label=name
#                     )
#                     cf_all.append(odf["cf"].to_numpy())

#         if cf_all:
#             flat = np.concatenate(cf_all)
#             m = 0.05*(flat.max() - flat.min())
#             ax_cf.set_ylim(flat.min()-m, flat.max()+m)

#         ax_cf.set_ylabel(r'C$_f$', fontsize=12)
#         ax_cf.set_xlim(0, dfs[0]["x"].max())
#         stylize(
#             ax_cf,
#             show_xlabel=(bottom_most_panel=="cf"),
#             show_xticks=(bottom_most_panel=="cf"),
#             show_bottom_spine=(bottom_most_panel=="cf")
#         )
#         if show_legends_cf and ax_cf.get_legend_handles_labels()[1]:
#             ax_cf.legend(
#                 fontsize=9, loc="upper right",
#                 handlelength=2, borderpad=0.3, labelspacing=0.3
#             )

#     # -------------------------
#     # IT PANEL
#     if "it" in section_order:
#         ax_it = fig.add_axes([0.12, bottoms["it"], 0.75, section_heights["it"]])
#         for df, lbl in zip(dfs, labels):
#             if "Turb_index" in df.columns:
#                 ax_it.plot(
#                     df["x"], df["Turb_index"].clip(0,1),
#                     'k-', lw=0.5,
#                     label=lbl if label_style!="metadata" else ""
#                 )
#         ax_it.set_ylabel(r'i$_t$', fontsize=12)
#         ax_it.set_xlim(0,1)
#         ax_it.set_ylim(0,1.05)
#         stylize(
#             ax_it,
#             show_xlabel=(bottom_most_panel=="it"),
#             show_xticks=(bottom_most_panel=="it"),
#             show_bottom_spine=(bottom_most_panel=="it")
#         )
#         if show_legends_it and ax_it.get_legend_handles_labels()[1]:
#             ax_it.legend(fontsize=9, loc="best")

#     # -------------------------
#     # AIRFOIL PANEL
#     if "airfoil" in section_order:
#         ax_af = fig.add_axes([0.12, bottoms["airfoil"], 0.75, section_heights["airfoil"]])
#         for df in dfs:
#             ax_af.plot(df["x"], df["y"], '-', lw=0.6, color="black")
#         ax_af.set_xlim(0,1)
#         ax_af.set_ylim(min(d["y"].min() for d in dfs)-0.01,
#                        max(d["y"].max() for d in dfs)+0.01)
#         ax_af.set_aspect("equal")
#         ax_af.axis("off")

#     # -------------------------
#     # MAP PANEL
#     if "map" in section_order and "Turb_index" in dfs[0].columns:
#         df0 = dfs[0]
#         x, y = df0["x"], df0["y"]
#         it = df0["Turb_index"].clip(0,1)
#         pts = np.array([x,y]).T.reshape(-1,1,2)
#         segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
#         lc = LineCollection(segs, cmap="viridis_r", norm=plt.Normalize(0,1))
#         lc.set_array(it)
#         lc.set_linewidth(1.2)

#         ax_map = fig.add_axes([0.12, bottoms["map"], 0.75, section_heights["map"]])
#         ax_map.add_collection(lc)
#         ax_map.set_xlim(-0.005,1)              # ← restored from original
#         ax_map.set_ylim(y.min()-0.01, y.max()+0.01)
#         ax_map.set_aspect("equal")
#         ax_map.axis("off")

#         cax = fig.add_axes([0.9, bottoms["map"]+0.06, 0.02, 0.16])
#         cbar = plt.colorbar(lc, cax=cax)
#         cbar.set_label(r'Turbulence Index, i$_t$', fontsize=10)
#         cbar.set_ticks([0,0.5,1])
#         cbar.ax.tick_params(labelsize=9)

#     return fig




# import pandas as pd
# import numpy as np
# from pathlib import Path
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback


# def plot_cp_cf_it_multi(
#     dfs,
#     labels,
#     merge_slider=0.03,
#     show_cp=True,
#     show_cf=True,
#     show_it=True,
#     show_map=True,
#     forces_files=None,
#     xfoil_data=None,
#     exp_data=None,
#     label_style="short",
#     show_legends_cp=True,
#     show_legends_cf=False,
#     show_legends_it=False,
# ):
#     if len(dfs) != len(labels):
#         raise ValueError("dfs and labels must match in length.")
#     if forces_files and len(forces_files) != len(dfs):
#         raise ValueError("forces_files must be None or match dfs length.")

#     single_case = len(dfs) == 1
#     merge_slider = np.clip(merge_slider, 0.0, 1.0)

#     section_order = []
#     if show_cp: section_order.append("cp")
#     if show_cf: section_order.append("cf")
#     if show_it and any("Turb_index" in df.columns for df in dfs): section_order.append("it")
#     if show_map: section_order.append("map")

#     section_heights = {
#         "cp": 0.28,
#         "cf": 0.22,
#         "it": 0.18,
#         "map": 0.25
#     }
#     spacing = 0.1 * merge_slider

#     bottoms = {}
#     current_bottom = 1.0
#     for name in section_order:
#         height = section_heights[name]
#         bottoms[name] = current_bottom - height
#         current_bottom = bottoms[name] - spacing

#     fig_height = sum(section_heights[name] for name in section_order) + len(section_order) * spacing
#     fig = plt.figure(figsize=(5, fig_height * 9))

#     def stylize(ax, show_xlabel=False, show_xticks=False, show_bottom_spine=False):
#         ax.set_facecolor("none")
#         ax.grid(False)
#         ax.tick_params(axis='x', direction='out', width=0.8, bottom=show_xticks, labelbottom=show_xticks)
#         ax.tick_params(axis='y', direction='out', width=0.8)
#         ax.spines['top'].set_visible(False)
#         ax.spines['left'].set_visible(True)
#         ax.spines['left'].set_linewidth(0.8)
#         ax.spines['right'].set_visible(True)
#         ax.spines['right'].set_linewidth(0.8)
#         ax.spines['bottom'].set_visible(show_bottom_spine)
#         ax.spines['bottom'].set_linewidth(0.8 if show_bottom_spine else 0.0)
#         if show_xlabel:
#             ax.set_xlabel(r'x/c', fontsize=12)

#     bottom_most_panel = section_order[-2] if "map" in section_order else section_order[-1]

#     if "cp" in section_order:
#         ax_cp = fig.add_axes([0.12, bottoms["cp"], 0.75, section_heights["cp"]])
#         cp_all = []

#         for df, label in zip(dfs, labels):
#             if label_style != "metadata" and label:
#                 ax_cp.plot(df["x"], df["Pressure_Coefficient"], 'k-', lw=0.5, label=label)
#             else:
#                 ax_cp.plot(df["x"], df["Pressure_Coefficient"], 'k-', lw=0.5)
#             cp_all.append(df["Pressure_Coefficient"].to_numpy())

#         for overlay_name, overlay_src in [("XFOIL", xfoil_data), ("EXP", exp_data)]:
#             if isinstance(overlay_src, dict):
#                 for label, df in overlay_src.items():
#                     df.columns = [str(c).strip() for c in df.columns]
#                     if "x" in df.columns and "Cp" in df.columns:
#                         ax_cp.plot(df["x"], df["Cp"], '-.', lw=1.0, color='black' if overlay_name == "XFOIL" else 'gray', label=f"{overlay_name}: {label}")
#                         cp_all.append(df["Cp"].to_numpy())
#             elif isinstance(overlay_src, pd.DataFrame):
#                 df = overlay_src.copy()
#                 df.columns = [str(c).strip() for c in df.columns]
#                 if "x" in df.columns and "Cp" in df.columns:
#                     ax_cp.plot(df["x"], df["Cp"], '-.', lw=1.0, color='black' if overlay_name == "XFOIL" else 'gray', label=f"{overlay_name}")
#                     cp_all.append(df["Cp"].to_numpy())

#         if cp_all:
#             cp_flat = np.concatenate(cp_all)
#             margin = 0.05 * (cp_flat.max() - cp_flat.min())
#             ax_cp.set_ylim(cp_flat.max() + margin, cp_flat.min() - margin)

#         ax_cp.set_ylabel(r'C$_p$', fontsize=12)
#         ax_cp.set_xlim(0, df["x"].max())
#         stylize(ax_cp, show_xticks=False)

#         if single_case and forces_files and label_style in ("full", "metadata"):
#             force_path = Path(forces_files[0])
#             if force_path.exists():
#                 label_box, _ = extract_case_metadata_fallback(force_path, return_metadata=True)
#                 try:
#                     ax_cp.text(
#                         1.2, 1.15, label_box,
#                         fontsize=10, ha='right', va='top',
#                         transform=ax_cp.transAxes,
#                         bbox=dict(facecolor='white', edgecolor='black', linewidth=0.3, boxstyle='round, pad=.2', alpha=1)
#                     )
#                 except Exception as e:
#                     print(f"[WARN] Failed to render label box: {e}")

#         if show_legends_cp and ax_cp.get_legend_handles_labels()[1]:
#             ax_cp.legend(fontsize=9, loc="lower right")

#         ax_cp.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.1)

#     if "cf" in section_order:
#         ax_cf = fig.add_axes([0.12, bottoms["cf"], 0.75, section_heights["cf"]])
#         cf_all = []

#         for df, label in zip(dfs, labels):
#             if label_style != "metadata" and label:
#                 ax_cf.plot(df["x"], df["Skin_Friction_Coefficient_x"], 'k-', lw=0.5, label=label)
#             else:
#                 ax_cf.plot(df["x"], df["Skin_Friction_Coefficient_x"], 'k-', lw=0.5)
#             cf_all.append(df["Skin_Friction_Coefficient_x"].to_numpy())

#         for overlay_name, overlay_src in [("XFOIL", xfoil_data), ("EXP", exp_data)]:
#             if isinstance(overlay_src, dict):
#                 for label, df in overlay_src.items():
#                     df.columns = [str(c).strip() for c in df.columns]
#                     if "x" in df.columns and "cf" in df.columns:
#                         ax_cf.plot(df["x"], df["cf"], '--', lw=1.0, label=f"{overlay_name}: {label}")
#                         cf_all.append(df["cf"].to_numpy())
#             elif isinstance(overlay_src, pd.DataFrame):
#                 df = overlay_src.copy()
#                 df.columns = [str(c).strip() for c in df.columns]
#                 if "x" in df.columns and "cf" in df.columns:
#                     ax_cf.plot(df["x"], df["cf"], '--', lw=1.0, label=f"{overlay_name}")
#                     cf_all.append(df["cf"].to_numpy())

#         if cf_all:
#             cf_flat = np.concatenate(cf_all)
#             margin = 0.05 * (cf_flat.max() - cf_flat.min())
#             ax_cf.set_ylim(cf_flat.min() - margin, cf_flat.max() + margin)

#         ax_cf.set_ylabel(r'C$_f$', fontsize=12)
#         ax_cf.set_xlim(0, df["x"].max())
#         stylize(ax_cf, show_xlabel=(bottom_most_panel == "cf"), show_xticks=(bottom_most_panel == "cf"), show_bottom_spine=(bottom_most_panel == "cf"))

#         if show_legends_cf and ax_cf.get_legend_handles_labels()[1]:
#             ax_cf.legend(fontsize=9, loc="best")

#     if "it" in section_order:
#         ax_it = fig.add_axes([0.12, bottoms["it"], 0.75, section_heights["it"]])
#         for df, label in zip(dfs, labels):
#             if "Turb_index" in df.columns:
#                 if label_style != "metadata" and label:
#                     ax_it.plot(df["x"], df["Turb_index"].clip(0, 1), label=label, lw=0.5)
#                 else:
#                     ax_it.plot(df["x"], df["Turb_index"].clip(0, 1), lw=0.5)
#         ax_it.set_ylabel(r'i$_t$', fontsize=12)
#         ax_it.set_xlim(0, 1)
#         ax_it.set_ylim(0, 1.05)
#         stylize(ax_it, show_xlabel=(bottom_most_panel == "it"), show_xticks=(bottom_most_panel == "it"), show_bottom_spine=(bottom_most_panel == "it"))

#         if show_legends_it and ax_it.get_legend_handles_labels()[1]:
#             ax_it.legend(fontsize=9, loc="best")

#     if "map" in section_order and "Turb_index" in dfs[0].columns:
#         df0 = dfs[0]
#         x, y, it = df0["x"], df0["y"], df0["Turb_index"].clip(0, 1)
#         points = np.array([x, y]).T.reshape(-1, 1, 2)
#         segments = np.concatenate([points[:-1], points[1:]], axis=1)
#         # lc = LineCollection(segments, cmap="inferno_r", norm=plt.Normalize(0, 1))
#         lc = LineCollection(segments, cmap="viridis_r", norm=plt.Normalize(0, 1))
#         lc.set_array(it)
#         lc.set_linewidth(1.2)

#         ax_map = fig.add_axes([0.12, bottoms["map"], 0.75, section_heights["map"]])
#         ax_map.add_collection(lc)
#         ax_map.set_xlim(-0.005, 1)
#         ax_map.set_ylim(y.min() - 0.01, y.max() + 0.01)
#         ax_map.set_aspect("equal")
#         ax_map.axis("off")

#         cax = fig.add_axes([0.9, bottoms["map"] + 0.06, 0.02, 0.16])
#         cbar = plt.colorbar(lc, cax=cax)
#         cbar.set_label(r'Turbulence Index, i$_t$', fontsize=10)
#         cbar.set_ticks([0, 0.5, 1])
#         cbar.ax.tick_params(labelsize=9)

#     return fig

##########################################################################################################

# import pandas as pd
# import numpy as np
# from pathlib import Path
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_fallback
# from su2_postprocess.utils.parse_metadata import extract_case_metadata_from_log

# def plot_cp_cf_it_multi(
#     dfs,
#     labels,
#     merge_slider=0.03,
#     show_cp=True,
#     show_cf=True,
#     show_it=True,
#     show_map=True,
#     forces_files=None,
#     xfoil_data=None,
#     exp_data=None,
# ):
#     if len(dfs) != len(labels):
#         raise ValueError("dfs and labels must match in length.")
#     if forces_files and len(forces_files) != len(dfs):
#         raise ValueError("forces_files must be None or match dfs length.")

#     single_case = len(dfs) == 1
#     merge_slider = np.clip(merge_slider, 0.0, 1.0)

#     section_order = []
#     if show_cp: section_order.append("cp")
#     if show_cf: section_order.append("cf")
#     if show_it and any("Turb_index" in df.columns for df in dfs): section_order.append("it")
#     if show_map: section_order.append("map")

#     section_heights = {
#         "cp": 0.28,
#         "cf": 0.22,
#         "it": 0.18,
#         "map": 0.25
#     }
#     spacing = 0.1 * merge_slider

#     bottoms = {}
#     current_bottom = 1.0
#     for name in section_order:
#         height = section_heights[name]
#         bottoms[name] = current_bottom - height
#         current_bottom = bottoms[name] - spacing

#     fig_height = sum(section_heights[name] for name in section_order) + len(section_order) * spacing
#     fig = plt.figure(figsize=(5, fig_height * 9))

#     def stylize(ax, show_xlabel=False, show_xticks=False, show_bottom_spine=False):
#         ax.set_facecolor("none")
#         ax.grid(False)
#         ax.tick_params(axis='x', direction='out', width=0.8, bottom=show_xticks, labelbottom=show_xticks)
#         ax.tick_params(axis='y', direction='out', width=0.8)
#         ax.spines['top'].set_visible(False)
#         ax.spines['left'].set_visible(True)
#         ax.spines['left'].set_linewidth(0.8)
#         ax.spines['right'].set_visible(True)
#         ax.spines['right'].set_linewidth(0.8)
#         ax.spines['bottom'].set_visible(show_bottom_spine)
#         ax.spines['bottom'].set_linewidth(0.8 if show_bottom_spine else 0.0)
#         if show_xlabel:
#             ax.set_xlabel(r'x/c', fontsize=12)

#     bottom_most_panel = section_order[-2] if "map" in section_order else section_order[-1]

#     if "cp" in section_order:
#         ax_cp = fig.add_axes([0.12, bottoms["cp"], 0.75, section_heights["cp"]])
#         cp_all = []

#         for df, label in zip(dfs, labels):
#             ax_cp.plot(df["x"], df["Pressure_Coefficient"], 'k-', lw=0.5, label=label)
#             cp_all.append(df["Pressure_Coefficient"].to_numpy())

#         for overlay_name, overlay_src in [("XFOIL", xfoil_data), ("EXP", exp_data)]:
#             if isinstance(overlay_src, dict):
#                 for label, df in overlay_src.items():
#                     df.columns = [str(c).strip() for c in df.columns]
#                     if "x" in df.columns and "Cp" in df.columns:
#                         ax_cp.plot(df["x"], df["Cp"], '-.', lw=1.0, color='black' if overlay_name == "XFOIL" else 'gray', label=f"{overlay_name}: {label}")
#                         cp_all.append(df["Cp"].to_numpy())
#                     else:
#                         print(f"[WARN] Skipping {overlay_name}: {label}, missing 'x' or 'Cp' in columns: {df.columns.tolist()}")
#             elif isinstance(overlay_src, pd.DataFrame):
#                 df = overlay_src.copy()
#                 df.columns = [str(c).strip() for c in df.columns]
#                 if "x" in df.columns and "Cp" in df.columns:
#                     ax_cp.plot(df["x"], df["Cp"], '-.', lw=1.0, color='black' if overlay_name == "XFOIL" else 'gray', label=f"{overlay_name}")
#                     cp_all.append(df["Cp"].to_numpy())
#                 else:
#                     print(f"[WARN] Skipping {overlay_name} single input, missing 'x' or 'Cp': {df.columns.tolist()}")

#         if cp_all:
#             cp_flat = np.concatenate(cp_all)
#             margin = 0.05 * (cp_flat.max() - cp_flat.min())
#             ax_cp.set_ylim(cp_flat.max() + margin, cp_flat.min() - margin)

#         ax_cp.set_ylabel(r'C$_p$', fontsize=12)
#         ax_cp.set_xlim(0, df["x"].max())
#         stylize(ax_cp, show_xticks=False)

#         if single_case and forces_files:
#             force_path = Path(forces_files[0])
#             if force_path.exists():
#                 label_str, _ = extract_case_metadata_fallback(force_path, return_metadata=True)
#                 try:
#                     ax_cp.text(
#                         1.2, 1.15, label_str,
#                         fontsize=10, ha='right', va='top',
#                         transform=ax_cp.transAxes,
#                         bbox=dict(facecolor='white', edgecolor='black', linewidth=0.3, boxstyle='round, pad=.2', alpha=1)
#                     )
#                 except Exception as e:
#                     print(f"[WARN] Failed to render label: {label_str!r} due to: {e}")

#         ax_cp.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.1)
#         ax_cp.legend(fontsize=9, loc="lower right")

#     if "cf" in section_order:
#         ax_cf = fig.add_axes([0.12, bottoms["cf"], 0.75, section_heights["cf"]])
#         cf_all = []

#         for df, label in zip(dfs, labels):
#             ax_cf.plot(df["x"], df["Skin_Friction_Coefficient_x"], 'k-', lw=0.5, label=label)
#             cf_all.append(df["Skin_Friction_Coefficient_x"].to_numpy())

#         for overlay_name, overlay_src in [("XFOIL", xfoil_data), ("EXP", exp_data)]:
#             if isinstance(overlay_src, dict):
#                 for label, df in overlay_src.items():
#                     df.columns = [str(c).strip() for c in df.columns]
#                     if "x" in df.columns and "cf" in df.columns:
#                         ax_cf.plot(df["x"], df["cf"], '--', lw=1.0, label=f"{overlay_name}: {label}")
#                         cf_all.append(df["cf"].to_numpy())
#                     else:
#                         print(f"[WARN] Skipping {overlay_name} Cf: {label}, missing 'x' or 'cf' in columns: {df.columns.tolist()}")
#             elif isinstance(overlay_src, pd.DataFrame):
#                 df = overlay_src.copy()
#                 df.columns = [str(c).strip() for c in df.columns]
#                 if "x" in df.columns and "cf" in df.columns:
#                     ax_cf.plot(df["x"], df["cf"], '--', lw=1.0, label=f"{overlay_name}")
#                     cf_all.append(df["cf"].to_numpy())
#                 else:
#                     print(f"[WARN] Skipping {overlay_name} Cf single input, missing 'x' or 'cf': {df.columns.tolist()}")

#         if cf_all:
#             cf_flat = np.concatenate(cf_all)
#             margin = 0.05 * (cf_flat.max() - cf_flat.min())
#             ax_cf.set_ylim(cf_flat.min() - margin, cf_flat.max() + margin)

#         ax_cf.set_ylabel(r'C$_f$', fontsize=12)
#         ax_cf.set_xlim(0, df["x"].max())
#         stylize(ax_cf, show_xlabel=(bottom_most_panel == "cf"), show_xticks=(bottom_most_panel == "cf"), show_bottom_spine=(bottom_most_panel == "cf"))

#         if len(ax_cf.get_legend_handles_labels()[1]) > 1:
#             ax_cf.legend(fontsize=9, loc="best")

#     if "it" in section_order:
#         ax_it = fig.add_axes([0.12, bottoms["it"], 0.75, section_heights["it"]])
#         for df, label in zip(dfs, labels):
#             if "Turb_index" in df.columns:
#                 ax_it.plot(df["x"], df["Turb_index"].clip(0, 1), label=label, lw=0.5)
#         ax_it.set_ylabel(r'i$_t$', fontsize=12)
#         ax_it.set_xlim(0, 1)
#         ax_it.set_ylim(0, 1.05)
#         stylize(ax_it, show_xlabel=(bottom_most_panel == "it"), show_xticks=(bottom_most_panel == "it"), show_bottom_spine=(bottom_most_panel == "it"))
#         ax_it.legend(fontsize=9, loc="best")

#     if "map" in section_order and "Turb_index" in dfs[0].columns:
#         df0 = dfs[0]
#         x, y, it = df0["x"], df0["y"], df0["Turb_index"].clip(0, 1)
#         points = np.array([x, y]).T.reshape(-1, 1, 2)
#         segments = np.concatenate([points[:-1], points[1:]], axis=1)
#         lc = LineCollection(segments, cmap="viridis_r", norm=plt.Normalize(0, 1))
#         lc = LineCollection(segments, cmap="inferno_r", norm=plt.Normalize(0, 1))
#         # lc = LineCollection(segments, cmap="turbo", norm=plt.Normalize(0, 1))
#         lc.set_array(it)
#         lc.set_linewidth(1.2)

#         ax_map = fig.add_axes([0.12, bottoms["map"], 0.75, section_heights["map"]])
#         ax_map.add_collection(lc)
#         ax_map.set_xlim(-0.005, 1)
#         ax_map.set_ylim(y.min() - 0.01, y.max() + 0.01)
#         ax_map.set_aspect("equal")
#         ax_map.axis("off")

#         cax = fig.add_axes([0.9, bottoms["map"] + 0.06, 0.02, 0.16])
#         cbar = plt.colorbar(lc, cax=cax)
#         cbar.set_label(r'Turbulence Index, i$_t$', fontsize=10)
#         cbar.set_ticks([0, 0.5, 1])
#         cbar.ax.tick_params(labelsize=9)

#     return fig


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
# from matplotlib.colors import LinearSegmentedColormap
# from su2_postprocess.utils.parse_forces import extract_case_metadata_from_log

# def _warn_missing_overlay(labels, overlay, name="Overlay"):
#     if isinstance(overlay, dict):
#         missing = [l for l in labels if l not in overlay]
#         if missing:
#             print(f"[WARN] {name} missing for: {', '.join(missing)}")

# def plot_cp_cf_it_multi(
#     dfs,
#     labels,
#     merge_slider=0.03,
#     show_cp=True,
#     show_cf=True,
#     show_it=True,
#     show_map=True,
#     forces_files=None,
#     xfoil_data=None,
#     exp_data=None,
# ):
#     if len(dfs) != len(labels):
#         raise ValueError("dfs and labels must match in length.")
#     if forces_files and len(forces_files) != len(dfs):
#         raise ValueError("forces_files must be None or match dfs length.")

#     single_case = len(dfs) == 1
#     merge_slider = np.clip(merge_slider, 0.0, 1.0)

#     section_order = []
#     if show_cp: section_order.append("cp")
#     if show_cf: section_order.append("cf")
#     if show_it and any("Turb_index" in df.columns for df in dfs): section_order.append("it")
#     if show_map: section_order.append("map")

#     section_heights = {
#         "cp": 0.28,
#         "cf": 0.22,
#         "it": 0.18,
#         "map": 0.25
#     }
#     spacing = 0.1 * merge_slider

#     bottoms = {}
#     current_bottom = 1.0
#     for name in section_order:
#         height = section_heights[name]
#         bottoms[name] = current_bottom - height
#         current_bottom = bottoms[name] - spacing

#     fig_height = sum(section_heights[name] for name in section_order) + len(section_order) * spacing
#     fig = plt.figure(figsize=(5, fig_height * 9))

#     def stylize(ax, show_xlabel=False, show_xticks=False, show_bottom_spine=False):
#         ax.set_facecolor("none")
#         ax.grid(False)
#         ax.tick_params(axis='x', direction='out', width=0.8, bottom=show_xticks, labelbottom=show_xticks)
#         ax.tick_params(axis='y', direction='out', width=0.8)
#         ax.spines['top'].set_visible(False)
#         ax.spines['left'].set_visible(True)
#         ax.spines['left'].set_linewidth(0.8)
#         ax.spines['right'].set_visible(True)
#         ax.spines['right'].set_linewidth(0.8)
#         ax.spines['bottom'].set_visible(show_bottom_spine)
#         ax.spines['bottom'].set_linewidth(0.8 if show_bottom_spine else 0.0)
#         if show_xlabel:
#             ax.set_xlabel(r'x/c', fontsize=12)

#     bottom_most_panel = section_order[-2] if "map" in section_order else section_order[-1]

#     if "cp" in section_order:
#         ax_cp = fig.add_axes([0.12, bottoms["cp"], 0.75, section_heights["cp"]])
#         cp_all = []

#         for df, label in zip(dfs, labels):
#             ax_cp.plot(df["x"], df["Pressure_Coefficient"], 'k-', lw=0.5, label=label)
#             cp_all.append(df["Pressure_Coefficient"].to_numpy())

#         if isinstance(xfoil_data, dict):
#             for label, df in xfoil_data.items():
#                 if {"x", "Cp"}.issubset(df.columns):
#                     ax_cp.plot(df["x"], df["Cp"], '-.', lw=1.0, color='black', label=f"XFOIL: {label}")
#                     cp_all.append(df["Cp"].to_numpy())
#         elif xfoil_data is not None and {"x", "Cp"}.issubset(xfoil_data.columns):
#             ax_cp.plot(xfoil_data["x"], xfoil_data["Cp"], '-.', lw=1.0, color='black', label="XFOIL")
#             cp_all.append(xfoil_data["Cp"].to_numpy())

#         if isinstance(exp_data, dict):
#             for label, df in exp_data.items():
#                 if {"x", "Cp"}.issubset(df.columns):
#                     ax_cp.plot(df["x"], df["Cp"], ':', lw=1.0, color='gray', label=f"EXP: {label}")
#                     cp_all.append(df["Cp"].to_numpy())
#         elif exp_data is not None and {"x", "Cp"}.issubset(exp_data.columns):
#             ax_cp.plot(exp_data["x"], exp_data["Cp"], ':', lw=1.0, color='gray', label="EXP")
#             cp_all.append(exp_data["Cp"].to_numpy())

#         cp_flat = np.concatenate(cp_all)
#         margin = 0.05 * (cp_flat.max() - cp_flat.min())
#         ax_cp.set_ylim(cp_flat.max() + margin, cp_flat.min() - margin)

#         ax_cp.set_ylabel(r'C$_p$', fontsize=12)
#         ax_cp.set_xlim(0, 1)
#         stylize(ax_cp, show_xticks=False)

#         if single_case and forces_files and forces_files[0].exists():
#             label = extract_case_metadata_from_log(forces_files[0])
#             ax_cp.text(
#                 1.2, 1.15, label,
#                 fontsize=10, ha='right', va='top',
#                 transform=ax_cp.transAxes,
#                 bbox=dict(facecolor='white', edgecolor='gray', linewidth=0.1, boxstyle='round, pad=.2', alpha=1)
#             )

#         ax_cp.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.1)
#         ax_cp.legend(fontsize=9, loc="lower right")

#     if "cf" in section_order:
#         ax_cf = fig.add_axes([0.12, bottoms["cf"], 0.75, section_heights["cf"]])
#         cf_all = []

#         for df, label in zip(dfs, labels):
#             ax_cf.plot(df["x"], df["Skin_Friction_Coefficient_x"], 'k-', lw=0.5, label=label)
#             cf_all.append(df["Skin_Friction_Coefficient_x"].to_numpy())

#         if isinstance(xfoil_data, dict):
#             for label, df in xfoil_data.items():
#                 if {"x", "cf"}.issubset(df.columns):
#                     ax_cf.plot(df["x"], df["cf"], '--', lw=1.0, label=f"XFOIL: {label}")
#                     cf_all.append(df["cf"].to_numpy())
#         elif xfoil_data is not None and {"x", "cf"}.issubset(xfoil_data.columns):
#             ax_cf.plot(xfoil_data["x"], xfoil_data["cf"], '--', lw=1.0, label="XFOIL")
#             cf_all.append(xfoil_data["cf"].to_numpy())

#         if isinstance(exp_data, dict):
#             for label, df in exp_data.items():
#                 if {"x", "cf"}.issubset(df.columns):
#                     ax_cf.plot(df["x"], df["cf"], 'o', ms=2.5, label=f"EXP: {label}")
#                     cf_all.append(df["cf"].to_numpy())
#         elif exp_data is not None and {"x", "cf"}.issubset(exp_data.columns):
#             ax_cf.plot(exp_data["x"], exp_data["cf"], 'o', ms=2.5, label="EXP")
#             cf_all.append(exp_data["cf"].to_numpy())

#         cf_flat = np.concatenate(cf_all)
#         margin = 0.05 * (cf_flat.max() - cf_flat.min())
#         ax_cf.set_ylim(cf_flat.min() - margin, cf_flat.max() + margin)

#         ax_cf.set_ylabel(r'C$_f$', fontsize=12)
#         ax_cf.set_xlim(0, 1)
#         stylize(ax_cf, show_xlabel=(bottom_most_panel == "cf"), show_xticks=(bottom_most_panel == "cf"), show_bottom_spine=(bottom_most_panel == "cf"))

#         if len(ax_cf.get_legend_handles_labels()[1]) > 1:
#             ax_cf.legend(fontsize=9, loc="best")

#     if "it" in section_order:
#         ax_it = fig.add_axes([0.12, bottoms["it"], 0.75, section_heights["it"]])
#         for df, label in zip(dfs, labels):
#             if "Turb_index" in df.columns:
#                 ax_it.plot(df["x"], df["Turb_index"].clip(0, 1), label=label, lw=0.5)
#         ax_it.set_ylabel(r'i$_t$', fontsize=12)
#         ax_it.set_xlim(0, 1)
#         ax_it.set_ylim(0, 1.05)
#         stylize(ax_it, show_xlabel=(bottom_most_panel == "it"), show_xticks=(bottom_most_panel == "it"), show_bottom_spine=(bottom_most_panel == "it"))
#         ax_it.legend(fontsize=9, loc="best")

#     if "map" in section_order and "Turb_index" in dfs[0].columns:
#         df0 = dfs[0]
#         x, y, it = df0["x"], df0["y"], df0["Turb_index"].clip(0, 1)
#         points = np.array([x, y]).T.reshape(-1, 1, 2)
#         segments = np.concatenate([points[:-1], points[1:]], axis=1)
#         lc = LineCollection(segments, cmap="viridis_r", norm=plt.Normalize(0, 1))
#         lc.set_array(it)
#         lc.set_linewidth(1.2)

#         ax_map = fig.add_axes([0.12, bottoms["map"], 0.75, section_heights["map"]])
#         ax_map.add_collection(lc)
#         ax_map.set_xlim(-0.005, 1)
#         ax_map.set_ylim(y.min() - 0.01, y.max() + 0.01)
#         ax_map.set_aspect("equal")
#         ax_map.axis("off")

#         cax = fig.add_axes([0.9, bottoms["map"] + 0.06, 0.02, 0.16])
#         cbar = plt.colorbar(lc, cax=cax)
#         cbar.set_label(r'Turbulence Index, i$_t$', fontsize=10)
#         cbar.set_ticks([0, 0.5, 1])
#         cbar.ax.tick_params(labelsize=9)

#     return fig









# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
# from matplotlib.colors import LinearSegmentedColormap
# from su2_postprocess.utils.parse_forces import extract_case_metadata_from_log

# def _warn_missing_overlay(labels, overlay, name="Overlay"):
#     if isinstance(overlay, dict):
#         missing = [l for l in labels if l not in overlay]
#         if missing:
#             print(f"[WARN] {name} missing for: {', '.join(missing)}")


# def plot_cp_cf_it_multi(
#     dfs,
#     labels,
#     merge_slider=0.03,
#     show_cp=True,
#     show_cf=True,
#     show_it=True,
#     show_map=True,
#     forces_files=None,
#     xfoil_data=None,
#     exp_data=None,
# ):
#     if len(dfs) != len(labels):
#         raise ValueError("dfs and labels must match in length.")
#     if forces_files and len(forces_files) != len(dfs):
#         raise ValueError("forces_files must be None or match dfs length.")

#     single_case = len(dfs) == 1
#     merge_slider = np.clip(merge_slider, 0.0, 1.0)

#     section_order = []
#     if show_cp: section_order.append("cp")
#     if show_cf: section_order.append("cf")
#     if show_it and any("Turb_index" in df.columns for df in dfs): section_order.append("it")
#     if show_map: section_order.append("map")

#     section_heights = {
#         "cp": 0.28,
#         "cf": 0.22,
#         "it": 0.18,
#         "map": 0.25
#     }
#     spacing = 0.1 * merge_slider

#     bottoms = {}
#     current_bottom = 1.0
#     for name in section_order:
#         height = section_heights[name]
#         bottoms[name] = current_bottom - height
#         current_bottom = bottoms[name] - spacing

#     fig_height = sum(section_heights[name] for name in section_order) + len(section_order) * spacing
#     fig = plt.figure(figsize=(5, fig_height * 9))

#     def stylize(ax, show_xlabel=False, show_xticks=False, show_bottom_spine=False):
#         ax.set_facecolor("none")
#         ax.grid(False)
#         ax.tick_params(axis='x', direction='out', width=0.8, bottom=show_xticks, labelbottom=show_xticks)
#         ax.tick_params(axis='y', direction='out', width=0.8)
#         ax.spines['top'].set_visible(False)
#         ax.spines['left'].set_visible(True)
#         ax.spines['left'].set_linewidth(0.8)
#         ax.spines['bottom'].set_visible(show_bottom_spine)
#         ax.spines['bottom'].set_linewidth(0.8 if show_bottom_spine else 0.0)
#         if show_xlabel:
#             ax.set_xlabel(r'x/c', fontsize=12)

#     # -- Cp Panel
#     if "cp" in section_order:
#         ax_cp = fig.add_axes([0.12, bottoms["cp"], 0.75, section_heights["cp"]])
#         cp_all = []

#         for i, (df, label) in enumerate(zip(dfs, labels)):
#             ax_cp.plot(df["x"], df["Pressure_Coefficient"], 'k-', lw=0.5, label=label)
#             cp_all.append(df["Pressure_Coefficient"].to_numpy())

#         # Overlay support
#         if isinstance(xfoil_data, dict):
#             for label, df in xfoil_data.items():
#                 if {"x", "Cp"}.issubset(df.columns):
#                     ax_cp.plot(df["x"], df["Cp"], '-.', lw=1.0, color='black', label=f"XFOIL: {label}")
#                     cp_all.append(df["Cp"].to_numpy())
#         elif xfoil_data is not None and {"x", "Cp"}.issubset(xfoil_data.columns):
#             ax_cp.plot(xfoil_data["x"], xfoil_data["Cp"], '-.', lw=1.0, color='black', label="XFOIL")
#             cp_all.append(xfoil_data["Cp"].to_numpy())

#         if isinstance(exp_data, dict):
#             for label, df in exp_data.items():
#                 if {"x", "Cp"}.issubset(df.columns):
#                     ax_cp.plot(df["x"], df["Cp"], ':', lw=1.0, color='gray', label=f"EXP: {label}")
#                     cp_all.append(df["Cp"].to_numpy())
#         elif exp_data is not None and {"x", "Cp"}.issubset(exp_data.columns):
#             ax_cp.plot(exp_data["x"], exp_data["Cp"], ':', lw=1.0, color='gray', label="EXP")
#             cp_all.append(exp_data["Cp"].to_numpy())

#         # Autoscaling
#         cp_flat = np.concatenate(cp_all)
#         margin = 0.05 * (cp_flat.max() - cp_flat.min())
#         ax_cp.set_ylim(cp_flat.max() + margin, cp_flat.min() - margin)

#         ax_cp.set_ylabel(r'C$_p$', fontsize=12)
#         ax_cp.set_xlim(0, 1)
#         stylize(ax_cp, show_xticks=False)

#         # Metadata annotation
#         if single_case and forces_files and forces_files[0].exists():
#             label = extract_case_metadata_from_log(forces_files[0])
#             ax_cp.text(
#                 1.2, 1.15, label,
#                 fontsize=10, ha='right', va='top',
#                 transform=ax_cp.transAxes,
#                 bbox=dict(facecolor='white', edgecolor='gray', linewidth=0.1, boxstyle='round, pad=.2', alpha=1)
#             )

#         ax_cp.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.1)
#         ax_cp.legend(fontsize=9, loc="lower right")

#     # -- Cf Panel
#     if "cf" in section_order:
#         ax_cf = fig.add_axes([0.12, bottoms["cf"], 0.75, section_heights["cf"]])
#         cf_all = []

#         for i, (df, label) in enumerate(zip(dfs, labels)):
#             ax_cf.plot(df["x"], df["Skin_Friction_Coefficient_x"], 'k-', lw=0.5, label=label)
#             cf_all.append(df["Skin_Friction_Coefficient_x"].to_numpy())

#         # Overlays
#         if isinstance(xfoil_data, dict):
#             for label, df in xfoil_data.items():
#                 if {"x", "cf"}.issubset(df.columns):
#                     ax_cf.plot(df["x"], df["cf"], '--', lw=1.0, label=f"XFOIL: {label}")
#                     cf_all.append(df["cf"].to_numpy())
#         elif xfoil_data is not None and {"x", "cf"}.issubset(xfoil_data.columns):
#             ax_cf.plot(xfoil_data["x"], xfoil_data["cf"], '--', lw=1.0, label="XFOIL")
#             cf_all.append(xfoil_data["cf"].to_numpy())

#         if isinstance(exp_data, dict):
#             for label, df in exp_data.items():
#                 if {"x", "cf"}.issubset(df.columns):
#                     ax_cf.plot(df["x"], df["cf"], 'o', ms=2.5, label=f"EXP: {label}")
#                     cf_all.append(df["cf"].to_numpy())
#         elif exp_data is not None and {"x", "cf"}.issubset(exp_data.columns):
#             ax_cf.plot(exp_data["x"], exp_data["cf"], 'o', ms=2.5, label="EXP")
#             cf_all.append(exp_data["cf"].to_numpy())

#         # Autoscaling
#         cf_flat = np.concatenate(cf_all)
#         margin = 0.05 * (cf_flat.max() - cf_flat.min())
#         ax_cf.set_ylim(cf_flat.min() - margin, cf_flat.max() + margin)

#         ax_cf.set_ylabel(r'C$_f$', fontsize=12)
#         ax_cf.set_xlim(0, 1)
#         stylize(ax_cf, show_xlabel=True, show_xticks=True, show_bottom_spine=True)

#         if len(ax_cf.get_legend_handles_labels()[1]) > 1:
#             ax_cf.legend(fontsize=9, loc="best")

#     # -- i_t Panel
#     if "it" in section_order:
#         ax_it = fig.add_axes([0.12, bottoms["it"], 0.75, section_heights["it"]])
#         for df, label in zip(dfs, labels):
#             if "Turb_index" in df.columns:
#                 ax_it.plot(df["x"], df["Turb_index"].clip(0, 1), label=label, lw=0.5)
#         ax_it.set_ylabel(r'i$_t$', fontsize=12)
#         ax_it.set_xlim(0, 1)
#         ax_it.set_ylim(0, 1.05)
#         stylize(ax_it, show_xlabel=True, show_xticks=True, show_bottom_spine=True)
#         ax_it.legend(fontsize=9, loc="best")

#     # -- Transition map
#     if "map" in section_order and "Turb_index" in dfs[0].columns:
#         df0 = dfs[0]
#         x, y, it = df0["x"], df0["y"], df0["Turb_index"].clip(0, 1)
#         points = np.array([x, y]).T.reshape(-1, 1, 2)
#         segments = np.concatenate([points[:-1], points[1:]], axis=1)
#         lc = LineCollection(segments, cmap="viridis_r", norm=plt.Normalize(0, 1))
#         lc.set_array(it)
#         lc.set_linewidth(1.2)

#         ax_map = fig.add_axes([0.12, bottoms["map"], 0.75, section_heights["map"]])
#         ax_map.add_collection(lc)
#         ax_map.set_xlim(-0.005, 1)
#         ax_map.set_ylim(y.min() - 0.01, y.max() + 0.01)
#         ax_map.set_aspect("equal")
#         ax_map.axis("off")

#         cax = fig.add_axes([0.9, bottoms["map"] + 0.06, 0.02, 0.16])
#         cbar = plt.colorbar(lc, cax=cax)
#         cbar.set_label(r'Turbulence Index, i$_t$', fontsize=10)
#         cbar.set_ticks([0, 0.5, 1])
#         cbar.ax.tick_params(labelsize=9)

#     return fig
