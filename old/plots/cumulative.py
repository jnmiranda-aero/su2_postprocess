import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from su2_postprocess.utils.parse_forces import extract_case_metadata

def plot_cp_cf_it(df, merge_slider=0.03, show_cp=True, show_cf=True, show_it=True, show_map=True,
                  forces_file=None, xfoil_df=None, exp_df=None):
    merge_slider = np.clip(merge_slider, 0.0, 1.0)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    cp = df["Pressure_Coefficient"].to_numpy()
    cf = df["Skin_Friction_Coefficient_x"].to_numpy()
    has_it = "Turb_index" in df.columns
    it = df["Turb_index"].clip(0, 1).to_numpy() if has_it else None

    # xfoil_df = load_overlay_data(xfoil_file) if xfoil_file else None
    # exp_df   = load_overlay_data(exp_file)   if exp_file   else None

    section_order = []
    if show_cp: section_order.append("cp")
    if show_cf: section_order.append("cf")
    if show_it and has_it: section_order.append("it")
    # if show_map and has_it: section_order.append("map")
    if show_map: section_order.append("map")

    section_heights = {
        "cp": 0.28,
        "cf": 0.22,
        "it": 0.18,
        "map": 0.25
    }
    spacing = 0.1 * merge_slider

    bottoms = {}
    current_bottom = 1.0
    for name in section_order:
        height = section_heights[name]
        bottoms[name] = current_bottom - height
        current_bottom = bottoms[name] - spacing

    fig_height = sum([section_heights[name] for name in section_order]) + len(section_order) * spacing
    fig = plt.figure(figsize=(5, fig_height * 9))

    custom_cmap = LinearSegmentedColormap.from_list("blue_purple_red", [
        (0.0, "#0044ff"),  # blue
        (0.5, "#933fe0"),  # blue
        (1.0, "#ff0000"),  # red
    ])

    def stylize(ax, show_xlabel=False, show_xticks=False, show_bottom_spine=False, show_right_spine=False):

        ax.set_facecolor("none")
        ax.grid(False)
        ax.tick_params(axis='x', direction='out', width=0.8, bottom=show_xticks, top=False, labelbottom=show_xticks)
        ax.tick_params(axis='y', direction='out', width=0.8)
        for spine in ['top']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(0.8)
        if show_bottom_spine:
            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_linewidth(0.8)
        else:
            ax.spines['bottom'].set_visible(False)
        if show_xlabel:
            ax.set_xlabel(r'x/c', fontsize=12)

    if "cp" in section_order:
        ax_cp = fig.add_axes([0.12, bottoms["cp"], 0.75, section_heights["cp"]])
        ax_cp.plot(x, cp, 'k-', lw=0.5)
        ax_cp.set_ylabel(r'C$_p$', fontsize=12)
        ax_cp.invert_yaxis()
        cp_all = [cp]
        if xfoil_df is not None and "Cp" in xfoil_df.columns:
            cp_all.append(xfoil_df["Cp"].to_numpy())
        if exp_df is not None and "Cp" in exp_df.columns:
            cp_all.append(exp_df["Cp"].to_numpy())

        cp_all_flat = np.concatenate(cp_all)
        cp_ymax = np.max(cp_all_flat)
        cp_ymin = np.min(cp_all_flat)
        margin = 0.05 * (cp_ymax - cp_ymin)

        ax_cp.set_ylim(cp_ymax + margin, cp_ymin - margin)
        
        # ax_cp.set_ylim(cp.max(), cp.min() + 0.1 * cp.min())
        #ax_cp.set_ybound(lower=cp.max(), upper=(cp.min() + 0.1 * cp.min()))
        stylize(ax_cp, show_xticks=False) 

        ax_cp.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.1)
        # ax_cp.grid(which='minor', linestyle='-.', linewidth=0.1)
        # ax_cp.minorticks_on()        
        
        ax_cp.set_xlim(0, 1)
        
        if forces_file:
            label = extract_case_metadata(forces_file)
            ax_cp.text(
                1.2, 1.15, label,
                fontsize=10, ha='right', va='top',
                transform=ax_cp.transAxes,
                bbox=dict(facecolor='white', edgecolor='gray', linewidth=0.1, boxstyle='round, pad=.2', alpha = 1)
            )

        # if xfoil_df is not None and "x" in xfoil_df.columns and "cp" in xfoil_df.columns:
        #     ax_cp.plot(xfoil_df["x"], xfoil_df["cp"], '--', lw=1.0, label=xfoil_label)

        # if exp_df is not None and "x" in exp_df.columns and "cp" in exp_df.columns:
        #     ax_cp.plot(exp_df["x"], exp_df["cp"], 'o', ms=2.5, label=exp_label)

        # # Add legend if any overlays present
        # if xfoil_df is not None or exp_df is not None:
        #     ax_cp.legend(fontsize=9, loc="upper right")

    if xfoil_df is not None and {"x", "Cp"}.issubset(xfoil_df.columns):
        ax_cp.plot(xfoil_df["x"], xfoil_df["Cp"], ls="-.", lw=1.0, color="black", label="XFOIL")

    if exp_df is not None and {"x", "Cp"}.issubset(exp_df.columns):
        ax_cp.plot(exp_df["x"], exp_df["Cp"], ls=":", lw=1.0, color="gray", label="Experiment")

    if xfoil_df is not None or exp_df is not None:
        ax_cp.legend(fontsize=9, loc="lower right")

    if "cf" in section_order:
        ax_cf = fig.add_axes([0.12, bottoms["cf"], 0.75, section_heights["cf"]])

        # Always plot SU2 Cf data
        ax_cf.plot(x, cf, 'k-', lw=0.5, label="SU2")

        # Aggregate all Cf values for autoscaling
        cf_all = [cf]

        if xfoil_df is not None and {"x", "cf"}.issubset(xfoil_df.columns):
            ax_cf.plot(xfoil_df["x"], xfoil_df["cf"], '--', lw=1.0, label="XFOIL")
            cf_all.append(xfoil_df["cf"].to_numpy())

        if exp_df is not None and {"x", "cf"}.issubset(exp_df.columns):
            ax_cf.plot(exp_df["x"], exp_df["cf"], 'o', ms=2.5, label="Experiment")
            cf_all.append(exp_df["cf"].to_numpy())

        # Compute global y-limits across all sources
        cf_all_flat = np.concatenate(cf_all)
        cf_ymin = np.min(cf_all_flat)
        cf_ymax = np.max(cf_all_flat)
        cf_margin = 0.05 * (cf_ymax - cf_ymin)
        ax_cf.set_ylim(cf_ymin - cf_margin, cf_ymax + cf_margin)

        ax_cf.set_ylabel(r'C$_f$', fontsize=12)
        ax_cf.set_xlim(0, 1)

        stylize(ax_cf, show_xlabel=True, show_xticks=True, show_bottom_spine=True)
        ax_cf.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.1)

        # Legend only if more than one label exists
        if len(ax_cf.get_legend_handles_labels()[1]) > 1:
            ax_cf.legend(fontsize=9, loc="best")

    if xfoil_df is not None and {"x", "cf"}.issubset(xfoil_df.columns):
        ax_cf.plot(xfoil_df["x"], xfoil_df["cf"], '--', lw=1.0, label="XFOIL")

    if exp_df is not None and {"x", "cf"}.issubset(exp_df.columns):
        ax_cf.plot(exp_df["x"], exp_df["cf"], 'o', ms=2.5, label="Experiment")

    if xfoil_df is not None or exp_df is not None:
        ax_cf.legend(fontsize=9, loc="best")





    if "it" in section_order:
        ax_it = fig.add_axes([0.12, bottoms["it"], 0.75, section_heights["it"]])
        ax_it.plot(x, it, 'k-', lw=0.5)
        ax_it.set_ylabel(r'i$_t$', fontsize=12)
        ax_it.set_xlim(0, 1)
        ax_it.set_ylim(0, 1.05)
        stylize(ax_it, show_xlabel=True, show_xticks=True, show_bottom_spine=True)

    if "map" in section_order:
        ax_map = fig.add_axes([0.12, bottoms["map"], 0.75, section_heights["map"]])
        ax_map.set_xlim(-0.005, 1)
        ax_map.set_ylim(y.min() - 0.01, y.max() + 0.01)
        ax_map.set_aspect('equal')
        ax_map.axis('off')

        # Always create airfoil geometry segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # if has_it and show_it:
        if has_it and show_map:
            # Colored transition map with i_t
            lc = LineCollection(segments, cmap='viridis_r', norm=plt.Normalize(0, 1))
            # lc = LineCollection(segments, cmap=custom_cmap, norm=plt.Normalize(0, 1))
            lc.set_array(it)
            lc.set_linewidth(1.2)
            ax_map.add_collection(lc)

            # Colorbar
            cax = fig.add_axes([0.9, bottoms["map"] + 0.06, 0.02, 0.16])
            cbar = plt.colorbar(lc, cax=cax)
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels(['0', '0.5', '1'])
            cbar.set_label(r'Turbulence Index, i$_t$', fontsize=10)
            cbar.ax.tick_params(labelsize=9)
        else:
            # Black outline only
            ax_map.plot(x, y, 'k-', lw=1.0)

        
    # if "map" in section_order:
    #     ax_map = fig.add_axes([0.12, bottoms["map"], 0.75, section_heights["map"]])

    #     points = np.array([x, y]).T.reshape(-1, 1, 2)
    #     segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #     lc = LineCollection(segments, cmap='viridis_r', norm=plt.Normalize(0, 1))
    #     # lc = LineCollection(segments, cmap='coolwarm_r', norm=plt.Normalize(0, 1))
    #     lc.set_array(it)
    #     lc.set_linewidth(1.2)

    #     ax_map.add_collection(lc)
    #     ax_map.set_xlim(-0.005, 1)
    #     ax_map.set_ylim(y.min() - 0.01, y.max() + 0.01)
    #     ax_map.set_aspect('equal')
    #     ax_map.axis('off')

    #     cax = fig.add_axes([0.9, bottoms["map"] + 0.06, 0.02, 0.16])
    #     cbar = plt.colorbar(lc, cax=cax)
    #     cbar.set_ticks([0, 0.5, 1])
    #     cbar.set_ticklabels(['0', '0.5', '1'])
    #     cbar.set_label(r'Turbulence Index, i$_t$', fontsize=10)
    #     cbar.ax.tick_params(labelsize=9)


    return fig
'''
import matplotlib.pyplot as plt
import numpy as np


def plot_cp_cf_it(df, merge_slider=1.0):
    """
    Plots a stacked Cp–Cf–i_t plot. Adds airfoil transition inset if Turb_index is present.
    """
    merge_slider = np.clip(merge_slider, 0.0, 1.0)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    cp = df["Pressure_Coefficient"].to_numpy()
    cf = df["Skin_Friction_Coefficient_x"].to_numpy()
    has_it = "Turb_index" in df.columns
    it = df["Turb_index"].clip(0, 1).to_numpy() if has_it else None

    fig = plt.figure(figsize=(6, 8 if has_it else 5))

    cp_height = 0.4
    cf_height = 0.3
    it_height = 0.2 if has_it else 0.0
    spacing = -0.09#0.03 * merge_slider

    cp_bottom = 0.72 if has_it else 0.55
    cf_bottom = cp_bottom - cp_height - spacing
    it_bottom = cf_bottom - cf_height - spacing

    # Cp Panel
    ax_cp = fig.add_axes([0.15, cp_bottom, 0.65, cp_height])
    ax_cp.plot(x, cp, 'k-', lw=1)
    ax_cp.set_ylabel(r'$C_p$', fontsize=12)
    ax_cp.invert_yaxis()
    ax_cp.set_xlim(0, 1)
    ax_cp.set_aspect('auto')
    ax_cp.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_cp.spines['bottom'].set_visible(False)

    # Cf Panel
    ax_cf = fig.add_axes([0.15, cf_bottom, 0.65, cf_height], sharex=ax_cp)
    ax_cf.plot(x, cf, 'k-', lw=1)
    ax_cf.set_ylabel(r'$C_f$', fontsize=12)
    ax_cf.set_xlim(0, 1)
    ax_cf.set_aspect('auto') 
    ax_cf.tick_params(labelbottom=not has_it)
    ax_cf.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_cf.tick_params(axis='x', top=False, labeltop=False)
    ax_cf.spines['bottom'].set_visible(False)
    ax_cf.spines['top'].set_visible(False)

    # i_t Panel (optional)
    if has_it:
        ax_it = fig.add_axes([0.15, it_bottom, 0.65, it_height], sharex=ax_cp)
        ax_it.plot(x, it, 'k-', lw=1)
        ax_it.fill_between(x, 0, it, color='gray', alpha=0.3)
        ax_it.set_ylabel(r'$i_t$', fontsize=12)
        ax_it.set_xlabel(r'$x/c$', fontsize=12)
        ax_it.set_xlim(0, 1)
        ax_it.set_ylim(0, 1.05)
        ax_it.set_aspect('auto')
        ax_it.tick_params(axis='x', top=False, labeltop=False)
        ax_it.spines['bottom'].set_visible(False)
        ax_it.spines['top'].set_visible(False)
    else:
        print("Note: 'Turb_index' not found — generating Cp–Cf only.")

    # Airfoil Transition Inset (optional)
#    if has_it:
#        ax_airfoil = fig.add_axes([0.82, 0.45, 0.13, 0.13])
#        sc = ax_airfoil.scatter(x, y, c=it, cmap='viridis', s=4)
#        ax_airfoil.set_title("iₜ map", fontsize=10)
#        ax_airfoil.axis('equal')
#        ax_airfoil.axis('off')

    return fig
'''

'''
import matplotlib.pyplot as plt
import numpy as np

def plot_cp_cf_it(df, merge_slider=1.0):
    """
    Plots a stacked Cp–Cf–i_t plot. Adds a full-width transition map at the bottom.
    """
    merge_slider = np.clip(merge_slider, 0.0, 1.0)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    cp = df["Pressure_Coefficient"].to_numpy()
    cf = df["Skin_Friction_Coefficient_x"].to_numpy()
    has_it = "Turb_index" in df.columns
    it = df["Turb_index"].clip(0, 1).to_numpy() if has_it else None

    # Layout
    cp_height = 0.25
    cf_height = 0.20
    it_height = 0.15 if has_it else 0.0
    map_height = 0.28 if has_it else 0.0
    spacing = 0.025 * merge_slider

    # Panel bottoms from top
    cp_bottom = 0.95 - cp_height
    cf_bottom = cp_bottom - cf_height - spacing
    it_bottom = cf_bottom - it_height - spacing
    map_bottom = it_bottom - map_height - spacing

    fig_height = cp_height + cf_height + it_height + map_height + 4 * spacing
    fig = plt.figure(figsize=(7, fig_height * 10))

    # Cp Panel
    ax_cp = fig.add_axes([0.15, cp_bottom, 0.75, cp_height])
    ax_cp.plot(x, cp, 'k-', lw=1)
    ax_cp.set_ylabel(r'$C_p$', fontsize=12)
    ax_cp.invert_yaxis()
    ax_cp.set_xlim(0, 1)
    ax_cp.grid(True)
    ax_cp.tick_params(labelbottom=False)

    # Cf Panel
    ax_cf = fig.add_axes([0.15, cf_bottom, 0.75, cf_height], sharex=ax_cp)
    ax_cf.plot(x, cf, 'k-', lw=1)
    ax_cf.set_ylabel(r'$C_f$', fontsize=12)
    ax_cf.grid(True)
    ax_cf.tick_params(labelbottom=False)

    # i_t Panel (optional)
    if has_it:
        ax_it = fig.add_axes([0.15, it_bottom, 0.75, it_height], sharex=ax_cp)
        ax_it.plot(x, it, 'k-', lw=1)
        ax_it.fill_between(x, 0, it, color='gray', alpha=0.3)
        ax_it.set_ylabel(r'$i_t$', fontsize=12)
        ax_it.set_xlim(0, 1)
        ax_it.set_ylim(0, 1.05)
        ax_it.grid(True)
        ax_it.tick_params(labelbottom=False)

        # Transition map at the bottom
        ax_map = fig.add_axes([0.15, map_bottom, 0.75, map_height])
        sc = ax_map.scatter(x, y, c=it, cmap='viridis', s=4)
        ax_map.set_title("Transition Map (iₜ)", fontsize=10)
        ax_map.axis('equal')
        ax_map.axis('off')
    else:
        print("Note: 'Turb_index' not found — generating Cp–Cf only.")

    return fig
'''

'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

def plot_cp_cf_it(df, merge_slider=1.0):
    merge_slider = np.clip(merge_slider, 0.0, 1.0)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    cp = df["Pressure_Coefficient"].to_numpy()
    cf = df["Skin_Friction_Coefficient_x"].to_numpy()
    has_it = "Turb_index" in df.columns
    it = df["Turb_index"].clip(0, 1).to_numpy() if has_it else None

    # Layout
    cp_height = 0.28
    cf_height = 0.22
    it_height = 0.18 if has_it else 0.0
    map_height = 0.25 if has_it else 0.0
    spacing = 0.005 * merge_slider

    cp_bottom = 0.95 - cp_height
    cp_bottom = 1 - cp_height
    cf_bottom = cp_bottom - cf_height - spacing
    it_bottom = cf_bottom - it_height - spacing
    map_bottom = it_bottom - map_height - spacing

    fig_height = cp_height + cf_height + it_height + map_height + 4 * spacing
    fig = plt.figure(figsize=(5, fig_height * 9))  # Wider to accommodate external legend
 
    def stylize(ax, show_xlabel=False, show_xticks=False, show_bottom_spine=False):
        ax.set_facecolor("none")
        ax.grid(False)
        ax.tick_params(axis='x', direction='in', width=0.8, bottom=show_xticks, top=False, labelbottom=show_xticks)
        ax.tick_params(axis='y', direction='in', width=0.8)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(0.8)
        if show_bottom_spine:
            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_linewidth(0.8)
        else:
            ax.spines['bottom'].set_visible(False)
        if show_xlabel:
            ax.set_xlabel(r'x/c', fontsize=12)

    # Cp panel
    ax_cp = fig.add_axes([0.12, cp_bottom, 0.75, cp_height])
    ax_cp.plot(x, cp, 'k-', lw=0.5)
    ax_cp.set_ylabel(r'C$_p$', fontsize=12)
    ax_cp.invert_yaxis()
    ax_cp.set_ylim(cp.max(),cp.min()+ 0.1*cp.min())    
    ax_cp.set_xlim(0, 1)
    stylize(ax_cp)

    # Cf panel
    ax_cf = fig.add_axes([0.12, cf_bottom, 0.75, cf_height], sharex=ax_cp)
    ax_cf.plot(x, cf, 'k-', lw=0.5)
    ax_cf.set_ylabel(r'C$_f$', fontsize=12)
    stylize(ax_cf)

    if has_it:
        # i_t panel
        ax_it = fig.add_axes([0.12, it_bottom, 0.75, it_height], sharex=ax_cp)
        ax_it.plot(x, it, 'k-', lw=0.5)
        #ax_it.fill_between(x, 0, it, color='gray', alpha=0.3)
        ax_it.set_ylabel(r'i$_t$', fontsize=12)
        ax_it.set_xlim(0, 1)
        ax_it.set_ylim(0, 1.05)
        stylize(ax_it, show_xlabel=True, show_xticks=True, show_bottom_spine=True)

        # Airfoil transition map — full width
        ax_map = fig.add_axes([0.12, map_bottom, 0.75, map_height])
#        sc = ax_map.scatter(x, y, c=it, cmap='viridis', s=1, vmin=0, vmax=1)


        # Create segments for line coloring
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a colored line collection
        lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, 1))
        lc = LineCollection(segments, cmap='plasma', norm=plt.Normalize(0, 1))
        lc = LineCollection(segments, cmap='inferno', norm=plt.Normalize(0, 1))
        lc = LineCollection(segments, cmap='cividis', norm=plt.Normalize(0, 1))
        lc = LineCollection(segments, cmap='seismic', norm=plt.Normalize(0, 1))
        lc.set_array(it)
        lc.set_linewidth(1.2)

        # Add to plot
        ax_map.add_collection(lc)
        ax_map.set_xlim(-0.005, 1)
        ax_map.set_ylim(y.min() - 0.01, y.max() + 0.01)
        ax_map.set_aspect('equal')
        ax_map.axis('off')

        ax_map.set_xlim(-0.005, 1)  # match airfoil to flow panels
        ax_map.axis('off')
        ax_map.set_aspect('equal')

        # External colorbar with fixed ticks
        cax = fig.add_axes([0.9, map_bottom + 0.06, 0.02, 0.16])
        cbar = plt.colorbar(lc, cax=cax)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['0', '0.5', '1'])
        cbar.set_label(r'Turbulence Index, i$_t$', fontsize=10)
        cbar.ax.tick_params(labelsize=9)
    else:
        stylize(ax_cf, show_xlabel=True, show_xticks=True, show_bottom_spine=True)
        print("Note: 'Turb_index' not found — generating Cp–Cf only.")

    return fig
'''

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.collections import LineCollection
# from su2_postprocess.utils.parse_forces import extract_case_metadata

# def plot_cp_cf_it(df, merge_slider=0.05, show_cp=True, show_cf=True, show_it=True, show_map=True, forces_file=None):
#     merge_slider = np.clip(merge_slider, 0.0, 1.0)

#     x = df["x"].to_numpy()
#     y = df["y"].to_numpy()
#     cp = df["Pressure_Coefficient"].to_numpy()
#     cf = df["Skin_Friction_Coefficient_x"].to_numpy()
#     has_it = "Turb_index" in df.columns
#     it = df["Turb_index"].clip(0, 1).to_numpy() if has_it else None

#     section_order = []
#     if show_cp: section_order.append("cp")
#     if show_cf: section_order.append("cf")
#     if show_it and has_it: section_order.append("it")
#     # if show_map and has_it: section_order.append("map")
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

#     fig_height = sum([section_heights[name] for name in section_order]) + len(section_order) * spacing
#     fig = plt.figure(figsize=(5, fig_height * 9))

#     def stylize(ax, show_xlabel=False, show_xticks=False, show_bottom_spine=False, show_right_spine=False):
#         ax.set_facecolor("none")
#         ax.grid(False)
#         ax.tick_params(axis='x', direction='in', width=0.8, bottom=show_xticks, top=False, labelbottom=show_xticks)
#         ax.tick_params(axis='y', direction='in', width=0.8)
#         for spine in ['top']:
#             ax.spines[spine].set_visible(False)
#         ax.spines['left'].set_visible(True)
#         ax.spines['left'].set_linewidth(0.8)
#         if show_bottom_spine:
#             ax.spines['bottom'].set_visible(True)
#             ax.spines['bottom'].set_linewidth(0.8)
#         else:
#             ax.spines['bottom'].set_visible(False)
#         if show_xlabel:
#             ax.set_xlabel(r'x/c', fontsize=12)

#     if "cp" in section_order:
#         ax_cp = fig.add_axes([0.12, bottoms["cp"], 0.75, section_heights["cp"]])
#         ax_cp.plot(x, cp, 'k-', lw=0.5)
#         ax_cp.set_ylabel(r'C$_p$', fontsize=12)
#         ax_cp.invert_yaxis()
#         ax_cp.set_ylim(cp.max(), cp.min() + 0.1 * cp.min())
#         #ax_cp.set_ybound(lower=cp.max(), upper=(cp.min() + 0.1 * cp.min()))
#         stylize(ax_cp, show_xticks=False) 

#         ax_cp.grid(which='major', linestyle='--', linewidth=0.3)
#         # ax_cp.grid(which='minor', linestyle='-.', linewidth=0.1)
#         # ax_cp.minorticks_on()        
        
#         ax_cp.set_xlim(0, 1)
        
#         if forces_file:
#             label = extract_case_metadata(forces_file)
#             ax_cp.text(
#                 1.2, 1.1, label,
#                 fontsize=10, ha='right', va='top',
#                 transform=ax_cp.transAxes,
#                 bbox=dict(facecolor='white', edgecolor='white', boxstyle='round, pad=0.0', alpha = 1)
#             )

#     if "cf" in section_order:
#         ax_cf = fig.add_axes([0.12, bottoms["cf"], 0.75, section_heights["cf"]])
#         ax_cf.plot(x, cf, 'k-', lw=0.5)
#         ax_cf.set_ylabel(r'C$_f$', fontsize=12)
#         ax_cf.set_xlim(0, 1)

#         stylize(ax_cf, show_xlabel=True, show_xticks=True, show_bottom_spine=True)        
#         #stylize(ax_cf)#, show_xlabel=True, show_xticks=True, show_bottom_spine=True)
        
#         ax_cf.grid(which='major', linestyle='--', linewidth=0.3)
#         # ax_cf.grid(which='minor', linestyle='-.', linewidth=0.1)
#         # ax_cf.minorticks_on()

#     if "it" in section_order:
#         ax_it = fig.add_axes([0.12, bottoms["it"], 0.75, section_heights["it"]])
#         ax_it.plot(x, it, 'k-', lw=0.5)
#         ax_it.set_ylabel(r'i$_t$', fontsize=12)
#         ax_it.set_xlim(0, 1)
#         ax_it.set_ylim(0, 1.05)
#         stylize(ax_it, show_xlabel=True, show_xticks=True, show_bottom_spine=True)

#     if "map" in section_order:
#         ax_map = fig.add_axes([0.12, bottoms["map"], 0.75, section_heights["map"]])
#         ax_map.set_xlim(-0.005, 1)
#         ax_map.set_ylim(y.min() - 0.01, y.max() + 0.01)
#         ax_map.set_aspect('equal')
#         ax_map.axis('off')

#         # Always create airfoil geometry segments
#         points = np.array([x, y]).T.reshape(-1, 1, 2)
#         segments = np.concatenate([points[:-1], points[1:]], axis=1)

#         # if has_it and show_it:
#         if has_it and show_map:
#             # Colored transition map with i_t
#             lc = LineCollection(segments, cmap='viridis_r', norm=plt.Normalize(0, 1))
#             lc.set_array(it)
#             lc.set_linewidth(1.2)
#             ax_map.add_collection(lc)

#             # Colorbar
#             cax = fig.add_axes([0.9, bottoms["map"] + 0.06, 0.02, 0.16])
#             cbar = plt.colorbar(lc, cax=cax)
#             cbar.set_ticks([0, 0.5, 1])
#             cbar.set_ticklabels(['0', '0.5', '1'])
#             cbar.set_label(r'Turbulence Index, i$_t$', fontsize=10)
#             cbar.ax.tick_params(labelsize=9)
#         else:
#             # Black outline only
#             ax_map.plot(x, y, 'k-', lw=1.0)

        
#     # if "map" in section_order:
#     #     ax_map = fig.add_axes([0.12, bottoms["map"], 0.75, section_heights["map"]])

#     #     points = np.array([x, y]).T.reshape(-1, 1, 2)
#     #     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     #     lc = LineCollection(segments, cmap='viridis_r', norm=plt.Normalize(0, 1))
#     #     # lc = LineCollection(segments, cmap='coolwarm_r', norm=plt.Normalize(0, 1))
#     #     lc.set_array(it)
#     #     lc.set_linewidth(1.2)

#     #     ax_map.add_collection(lc)
#     #     ax_map.set_xlim(-0.005, 1)
#     #     ax_map.set_ylim(y.min() - 0.01, y.max() + 0.01)
#     #     ax_map.set_aspect('equal')
#     #     ax_map.axis('off')

#     #     cax = fig.add_axes([0.9, bottoms["map"] + 0.06, 0.02, 0.16])
#     #     cbar = plt.colorbar(lc, cax=cax)
#     #     cbar.set_ticks([0, 0.5, 1])
#     #     cbar.set_ticklabels(['0', '0.5', '1'])
#     #     cbar.set_label(r'Turbulence Index, i$_t$', fontsize=10)
#     #     cbar.ax.tick_params(labelsize=9)

#     return fig


