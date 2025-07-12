import matplotlib as mpl

USE_TEX = bool(mpl.rcParams.get("text.usetex", False))

# ------------------------------------------------------------------
# Legend label
# ------------------------------------------------------------------
def legend_label(meta: dict, style: str, case_name: str, n_cases: int) -> str:
    turb  = meta.get("turbulence_model", "???")
    trans = meta.get("transition_model", "???")
    correl = meta.get("correlation_model", "???")

    if style == "short":
        # return f"{turb}-{trans}{corr}"
        return f"{turb}-{trans}{correl}"

    if style == "full":
        if USE_TEX:
            return (f"{turb}-{trans}"
                    rf"_M{meta['mach']:.2f}"
                    rf"_Re{int(meta['reynolds']/1e6)}\times10^6"
                    rf"_\alpha{meta['alpha']:.3f}^{{\\circ}}")
        else:
            return (f"{turb}-{trans}"
                    f"_M{meta['mach']:.2f}"
                    f"_Re{int(meta['reynolds']/1e6)}e6"
                    f"_α{meta['alpha']:.1f}°")

    if style == "metadata":
        return flow_metadata_text(meta).replace("\n", r"\\")  # rarely used

    if style == "auto":
        if n_cases > 0:
            return f"{turb}-{trans}"
        return flow_metadata_text(meta, one_line=True)

    if style == "sense":
        grid_lvl = f"$_{{L_{n_cases-1}}}$" 
        return rf"{turb}-{trans}{grid_lvl}"
    
    else:
        print('ERROR')
        
    return case_name


# ------------------------------------------------------------------
# Flow-condition box
# ------------------------------------------------------------------
def flow_metadata_text(meta: dict, *, one_line: bool = False) -> str:
    """Return Re, M, α (and Tu if present) formatted for either TeX or Unicode.

    one_line=True → comma-separated single line (for legend ‘auto’ case)
    """
    if USE_TEX:
        parts = [
            rf"$\mathrm{{Re}}_\infty = {int(meta['reynolds']/1e6)}\times10^{6}$",
            rf"$\mathrm{{M}}_\infty = {meta['mach']:.2f}$",
            rf"$\alpha = {meta['alpha']:.3f}^{{\circ}}$",
        ]
        if meta.get("tu") is not None:
            parts.insert(1, rf"$\mathrm{{Tu}}_\infty = {meta['tu']:.3f}\%$")
    else:
        parts = [
            rf"Re = {int(meta['reynolds']/1e6)}e6",
            rf"M  = {meta['mach']:.2f}",
            rf"α  = {meta['alpha']:.3f}°",
        ]
        if meta.get("tu") is not None:
            parts.insert(0, rf"Tu∞ = {meta['tu']*100:.3f}%")

    return ", ".join(parts) if one_line else "\n".join(parts)


# # ----------------------------------------------------------------------
# # Legend helpers
# # ----------------------------------------------------------------------
# def legend_label(meta: dict, style: str) -> str:
#     """Return the text that should appear next to each curve."""
#     turb  = meta.get("turbulence_model", "???")
#     trans = meta.get("transition_model", "???")

#     if style == "short":                  # e.g. compare-surface default
#         return f"{turb}-{trans}"

#     if style == "full":
#         return (f"{turb}-{trans}"
#                 f"_M{meta['mach']:.2f}"
#                 f"_Re{int(meta['reynolds']/1e6)}e6"
#                 f"_AoA{meta['alpha']:.1f}")

#     # fallback
#     return turb

# def flow_metadata_text(meta: dict) -> str:
#     """Compact block used in the annotation box."""
#     txt  = rf"Re = {int(meta['reynolds']/1e6)}e6" + "\n"
#     txt += rf"M  = {meta['mach']:.2f}"            + "\n"
#     txt += rf"α  = {meta['alpha']:.1f}°"
#     if "tu" in meta:
#         txt = rf"Tu₈ = {meta['tu']*100:.2f}%\n" + txt
#     return txt
