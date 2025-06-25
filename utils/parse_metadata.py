import re
import numpy as np
from pathlib import Path
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# Aliases
TURB_ALIASES = {
    "spalart-allmaras-noft2": r"SA-noft2",
    "spalart allmaras": r"SA",
    "menter's sst": r"$k\!$ - $\!\omega$ $\mathrm{SST}\!$ - $\!\mathrm{v2003m}$",
    "menter's k-omega sst-2003m": r"$k\!$ - $\!\omega$ $\mathrm{SST}\!$ - $\!\mathrm{v2003m}$",
    "k-omega": "k-w",
    "k-epsilon": "k-e",
    "none": "None",
    "unknown": "Unknown",
}

TRANS_ALIASES = {
    "langtry and menter's 4 equation model (2009)": {
        "standard": r"$\gamma\!$ - $\!Re_{\theta}\!$ - $\!\mathrm{LM2009}$",
        "cc": r"$\gamma\!$ - $\!Re_{\theta}\!$ - $\!\mathrm{LM2009}_{\mathit{cc}}$",
    },
    "langtry and menter's transition (2009)": r"$\gamma\!$ - $\!Re_{\theta}\!$ - $\!\mathrm{LM2009}$",
    "langtry and menter": r"$\gamma\!$ - $\!Re_{\theta}\!$ - $\!\mathrm{LM2009}$",
    "medida and baeder": "MB",
    "medida-baeder": "MB",
    "malan et al.": "Malan 2009",
    "none": "None",
}

CORRELATION_ALIASES = {
    "menter and langtry": " (Langtry-Menter Correlation)",
    "medida and baeder": " (Medida-Baeder Correlation)",
    "malan et al.": " (Malan Correlation)",
}


def apply_alias(model_str, alias_map):
    if not model_str:
        return "Unknown"
    normalized = " ".join(model_str.strip().lower().split())
    for key, alias in alias_map.items():
        if key in normalized:
            return alias
    return model_str.strip()


def get_transition_label(base_model, compressibility=False):
    normalized = " ".join(base_model.strip().lower().split())
    match = TRANS_ALIASES.get(normalized)
    if isinstance(match, dict):
        return match["cc"] if compressibility else match["standard"]
    for key, alias in TRANS_ALIASES.items():
        if key in normalized:
            return alias["cc"] if compressibility and isinstance(alias, dict) else (
                alias["standard"] if isinstance(alias, dict) else alias)
    return base_model.strip()

def flow_metadata_text(meta: dict, fields: list[str] = None) -> str:
    """Return multi-line TeX-formatted string of metadata. Fields is a whitelist."""
    if fields is None:
        fields = ["turbulence_model", "transition_model", "correlation_model", "reynolds", "tu", "mach", "alpha"]

    lines = []
    if "turbulence_model" in fields and meta.get("turbulence_model"):
        lines.append(meta["turbulence_model"])
    if "transition_model" in fields and meta.get("transition_model"):
        lines.append(meta["transition_model"])
    if "correlation_model" in fields and meta.get("correlation_model"):
        lines.append(meta["correlation_model"])
    if "reynolds" in fields:
        lines.append(
            rf"$\mathrm{{Re}}_\infty = {meta['reynolds']/1e6:.1f}\!\!\times\!\!10^6$"
            if meta.get("reynolds") else "Re unavailable")
    if "tu" in fields:
        lines.append(
            rf"$\mathrm{{Tu}}_\%$ = {meta['tu']:.3f}\%" if meta.get("tu") is not None else "Tu unavailable")
    if "mach" in fields:
        lines.append(
            rf"$\mathrm{{M}}_\infty = {meta['mach']:.2f}$" if meta.get("mach") is not None else "Mach unavailable")
    if "alpha" in fields:
        lines.append(
            rf"$\alpha = {meta['alpha']:.3f}^{{\circ}}$" if meta.get("alpha") is not None else "AoA unavailable")

    return "\n".join(lines)


def extract_case_metadata_from_log(log_file, minimal=False, return_metadata=False, fields=None):
    try:
        turb_model, trans_model, corr_model = "Unknown", "None", None
        mach, re_val, aoa, fsti_p = None, None, None, None
        gamma_val = None                  
        compressibility_flag = False

        with open(log_file, "r") as f:
            lines = list(f)
            for i, line in enumerate(lines):
                line_l = line.lower()

                if "turbulence model:" in line_l:
                    turb_model = line.split(":", 1)[1].strip()
                elif "transition model:" in line_l:
                    trans_model = line.split(":", 1)[1].strip()
                    for j in range(1, 4):
                        if i + j < len(lines) and "compressibility" in lines[i + j].lower():
                            compressibility_flag = True
                            break
                elif "correlation functions:" in line_l:
                    corr_model = line.split(":", 1)[1].strip()
                elif "mach number:" in line_l:
                    m = re.search(r"mach number:\s*([\d.]+)", line, re.I)
                    if m:
                        mach = float(m.group(1).rstrip('.'))
                elif "angle of attack" in line_l:
                    m = re.search(r"aoa\):\s*([-\d.]+)", line, re.I)
                    if m:
                        aoa = float(m.group(1).rstrip('.'))
                elif "reynolds number:" in line_l:
                    m = re.search(r"reynolds number:\s*([\deE.+-]+)", line, re.I)
                    if m:
                        re_val = float(m.group(1).rstrip('.'))
                elif "turb. kin. energy" in line_l and "non-dim" not in line_l:
                    try:
                        parts = [p.strip() for p in line.strip().split("|") if p.strip()]
                        if len(parts) >= 5:
                            val = float(parts[-1])
                            fsti_p = 100 * np.sqrt(2 / 3 * val)
                    except Exception:
                        pass
                elif "ratio of specific heats" in line_l or "specific heat ratio" in line_l:
                    m = re.search(r"([0-9.]+)", line)
                    if m:
                        gamma_val = float(m.group(1))

        short_turb = apply_alias(turb_model, TURB_ALIASES)
        short_trans = get_transition_label(trans_model, compressibility_flag)
        short_corr = apply_alias(corr_model, CORRELATION_ALIASES) if corr_model else None

        meta = {
            "mach": mach,
            "reynolds": re_val,
            "alpha": aoa,
            "tu": fsti_p,
            "gamma": gamma_val,  
            "transition_model": short_trans,
            "turbulence_model": short_turb,
            "compressibility": compressibility_flag,
            "correlation_model": short_corr,
        }

        if minimal:
            return Path(log_file).parent.name
        elif return_metadata:
            return flow_metadata_text(meta, fields), meta
        else:
            return flow_metadata_text(meta, fields)

    except Exception as e:
        print(f"[ERROR] Could not parse log: {log_file} → {e}")
        if return_metadata:
            return "Unknown", {}
        return "Unknown"

def extract_case_metadata_fallback(forces_file, minimal=False, return_metadata=False, fields=None):
    forces_path = Path(forces_file)
    parent_dir = forces_path.parent

    if not forces_path.name.startswith("forces_bdwn_"):
        print(f"[ERROR] Invalid forces file: {forces_path.name}. Must start with 'forces_bdwn_'.")
        label = "Metadata unavailable"
        meta = {
            "mach": None,
            "reynolds": None,
            "alpha": None,
            "tu": None,
            "transition_model": None,
            "turbulence_model": None,
            "compressibility": None,
            "correlation_model": None,
        }
        return (label, meta) if return_metadata else label

    log_files = list(parent_dir.glob("log_*.log"))
    if log_files:
        log_file = max(log_files, key=lambda p: p.stat().st_mtime)
        print(f"[INFO] Using metadata from: {log_file}")
        return extract_case_metadata_from_log(
            log_file, minimal=minimal, return_metadata=return_metadata, fields=fields
        )

    print(f"[WARN] No log_*.log file found in {parent_dir}")
    return extract_case_metadata_from_forces(forces_file, minimal=minimal, return_metadata=return_metadata, fields=fields)


def extract_case_metadata_from_forces(forces_file, minimal=False, return_metadata=False, fields=None):
    try:
        turb_model  = "Unknown"
        trans_model = "None"
        mach        = "?"
        re_val      = "?"
        aoa         = "?"
        fsti        = "?"
        fsti_p      = "?"

        with open(forces_file, "r") as f:
            for line in f:
                if "Turbulence model:" in line:
                    turb_model = line.split(":", 1)[1].strip()
                elif "Transition model:" in line:
                    trans_model = line.split(":", 1)[1].strip()
                elif "Mach number" in line:
                    m = re.search(r"Mach number:\s*([\d.]+)", line)
                    if m:
                        mach = m.group(1).rstrip(".")
                elif "Angle of attack" in line:
                    m = re.search(r"AoA\):\s*([-\d.]+)", line)
                    if m:
                        aoa = m.group(1)
                elif "Reynolds number" in line:
                    m = re.search(r"Reynolds number:\s*([\deE.+-]+)", line)
                    if m:
                        cleaned = m.group(1).rstrip(".")
                        re_val = f"{float(cleaned):.1f}"
                elif "Free-stream turb. kinetic energy (non-dim):" in line:
                    m = re.search(r"Free-stream turb\. kinetic energy \(non-dim\):\s*([\deE.+-]+)", line)
                    if m:
                        try:
                            fsti = float(m.group(1).rstrip("."))
                            fsti_p = f"{100 * np.sqrt(2 / 3 * fsti):.3f}"
                        except ValueError:
                            fsti = "?"
                            fsti_p = "?"

        def apply_alias(model_str, alias_map):
            model_str = model_str.lower()
            for key, alias in alias_map.items():
                if key.lower() in model_str:
                    return alias
            return model_str.strip()

        short_turb = apply_alias(turb_model, TURB_ALIASES)
        short_trans = apply_alias(trans_model, TRANS_ALIASES)

        meta = {
            "mach": float(mach) if mach != "?" else None,
            "reynolds": float(re_val) * 1e6 if re_val != "?" else None,
            "alpha": float(aoa) if aoa != "?" else None,
            "tu": float(fsti_p) if fsti_p != "?" else None,
            "transition_model": short_trans,
            "turbulence_model": short_turb,
            "compressibility": False,
            "correlation_model": None,
        }

        if minimal:
            return Path(forces_file).parent.name
        elif return_metadata:
            return flow_metadata_text(meta, fields), meta
        else:
            return flow_metadata_text(meta, fields)

    except Exception as e:
        print("Error in extract_case_metadata_from_forces:", e)
        if return_metadata:
            return "Metadata unavailable", {
                "mach": None,
                "reynolds": None,
                "alpha": None,
                "tu": None,
                "transition_model": None,
                "turbulence_model": None,
                "compressibility": None,
                "correlation_model": None,
            }
        return "Metadata unavailable"





###################################################################################

# def extract_case_metadata_from_log(log_file, minimal=False, return_metadata=False):
#     try:
#         turb_model, trans_model, corr_model = "Unknown", "None", None
#         mach, re_val, aoa, fsti_p = None, None, None, None
#         compressibility_flag = False

#         with open(log_file, "r") as f:
#             lines = list(f)
#             for i, line in enumerate(lines):
#                 line_l = line.lower()

#                 if "turbulence model:" in line_l:
#                     turb_model = line.split(":", 1)[1].strip()
#                     print("\t\t Turbulence Model:", turb_model)

#                 elif "transition model:" in line_l:
#                     trans_model = line.split(":", 1)[1].strip()
#                     print("\t\t Transition Model:", trans_model)
#                     for j in range(1, 4):
#                         if i + j < len(lines) and "compressibility" in lines[i + j].strip().lower():
#                             compressibility_flag = True
#                             print("\t\t Compressibility Flag for Transition Model:", compressibility_flag)
#                             break

#                 elif "correlation functions:" in line_l:
#                     corr_model = line.split(":", 1)[1].strip()
#                     print("\t\t Correlation Function for Transition Model:", corr_model)

#                 elif "mach number:" in line_l:
#                     m = re.search(r"mach number:\s*([\d.]+)", line, re.I)
#                     if m:
#                         mach = float(m.group(1).rstrip('.'))
#                         print("\t\t M:", mach)

#                 elif "angle of attack" in line_l:
#                     m = re.search(r"aoa\):\s*([-\d.]+)", line, re.I)
#                     if m:
#                         aoa = float(m.group(1).rstrip('.'))
#                         print("\t\t AoA:", aoa)

#                 elif "reynolds number:" in line_l:
#                     m = re.search(r"reynolds number:\s*([\deE.+-]+)", line, re.I)
#                     if m:
#                         re_val = float(m.group(1).rstrip('.'))
#                         print("\t\t Re:", re_val)

#                 elif "turb. kin. energy" in line_l and "non-dim" not in line_l:
#                     try:
#                         parts = [p.strip() for p in line.strip().split("|") if p.strip()]
#                         if len(parts) >= 5:
#                             val = float(parts[-1])
#                             fsti_p = 100 * np.sqrt(2 / 3 * val)
#                             print("\t\t Tu%:", fsti_p)
#                     except Exception:
#                         pass

#         short_turb = apply_alias(turb_model, TURB_ALIASES)
#         short_trans = get_transition_label(trans_model, compressibility_flag)
#         short_corr = apply_alias(corr_model, CORRELATION_ALIASES) if corr_model else None

#         # if short_corr:
#             # label_lines.append(rf"Corr: {short_corr}")
#         # ----------------------------------------------------------
#         # Build multi-line label
#         # ----------------------------------------------------------
#         label_lines = [short_turb, short_trans]
#         if short_corr:
#             # keep it small; TeX or Unicode handled by apply_alias()
#             label_lines.append(rf"{short_corr}")            
#         label_lines.append(
#             rf"$\mathrm{{Re}}_\infty = {re_val/1e6:.1f}\!\!\times\!\!10^6$" if re_val else "Re unavailable")
#         label_lines.append(
#             rf"$\mathrm{{Tu}}_\%$ = {fsti_p:.3f}\%" if fsti_p is not None else "Tu unavailable")
#         label_lines.append(
#             rf"$\mathrm{{M}}_\infty = {mach:.2f}$" if mach is not None else "Mach unavailable")
#         label_lines.append(
#             rf"$\alpha = {aoa:.1f}^{{\circ}}$" if aoa is not None else "AoA unavailable")

#         label = "\n".join(label_lines)

#         if minimal:
#             return Path(log_file).parent.name
#         elif return_metadata:
#             return label, {
#                 "mach": mach,
#                 "reynolds": re_val,
#                 "alpha": aoa,
#                 "tu": fsti_p,
#                 "transition_model": short_trans,
#                 "turbulence_model": short_turb,
#                 "compressibility": compressibility_flag,
#                 "correlation_model": short_corr,
#             }
#         else:
#             return label


#     except Exception as e:
#         print(f"[ERROR] Failed to parse {log_file}: {e}")
#         if return_metadata:
#             return "Metadata unavailable", {
#                 "mach": None,
#                 "reynolds": None,
#                 "alpha": None,
#                 "tu":None,
#                 "transition_model": None,
#                 "turbulence_model": None,
#                 "compressibility": None,
#                 "correlation_model": None,
#             }
#         return "Metadata unavailable"


# def extract_case_metadata_fallback(forces_file, minimal=False, return_metadata=False):
#     forces_path = Path(forces_file)
#     parent_dir = forces_path.parent

#     # Confirm it's a valid forces file
#     if not forces_path.name.startswith("forces_bdwn_"):
#         print(f"[ERROR] Invalid forces file: {forces_path.name}. Must start with 'forces_bdwn_'.")
#         if return_metadata:
#             return "Metadata unavailable", {
#                 "mach": None,
#                 "reynolds": None,
#                 "alpha": None,
#                 "tu": None,
#                 "transition_model": None,
#                 "turbulence_model": None,
#                 "compressibility": None,
#                 "correlation_model": None,
#             }
#         return "Metadata unavailable"

#     # Search for any matching log_*.log file

#     # look for any log_*.log in the case directory
#     log_files = list(parent_dir.glob("log_*.log"))
#     if log_files:
#         # pick the most recently modified log file
#         log_file = max(log_files, key=lambda p: p.stat().st_mtime)
#         print(f"[INFO] Using metadata from: {log_file}")
#         return extract_case_metadata_from_log(
#             log_file, minimal=minimal, return_metadata=return_metadata
#         )

#     print(f"[WARN] No log_*.log file found in {parent_dir}")
#     if return_metadata:
#         return extract_case_metadata_from_forces(forces_file, minimal=minimal, return_metadata=True)
#     return extract_case_metadata_from_forces(forces_file, minimal=minimal)


# def extract_case_metadata_from_forces(forces_file, minimal=False, return_metadata=False):
#     try:
#         turb_model  = "Unknown"
#         trans_model = "None"
#         mach        = "?"
#         re_val      = "?"
#         aoa         = "?"
#         fsti        = "?"
#         fsti_p      = "?"

#         with open(forces_file, "r") as f:
#             for line in f:
#                 if "Turbulence model:" in line:
#                     turb_model = line.split(":", 1)[1].strip()
#                 elif "Transition model:" in line:
#                     trans_model = line.split(":", 1)[1].strip()
#                 elif "Mach number" in line:
#                     m = re.search(r"Mach number:\s*([\d.]+)", line)
#                     if m:
#                         mach = m.group(1).rstrip(".")
#                 elif "Angle of attack" in line:
#                     m = re.search(r"AoA\):\s*([-\d.]+)", line)
#                     if m:
#                         aoa = m.group(1)
#                 elif "Reynolds number" in line:
#                     m = re.search(r"Reynolds number:\s*([\deE.+-]+)", line)
#                     if m:
#                         cleaned = m.group(1).rstrip(".")
#                         re_val_raw = float(cleaned)
#                         re_val = f"{re_val_raw / 1e6:.1f}"
#                 elif "Free-stream turb. kinetic energy (non-dim):" in line:
#                     m = re.search(r"Free-stream turb\. kinetic energy \(non-dim\):\s*([\deE.+-]+)", line)
#                     if m:
#                         try:
#                             fsti = float(m.group(1).rstrip("."))
#                             fsti_p = f"{100 * np.sqrt(2 / 3 * fsti):.3f}"
#                         except ValueError:
#                             fsti = "?"
#                             fsti_p = "?"

#         def apply_alias(model_str, alias_map):
#             model_str = model_str.lower()
#             for key, alias in alias_map.items():
#                 if key.lower() in model_str:
#                     return alias
#             return model_str.strip()

#         short_turb = apply_alias(turb_model, TURB_ALIASES)
#         short_trans = apply_alias(trans_model, TRANS_ALIASES)

#         label_lines = [
#             rf"{short_turb}",
#             rf"{short_trans}",
#             rf"$\mathrm{{Re}}_\infty = {re_val}\!\!\times\!\!10^6$",
#             rf"$\mathrm{{Tu}}_\%$ = {fsti_p}\%",
#             rf"$\mathrm{{M}}_\infty = {mach}$",
#             rf"$\alpha = {aoa}^{{\circ}}$",
#         ]

#         label = "\n".join(label_lines)

#         if minimal:
#             try:
#                 return Path(forces_file).parent.name
#             except Exception:
#                 return "Unknown"

#         if return_metadata:
#             try:
#                 metadata = {
#                     "mach": float(mach),
#                     "reynolds": float(re_val) * 1e6,
#                     "alpha": float(aoa),
#                     "tu": float(fsti_p),
#                 }
#             except Exception:
#                 metadata = {
#                     "mach": None,
#                     "reynolds": None,
#                     "alpha": None,
#                     "tu": None
#                 }
#             return label, metadata

#         return label

#     except Exception as e:
#         print("Error in extract_case_metadata:", e)
#         return "Metadata unavailable"



# import re
# import numpy as np
# import matplotlib as mpl
# from pathlib import Path
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['font.family'] = 'serif'  # optional: can set to 'lmodern' or 'cm'


# TURB_ALIASES = {
#     "Spalart Allmaras": "SA-noft2",
#     # "Menter's SST": r"k-\omega SST-v2003m",
#     "Menter's SST": r"$k\!-\!\omega \mathrm{SST\!-\!v2003m}$",
#     "k-omega": "k-w",
#     "k-epsilon": "k-e",
#     "NONE": "None"
# }

# TRANS_ALIASES = {
#     # "Langtry and Menter's transition (2009)": r"$\gamma\text{-Re}_{\theta}\text{-}\text{LM2009}$",
#     "Langtry and Menter's transition (2009)": r"$\gamma\!-\!Re_{\theta}\!-\!\mathrm{LM2009}$",
#     "Medida-Baeder": "MB",
#     "None": "None"
# }


# def extract_case_metadata(forces_file, minimal=False):
#     if minimal:
#         try:
#             return Path(forces_file).parent.name
#         except Exception as e:
#             print(f"[WARN] Could not extract minimal label: {e}")
#             return "Unknown"
    
#     try:
#         turb_model = "Unknown"
#         trans_model = "None"
#         mach = "?"
#         re_val = "?"
#         aoa = "?"
#         fsti = "?"

#         with open(forces_file, "r") as f:
#             for line in f:
#                 if "Turbulence model:" in line:
#                     turb_model = line.split(":", 1)[1].strip()
#                 elif "Transition model:" in line:
#                     trans_model = line.split(":", 1)[1].strip()
#                 elif "Mach number" in line:
#                     m = re.search(r"Mach number:\s*([\d.]+)", line)
#                     if m:
#                         mach = m.group(1).rstrip(".")
#                 elif "Angle of attack" in line:
#                     m = re.search(r"AoA\):\s*([-\d.]+)", line)
#                     if m:
#                         aoa = m.group(1)
#                 elif "Reynolds number" in line:
#                     m = re.search(r"Reynolds number:\s*([\deE.+-]+)", line)
#                     if m:
#                         cleaned = m.group(1).rstrip(".")
#                         re_val_raw = float(cleaned)
#                         re_val = f"{re_val_raw / 1e6:.1f}"
#                 elif "Free-stream turb. kinetic energy (non-dim):" in line:
#                     m = re.search(r"Free-stream turb\. kinetic energy \(non-dim\):\s*([\deE.+-]+)", line)
#                     if m:
#                         try:
#                             fsti   = float(m.group(1).rstrip("."))
#                             fsti_p = f"{100*np.sqrt(2 / 3 * fsti)}"
#                         except ValueError:
#                             fsti = "?"

#         def apply_alias(model_str, alias_map):
#             model_str = model_str.lower()
#             for key, alias in alias_map.items():
#                 if key.lower() in model_str:
#                     return alias
#             return model_str.strip()

#         short_turb = apply_alias(turb_model, TURB_ALIASES)
#         short_trans = apply_alias(trans_model, TRANS_ALIASES)

#         # label_lines = [
#         #     rf"{short_turb} "rf"{short_trans}",
#         #     rf"Tu$_\%$ = {fsti_p}%",
#         #     r"$\text{Re}_\infty $= " rf"${re_val}$",
#         #     r"$\text{M}_\infty $= " rf"${mach}$",
#         #     rf"$\alpha = {aoa}^{{\circ}}$",
#         # ]
#         label_lines = [
#             rf"{short_turb}",
#             rf"{short_trans}",
#             rf"$\mathrm{{Re}}_\infty = {re_val}\!\!\times\!\!10^6$",
#             rf"$\mathrm{{Tu}}_\%$ = {fsti_p}\%",
#             rf"$\mathrm{{M}}_\infty = {mach}$",
#             rf"$\alpha = {aoa}^{{\circ}}$",
#         ]

#         label = "\n".join(label_lines)

#         return label

#     except Exception as e:
#         print("Error in extract_case_metadata:", e)
#         return "Metadata unavailable"


