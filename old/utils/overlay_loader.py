# su2_postprocess/utils/overlay_loader.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------------
# Core reader
# ---------------------------------------------------------------------------
def _read_dat(path: Path) -> pd.DataFrame:
    """
    Read any whitespace‐delimited file, skip comment lines,
    take col0 as x, colN as y, coerce both to numeric,
    drop rows with NaNs, keep only 0<=x<=1, preserve input order.
    """
    try:
        # 1) Read everything, no usecols
        df = pd.read_csv(
            path,
            sep=r"\s+",
            comment="#",
            header=None,
            engine="python",
        )

        # 2) Pick first & last columns
        df = df.loc[:, [0, df.columns[-1]]]
        df.columns = ["x", "y"]

        # 3) Coerce to floats (bad entries → NaN) and drop them
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna(subset=["x", "y"])

        # 4) Filter to 0<=x<=1
        df = df[(df["x"] >= 0.0) & (df["x"] <= 1.0)]

        return df.reset_index(drop=True)

    except Exception as exc:
        print(f"[WARN] overlay read failed for {path.name}: {exc}")
        return pd.DataFrame(columns=["x", "y"])


# ---------------------------------------------------------------------------
# Public helpers (back-compat)
# ---------------------------------------------------------------------------
def load_overlay_file(path: Path) -> pd.DataFrame:
    stem = path.stem.lower()
    # if it's a CP file with three columns, cp is in column 2
    if stem.startswith("cp_") and (len(path.read_text().splitlines()[1].split()) > 2):
        return _read_dat(path, ycol=2)
    # if it's a CF file with five columns, cf is in column 4
    if stem.startswith("cf_") and (len(path.read_text().splitlines()[1].split()) > 2):
        return _read_dat(path, ycol=4)
    # fallback to first two columns
    return _read_dat(path, ycol=1)


def load_overlay_dir_by_prefix(directory: Path):
    """
    Scan *directory* for files whose stem starts with  **cp_**  or  **cf_**
    (case-insensitive).  Accepts .dat **and** .txt.  Returns two dicts
    (cp_dict, cf_dict), each keyed by lower-case stem.
    """
    cp_dict, cf_dict = {}, {}
    if not (directory and directory.is_dir()):
        print(f"[WARN] overlay dir not found: {directory}")
        return cp_dict, cf_dict

    for ext in ("*.dat", "*.txt"):
        for f in directory.glob(ext):
            stem = f.stem.lower()
            if stem.startswith("cp_"):
                df = _read_dat(f)
                if not df.empty:
                    cp_dict[stem] = df
            elif stem.startswith("cf_"):
                df = _read_dat(f)
                if not df.empty:
                    cf_dict[stem] = df
    return cp_dict, cf_dict


__all__ = ["load_overlay_file", "load_overlay_dir_by_prefix"]



# # su2_postprocess/utils/overlay_loader.py

# from pathlib import Path
# import pandas as pd

# from pathlib import Path
# import pandas as pd

# def load_overlay_file(path: Path) -> pd.DataFrame:
#     """
#     Read the first two numeric columns of any .dat file, skip comments,
#     and return a DataFrame with columns ['x','y'], filtered to 0<=x<=1
#     and sorted by x.
#     """
#     try:
#         # read raw
#         df = pd.read_csv(
#             path,
#             sep=r'\s+',
#             comment='#',
#             header=None,
#             usecols=[0, 1],
#             engine='python'
#         )
#         df.columns = ['x', 'y']

#         # drop any rows with NaNs
#         df = df.dropna(subset=['x','y'])

#         # keep only 0 <= x <= 1
#         df = df[(df['x'] >= 0.0) & (df['x'] <= 1.0)]

#         # sort by x
#         df = df.sort_values('x').reset_index(drop=True)

#         return df

#     except Exception as e:
#         print(f"[WARN] Failed to load overlay {path.name}: {e}")
#         return pd.DataFrame(columns=['x', 'y'])



# def load_overlay_dir_by_prefix(directory: Path):
#     """
#     Scan `directory` for *.dat files whose stem *starts* with 'cp_' or 'cf_' (case-insensitive).
#     Returns two dicts (cp_dict, cf_dict), keyed by lower-case file-stem.
#     """
#     cp_dict, cf_dict = {}, {}
#     if not directory or not directory.exists():
#         print("Does not exist")
#         return cp_dict, cf_dict

#     for file in sorted(directory.glob("*.dat")):
#         stem = file.stem.lower()
#         if stem.startswith("cp_"):
#             print("cp")
#             df = load_overlay_file(file)
#             if not df.empty:
#                 cp_dict[stem] = df
#         elif stem.startswith("cf_"):
#             print("cf")
#             df = load_overlay_file(file)
#             if not df.empty:
#                 cf_dict[stem] = df
#         # anything else → ignore

#     return cp_dict, cf_dict


# # --- overlay_loader.py ---
# from pathlib import Path
# import pandas as pd
# import re

# def load_overlay_file(path: Path) -> pd.DataFrame:
#     try:
#         df = pd.read_csv(path, delim_whitespace=True, comment="#")
#         df = df[df.apply(lambda row: all(pd.api.types.is_number(x) for x in row), axis=1)]
#         df = df.sort_values(by=df.columns[0])  # assumes x is first col
#         return df.reset_index(drop=True)
#     except Exception:
#         return pd.DataFrame()

# def load_overlay_dir_by_prefix(directory: Path):
#     if not directory or not directory.exists():
#         return {}

#     cp_files = {}
#     cf_files = {}

#     for file in directory.glob("*.dat"):
#         name = file.stem
#         df = load_overlay_file(file)
#         if df.empty:
#             continue

#         if "cp" in name.lower():
#             cp_files[name] = df
#         elif "cf" in name.lower():
#             cf_files[name] = df
#         else:
#             cp_files[name] = df  # fallback

#     return cp_files, cf_files



# import re
# import pandas as pd
# from pathlib import Path
# from typing import Dict, Tuple, Optional, Union

# def load_overlay_file(file_path: Union[str, Path]) -> pd.DataFrame:
#     df = pd.read_csv(file_path, sep=r"\s+", comment="#", engine="python", header=None)
#     first_row = df.iloc[0].astype(str)
#     if not all(first_row.apply(lambda x: x.replace('.', '', 1).replace('-', '', 1).isdigit())):
#         df = pd.read_csv(file_path, sep=r"\s+", comment="#", engine="python")
#     df.columns = [str(col).strip().lstrip("#") for col in df.columns]
#     return df

# def extract_label_from_filename(name: str) -> Optional[str]:
#     pattern = re.compile(r"[cpf]{2}_M(?P<mach>[\d.]+)_Re(?P<re>\d+e\d+)_AoA(?P<aoa>-?\d+)")
#     match = pattern.search(name)
#     if match:
#         mach = float(match.group("mach"))
#         re_val = match.group("re")
#         aoa = int(match.group("aoa"))
#         return f"M{mach:.2f}_Re{re_val}_AoA{aoa}"
#     return None

# def load_overlay_dir_by_prefix(directory: Union[str, Path]) -> Optional[Union[Dict[str, pd.DataFrame], Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]]]:
#     directory = Path(directory)
#     if not directory.exists():
#         return None

#     cp_files = list(directory.glob("cp_*.dat")) + list(directory.glob("cp_*.txt"))
#     cf_files = list(directory.glob("cf_*.dat")) + list(directory.glob("cf_*.txt"))

#     cp_dict = {}
#     cf_dict = {}

#     def insert_with_unique_label(d: Dict[str, pd.DataFrame], label: str, df: pd.DataFrame):
#         if label not in d:
#             d[label] = df
#         else:
#             i = 2
#             new_label = f"{label} ({i})"
#             while new_label in d:
#                 i += 1
#                 new_label = f"{label} ({i})"
#             d[new_label] = df

#     for f in cp_files:
#         label = extract_label_from_filename(f.name)
#         if label:
#             try:
#                 insert_with_unique_label(cp_dict, label, load_overlay_file(f))
#             except Exception:
#                 continue

#     for f in cf_files:
#         label = extract_label_from_filename(f.name)
#         if label:
#             try:
#                 insert_with_unique_label(cf_dict, label, load_overlay_file(f))
#             except Exception:
#                 continue

#     if cp_dict and not cf_dict:
#         return cp_dict
#     elif cf_dict and not cp_dict:
#         return cf_dict
#     elif cp_dict and cf_dict:
#         return cp_dict, cf_dict

#     return None

# def find_xfoil_by_metadata(directory: Path, label_str: str) -> Optional[Dict[str, pd.DataFrame]]:
#     if not directory.exists():
#         return None
#     matches = list(directory.glob(f"cp*{label_str}*.dat")) + list(directory.glob(f"cp*{label_str}*.txt"))
#     if not matches:
#         return None
#     overlays = {}
#     for idx, match in enumerate(matches, start=1):
#         label = f"{label_str}" if idx == 1 else f"{label_str} ({idx})"
#         try:
#             overlays[label] = load_overlay_file(match)
#         except Exception:
#             continue
#     return overlays if overlays else None

# import re
# import pandas as pd
# from pathlib import Path
# from typing import Dict, Tuple, Optional, Union

# def load_overlay_file(file_path: Union[str, Path]) -> pd.DataFrame:
#     df_raw = pd.read_csv(file_path, sep=r"\s+", comment="#", engine="python", header=None)
    
#     # Check if first row contains strings (header) or numbers
#     first_row = df_raw.iloc[0].astype(str)
#     is_header = not all(first_row.apply(lambda x: x.replace('.', '', 1).replace('-', '', 1).isdigit()))
    
#     if is_header:
#         df = pd.read_csv(file_path, sep=r"\s+", comment="#", engine="python")
#         df.columns = [col.strip().lstrip("#") for col in df.columns]
#     else:
#         # Assign default column names
#         if df_raw.shape[1] == 2:
#             df_raw.columns = ["x", "Cp"]
#         elif df_raw.shape[1] >= 3:
#             df_raw.columns = ["x", "Cp", "cf"][:df_raw.shape[1]]
#         else:
#             df_raw.columns = [str(i) for i in range(df_raw.shape[1])]
#         df = df_raw

#     return df



# def extract_label_from_filename(name: str) -> Optional[str]:
#     pattern = re.compile(r"[cpf]{2}_M(?P<mach>[\d.]+)_Re(?P<re>\d+e\d+)_AoA(?P<aoa>-?\d+)")
#     match = pattern.search(name)
#     if match:
#         mach = float(match.group("mach"))
#         re_val = match.group("re")
#         aoa = int(match.group("aoa"))
#         return f"M{mach:.2f}_Re{re_val}_AoA{aoa}"
#     return None

# def load_overlay_dir_by_prefix(directory: Union[str, Path]) -> Optional[Union[Dict[str, pd.DataFrame], Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]]]:
#     directory = Path(directory)
#     if not directory.exists():
#         return None

#     cp_files = list(directory.glob("cp_*.dat")) + list(directory.glob("cp_*.txt"))
#     cf_files = list(directory.glob("cf_*.dat")) + list(directory.glob("cf_*.txt"))

#     cp_dict = {}
#     cf_dict = {}

#     for f in cp_files:
#         label = extract_label_from_filename(f.name)
#         if label:
#             cp_dict[label] = load_overlay_file(f)

#     for f in cf_files:
#         label = extract_label_from_filename(f.name)
#         if label:
#             cf_dict[label] = load_overlay_file(f)

#     if cp_dict and not cf_dict:
#         return cp_dict
#     elif cf_dict and not cp_dict:
#         return cf_dict
#     elif cp_dict and cf_dict:
#         return cp_dict, cf_dict

#     return None

# def find_xfoil_by_metadata(directory: Path, label_str: str) -> Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
#     if not directory.exists():
#         return None
#     matches = list(directory.glob(f"cp*{label_str}*.dat")) + list(directory.glob(f"cp*{label_str}*.txt"))
#     if not matches:
#         return None
#     try:
#         return load_overlay_file(matches[0])
#     except Exception:
#         return None
