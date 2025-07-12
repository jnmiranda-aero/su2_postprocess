#path_helpers.py
from pathlib import Path
import os


def get_comparison_save_path(paths: list[Path], mode: str = "A") -> Path:
    """
    Determine the directory to save comparison plots based on a save mode.

    Modes:
        A: Deepest common ancestor + "_compare"         [default]
        B: Current working directory + "_compare"
        C: User home + "su2_postprocess_compare"
        D: Parent of the first path + "_compare"

    Args:
        paths: List of case paths (Path objects)
        mode: Save mode ('A', 'B', 'C', 'D')

    Returns:
        A Path object pointing to the desired save directory.
    """
    if not paths:
        raise ValueError("No paths provided for comparison output.")

    resolved = [p.resolve() for p in paths]
    mode = mode.upper()

    if mode == "A":
        common_prefix = Path(os.path.commonpath([str(p) for p in resolved]))
        return common_prefix.parent / (common_prefix.name + "_compare")
    elif mode == "B":
        return Path.cwd() / "_compare"
    elif mode == "C":
        return Path.home() / "su2_postprocess_compare"
    elif mode == "D":
        return resolved[0].parent / "_compare"
    else:
        raise ValueError(f"Unknown save mode '{mode}'. Use A, B, C, or D.")

# from pathlib import Path
# import os

# def get_comparison_save_path(paths: list[Path]) -> Path:
#     """
#     Given a list of paths, return a directory path to save comparison plots.
#     It uses the deepest common ancestor and appends '_compare' to it.

#     Example:
#         Input:  [/sim/A/Mach0.7/Case1, /sim/A/Mach0.7/Case2]
#         Output: /sim/A/Mach0.7_compare
#     """
#     if not paths:
#         raise ValueError("No paths provided for comparison output.")

#     resolved = [p.resolve() for p in paths]
#     common_prefix = Path(os.path.commonpath([str(p) for p in resolved]))
#     return common_prefix.parent / (common_prefix.name + "_compare")
