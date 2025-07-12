"""
Light-weight 1-D smoothing helpers shared by BL / Cp / Cf utilities
-------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal
import numpy as np
from scipy.signal import savgol_filter

__all__ = [
    "savgol_1d_safe",
    "moving_average_1d",
]

# ------------------------------------------------------------------
# Savitzky–Golay with graceful fallback
# ------------------------------------------------------------------


def savgol_1d_safe(arr,
                   *,
                   window_length: int = 5,
                   polyorder: int = 2):
    """
    Apply a Savitzky–Golay filter *only* when there are enough finite samples.

    If not, the input array is returned unchanged.
    """
    arr = np.asarray(arr, dtype=float)
    if np.count_nonzero(np.isfinite(arr)) < window_length:
        return arr
    # window_length must be odd
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(arr, window_length, polyorder, mode="interp")


# ------------------------------------------------------------------
# NaN-aware centred moving-average
# ------------------------------------------------------------------


def moving_average_1d(arr: np.ndarray,
                      window: int = 5,
                      *,
                      nan_policy: Literal["interp", "ignore"] = "interp"
                      ) -> np.ndarray:
    """
    Centred moving average with two flavours of NaN handling.

    * **interp**  – interpolate NaNs first, then average (default)  
    * **ignore**  – compute average of *valid* samples inside each window
                   (equivalent to ``np.nanmean``); returns NaN where the
                   entire window is NaN.
    """
    arr = np.asarray(arr, dtype=float)
    n   = arr.size
    if window < 3 or window > n:
        return arr.copy()

    if nan_policy not in {"interp", "ignore"}:
        raise ValueError("nan_policy must be 'interp' or 'ignore'")

    # ----------------------------------------------------------------------
    # preprocess NaNs
    # ----------------------------------------------------------------------
    if np.isnan(arr).any():
        if nan_policy == "interp":
            valid = ~np.isnan(arr)
            if not valid.any():
                return arr.copy()
            arr = np.interp(np.arange(n), np.flatnonzero(valid), arr[valid])
        # else → handled later

    pad     = window // 2
    kernel  = np.ones(window)

    if nan_policy == "ignore":
        valid        = ~np.isnan(arr)
        padded_vals  = np.pad(np.where(valid, arr, 0.0), pad, mode="edge")
        padded_mask  = np.pad(valid.astype(float), pad, mode="edge")
        num          = np.convolve(padded_vals, kernel, mode="valid")
        den          = np.convolve(padded_mask, kernel, mode="valid")
        out          = num / den
        out[den == 0.0] = np.nan
        return out

    # standard average (all NaNs already removed/interpolated)
    padded = np.pad(arr, pad, mode="edge")
    return np.convolve(padded, kernel / window, mode="valid")
