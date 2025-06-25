from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Mapping, Optional, Any

@dataclass
class BLResult:
    # x‐coordinates along the surface
    x: Sequence[float]
    # boundary‐layer thickness δ, displacement δ*, momentum θ, shape factor H,
    # and edge Mach / velocity M_e, each keyed by the edge_method name
    delta:       Mapping[str, Sequence[float]]
    delta_star:  Mapping[str, Sequence[float]]
    theta:       Mapping[str, Sequence[float]]
    H:           Mapping[str, Sequence[float]]
    M_e:         Mapping[str, Sequence[float]]
    # optional LM‐Mach_e array if you compute it
    M_e_LM:      Optional[Sequence[float]]
    # velocity profiles at selected x_locs
    velocity_profiles: Mapping[float, Any]
