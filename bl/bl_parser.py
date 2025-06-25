# su2_postprocess/bl/bl_parser.py
from pathlib import Path
from typing import List
from .extract_new import extract_boundary_layer, BLResult

class BLParser:
    """
    Connects CLI args → extract_boundary_layer → writes tables & plots.
    """
    def __init__(
        self,
        root: Path,
        edge_method: str,
        plot_lm_mach: bool,
        x_locs: List[float],
        n_jobs: int,
    ):
        self.root = root
        self.edge_method = edge_method
        self.plot_lm_mach = plot_lm_mach
        self.x_locs = x_locs
        self.n_jobs = n_jobs
        self.result: BLResult

    def run(self) -> BLResult:
        self.result = extract_boundary_layer(
            root=self.root,
            edge_method=self.edge_method,
            plot_lm_mach=self.plot_lm_mach,
            x_locs=self.x_locs,
            n_jobs=self.n_jobs,
        )
        # here you would write out .dat tables, call plotting routines etc.
        return self.result
