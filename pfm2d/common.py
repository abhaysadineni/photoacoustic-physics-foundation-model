"""Shared definitions for the Cartesian-grid rounds (round 4): d = 2 and d = 3.

Format
------
A trajectory is an array (frames, 3, *grid) float32, grid = (64, 64) for d = 2 and
(32, 32, 32) for d = 3. Channels are fixed by quantity exactly as in 1D (pfm/common.py):
0 fluence rate, 1 temperature rise, 2 pressure. Unused channels are zero.

Index convention: the last d axes of every array are spatial; axis a (a = 0 .. d-1,
counted from the first spatial axis) is coordinate x_a (x, y, z). Element [..., i, j(, k)]
is the cell centred at ((i + 1/2) h, (j + 1/2) h(, (k + 1/2) h)).

Geometry (modelling choices, stated):
    d = 2: output 64 x 64 cells, internal grid 128 x 128 by default, 32 frames.
    d = 3: output 32^3 cells, internal grid 64^3 by default, 24 frames.
Tissue systems use the 1D physical window L = 12.8 mm in every direction (output cell
0.2 mm for d = 2 and 0.4 mm for d = 3); the generic system is dimensionless on [0, 1)^d.
Downsampling from the internal grid is the cell average over f^d blocks (f = 2 by
default), exact for cell-averaged finite-volume fields.

Boundary conditions apply to every face of the box (all 2d walls have the same type):
    scalar  periodic, dirichlet (zero), neumann (zero flux)   train: periodic, dirichlet;
                                                              held out: neumann
    sound   periodic, pressure_release, rigid                 train: periodic,
                                                              pressure_release; held out: rigid
These are the 1D sets (pfm.common.TRAIN_BCS_SCALAR etc.), imported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pfm import common as C1

ROOT = C1.ROOT
N_CH = C1.N_CH
CH_LIGHT, CH_HEAT, CH_PRESSURE = C1.CH_LIGHT, C1.CH_HEAT, C1.CH_PRESSURE
TRAIN_BCS_SCALAR, TEST_BCS_SCALAR = C1.TRAIN_BCS_SCALAR, C1.TEST_BCS_SCALAR
TRAIN_BCS_SOUND, TEST_BCS_SOUND = C1.TRAIN_BCS_SOUND, C1.TEST_BCS_SOUND
BCS_SCALAR = ("periodic", "dirichlet", "neumann")
BCS_SOUND = ("periodic", "pressure_release", "rigid")

GEOM = {2: {"n_out": 64, "n_internal": 128, "n_frames": 32},
        3: {"n_out": 32, "n_internal": 64, "n_frames": 24}}
LENGTH = 12.8e-3              # m, tissue systems (the 1D window, modelling choice)

RESULTS = ROOT / "results" / "verification2d"


def results_dir(dim: int):
    """results/verification2d for d = 2, results/verification3d for d = 3."""
    return ROOT / "results" / f"verification{dim}d"


DATA_ROOT = ROOT / "data2d"


def data_dir(dim: int):
    """data2d/<system>/ for d = 2, data2d/d3/<system>/ for d = 3."""
    return DATA_ROOT if dim == 2 else DATA_ROOT / "d3"


convergence_report = C1.convergence_report
save_json = C1.save_json


def centres(n: int, length: float) -> np.ndarray:
    return (np.arange(n) + 0.5) * (length / n)


def mesh(n, length: float, d: int, stagger: int | None = None):
    """List of d coordinate arrays, broadcastable to the spatial shape (n,)*d (or to the
    face shape of axis `stagger`, which has n + 1 points along that axis)."""
    out = []
    for a in range(d):
        if a == stagger:
            x = np.arange(n + 1) * (length / n)
        else:
            x = centres(n, length)
        shp = [1] * d
        shp[a] = x.size
        out.append(x.reshape(shp))
    return out


def downsample(u: np.ndarray, d: int, n_out: int) -> np.ndarray:
    """Cell average of the last d axes (each a multiple of n_out) down to n_out cells."""
    lead = u.shape[:-d]
    sp = u.shape[-d:]
    shp = list(lead)
    for n in sp:
        if n % n_out:
            raise ValueError(f"internal grid {n} is not a multiple of {n_out}")
        shp += [n_out, n // n_out]
    v = u.reshape(shp)
    axes = tuple(len(lead) + 2 * a + 1 for a in range(d))
    return v.mean(axis=axes)


@dataclass
class Trajectory:
    fields: np.ndarray           # (frames, 3, *grid) float32
    system: str
    params: dict
    dt_frame: float
    length: float
    bc: str
    channels: tuple
    dim: int


class Solver:
    """Interface of every d-dimensional system (as pfm.common.Solver, plus `dim`)."""
    name = "base"
    channels: tuple | str = ()
    order = 2.0
    family = "scalar"            # or "sound": selects the boundary-condition sets

    def __init__(self, dim: int = 2):
        if dim not in GEOM:
            raise ValueError("dim must be 2 or 3")
        self.dim = int(dim)
        g = GEOM[self.dim]
        self.n_out, self.n_internal, self.n_frames = g["n_out"], g["n_internal"], g["n_frames"]

    def _defaults(self, n_frames, n_internal):
        return (self.n_frames if n_frames is None else int(n_frames),
                self.n_internal if n_internal is None else int(n_internal))

    def check_bc(self, bc):
        allowed = BCS_SCALAR if self.family == "scalar" else BCS_SOUND
        if bc not in allowed:
            raise ValueError(f"{self.name}: bc {bc!r} not in {allowed}")

    def sample_params(self, rng, regime="train"):
        raise NotImplementedError

    def simulate_batch(self, params_list, bc, n_frames=None, rng=None, n_internal=None):
        raise NotImplementedError

    def verify(self, n_list=None):
        raise NotImplementedError


def default_n_list(d: int):
    return (64, 128, 256) if d == 2 else (32, 64, 128)
