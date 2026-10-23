"""Shared definitions: output format, grids, the Solver interface and the convergence harness.

Every solver module defines one subclass of Solver. Downstream code (data generation,
experiments) only ever talks to this interface.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

N_OUT = 128          # grid points in the stored format
N_CH = 3             # channels in the stored format
CH_LIGHT, CH_HEAT, CH_PRESSURE = 0, 1, 2

TRAIN_BCS_SCALAR = ("periodic", "dirichlet")
TEST_BCS_SCALAR = ("neumann",)
TRAIN_BCS_SOUND = ("periodic", "pressure_release")
TEST_BCS_SOUND = ("rigid",)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def cell_centres(n: int, length: float) -> np.ndarray:
    """Cell-centred grid on [0, length): x_i = (i + 1/2) * length / n."""
    dx = length / n
    return (np.arange(n) + 0.5) * dx


def downsample(u: np.ndarray, n_out: int = N_OUT) -> np.ndarray:
    """Average a cell-centred field of size n (a multiple of n_out) down to n_out cells.

    Cell averaging is exact for cell-averaged quantities and keeps conservation.
    """
    n = u.shape[-1]
    if n % n_out:
        raise ValueError(f"internal grid {n} is not a multiple of {n_out}")
    f = n // n_out
    return u.reshape(*u.shape[:-1], n_out, f).mean(-1)


@dataclass
class Trajectory:
    """One simulated trajectory in the common format."""
    fields: np.ndarray                 # (frames, 3, 128) float32
    system: str
    params: dict
    dt_frame: float                    # physical seconds between frames (or dimensionless for generic)
    length: float                      # physical domain length in metres (or dimensionless)
    bc: str
    channels: tuple                    # channels this system occupies


class Solver:
    """Interface every system implements.

    name        short identifier
    channels    which output channels it writes ("random" for generic scalar systems)
    order       promised order of convergence of the discretisation (in space, with
                time step tied to space step), used by the verification harness
    """
    name: str = "base"
    channels: tuple | str = ()
    order: float = 2.0

    def sample_params(self, rng: np.random.Generator, regime: str = "train") -> dict:
        """Draw physical parameters. regime is "train" or "ood" (outside training ranges)."""
        raise NotImplementedError

    def simulate(self, params: dict, bc: str, n_frames: int, rng: np.random.Generator,
                 n_internal: int = 512) -> Trajectory:
        """Run the solver and return a Trajectory in the common format."""
        raise NotImplementedError

    def simulate_batch(self, params_list: list, bc: str, n_frames: int,
                       rng: np.random.Generator, n_internal: int = 512) -> list:
        """Simulate several trajectories that share one boundary condition.

        Solvers should override this with a version vectorised over a leading batch axis;
        data generation calls it for speed. The default simply loops.
        """
        return [self.simulate(p, bc, n_frames, rng, n_internal) for p in params_list]

    def verify(self, n_list=(128, 256, 512)) -> dict:
        """Run the exact-solution check at several internal resolutions.

        Returns {"system", "check", "n": [...], "error": [...], "observed_order": [...],
        "promised_order", "passed"}.
        """
        raise NotImplementedError


def convergence_report(name: str, check: str, n_list, errors, promised: float,
                       tol: float = 0.3) -> dict:
    """Observed order between successive refinements; passes if the finest observed order
    is within tol of the promised order (or better) and errors fall monotonically."""
    errors = [float(e) for e in errors]
    orders = []
    for (n1, e1), (n2, e2) in zip(zip(n_list, errors), zip(n_list[1:], errors[1:])):
        orders.append(math.log(e1 / e2) / math.log(n2 / n1))
    monotone = all(b < a for a, b in zip(errors, errors[1:]))
    passed = monotone and orders[-1] >= promised - tol
    return {"system": name, "check": check, "n": list(n_list), "error": errors,
            "observed_order": orders, "promised_order": promised, "passed": bool(passed)}


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def rel_l2(pred: np.ndarray, true: np.ndarray, axis=None) -> np.ndarray:
    """Relative L2 error ||pred - true|| / ||true||."""
    num = np.sqrt(np.sum((pred - true) ** 2, axis=axis))
    den = np.sqrt(np.sum(true ** 2, axis=axis))
    return num / np.maximum(den, 1e-12)
