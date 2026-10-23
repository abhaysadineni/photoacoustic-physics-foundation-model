"""Stored-data accuracy of the grid sound systems against the internal grid (decides the
internal grid of the sound family; PLAN.md section 23).

Cases per dimension (production path simulate_batch, full record, periodic box):
  uniform medium, one launched Gaussian pulse of 1/e half-width w (the training range is
  w in [0.3, 1.0] mm; w = 0.3 mm is the worst case), c and rho at the training midpoint,
  for w = 0.3, 0.4, 0.5, 0.7 mm; and one random training draw of SoundLayered (seed 500).
Internal grids: d = 2: 128, 256, 512; d = 3: 64, 96, 128. Widths: the 1D minimum 0.3 mm, and
3 and 6.5 output cells (the minimum and middle of the cell-scaled range adopted 21 Oct 2026).
An earlier run (21 Oct 2026, 13:27 PDT, log only) with 64, 96, 128, 192 for the 0.3 mm pulse in
3D gave errors 0.795, 0.522, 0.328, 0.146 (observed order 1.98). Reference: Richardson extrapolation of the two finest grids assuming
second order, u_ref = (n2^2 u2 - n1^2 u1) / (n2^2 - n1^2); the error of grid n is the relative
L2 difference to u_ref over all output frames (the quantity the network is trained on). The
finest-grid pair also gives the observed order between the three finest grids.
Writes results/verification<d>d/sound_resolution_d<d>.json, and adds the summary block
"resolution_study" to results/verification<d>d/sound_uniform_d<d>.json (attach(d); also called
by tests2d/verify_sound.py, so either order of the two scripts gives the same file).
Usage: python tests2d/resolution_sound.py 3 [2]
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np                                      # noqa: E402

from pfm2d import common as C                           # noqa: E402
from pfm2d.solvers.sound import SoundUniform, SoundLayered, SOUND_INTERNAL, PULSE_WIDTH_CELLS   # noqa: E402

GRIDS = {2: [128, 256, 512], 3: [64, 96, 128]}
# the 1D millimetre minimum (0.3 mm) and the cell-scaled minimum (3 output cells) and middle
WIDTHS = {2: [0.3e-3, 0.6e-3, 1.3e-3], 3: [0.3e-3, 1.2e-3, 2.6e-3]}


def cases(d):
    su = SoundUniform(d)
    out = []
    for w in WIDTHS[d]:
        dirn = [1.0 / math.sqrt(d)] * d
        p = {"regime": "train", "dim": d, "c": 1561.0, "rho": 1050.0,
             "pulses": [{"u_centre": [0.37, 0.52, 0.45][:d], "width": w, "amp": 1.0,
                         "direction": dirn}]}
        out.append((f"uniform_launched_w{w * 1e3:.1f}mm", su, p))
    sl = SoundLayered(d)
    out.append(("layered_random_seed500", sl, sl.sample_params(np.random.default_rng(500), "train")))
    out.append(("uniform_random_seed501", su, su.sample_params(np.random.default_rng(501), "train")))
    return out


def run(d):
    rep = {"dim": d, "grids": GRIDS[d], "cases": {}}
    for name, solver, p in cases(d):
        fields, secs = {}, {}
        for n in GRIDS[d]:
            t0 = time.perf_counter()
            tr = solver.simulate_batch([json.loads(json.dumps(p))], "periodic", None, None, n)[0]
            secs[n] = time.perf_counter() - t0
            fields[n] = tr.fields[:, C.CH_PRESSURE].astype(float)
        n1, n2 = GRIDS[d][-2], GRIDS[d][-1]
        ref = (n2 ** 2 * fields[n2] - n1 ** 2 * fields[n1]) / (n2 ** 2 - n1 ** 2)
        nr = np.sqrt(np.sum(ref ** 2))
        err = {n: float(np.sqrt(np.sum((fields[n] - ref) ** 2)) / nr) for n in GRIDS[d]}
        g = GRIDS[d][-3:]
        e1 = np.sqrt(np.sum((fields[g[0]] - fields[g[1]]) ** 2))
        e2 = np.sqrt(np.sum((fields[g[1]] - fields[g[2]]) ** 2))
        # successive differences shrink by (1 - r1^-p)/(r1^p ... ) ; for ratios r1 = g1/g0,
        # r2 = g2/g1 and order p: e1/e2 = (g0^-p - g1^-p) / (g1^-p - g2^-p); solve for p
        f = lambda q: (g[0] ** -q - g[1] ** -q) / (g[1] ** -q - g[2] ** -q) - e1 / e2
        lo, hi = 0.1, 6.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        rep["cases"][name] = {"rel_L2_error_vs_richardson": {str(k): v for k, v in err.items()},
                              "observed_order_three_finest": 0.5 * (lo + hi),
                              "seconds": {str(k): v for k, v in secs.items()},
                              "params": p}
        print(d, name, {k: f"{v:.4f}" for k, v in err.items()}, f"order {0.5 * (lo + hi):.2f}",
              {k: round(v, 1) for k, v in secs.items()}, flush=True)
    rep["note"] = ("errors relative to a Richardson extrapolation of the two finest grids; the "
                   "finest grid's own error is therefore not measured (reported as near zero by "
                   "construction) and the coarser grids' errors are the meaningful values")
    C.save_json(rep, C.results_dir(d) / f"sound_resolution_d{d}.json")
    attach(d)


def study(d):
    """Summary of sound_resolution_d<d>.json at the production internal grid."""
    sr = json.loads((C.results_dir(d) / f"sound_resolution_d{d}.json").read_text())
    n = SOUND_INTERNAL[d]
    lo, hi = (int(v) for v in PULSE_WIDTH_CELLS)
    return {"source": f"results/verification{d}d/sound_resolution_d{d}.json (tests2d/resolution_sound.py)",
            "chosen_internal_grid": n,
            "pulse_width_rule": f"1/e half-width uniform in [{lo}, {hi}] output cells",
            "worst_case": (f"launched Gaussian of the minimum width ({lo} output cells), uniform medium, "
                           "full record, relative L2 over all frames against a Richardson extrapolation"),
            "cases": {k: {"error_at_chosen_internal_grid": v["rel_L2_error_vs_richardson"][str(n)],
                          "all_grids": v["rel_L2_error_vs_richardson"]}
                      for k, v in sr["cases"].items()}}


def attach(d):
    """Add the resolution_study block to sound_uniform_d<d>.json when both files exist."""
    f = C.results_dir(d) / f"sound_uniform_d{d}.json"
    if not (f.exists() and (C.results_dir(d) / f"sound_resolution_d{d}.json").exists()):
        return False
    rep = json.loads(f.read_text())
    rep["resolution_study"] = study(d)
    C.save_json(rep, f)
    return True


if __name__ == "__main__":
    for a in sys.argv[1:] or ["3", "2"]:
        run(int(a))
