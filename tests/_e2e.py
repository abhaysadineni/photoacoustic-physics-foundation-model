"""Shared driver for tests/e2e_<name>.py: end-to-end exact checks of the generic solvers
through the production path (audit round 1, MINOR 6, 7 and 8).

What is exercised
-----------------
solver.simulate_batch(params_list, bc, n_frames=64, rng, n_internal) exactly as data
generation calls it: per-trajectory parameter arrays, per-trajectory substep counts
(_mol.n_substeps), the batch sorted by substep count inside _mol.rk4_frames, the production
ghost cells for the boundary condition, cell-average downsampling to 128 output cells,
placement in the trajectory's channel of the common (64, 3, 128) float32 format. Several
cases with different substep counts share each simulate_batch call, so the sorted-substep
bookkeeping is exercised.

Initial conditions
------------------
Explicit initial conditions go through params["ic"], as in the stored data. A Gaussian ic
dict is evaluated by the production _mol.eval_gaussians. For exact solutions whose initial
data are not a Gaussian sum (cosine mode, Cole-Hopf, tanh, Ablowitz-Zeppetella, uniform
state) the test process replaces _mol.eval_gaussians IN MEMORY ONLY with a wrapper that
returns the exact initial data at the internal cell centres when the ic dict carries the key
"exact_ic", and delegates to the original function otherwise. No module file is changed.
This wrapper is the only departure from the production path; everything downstream of the
initial-condition evaluation (rescaling in Burgers, clipping in Fisher, time stepping,
ghosts, downsampling, storage) is the production code. Burgers rescales u0 by
u_max / max|u0|; u_max is set to max|u0| on the same internal grid, so the factor is
exactly 1.0 in IEEE arithmetic.

Reference
---------
The exact solution at t_k = k * DT_FRAME (k = 0 .. 63) averaged over each of the 128
output cells by composite Gauss-Legendre quadrature (4 sub-intervals x 10 nodes per output
cell). A second quadrature (8 x 14) gives the quadrature error, reported per case.

Metric
------
Relative L2 per frame, ||stored - exact_avg|| / ||exact_avg|| over the 128 cells of the
occupied channel (stored values are float32, the reference float64). Reported: maximum and
mean over the 64 frames, at n_internal = 512 and 1024, and the factor e(512) / e(1024).
Secondary metric (all cases, not used for pass/fail): ||stored_k - exact_k|| divided by the
largest exact frame norm of the trajectory, max and mean over frames. It stays meaningful
when an oscillating mode passes through zero amplitude, where the per-frame relative error
is ill-conditioned. Diagnostic (not used for pass/fail): the same error computed on the float64 frames that
common.downsample returns, captured in memory just before the production code casts them to
float32 (the capture is checked to equal the stored frames after the cast). It separates the
discretisation error from float32 rounding. Also reported: the float32 storage floor ||float32(exact_avg) - exact_avg|| / ||exact_avg||,
and that every unused channel is exactly zero.

Pass rule (per case): max error at 512 below 1e-2, and both the max and the mean error fall
from 512 to 1024. Exception, stated in the output whenever it is used: if both errors are
within 4 x the float32 storage floor, refinement cannot be resolved in stored data, and the
case is marked "passed (at float32 storage floor)".
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pfm import common  # noqa: E402
from pfm.solvers import _mol  # noqa: E402

N_FRAMES = 64
N_LIST = (512, 1024)
TOL_512 = 1e-2
FLOOR_FACTOR = 4.0
QUAD = (4, 10)
QUAD_CHECK = (8, 14)

# ------------------------------------------------------------------ exact-IC hook (in memory)
EXACT_IC: dict = {}
_ORIG_EVAL = _mol.eval_gaussians


def _eval_hook(ic, x, periodic):
    if isinstance(ic, dict) and "exact_ic" in ic:
        return np.asarray(EXACT_IC[ic["exact_ic"]](x), dtype=float) * np.ones_like(x)
    return _ORIG_EVAL(ic, x, periodic)


_mol.eval_gaussians = _eval_hook

# float64 frames before the float32 cast (diagnostic only; the stored output is unchanged):
# each on_frame call of the generic solvers passes through common.downsample exactly once.
_CAPTURE: list = []
_ORIG_DOWNSAMPLE = common.downsample


def _downsample_hook(u, n_out=common.N_OUT):
    r = _ORIG_DOWNSAMPLE(u, n_out)
    _CAPTURE.append(np.array(r, dtype=float, copy=True))
    return r


common.downsample = _downsample_hook


def cell_average(f, n_out=common.N_OUT, length=_mol.LENGTH, quad=QUAD):
    """Average of f over each of n_out equal cells by composite Gauss-Legendre quadrature."""
    sub, order = quad
    xg, wg = np.polynomial.legendre.leggauss(order)
    H = length / n_out
    h = H / sub
    left = np.arange(n_out * sub) * h
    nodes = left[:, None] + 0.5 * (xg[None, :] + 1.0) * h
    integ = (f(nodes) * wg).sum(-1) * 0.5 * h
    return integ.reshape(n_out, sub).sum(-1) / H


def rel(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2)) / max(np.sqrt(np.sum(b ** 2)), 1e-300))


def run_group(solver, dt_frame, bc, cases, rng_seed=0):
    """Run every case of one boundary condition in one simulate_batch call per resolution.

    cases: list of dicts with "name", "params" (passed to the solver; must hold "ic" and
    "channel" for generic systems), "exact" (callable (x, t) -> u, vectorised), "channel",
    and optional "notes" and "params_by_n" (callable n -> extra params, e.g. u_max).
    Returns a dict of results keyed by case name.
    """
    out = {c["name"]: {"bc": bc, "notes": c.get("notes", ""), "by_n": {}} for c in cases}
    t_k = np.arange(N_FRAMES) * dt_frame
    refs = {}
    for c in cases:
        ex = np.stack([cell_average(lambda xx, tt=tt: c["exact"](xx, tt)) for tt in t_k])
        ex2 = np.stack([cell_average(lambda xx, tt=tt: c["exact"](xx, tt), quad=QUAD_CHECK) for tt in t_k])
        refs[c["name"]] = ex
        floor = [rel(e.astype(np.float32).astype(float), e) for e in ex]
        out[c["name"]].update({
            "quadrature": f"composite Gauss-Legendre {QUAD[0]} x {QUAD[1]} per output cell",
            "quadrature_check_max_abs_diff_vs_{}x{}".format(*QUAD_CHECK): float(np.abs(ex - ex2).max()),
            "float32_floor_max": float(max(floor)), "float32_floor_mean": float(np.mean(floor)),
            "exact_max_abs_over_frames": float(np.abs(ex).max()),
            "exact_min_norm_ratio_over_frames": float(np.min(np.linalg.norm(ex, axis=1)) / np.linalg.norm(ex[0])),
        })
    for n in N_LIST:
        plist = []
        for c in cases:
            p = dict(c["params"])
            if "params_by_n" in c:
                p.update(c["params_by_n"](n))
            plist.append(p)
        _CAPTURE.clear()
        t0 = time.perf_counter()
        trajs = solver.simulate_batch(plist, bc, N_FRAMES, np.random.default_rng(rng_seed), n_internal=n)
        secs = time.perf_counter() - t0
        assert len(_CAPTURE) == N_FRAMES, len(_CAPTURE)
        f64 = np.stack(_CAPTURE, axis=1)                     # (B, frames, 128), float64
        for b, (c, tr) in enumerate(zip(cases, trajs)):
            f = tr.fields
            ch = c["channel"]
            assert f.shape == (N_FRAMES, common.N_CH, common.N_OUT) and f.dtype == np.float32
            assert tuple(tr.channels) == (ch,), (tr.channels, ch)
            assert abs(tr.dt_frame - dt_frame) == 0.0
            others = [k for k in range(common.N_CH) if k != ch]
            unused_zero = bool(np.all(f[:, others, :] == 0.0))
            errs = np.array([rel(f[k, ch].astype(float), refs[c["name"]][k]) for k in range(N_FRAMES)])
            enorm = np.linalg.norm(f[:, ch].astype(float) - refs[c["name"]], axis=1)
            scale = np.linalg.norm(refs[c["name"]], axis=1).max()
            assert np.array_equal(f64[b].astype(np.float32), f[:, ch]), "float64 capture does not match storage"
            e64 = np.array([rel(f64[b, k], refs[c["name"]][k]) for k in range(N_FRAMES)])
            nsub = int(round(dt_frame / tr.params["dt"]))
            out[c["name"]]["by_n"][str(n)] = {
                "max_rel_l2": float(errs.max()), "mean_rel_l2": float(errs.mean()),
                "max_err_over_max_frame_norm": float(enorm.max() / scale),
                "mean_err_over_max_frame_norm": float(enorm.mean() / scale),
                "frame_of_max": int(errs.argmax()),
                "float64_before_storage_max_rel_l2": float(e64.max()),
                "float64_before_storage_mean_rel_l2": float(e64.mean()), "rel_l2_frame0": float(errs[0]),
                "rel_l2_frame63": float(errs[-1]), "substeps_per_frame": nsub,
                "dt": float(tr.params["dt"]), "unused_channels_exactly_zero": unused_zero,
                "batch_seconds": secs, "batch_size": len(cases),
                "per_frame_rel_l2": [float(e) for e in errs],
            }
    for c in cases:
        r = out[c["name"]]
        a, b = r["by_n"]["512"], r["by_n"]["1024"]
        fmax, fmean = a["max_rel_l2"] / b["max_rel_l2"], a["mean_rel_l2"] / b["mean_rel_l2"]
        r["factor_max_512_over_1024"] = fmax
        r["factor_mean_512_over_1024"] = fmean
        r["observed_order_max"] = math.log2(fmax)
        r["observed_order_mean"] = math.log2(fmean)
        r["factor_float64_max_512_over_1024"] = (a["float64_before_storage_max_rel_l2"]
                                                 / max(b["float64_before_storage_max_rel_l2"], 1e-300))
        below = a["max_rel_l2"] < TOL_512
        falls = b["max_rel_l2"] < a["max_rel_l2"] and b["mean_rel_l2"] < a["mean_rel_l2"]
        floor = FLOOR_FACTOR * r["float32_floor_max"]
        at_floor = a["max_rel_l2"] <= floor and b["max_rel_l2"] <= floor
        zero_ok = a["unused_channels_exactly_zero"] and b["unused_channels_exactly_zero"]
        r["below_1e-2_at_512"] = bool(below)
        r["falls_with_refinement"] = bool(falls)
        r["at_float32_floor"] = bool(at_floor)
        if below and falls and zero_ok:
            r["passed"], r["verdict"] = True, "passed"
        elif below and at_floor and zero_ok:
            r["passed"], r["verdict"] = True, "passed (at float32 storage floor; refinement not resolvable)"
        else:
            why = []
            if not below:
                why.append("max error at 512 not below 1e-2")
            if not (falls or at_floor):
                why.append("error does not fall with refinement")
            if not zero_ok:
                why.append("unused channel not exactly zero")
            r["passed"], r["verdict"] = False, "FAILED: " + "; ".join(why)
    return out


def summarise(system, dt_frame, groups, unsupported, extra=None):
    cases = {}
    for g in groups:
        cases.update(g)
    rep = {
        "system": system,
        "check": "end-to-end exact check through simulate_batch (production path), "
                 "audit round 1 MINOR 6/7/8",
        "n_internal": list(N_LIST), "n_frames": N_FRAMES, "dt_frame": dt_frame,
        "metric": "relative L2 per stored frame (occupied channel, 128 output cells) against the exact "
                  "solution cell-averaged by composite Gauss-Legendre quadrature; max and mean over "
                  "the 64 frames (frame 0 included)",
        "pass_rule": f"max error at 512 < {TOL_512} and max and mean errors fall from 512 to 1024; "
                     f"or both within {FLOOR_FACTOR:g} x the float32 storage floor (stated per case)",
        "initial_condition_path": "Gaussian ic dicts: production _mol.eval_gaussians. Non-Gaussian exact "
                                  "initial data: in-memory wrapper around _mol.eval_gaussians keyed by "
                                  "ic['exact_ic'] (no module file changed); all later steps are production code",
        "cases": cases,
        "not_run_through_production_path": unsupported,
        "passed": bool(all(c["passed"] for c in cases.values())),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "host_cpus": os.cpu_count(), "niceness": os.nice(0)},
    }
    if extra:
        rep.update(extra)
    path = common.RESULTS / "verification" / f"e2e_{system}.json"
    common.save_json(rep, path)
    print_table(rep)
    print(f"saved {path.relative_to(ROOT)}; overall {'PASS' if rep['passed'] else 'FAIL'}")
    return rep


def print_table(rep):
    print(f"\n{rep['system']}: dt_frame = {rep['dt_frame']:.6g}, {rep['n_frames']} frames")
    hdr = f"{'case':34s} {'bc':16s} {'max512':>10s} {'mean512':>10s} {'max1024':>10s} {'mean1024':>10s} {'fac max':>8s} {'fac mean':>8s} {'f32 floor':>10s} {'f64max512':>10s} {'f64max1024':>10s}  verdict"
    print(hdr)
    for name, r in rep["cases"].items():
        a, b = r["by_n"]["512"], r["by_n"]["1024"]
        print(f"{name:34s} {r['bc']:16s} {a['max_rel_l2']:10.3e} {a['mean_rel_l2']:10.3e} "
              f"{b['max_rel_l2']:10.3e} {b['mean_rel_l2']:10.3e} {r['factor_max_512_over_1024']:8.3f} "
              f"{r['factor_mean_512_over_1024']:8.3f} {r['float32_floor_max']:10.2e} "
              f"{a['float64_before_storage_max_rel_l2']:10.3e} {b['float64_before_storage_max_rel_l2']:10.3e}  {r['verdict']}")


def fd_residual(u, pde, x, t, h=1e-5, k=1e-6):
    """Max |PDE residual| of a closed form by centred finite differences (sanity check of
    the closed form, independent of the solver). pde(u, ut, ux, uxx) -> residual."""
    ut = (u(x, t + k) - u(x, t - k)) / (2 * k)
    ux = (u(x + h, t) - u(x - h, t)) / (2 * h)
    uxx = (u(x + h, t) - 2 * u(x, t) + u(x - h, t)) / h ** 2
    res = pde(u(x, t), ut, ux, uxx)
    return float(np.abs(res).max()), float(np.abs(ut).max())
