"""Shared driver for tests2d/verify_<system>.py (round 4, d = 2 and d = 3).

For one solver and one dimension:
1. verify(): exact-solution checks at internal n = 64, 128, 256 (d = 2) or 32, 64, 128
   (d = 3; two levels where a run would not fit the time budget, stated in the output).
2. Smoke test, per allowed boundary condition, training (and ood) parameters, default
   internal grid: 4 trajectories (d = 2) or 2 (d = 3) x the full record. Asserts shape
   (frames, 3, *grid), float32, finite, unused channels exactly zero, bc recorded, and a
   physical bound:
     light   0 <= phi <= max phi(0) on the internal grid x (1 + 1e-9), min >= -1e-9 max
             (maximum principle; no source during the record)
     heat    -1e-9 max <= T <= max T(0) + sum of heater peaks (K/s) x record length
     sound   energy(last) / energy(0) <= 1 + 1e-9 (discrete energy)
     chain family  phi >= 0, T >= -1e-9 max T; T <= the recorded bound H_max/(rho C_p) T_p/2
             x 1.01 (the heating integral of a sin^2 pulse is T_p/2)
     advdiff max |u| <= 1.5 max |u(0)| (the 1D harness blow-up rule)
3. Resolution check (stored-data accuracy, reported, not pass/fail): the same 2 (d = 2) or
   1 (d = 3) training trajectories, periodic bc, at the default internal grid and at twice
   it; relative L2 difference over all frames of the output fields. For a second-order
   scheme the error of the default grid is about 4/3 of this difference.
4. Timing: one simulate_batch call of 16 trajectories x the full record at the default
   internal grid, first training bc (wall clock; the machine was shared with two training
   jobs, so times are upper bounds for an idle machine).
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfm2d import common as C  # noqa: E402


def _bound_check(solver, trajs, f, fails):
    name = solver.name
    if name == "light":
        for t, ff in zip(trajs, f):
            m = t.params["phi_init_max"]
            if ff.max() > m * (1 + 1e-9) or ff.min() < -1e-9 * m:
                fails.append(f"light max principle: [{ff.min()}, {ff.max()}] vs {m}")
    elif name == "heat":
        for t, ff in zip(trajs, f):
            T0 = sum(hs["amp"] for hs in t.params["hot_spots"]) * (1 + 1e-6)
            rec = (ff.shape[0] - 1) * t.dt_frame
            bound = T0 * 1.0 + sum(h["K_per_s"] for h in t.params["heaters"]) * rec
            if ff.max() > bound * (1 + 1e-9) or ff.min() < -1e-9 * ff.max():
                fails.append(f"heat bound: [{ff.min()}, {ff.max()}] vs {bound}")
    elif name.startswith("sound"):
        for t in trajs:
            if t.params["energy_ratio"] > 1 + 1e-9:
                fails.append(f"energy grew: {t.params['energy_ratio']}")
    elif name in ("chain", "photothermal", "thermoacoustic"):
        for t, ff in zip(trajs, f):
            if ff[:, 0].min() < 0:
                fails.append("negative fluence")
            Tm = ff[:, 1]
            if Tm.min() < -1e-9 * max(Tm.max(), 1e-30):
                fails.append(f"negative temperature {Tm.min()}")
            if Tm.max() > 1.01 * t.params["T_bound_K"]:
                fails.append(f"T above bound {Tm.max()} > {t.params['T_bound_K']}")
    elif name == "advdiff":
        for ff in f:
            a0 = np.abs(ff[0]).max()
            if np.abs(ff).max() > 1.5 * a0:
                fails.append("advdiff blow-up")


def smoke(solver, bcs, n_traj, seed=0, regime="train"):
    rows, ok = [], True
    for i, bc in enumerate(bcs):
        rng = np.random.default_rng(seed + i)
        params = [solver.sample_params(rng, regime) for _ in range(n_traj)]
        t0 = time.perf_counter()
        trajs = solver.simulate_batch(params, bc, None, rng)
        el = time.perf_counter() - t0
        fails = []
        f = np.stack([t.fields for t in trajs])
        exp_shape = (solver.n_frames, C.N_CH) + (solver.n_out,) * solver.dim
        for t in trajs:
            if t.fields.shape != exp_shape:
                fails.append(f"shape {t.fields.shape}")
            if t.fields.dtype != np.float32:
                fails.append(f"dtype {t.fields.dtype}")
            unused = [c for c in range(C.N_CH) if c not in t.channels]
            if np.any(t.fields[:, unused] != 0):
                fails.append("non-zero data in an unused channel")
            if t.bc != bc or t.dim != solver.dim:
                fails.append("bc/dim not recorded")
        finite = bool(np.isfinite(f).all())
        if not finite:
            fails.append("non-finite values")
        else:
            _bound_check(solver, trajs, f, fails)
        row = {"bc": bc, "regime": regime, "n_traj": n_traj, "shape": list(trajs[0].fields.shape),
               "finite": finite, "min": float(f.min()), "max": float(f.max()),
               "channels": [list(t.channels) for t in trajs],
               "internal_dt": [t.params["dt"] for t in trajs], "seconds": el,
               "failures": fails, "passed": not fails}
        if solver.name.startswith("sound"):
            row["energy_ratio"] = [t.params["energy_ratio"] for t in trajs]
        if "mua_over_musp_max" in trajs[0].params:
            row["mua_over_musp_max"] = [t.params["mua_over_musp_max"] for t in trajs]
        if "fluence_solve" in trajs[0].params:
            row["fluence_relative_residual_max"] = max(
                t.params["fluence_solve"]["relative_residual"] for t in trajs)
            row["fluence_method"] = trajs[0].params["fluence_solve"]["method"]
        ok &= not fails
        rows.append(row)
    return rows, ok


def resolution(solver, n_traj, seed=500, bc=None):
    bc = bc or ("periodic")
    rng = np.random.default_rng(seed)
    params = [solver.sample_params(rng, "train") for _ in range(n_traj)]
    n0 = solver.n_internal
    out = {}
    for n in (n0, 2 * n0):
        r2 = np.random.default_rng(seed + 1)
        t0 = time.perf_counter()
        tr = solver.simulate_batch([dict(p) for p in params], bc, None, r2, n)
        out[n] = (np.stack([t.fields for t in tr]).astype(float), time.perf_counter() - t0)
    a, b = out[n0][0], out[2 * n0][0]
    per = [float(np.sqrt(np.sum((a[i] - b[i]) ** 2) / np.sum(b[i] ** 2))) for i in range(n_traj)]
    return {"bc": bc, "n": [n0, 2 * n0], "n_traj": n_traj,
            "rel_L2_diff_all_frames": per, "max": max(per),
            "estimated_error_default_grid": [4.0 / 3.0 * x for x in per],
            "seconds": [out[n0][1], out[2 * n0][1]]}


GEN_BATCH = {2: 16, 3: 4}      # trajectories per simulate_batch call in data generation


def timing(solver, bc, n_traj=16, seed=100):
    """16 trajectories in calls of GEN_BATCH[d] (16 for d = 2, one call; 4 for d = 3, four
    calls: smaller batches are faster per cell on this machine)."""
    rng = np.random.default_rng(seed)
    params = [solver.sample_params(rng, "train") for _ in range(n_traj)]
    t0 = time.perf_counter()
    k = GEN_BATCH[solver.dim]
    trajs = []
    for i in range(0, n_traj, k):
        trajs += solver.simulate_batch(params[i:i + k], bc, None, rng)
    el = time.perf_counter() - t0
    dts = np.array([t.params["dt"] for t in trajs])
    return {"bc": bc, "n_traj": n_traj, "n_frames": solver.n_frames,
            "n_internal": solver.n_internal, "seconds": el, "seconds_per_trajectory": el / n_traj,
            "batch_per_call": k, "rk4_substeps_per_frame": [int(round(trajs[0].dt_frame / dts.max())),
                                       int(round(trajs[0].dt_frame / dts.min()))]}


def environment():
    load = os.getloadavg() if hasattr(os, "getloadavg") else None
    return {"python": platform.python_version(), "numpy": np.__version__,
            "cpu_count": os.cpu_count(), "load_average_at_end": load,
            "nice": os.nice(0)}


def run(solver, verify_kwargs=None, do_resolution=True, res_traj=None, smoke_traj=None,
        extra=None, tag=None, verify_result=None):
    d = solver.dim
    bcs = C.BCS_SCALAR if solver.family == "scalar" else C.BCS_SOUND
    rep = {}
    t0 = time.perf_counter()
    rep["verify"] = verify_result if verify_result is not None else solver.verify(**(verify_kwargs or {}))
    rep["verify_seconds"] = time.perf_counter() - t0
    st = smoke_traj or (4 if d == 2 else 2)
    rows, ok = smoke(solver, bcs, st)
    rows2, ok2 = smoke(solver, bcs[:2], st, seed=50, regime="ood")
    rep["smoke"] = {"passed": ok and ok2, "rows": rows + rows2}
    if do_resolution:
        rep["resolution"] = resolution(solver, res_traj or (2 if d == 2 else 1))
    rep["timing"] = timing(solver, bcs[0])
    if extra:
        rep.update(extra)
    rep["environment"] = environment()
    rep["passed"] = bool(rep["verify"]["passed"] and rep["smoke"]["passed"])
    out = C.results_dir(d) / f"{tag or solver.name}_d{d}.json"
    C.save_json(rep, out)
    v = rep["verify"]
    print(f"{solver.name} d={d}: verify passed={v['passed']}, smoke passed={rep['smoke']['passed']}")
    for k, c in v.get("cases", {"main": v}).items():
        if "error" in c:
            print(f"  {k}: n {c['n']} errors {[f'{e:.3e}' for e in c['error']]} orders "
                  f"{[round(o, 3) for o in (c['observed_order'] or [])]} passed={c['passed']}")
        else:
            print(f"  {k}: passed={c.get('passed')}")
    for r in rep["smoke"]["rows"]:
        if r["failures"]:
            print(f"  smoke {r['bc']} {r['regime']}: {r['failures']}")
    if do_resolution:
        print(f"  resolution {rep['resolution']['n']}: {rep['resolution']['rel_L2_diff_all_frames']}")
    print(f"  timing 16 x {solver.n_frames}: {rep['timing']['seconds']:.1f} s")
    print(f"  wrote {out}")
    return 0 if rep["passed"] else 1


def main_dims(argv):
    dims = [int(a) for a in argv[1:]] or [2, 3]
    return dims


def dump(obj, name):
    C.save_json(obj, C.RESULTS / name)
    print(json.dumps(obj, default=float)[:2000])
