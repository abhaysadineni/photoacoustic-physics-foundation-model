"""Shared driver for tests/verify_<name>.py.

1. Runs solver.verify() (internal n = 128, 256, 512).
2. Runs simulate_batch for 8 trajectories x 64 frames for every allowed boundary condition
   (train-range parameters) and asserts shape (64, 3, 128), float32, finite, no blow-up
   (max |u| over all frames <= BLOWUP_FACTOR * max(max |u| at frame 0, scale_floor);
   scale_floor = 1 for Fisher, whose logistic term legitimately grows small data up to the
   carrying capacity 1), channel bookkeeping, plus any system-specific bound.
3. Times one simulate_batch call of 64 trajectories x 64 frames at n_internal = 512 for each
   boundary condition (wall-clock, time.perf_counter, single process).
4. Saves everything to results/verification/<name>.json; exits non-zero if verify() did not
   pass or any batch assertion failed.
"""
from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfm import common  # noqa: E402

BLOWUP_FACTOR = 1.5
N_FRAMES = 64


def batch_check(solver, bcs, seed=0, bounds=None, scale_floor=0.0):
    rows, ok = [], True
    for i, bc in enumerate(bcs):
        rng = np.random.default_rng(seed + i)
        params = [solver.sample_params(rng, "train") for _ in range(8)]
        trajs = solver.simulate_batch(params, bc, N_FRAMES, rng)
        fails = []
        if len(trajs) != 8:
            fails.append(f"expected 8 trajectories, got {len(trajs)}")
        f = np.stack([t.fields for t in trajs])
        for t in trajs:
            if t.fields.shape != (N_FRAMES, common.N_CH, common.N_OUT):
                fails.append(f"shape {t.fields.shape}")
            if t.fields.dtype != np.float32:
                fails.append(f"dtype {t.fields.dtype}")
            unused = [c for c in range(common.N_CH) if c not in t.channels]
            if np.any(t.fields[:, unused] != 0):
                fails.append("non-zero data in an unused channel")
            if t.bc != bc:
                fails.append("bc not recorded")
        finite = bool(np.isfinite(f).all())
        if not finite:
            fails.append("non-finite values")
        amp0 = np.maximum(np.abs(f[:, 0]).max(axis=(1, 2)), scale_floor)
        ampmax = np.abs(f).max(axis=(1, 2, 3))
        growth = float(np.max(ampmax / np.maximum(amp0, 1e-30)))
        if growth > BLOWUP_FACTOR:
            fails.append(f"blow-up: max|u| / max(max|u0|, {scale_floor}) = {growth:.3g}")
        row = {"bc": bc, "n_traj": len(trajs), "shape": list(trajs[0].fields.shape),
               "dtype": str(trajs[0].fields.dtype), "finite": finite,
               "max_growth_ratio": growth, "min_value": float(f.min()),
               "max_value": float(f.max()),
               "channels": [list(t.channels) for t in trajs],
               "internal_dt": [t.params["dt"] for t in trajs]}
        if bounds is not None:
            lo, hi = bounds
            row["bounds"] = [lo, hi]
            if f.min() < lo or f.max() > hi:
                fails.append(f"out of bounds [{lo}, {hi}]: [{f.min()}, {f.max()}]")
        # mean frame-to-frame change relative to field scale (visible-change check, reported)
        d = np.abs(np.diff(f, axis=1)).mean(axis=(2, 3)) / np.maximum(
            np.abs(f).mean(axis=(1, 2, 3))[:, None], 1e-30)
        row["mean_rel_change_per_frame_frames_1to8"] = float(d[:, :8].mean())
        row["mean_rel_change_per_frame_frames_56to63"] = float(d[:, -8:].mean())
        row["failures"] = fails
        row["passed"] = not fails
        ok &= not fails
        rows.append(row)
    return rows, ok


def timing(solver, bcs, n_traj=64, seed=100):
    rows = []
    for i, bc in enumerate(bcs):
        rng = np.random.default_rng(seed + i)
        params = [solver.sample_params(rng, "train") for _ in range(n_traj)]
        t0 = time.perf_counter()
        trajs = solver.simulate_batch(params, bc, N_FRAMES, rng)
        el = time.perf_counter() - t0
        dts = np.array([t.params["dt"] for t in trajs])
        rows.append({"bc": bc, "n_traj": n_traj, "n_frames": N_FRAMES, "n_internal": 512,
                     "seconds": el, "seconds_per_trajectory": el / n_traj,
                     "rk4_substeps_per_frame_max": int(round(trajs[0].dt_frame / dts.min())),
                     "rk4_substeps_per_frame_min": int(round(trajs[0].dt_frame / dts.max()))})
    return rows


def run(solver, bcs, bounds=None, scale_floor=0.0):
    t0 = time.perf_counter()
    rep = solver.verify()
    rep["verify_seconds"] = time.perf_counter() - t0
    rows, batch_ok = batch_check(solver, bcs, bounds=bounds, scale_floor=scale_floor)
    rep["batch_check"] = {"passed": batch_ok, "blowup_factor": BLOWUP_FACTOR,
                          "blowup_rule": "max|u| over frames <= blowup_factor * "
                                         "max(max|u| at frame 0, scale_floor)",
                          "scale_floor": scale_floor, "rows": rows}
    rep["timing_64x64"] = timing(solver, bcs)
    rep["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                          "cpu_count": os.cpu_count(), "machine": platform.machine()}
    out = common.RESULTS / "verification" / f"{solver.name}.json"
    common.save_json(rep, out)
    print(f"{solver.name}: errors {rep['error']}, orders {rep['observed_order']}, "
          f"verify passed={rep['passed']}, batch passed={batch_ok}")
    for k, c in rep.get("cases", {}).items():
        print(f"  case {k}: errors {c['error']}, orders {c['observed_order']}, passed={c['passed']}")
    for r in rows:
        if r["failures"]:
            print(f"  batch {r['bc']}: {r['failures']}")
    for r in rep["timing_64x64"]:
        print(f"  timing {r['bc']}: {r['seconds']:.2f} s for 64 x 64 "
              f"(substeps/frame {r['rk4_substeps_per_frame_min']}-{r['rk4_substeps_per_frame_max']})")
    print(f"  wrote {out}")
    return 0 if (rep["passed"] and batch_ok) else 1
