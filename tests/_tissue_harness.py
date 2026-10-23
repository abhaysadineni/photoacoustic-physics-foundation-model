"""Shared driver for the tissue-physics tests (verify_light, verify_heat, verify_sound,
verify_chain).

For each solver:
1. solver.verify() at internal n = 128, 256, 512 (orders by common.convergence_report).
2. Smoke test: simulate_batch for 8 trajectories x 64 frames per allowed boundary condition
   (train-range parameters, n_internal = 512). Asserts shape (64, 3, 128), float32, finite,
   unused channels zero, and a PHYSICAL no-blow-up bound per trajectory:
     light   0 <= phi <= max(phi(0)) + integral of the source peak over time
             (maximum principle; min >= -1e-6 max allowed for round-off)
     heat    0 <= T <= max(T(0)) + sum of heater peaks x record length (maximum principle)
     sound   discrete acoustic energy at the last frame <= energy at frame 0 (x(1 + 1e-9))
     chain   phi, T >= 0; d = 1: |p| <= 1.05 Gamma max(H) T_p / 2 (d'Alembert with
             images: the pressure is an average of the heating integral, so its sup norm
             cannot exceed Gamma max H int E dt; 5 per cent for dispersion);
             d = 2, 3 (geometric focusing voids that bound): the n = 256 and n = 512 runs
             of the same trajectories agree to 5 per cent in relative L2.
   A fixed growth factor relative to frame 0 is not used: chain frames start at zero and
   light and heat contain sources.
3. Timing: one simulate_batch call of 64 trajectories x 64 frames, n_internal = 512, per
   boundary condition (wall clock, time.perf_counter, single process).
"""
from __future__ import annotations

import copy
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfm import common  # noqa: E402

N_FRAMES = 64


def _bound_fail(solver, t):
    f = t.fields
    p = t.params
    name = solver.name
    if name == "light":
        phi = f[:, common.CH_LIGHT]
        hi = p["phi_init_max"]            # no source during the record: maximum principle
        if phi.min() < -1e-6 * phi.max():
            return f"negative fluence {phi.min()}"
        if phi.max() > hi * (1 + 1e-6):
            return f"fluence {phi.max()} above maximum-principle bound {hi}"
        if p["mua_over_musp_max"] > p["mua_over_musp_cap"] + 1e-12:
            return f"mu_a/mu_s' {p['mua_over_musp_max']} above cap"
        if p["start_over_t_tr"] < 10.0:
            return f"record starts at {p['start_over_t_tr']} transport times"
    elif name == "heat":
        T = f[:, common.CH_HEAT]
        hi = sum(h["amp"] for h in p["hot_spots"]) + sum(
            h["K_per_frame"] for h in p["heaters"]) * (N_FRAMES - 1)
        if T.min() < -1e-6 * T.max():
            return f"negative temperature rise {T.min()}"
        if T.max() > hi * (1 + 1e-6):
            return f"temperature {T.max()} above maximum-principle bound {hi}"
    elif name.startswith("sound"):
        if p["energy_ratio"] > 1 + 1e-9:
            return f"acoustic energy grew: ratio {p['energy_ratio']}"
    elif name in ("photothermal", "thermoacoustic"):
        if p["mua_over_musp_max"] > p["mua_over_musp_cap"] + 1e-12:
            return f"mu_a/mu_s' {p['mua_over_musp_max']} above cap"
        T = f[:, common.CH_HEAT]
        if T.min() < -1e-12 * max(T.max(), 1e-300):
            return f"negative temperature rise {T.min()}"
        # maximum principle: T <= (max H / (rho C_p)) int E dt, int E dt <= T_p / 2
        if T.max() > p["T_bound_K"] * (1 + 1e-6):
            return f"T {T.max()} above bound {p['T_bound_K']}"
        if name == "photothermal" and f[:, common.CH_LIGHT].min() < 0:
            return f"negative fluence {f[:, common.CH_LIGHT].min()}"
        if (name == "thermoacoustic" and p.get("dim", 1) == 1
                and np.abs(f[:, 2]).max() > 1.05 * p["p_bound_1d_Pa"]):
            return f"|p| {np.abs(f[:, 2]).max()} above bound {p['p_bound_1d_Pa']}"
    elif name == "chain":
        if p["mua_over_musp_max"] > p["mua_over_musp_cap"] + 1e-12:
            return f"mu_a/mu_s' {p['mua_over_musp_max']} above cap"
        if f[:, 0].min() < 0 or f[:, 1].min() < -1e-12 * max(f[:, 1].max(), 1e-300):
            return f"negative fluence or temperature ({f[:, 0].min()}, {f[:, 1].min()})"
        if p["dim"] == 1 and np.abs(f[:, 2]).max() > 1.05 * p["p_bound_1d_Pa"]:
            return f"|p| {np.abs(f[:, 2]).max()} above bound {p['p_bound_1d_Pa']}"
    return None


def batch_check(solver, bcs, seed=0, extra_params=None, regime="train", sample_kw=None):
    rows, ok = [], True
    for i, bc in enumerate(bcs):
        rng = np.random.default_rng(seed + i)
        params = [solver.sample_params(rng, regime, **(sample_kw or {})) for _ in range(8)]
        if extra_params:
            for q in params:
                q.update(extra_params)
        trajs = solver.simulate_batch(copy.deepcopy(params), bc, N_FRAMES,
                                      np.random.default_rng(seed + 50 + i))
        fails = []
        f = np.stack([t.fields for t in trajs])
        for t in trajs:
            if t.fields.shape != (N_FRAMES, common.N_CH, common.N_OUT):
                fails.append(f"shape {t.fields.shape}")
            if t.fields.dtype != np.float32:
                fails.append(f"dtype {t.fields.dtype}")
            unused = [c for c in range(common.N_CH) if c not in t.channels]
            if unused and np.any(t.fields[:, unused] != 0):
                fails.append("non-zero data in an unused channel")
            if t.bc != bc:
                fails.append("bc not recorded")
            if not np.isfinite(t.fields).all():
                fails.append("non-finite values")
            else:
                b = _bound_fail(solver, t)
                if b:
                    fails.append(b)
        row = {"bc": bc, "regime": regime, "n_traj": len(trajs),
               "shape": list(trajs[0].fields.shape),
               "dtype": str(trajs[0].fields.dtype), "finite": bool(np.isfinite(f).all()),
               "channels": sorted({tuple(t.channels) for t in trajs})}
        for c in trajs[0].channels:
            fc = f[:, :, c]
            row[f"ch{c}_min"] = float(fc.min()); row[f"ch{c}_max"] = float(fc.max())
            d = np.abs(np.diff(fc, axis=1)).mean(axis=2) / np.maximum(
                np.abs(fc).mean(axis=(1, 2))[:, None], 1e-300)
            row[f"ch{c}_mean_rel_change_per_frame_1to8"] = float(d[:, :8].mean())
            row[f"ch{c}_mean_rel_change_per_frame_56to63"] = float(d[:, -8:].mean())
        if solver.name.startswith("sound"):
            row["energy_ratio_max"] = max(t.params["energy_ratio"] for t in trajs)
            row["energy_ratio_min"] = min(t.params["energy_ratio"] for t in trajs)
        if solver.name in ("chain", "thermoacoustic") and trajs[0].params.get("dim", 1) > 1:
            lo = solver.simulate_batch(copy.deepcopy(params), bc, N_FRAMES,
                                       np.random.default_rng(seed + 50 + i), n_internal=256)
            diffs = [float(common.rel_l2(a.fields[:, 2], b.fields[:, 2])) for a, b in zip(trajs, lo)]
            row["p_rel_l2_n256_vs_n512_max"] = max(diffs)
            if max(diffs) > 0.05:
                fails.append(f"n = 256 vs 512 pressure differs by {max(diffs):.3g}")
        if "mua_over_musp_max" in trajs[0].params:
            row["mua_over_musp_max"] = max(t.params["mua_over_musp_max"] for t in trajs)
            row["wavelengths_nm"] = sorted({t.params["wavelength_nm"] for t in trajs})
            row["ood_kinds"] = sorted({str(t.params.get("ood_kind")) for t in trajs})
        if solver.name == "light":
            row["start_over_t_tr_min"] = min(t.params["start_over_t_tr"] for t in trajs)
            row["init_width_over_l_tr_min"] = min(t.params["init_width_over_l_tr"] for t in trajs)
        if solver.name in ("photothermal", "thermoacoustic"):
            row["T_over_bound_max"] = max(float(t.fields[:, 1].max() / t.params["T_bound_K"])
                                          for t in trajs)
        if solver.name in ("chain", "thermoacoustic"):
            row["p_over_bound_max"] = max(float(np.abs(t.fields[:, 2]).max() / t.params["p_bound_1d_Pa"])
                                          for t in trajs)
        row["internal_dt_s"] = [t.params["dt"] for t in trajs]
        row["failures"] = fails
        row["passed"] = not fails
        ok &= not fails
        rows.append(row)
    return rows, ok


def timing(solver, bcs, n_traj=64, seed=100, extra_params=None, sample_kw=None):
    rows = []
    for i, bc in enumerate(bcs):
        rng = np.random.default_rng(seed + i)
        params = [solver.sample_params(rng, "train", **(sample_kw or {}))
                  for _ in range(n_traj)]
        if extra_params:
            for q in params:
                q.update(extra_params)
        t0 = time.perf_counter()
        trajs = solver.simulate_batch(params, bc, N_FRAMES, rng)
        el = time.perf_counter() - t0
        dts = np.array([t.params["dt"] for t in trajs])
        rows.append({"bc": bc, **({"sample_kw": sample_kw} if sample_kw else {}),
                     "n_traj": n_traj, "n_frames": N_FRAMES, "n_internal": 512,
                     "seconds": el, "seconds_per_trajectory": el / n_traj,
                     "rk4_substeps_per_frame_max": int(round(trajs[0].dt_frame / dts.min())),
                     "rk4_substeps_per_frame_min": int(round(trajs[0].dt_frame / dts.max()))})
    return rows


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "cpu_count": os.cpu_count(), "machine": platform.machine()}


def print_rep(rep, indent=""):
    print(f"{indent}{rep['system']} {rep['check']}: errors {rep['error']}, "
          f"orders {rep['observed_order']}, passed={rep['passed']}")
    for k, c in rep.get("cases", {}).items():
        print(f"{indent}  case {k}: errors {c['error']}, orders {c['observed_order']}, "
              f"passed={c['passed']}")


def sampling_stats(sample, regimes=("train", "ood"), n=5000, seed=2026):
    """Rejection rate of the mu_a/mu_s' cap and the light time-window ratios over n draws
    per regime (sampling only, no simulation). sample(rng, regime) -> params."""
    import collections
    from pfm.solvers import light as L
    out = {}
    for reg in regimes:
        rng = np.random.default_rng(seed)
        acc, rej = collections.Counter(), collections.Counter()
        st, wd, rat = [], [], []
        for _ in range(n):
            p = sample(rng, reg)
            k = str(p.get("ood_kind"))
            acc[k] += 1; rej[k] += p["n_rejected"]
            ttr = L.transport_time(p)
            ltr = 1.0 / (L.mua_background(p) + L.musp(p))
            D = L.speed_of_light(p) * ltr / 3.0
            st.append(L.T_START / ttr); wd.append(math.sqrt(4 * D * L.T_START) / ltr)
            rat.append(p["mua_over_musp_max_sampled"])
        tot = sum(rej.values())
        out[reg] = {"n_accepted": n, "n_rejected": int(tot),
                    "rejection_rate": tot / (tot + n),
                    "by_ood_kind": {k: {"accepted": acc[k], "rejected": int(rej[k]),
                                        "rejection_rate": rej[k] / (rej[k] + acc[k])}
                                    for k in acc},
                    "mua_over_musp_max_accepted": max(rat),
                    "light_start_over_t_tr_min": min(st),
                    "light_init_width_over_l_tr_min": min(wd),
                    "note": "light_* ratios refer to light.T_START and the background medium"}
    return out


def run(solver, bcs, module=None):
    t0 = time.perf_counter()
    rep = solver.verify()
    rep["verify_seconds"] = time.perf_counter() - t0
    rows, batch_ok = batch_check(solver, bcs)
    if solver.name == "light":
        rows_o, ok_o = batch_check(solver, bcs, seed=500, regime="ood")
        rows += rows_o
        batch_ok &= ok_o
    rep["smoke_test"] = {"passed": batch_ok, "rows": rows}
    rep["timing_64x64"] = timing(solver, bcs)
    rep["dt_frame_s"] = getattr(module, "DT_FRAME", None)
    rep["length_m"] = getattr(module, "LENGTH", None)
    if module is not None:
        rep["param_ranges"] = module.PARAM_RANGES
        rep["param_sources"] = module.PARAM_SOURCES
    rep["environment"] = environment()
    return rep, bool(rep["passed"] and batch_ok)


def report_and_exit(name, rep, ok):
    out = common.RESULTS / "verification" / f"{name}.json"
    common.save_json(rep, out)
    for k in ("smoke_test",):
        for r in rep[k]["rows"]:
            if r["failures"]:
                print(f"  smoke {r['bc']}: {r['failures']}")
    for r in rep["timing_64x64"]:
        print(f"  timing {r['bc']}: {r['seconds']:.2f} s for 64 x 64 "
              f"(substeps/frame {r['rk4_substeps_per_frame_min']}-{r['rk4_substeps_per_frame_max']})")
    print(f"  {name}: verify+smoke passed={ok}; wrote {out}")
    return 0 if ok else 1
