"""Error metrics, baselines and bootstrap intervals for d = 2, 3, as pfm/evaluate.py.

Error metric: e_t = ||pred_t - true_t|| / S, S = sqrt(mean over the trajectory's recorded
frames of ||true_f||^2), L2 norm over the full 3 x grid field (unused channels included),
in the system's reference units (each channel divided by its training-split RMS).

Input scaling for zero-shot tests:
  seed      k_c = r_c / s_c from the trajectory's own four seed frames (s_c their RMS of
            channel c; r_c the median seed-frame RMS of channel c over training-system
            trajectories occupying it); channels zero in the seed frames are left unscaled.
  physical  (PLAN section 14) each channel in its physical unit is divided by a constant
            from TRAINING systems carrying that channel in the same unit: channel 0 by the
            photothermal training RMS, channel 1 by the mean of the photothermal and
            thermoacoustic training RMS, channel 2 by the thermoacoustic training RMS. In
            reference units this is k_c = s_held,c / ref_c (physical_factors).
Outputs are divided by k before scoring.

One-step error: every start t = 0 .. F-5, mean over t per trajectory. Rollout: seeds 0..3,
errors at ROLL_STEPS (d = 2: 10, 25, 28 steps of the 32-frame record; d = 3: 10, 20 of the
24-frame record; the 1D steps 25 and 50 do not fit, 28 and 20 are the longest possible) and
the full curve. Mean and median over trajectories with 95 per cent percentile bootstrap
intervals (2000 resamples, fixed seed); paired bootstrap for differences on the same
trajectories.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import common as C
from .data import Store, system_dir, TRAIN_SYSTEMS
from .model import Surrogate, N_IN_FRAMES

ROLL_STEPS = {2: (10, 25, 28), 3: (10, 20)}


def load_model(path, dim=None, use_derivs=True, arch=None):
    lg = Path(path).parent / "log.json"
    if lg.exists():
        L = json.loads(lg.read_text())
        arch = arch or L.get("arch")
        use_derivs = L.get("use_derivs", use_derivs)
        dim = dim or L.get("dim")
    m = Surrogate(dim or 2, use_derivs=use_derivs, **(arch or {}))
    m.load_state_dict(torch.load(path, weights_only=True))
    m.eval()
    return m


def _flat(a, lead):
    return a.reshape(*a.shape[:lead], -1)


def traj_scale(fields):
    n2 = (_flat(fields, 2) ** 2).sum(-1)
    return np.sqrt(n2.mean(1))


def _err(pred, true, scale):
    num = np.sqrt((_flat(pred - true, 1) ** 2).sum(1))
    return num / np.maximum(scale, 1e-30)


def _bk(k, d):
    """k (N, 3) -> broadcast over (N, [frames,] 3, *grid)."""
    return k.reshape(k.shape + (1,) * d)


def seed_factors(fields, ref_seed_rms):
    d = fields.ndim - 3
    s = np.sqrt((fields[:, :N_IN_FRAMES] ** 2).mean(axis=(1,) + tuple(range(3, 3 + d))))
    k = np.ones_like(s)
    nz = s > 0
    k[nz] = np.broadcast_to(ref_seed_rms[None, :], s.shape)[nz] / s[nz]
    return k.astype(np.float32)


def physical_reference_scale(dim, root=None):
    """ref_c from the training systems' norm.json (photothermal, thermoacoustic)."""
    sc = lambda s: np.array(json.loads((system_dir(s, dim, root) / "norm.json").read_text())["scale"])
    pt, ta = sc("photothermal"), sc("thermoacoustic")
    return np.array([pt[0], 0.5 * (pt[1] + ta[1]), ta[2]], np.float32)


def physical_factors(held_scale, ref_scale, n):
    k = np.asarray(held_scale, np.float32) / np.asarray(ref_scale, np.float32)
    return np.repeat(k[None], n, 0)


def reference_seed_rms(dim, systems=TRAIN_SYSTEMS, root=None, split="train"):
    vals = {0: [], 1: [], 2: []}
    for s in systems:
        st = Store(s, split, dim, root)
        for i in range(st.n):
            w = st.window(i, 0, N_IN_FRAMES)
            rms = np.sqrt((w ** 2).reshape(N_IN_FRAMES, 3, -1).mean(axis=(0, 2)))
            for c in st.chan[i]:
                vals[int(c)].append(float(rms[int(c)]))
    return np.array([float(np.median(vals[c])) if vals[c] else 1.0 for c in range(3)], np.float32)


def bootstrap(per_traj, n_boot=2000, seed=0):
    per_traj = np.asarray(per_traj, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(per_traj), size=(n_boot, len(per_traj)))
    res = per_traj[idx]
    means, meds = res.mean(1), np.median(res, 1)
    return {"mean": float(per_traj.mean()),
            "mean_ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "median": float(np.median(per_traj)),
            "median_ci": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))],
            "n": int(len(per_traj))}


def paired_bootstrap(a, b, n_boot=2000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(d), size=(n_boot, len(d)))
    dm = d[idx].mean(1)
    medd = np.median(a[idx], 1) - np.median(b[idx], 1)
    return {"mean_diff": float(d.mean()),
            "mean_diff_ci": [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))],
            "median_diff": float(np.median(a) - np.median(b)),
            "median_diff_ci": [float(np.percentile(medd, 2.5)), float(np.percentile(medd, 97.5))],
            "frac_a_better": float(np.mean(a < b)), "n": int(len(d))}


@torch.no_grad()
def _predict(model, x, k=None, batch=32):
    d = x.ndim - 3
    if k is not None:
        x = x * k.reshape(k.shape[0], 1, 3, *([1] * d))
    out = np.concatenate([model(torch.from_numpy(np.ascontiguousarray(x[j:j + batch]))).numpy()
                          for j in range(0, len(x), batch)])
    if k is not None:
        out = out / _bk(k, d)
    return out


@torch.no_grad()
def one_step(model, fields, k=None):
    n, f = fields.shape[:2]
    S = traj_scale(fields)
    starts = np.arange(f - N_IN_FRAMES)
    errs = np.zeros((n, len(starts)))
    for i in range(n):
        x = np.stack([fields[i, t:t + N_IN_FRAMES] for t in starts])
        y = fields[i, starts + N_IN_FRAMES]
        if model is None:
            p = x[:, -1]
        else:
            kk = None if k is None else np.repeat(k[i:i + 1], len(x), 0)
            p = _predict(model, x, kk)
        errs[i] = _err(p, y, np.full(len(x), S[i]))
    return errs.mean(1)


@torch.no_grad()
def rollout_pred(model, fields, n_steps, k=None, batch=16):
    d = fields.ndim - 3
    if model is None:
        return np.repeat(fields[:, N_IN_FRAMES - 1:N_IN_FRAMES], n_steps, axis=1)
    x0 = fields[:, :N_IN_FRAMES]
    if k is not None:
        x0 = x0 * k.reshape(k.shape[0], 1, 3, *([1] * d))
    pred = np.concatenate([model.rollout(torch.from_numpy(np.ascontiguousarray(x0[j:j + batch])),
                                         n_steps).numpy() for j in range(0, len(x0), batch)])
    if k is not None:
        pred = pred / k.reshape(k.shape[0], 1, 3, *([1] * d))
    return pred


def rollout(model, fields, steps=None, k=None):
    d = fields.ndim - 3
    steps = steps or ROLL_STEPS[d]
    n_max = max(steps)
    S = traj_scale(fields)
    pred = rollout_pred(model, fields, n_max, k)
    curve = np.stack([_err(pred[:, j], fields[:, N_IN_FRAMES + j], S) for j in range(n_max)], 1)
    out = {s: curve[:, s - 1] for s in steps}
    out["curve"] = curve
    return out


def per_channel(model, fields, k=None, steps=None):
    d = fields.ndim - 3
    steps = steps or (1,) + ROLL_STEPS[d]
    pred = rollout_pred(model, fields, max(steps), k)
    out = {}
    for c in range(fields.shape[2]):
        S = np.sqrt((_flat(fields[:, :, c], 2) ** 2).sum(-1).mean(1))
        if not np.any(S > 0):
            continue
        r = {}
        for s in steps:
            e = np.sqrt((_flat(pred[:, s - 1, c] - fields[:, N_IN_FRAMES - 1 + s, c], 1) ** 2).sum(-1))
            r[str(s)] = bootstrap(e / np.maximum(S, 1e-30))
        out[str(c)] = r
    return out


def summarise(model, fields, k=None):
    d = fields.ndim - 3
    os_ = one_step(model, fields, k)
    ro = rollout(model, fields, k=k)
    return {"one_step": bootstrap(os_),
            "rollout": {str(s): bootstrap(ro[s]) for s in ROLL_STEPS[d]},
            "rollout_curve_mean": ro["curve"].mean(0).tolist(),
            "rollout_curve_median": np.median(ro["curve"], 0).tolist(),
            "per_traj_one_step": os_.tolist(),
            "per_traj_roll": {str(s): ro[s].tolist() for s in ROLL_STEPS[d]}}


@torch.no_grad()
def time_model(model, n_samples, n_steps=10, repeats=3, threads=1, seed_frames=None):
    torch.set_num_threads(threads)
    d = model.dim
    if seed_frames is not None:
        x0 = torch.from_numpy(np.ascontiguousarray(seed_frames[:n_samples]))
    else:
        x0 = torch.randn(n_samples, N_IN_FRAMES, 3, *([model.n_x] * d))
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.rollout(x0, n_steps)
        best = min(best, time.perf_counter() - t0)
    return best


def time_train_step(dim, batch, threads=1, repeats=3, **arch):
    """Wall seconds of one training step (forward, backward, clip, AdamW) on random input,
    best of repeats after one warm-up step."""
    torch.set_num_threads(threads)
    m = Surrogate(dim, **arch)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    n = m.n_x
    x = torch.randn(batch, N_IN_FRAMES, 3, *([n] * dim))
    y = torch.randn(batch, 3, *([n] * dim))
    best = float("inf")
    for r in range(repeats + 1):
        t0 = time.perf_counter()
        loss = torch.mean((m(x) - y) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        el = time.perf_counter() - t0
        if r > 0:
            best = min(best, el)
    return best
