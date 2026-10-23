"""Error metrics, baselines and bootstrap intervals.

Error metric (revised after audit round 1, finding B1):
    e_t = ||pred_t - true_t|| / S,   S = sqrt( mean over the trajectory's recorded frames of ||true_f||^2 )
The norm is the L2 norm over the full 3 x 128 field. Dividing by a per-trajectory scale
rather than by the norm of the target frame keeps the error finite when a field leaves the
domain or decays (transport outflow, pressure release); the earlier per-frame denominator
let single trajectories dominate the mean by factors of 1e2 to 1e5. Unused channels are part
of the field, so non-zero predictions there count as error.

Scoring units: errors are always computed in the system's reference units (each channel
divided by its training-split RMS). For held-out systems that scale is used only to weight
channels in the score, never as network input (see input scaling below).

Input scaling for zero-shot tests (finding B2): with input_scale="seed", each trajectory's
channels are rescaled, before entering the network, by a factor computed only from that
trajectory's own four seed frames: k_c = r_c / s_c, where s_c is the RMS of channel c over
frames 0 to 3 and r_c is the median seed-frame RMS of channel c over all training-system
trajectories that occupy it. Outputs are divided by k_c before scoring. Channels that are
exactly zero in the seed frames are left unscaled.

One-step error: for every start t = 0..F-5, predict frame t+4 from frames t..t+3; mean over t
per trajectory. Rollout error: seed with frames 0..3, roll forward, error at steps 10, 25, 50
(frames 13, 28, 53). Summaries give mean and median over trajectories, each with a 95 per
cent percentile bootstrap interval over trajectories (2000 resamples, fixed seed).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .model import Surrogate, N_IN_FRAMES

ROLL_STEPS = (10, 25, 50)


def load_model(path, use_derivs=True, arch=None) -> Surrogate:
    """arch: dict(width, depth, heads); read from the run's log.json when not given."""
    if arch is None:
        import json
        from pathlib import Path
        lg = Path(path).parent / "log.json"
        arch = json.loads(lg.read_text()).get("arch") if lg.exists() else None
        if lg.exists():                       # a run trained without derivative channels
            use_derivs = json.loads(lg.read_text()).get("use_derivs", use_derivs)
    m = Surrogate(use_derivs=use_derivs, **(arch or {}))
    m.load_state_dict(torch.load(path, weights_only=True))
    m.eval()
    return m


def traj_scale(fields: np.ndarray) -> np.ndarray:
    """Per-trajectory scale S (N,): RMS over recorded frames of the frame L2 norm."""
    n2 = (fields ** 2).reshape(fields.shape[0], fields.shape[1], -1).sum(-1)
    return np.sqrt(n2.mean(1))


def _err(pred, true, scale):
    num = np.sqrt(((pred - true) ** 2).reshape(pred.shape[0], -1).sum(1))
    return num / np.maximum(scale, 1e-30)


def seed_factors(fields: np.ndarray, ref_seed_rms: np.ndarray) -> np.ndarray:
    """k (N, 3): per-trajectory, per-channel input factors from the four seed frames only."""
    s = np.sqrt((fields[:, :N_IN_FRAMES] ** 2).mean(axis=(1, 3)))      # (N, 3)
    k = np.ones_like(s)
    nz = s > 0
    k[nz] = (ref_seed_rms[None, :].repeat(len(s), 0))[nz] / s[nz]
    return k.astype(np.float32)


def bootstrap(per_traj: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
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


def paired_bootstrap(a, b, n_boot: int = 2000, seed: int = 0) -> dict:
    """Difference a - b of per-trajectory errors of two models on the SAME trajectories,
    with the same resample indices for both (audit round 1, MINOR 4)."""
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


def per_channel(model, fields, k=None, steps=(1, 10, 25, 50)) -> dict:
    """Per-channel error for multi-channel systems (audit MINOR 3): for each channel c,
    ||pred_c - true_c|| / S_c with S_c the RMS over recorded frames of ||true_c||, at the
    given rollout steps (step 1 = first predicted frame of the rollout)."""
    n_max = max(steps)
    pred = rollout_pred(model, fields, n_max, k)
    out = {}
    for c in range(fields.shape[2]):
        S = np.sqrt((fields[:, :, c] ** 2).sum(-1).mean(1))
        if not np.any(S > 0):
            continue
        r = {}
        for s in steps:
            e = np.sqrt(((pred[:, s - 1, c] - fields[:, N_IN_FRAMES - 1 + s, c]) ** 2).sum(-1))
            r[str(s)] = bootstrap(e / np.maximum(S, 1e-30))
        out[str(c)] = r
    return out


@torch.no_grad()
def _predict(model, x, k=None, batch=512):
    """x (B, 4, 3, 128) in scoring units; k (B, 3) input factors or None."""
    if k is not None:
        x = x * k[:, None, :, None]
    out = np.concatenate([model(torch.from_numpy(np.ascontiguousarray(x[j:j + batch]))).numpy()
                          for j in range(0, len(x), batch)])
    if k is not None:
        out = out / k[:, :, None]
    return out


@torch.no_grad()
def one_step(model, fields: np.ndarray, k=None) -> np.ndarray:
    """Per-trajectory mean one-step error. model=None gives the persistence baseline."""
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
def rollout_pred(model, fields: np.ndarray, n_steps: int, k=None) -> np.ndarray:
    if model is None:
        return np.repeat(fields[:, N_IN_FRAMES - 1:N_IN_FRAMES], n_steps, axis=1)
    x0 = fields[:, :N_IN_FRAMES]
    if k is not None:
        x0 = x0 * k[:, None, :, None]
    pred = model.rollout(torch.from_numpy(np.ascontiguousarray(x0)), n_steps).numpy()
    if k is not None:
        pred = pred / k[:, None, :, None]
    return pred


def rollout(model, fields: np.ndarray, steps=ROLL_STEPS, k=None) -> dict:
    n_max = max(steps)
    S = traj_scale(fields)
    pred = rollout_pred(model, fields, n_max, k)
    curve = np.stack([_err(pred[:, j], fields[:, N_IN_FRAMES + j], S) for j in range(n_max)], 1)
    out = {s: curve[:, s - 1] for s in steps}
    out["curve"] = curve
    return out


def summarise(model, fields, k=None) -> dict:
    os_ = one_step(model, fields, k)
    ro = rollout(model, fields, k=k)
    return {"one_step": bootstrap(os_),
            "rollout": {str(s): bootstrap(ro[s]) for s in ROLL_STEPS},
            "rollout_curve_mean": ro["curve"].mean(0).tolist(),
            "rollout_curve_median": np.median(ro["curve"], 0).tolist(),
            "per_traj_one_step": os_.tolist(),
            "per_traj_roll": {str(s): ro[s].tolist() for s in ROLL_STEPS}}


@torch.no_grad()
def time_model(model, n_samples: int, n_steps: int = 50, repeats: int = 3,
               threads: int = 1, seed_frames=None) -> float:
    """Wall seconds for an n_steps rollout of n_samples trajectories (best of repeats),
    with the thread count fixed so it matches the solver timing. seed_frames, if given, is
    an array (N, 4, 3, 128) of real seed frames (the cost does not depend on the values)."""
    torch.set_num_threads(threads)
    if seed_frames is not None:
        x0 = torch.from_numpy(np.ascontiguousarray(seed_frames[:n_samples]))
    else:
        x0 = torch.randn(n_samples, N_IN_FRAMES, 3, 128)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.rollout(x0, n_steps)
        best = min(best, time.perf_counter() - t0)
    return best
