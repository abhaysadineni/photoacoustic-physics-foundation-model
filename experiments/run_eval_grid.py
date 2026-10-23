"""Evaluation of the grid networks (rounds 4 and 5, d = 2 and d = 3) on the held-out grid chain,
mirroring experiments/run_eval_r23.py. Definitions as pfm2d/evaluate.py and PLAN.md sections
14 and 23.

Chain input condition (primary): physical-unit scaling. Each chain channel in its physical unit
is divided by a constant from TRAINING systems of the same dimension that carry the channel in
the same unit: channel 0 by the photothermal training RMS, channel 1 by the mean of the
photothermal and thermoacoustic training RMS, channel 2 by the thermoacoustic training RMS
(norm.json of those systems). Scores are in the chain's own training-split RMS units (channel
weights in the score only, never network input). Secondary condition: seed-frame scaling with
the reference seed-frame RMS of the training systems (results/datagen_grid/ref_seed_rms_d<d>.json,
written by the GPU data job from the training splits).

Per model and condition: one-step error (every start, mean over starts per trajectory), rollout
errors at pfm2d.evaluate.ROLL_STEPS (d = 2: 10, 25, 28; d = 3: 10, 20, the 24-frame record
leaves at most 20 rollout steps, so 25 does not exist in 3D), the median curve, per-channel
errors, paired bootstrap (2000 resamples) against persistence and between models; a group
(--group name=m1,m2,m3) is the per-trajectory mean error of its members (the three-seed mean).

Diagnostic condition "physmaskL" (pre-registered in PLAN.md section 23 after audit finding M4,
section 21a): physical-unit scaling as "phys", but the pressure output (channel 2) of every
step is taken from a second forward pass with the light input (channel 0, all four frames)
set to zero; channels 0 and 1 come from the ordinary pass; rollouts feed back the composite.
It uses knowledge of the chain's causal structure (pressure is not driven by light directly),
so it is reported as a diagnostic and never enters S1-g to S3-g.

Usage:
  python experiments/run_eval_grid.py --dim 2 --split test --models g2_s0 g2_s1 g2_s2 g2nc_s0 \
      g2oracle --group g2=g2_s0,g2_s1,g2_s2 --out results/eval_grid/dev_d2.json
Models are read from runs_grid/<name>/model.pt (log.json beside it gives the architecture).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

from pfm2d import common as C                                         # noqa: E402
from pfm2d.data import Store, system_dir                              # noqa: E402
from pfm2d import evaluate as E                                       # noqa: E402

torch.set_num_threads(int(os.environ.get("PFM_EVAL_THREADS", "1")))
RUNS = C.ROOT / "runs_grid"


class MaskLightForPressure(torch.nn.Module):
    """Composite predictor: channels 0, 1 from net(x); channel 2 from net(x with channel 0 = 0)."""

    def __init__(self, net):
        super().__init__()
        self.net, self.dim, self.n_x = net, net.dim, net.n_x

    def forward(self, x):
        full = self.net(x)
        xm = x.clone()
        xm[:, :, 0] = 0.0
        out = full.clone()
        out[:, 2] = self.net(xm)[:, 2]
        return out

    @torch.no_grad()
    def rollout(self, x0, n_steps):
        window, preds = x0, []
        for _ in range(n_steps):
            nxt = self(window)
            preds.append(nxt)
            window = torch.cat([window[:, 1:], nxt[:, None]], dim=1)
        return torch.stack(preds, dim=1)


def ensure_norm(system, dim):
    """norm.json of a system comes from its training split, which exists only on the GPU
    machine; the GPU data job copies it to results/datagen_grid/norm_d<d>_<system>.json. Install
    that copy as data2d/.../<system>/norm.json here if it is missing (never overwritten)."""
    dst = system_dir(system, dim) / "norm.json"
    src = C.ROOT / "results" / "datagen_grid" / f"norm_d{dim}_{system}.json"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text())


def norm(system, dim):
    ensure_norm(system, dim)
    return np.array(json.loads((system_dir(system, dim) / "norm.json").read_text())["scale"],
                    np.float32)


def summarise(model, f, k, dim):
    """One-step, rollouts and per-channel errors from ONE rollout (pfm2d.evaluate functions)."""
    steps = E.ROLL_STEPS[dim]
    os_ = E.one_step(model, f, k)
    n_max = max(steps)
    S = E.traj_scale(f)
    pred = E.rollout_pred(model, f, n_max, k)
    curve = np.stack([E._err(pred[:, j], f[:, E.N_IN_FRAMES + j], S) for j in range(n_max)], 1)
    pc = {}
    for c in range(f.shape[2]):
        Sc = np.sqrt((E._flat(f[:, :, c], 2) ** 2).sum(-1).mean(1))
        r = {}
        for s in (1,) + tuple(steps):
            e = np.sqrt((E._flat(pred[:, s - 1, c] - f[:, E.N_IN_FRAMES - 1 + s, c], 1) ** 2).sum(-1))
            r[str(s)] = E.bootstrap(e / np.maximum(Sc, 1e-30))
        pc[str(c)] = r
    return {"one_step": E.bootstrap(os_),
            "rollout": {str(s): E.bootstrap(curve[:, s - 1]) for s in steps},
            "rollout_curve_median": np.median(curve, 0).tolist(),
            "rollout_curve_mean": curve.mean(0).tolist(),
            "per_channel_rollout_from_seed": pc,
            "per_traj_one_step": os_.tolist(),
            "per_traj_roll": {str(s): curve[:, s - 1].tolist() for s in steps}}


def versus(a, b, dim):
    out = {"one_step": E.paired_bootstrap(a["per_traj_one_step"], b["per_traj_one_step"])}
    for s in E.ROLL_STEPS[dim]:
        out[f"roll{s}"] = E.paired_bootstrap(a["per_traj_roll"][str(s)], b["per_traj_roll"][str(s)])
    return out


def group_of(members, dim):
    g = {"per_traj_one_step": np.mean([m["per_traj_one_step"] for m in members], 0).tolist(),
         "per_traj_roll": {str(s): np.mean([m["per_traj_roll"][str(s)] for m in members], 0).tolist()
                           for s in E.ROLL_STEPS[dim]}}
    g["one_step"] = E.bootstrap(g["per_traj_one_step"])
    g["rollout"] = {s: E.bootstrap(v) for s, v in g["per_traj_roll"].items()}
    return g


def strip(d):
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if not k.startswith("per_traj")}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--group", action="append", default=[], help="name=m1,m2,...")
    ap.add_argument("--no-seed-condition", action="store_true")
    ap.add_argument("--no-mask-condition", action="store_true")
    ap.add_argument("--cache", default=None, help="development only: directory of per-model "
                    "results keyed by the model file's SHA-256, split, dim and condition")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cache and a.split == "final_locked":
        raise SystemExit("--cache is for development splits only")
    dim = a.dim
    t0 = time.time()
    models = {m: E.load_model(RUNS / m / "model.pt", dim=dim) for m in a.models}
    ensure_norm("chain", dim)
    st = Store("chain", a.split, dim)
    f = st.full()                                                   # chain's own RMS units
    own = st.scale
    pt, ta = norm("photothermal", dim), norm("thermoacoustic", dim)
    phys = np.array([pt[0], 0.5 * (pt[1] + ta[1]), ta[2]], np.float32)
    k_phys = np.repeat((own / phys)[None, :], len(f), 0).astype(np.float32)
    conds = {"phys": k_phys}
    ref_path = C.ROOT / "results" / "datagen_grid" / f"ref_seed_rms_d{dim}.json"
    if not a.no_seed_condition:
        ref = np.array(json.loads(ref_path.read_text())["ref_seed_rms"], np.float32)
        conds["seed"] = E.seed_factors(f, ref)
    res = {"dim": dim, "split": a.split, "n": int(len(f)), "models": a.models,
           "groups": a.group, "roll_steps": list(E.ROLL_STEPS[dim]),
           "chain_own_scale": own.tolist(), "phys_reference_scale": phys.tolist(),
           "phys_factor": (own / phys).tolist(), "bc_counts": {b: int((st.bc == b).sum())
                                                               for b in sorted(set(st.bc.tolist()))}}
    full = {"persistence": summarise(None, f, None, dim)}
    p = full["persistence"]
    for m, net in models.items():
        runs = [(c, net, k) for c, k in conds.items()]
        if not a.no_mask_condition:
            runs.append(("physmaskL", MaskLightForPressure(net).eval(), k_phys))
        for c, mdl, k in runs:
            cf = None
            if a.cache:
                import hashlib
                hsh = hashlib.sha256((RUNS / m / "model.pt").read_bytes()).hexdigest()[:16]
                cf = Path(a.cache) / f"{m}_{hsh}_d{dim}_{a.split}_{c}.json"
            if cf is not None and cf.exists():
                full[f"{m}_{c}"] = json.loads(cf.read_text())
            else:
                full[f"{m}_{c}"] = summarise(mdl, f, k, dim)
                if cf is not None:
                    cf.parent.mkdir(parents=True, exist_ok=True)
                    cf.write_text(json.dumps(full[f"{m}_{c}"], default=float))
            print(m, c, "one-step median", round(full[f"{m}_{c}"]["one_step"]["median"], 5),
                  f"{time.time() - t0:.0f} s", flush=True)
    if not a.no_mask_condition:
        conds["physmaskL"] = k_phys
    for gspec in a.group:
        gname, mem = gspec.split("=")
        for c in conds:
            full[f"{gname}_{c}"] = group_of([full[f"{m}_{c}"] for m in mem.split(",")], dim)
    keys = [k for k in full if k != "persistence"]
    comp = {}
    for k in keys:
        comp[f"{k}_vs_persistence"] = versus(full[k], p, dim)
        comp[f"{k}_vs_persistence"]["ratio_one_step_median"] = (
            full[k]["one_step"]["median"] / p["one_step"]["median"])
        comp[f"{k}_vs_persistence"]["ratio_roll10_median"] = (
            full[k]["rollout"]["10"]["median"] / p["rollout"]["10"]["median"])
    for c in conds:
        ck = [k for k in keys if k.endswith("_" + c)]
        for i, k1 in enumerate(ck):
            for k2 in ck[i + 1:]:
                comp[f"{k1}_minus_{k2}"] = versus(full[k1], full[k2], dim)
    per_bc = {}
    for bc in sorted(set(st.bc.tolist())):
        m = st.bc == bc
        pb = float(np.median(np.asarray(p["per_traj_one_step"])[m]))
        per_bc[bc] = {"n": int(m.sum()), "persistence_one_step_median": pb}
        for k in keys:
            v = float(np.median(np.asarray(full[k]["per_traj_one_step"])[m]))
            per_bc[bc][k] = {"one_step_median": v, "ratio_to_persistence": v / pb}
    res["per_bc"] = per_bc
    res["results"] = strip(full)
    res["comparisons"] = comp
    res["per_trajectory"] = {k: {"one_step": v["per_traj_one_step"], "roll": v["per_traj_roll"]}
                             for k, v in full.items()}
    res["seconds"] = time.time() - t0
    C.save_json(res, C.ROOT / a.out)
    print("wrote", a.out, f"{res['seconds']:.0f} s")


if __name__ == "__main__":
    main()
