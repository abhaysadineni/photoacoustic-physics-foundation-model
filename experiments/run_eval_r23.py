"""Evaluation for the coupled-link (v2) and geometry (v3) comparisons on the held-out chain,
in slab (chain), cylinder (chain_r2) and sphere (chain_r3) geometry, and on the telegraph
equation. Definitions as in pfm/evaluate.py and PLAN.md sections 13, 14 and 16.

Chain input condition (primary): physical-unit scaling. Each channel of the chain, in its
physical unit, is divided by a constant taken from TRAINING systems of the same geometry:
channel 0 by the photothermal training RMS, channel 1 by the mean of the photothermal and
thermoacoustic training RMS, channel 2 by the thermoacoustic training RMS. Scoring units are
the chain's own training-split RMS (channel weights in the score only, never network input).
Secondary condition: seed-frame scaling (reference RMS from the training systems).
Telegraph: seed-frame scaling (it shares no physical unit with any training system).

Usage:
    python experiments/run_eval_r23.py --split test --models v2_s0 v3_s0 ... --out results/eval_r23/dev.json
    python experiments/run_eval_r23.py --split final_locked --models ... --out results/eval_r23/locked.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfm.common import ROOT, save_json                               # noqa: E402
from pfm.data import DATA, load, TRAIN_SYSTEMS_V2                     # noqa: E402
from pfm.evaluate import (load_model, summarise, seed_factors,        # noqa: E402
                          paired_bootstrap, per_channel)
from pfm.model import N_IN_FRAMES                                     # noqa: E402

torch.set_num_threads(1)
GEOMS = {"chain": "", "chain_r2": "_r2", "chain_r3": "_r3"}


def norm(system):
    return np.array(json.loads((DATA / system / "norm.json").read_text())["scale"], np.float32)


def phys_scale(suffix):
    pt, ta = norm("photothermal" + suffix), norm("thermoacoustic" + suffix)
    return np.array([pt[0], 0.5 * (pt[1] + ta[1]), ta[2]], np.float32)


def raw(system, split):
    return np.load(DATA / system / f"{split}.npz")["fields"].astype(np.float32)


def ref_seed_rms(systems):
    vals = {0: [], 1: [], 2: []}
    for s in systems:
        f = load(s, "train")
        rms = np.sqrt((f[:, :N_IN_FRAMES] ** 2).mean(axis=(1, 3)))
        occ = np.abs(f).max(axis=(1, 3)) > 0
        for c in range(3):
            vals[c] += rms[occ[:, c], c].tolist()
    return np.array([float(np.median(vals[c])) for c in range(3)], np.float32)


def strip(s):
    return {k: v for k, v in s.items() if not k.startswith("per_traj")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--geoms", nargs="+", default=list(GEOMS))
    ap.add_argument("--telegraph", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    models = {m: load_model(ROOT / "runs" / m / "model.pt") for m in a.models}
    ref = ref_seed_rms(TRAIN_SYSTEMS_V2)
    res = {"split": a.split, "models": a.models, "geoms": {}}
    for g in a.geoms:
        suf = GEOMS[g]
        own, phys = norm(g), phys_scale(suf)
        f = raw(g, a.split) / own[None, None, :, None]
        k_phys = np.repeat((own / phys)[None, :], len(f), 0).astype(np.float32)
        k_seed = seed_factors(f, ref)
        r = {"n": int(len(f)), "phys_factor": (own / phys).tolist(),
             "persistence": summarise(None, f)}
        for m, net in models.items():
            r[f"{m}_phys"] = summarise(net, f, k_phys)
            r[f"{m}_seed"] = summarise(net, f, k_seed)
            r[f"{m}_phys_per_channel"] = per_channel(net, f, k_phys)
        p = r["persistence"]
        for m in models:
            for cond in ("phys", "seed"):
                s = r[f"{m}_{cond}"]
                r[f"{m}_{cond}_vs_persistence"] = {
                    "ratio_one_step_median": s["one_step"]["median"] / p["one_step"]["median"],
                    "one_step": paired_bootstrap(s["per_traj_one_step"], p["per_traj_one_step"]),
                    "roll10": paired_bootstrap(s["per_traj_roll"]["10"], p["per_traj_roll"]["10"])}
        ms = list(models)
        for i, m1 in enumerate(ms):
            for m2 in ms[i + 1:]:
                r[f"{m1}_minus_{m2}_phys"] = {
                    "one_step": paired_bootstrap(r[f"{m1}_phys"]["per_traj_one_step"],
                                                 r[f"{m2}_phys"]["per_traj_one_step"]),
                    "roll10": paired_bootstrap(r[f"{m1}_phys"]["per_traj_roll"]["10"],
                                               r[f"{m2}_phys"]["per_traj_roll"]["10"])}
        res["geoms"][g] = r
        print(g, "done", flush=True)
    if a.telegraph:
        f = load("telegraph", a.split)
        k = seed_factors(f, ref)
        r = {"persistence": summarise(None, f)}
        for m, net in models.items():
            r[m] = summarise(net, f, k)
        res["telegraph"] = r
    save_json(res, ROOT / a.out)


if __name__ == "__main__":
    main()
