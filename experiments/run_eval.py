"""Evaluate every trained network and baseline; write results/eval/<experiment>.json.

Experiments (PLAN.md section 9):
  exp1  in-distribution: main (all seeds), specialist, persistence on each training system's test split
  exp2  zero-shot: held-out systems (chain, telegraph) with seed-frame input scaling (primary)
        and reference-unit input (secondary); oracle; persistence. Held-out boundaries
        (test_bc) and out-of-range parameters (test_ood) for every training system.
  exp3  rollout curves (contained in every summary: steps 10, 25, 50 and full curves)
  exp4  scaling: 3, 5, 8 training systems on the held-out systems
  exp5  fine-tuning: k = 1, 5, 25 from main_s0 versus from scratch
  exp6  wall time: solver versus network, 1 and 64 trajectories, one thread
  exp7  derivative-channel ablation
  exp8  geometry: chain radial profiles in 2D and 3D, pressure release and rigid walls

Usage: python experiments/run_eval.py exp1 exp2 ...   (no argument runs all)
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfm.common import ROOT, RESULTS, save_json                       # noqa: E402
from pfm.data import (TRAIN_SYSTEMS, HELD_OUT, SCALING_SUBSETS, DATA,  # noqa: E402
                      load, get_solver, N_FRAMES)
from pfm.evaluate import (load_model, summarise, seed_factors,         # noqa: E402
                          time_model, bootstrap, paired_bootstrap, per_channel)
from pfm.model import N_IN_FRAMES                                      # noqa: E402

torch.set_num_threads(1)
RUNS = ROOT / "runs"
OUT = RESULTS / "eval"
SEEDS = [s for s in (0, 1, 2) if (RUNS / f"main_s{s}" / "model.pt").exists()]


def have(name):
    return (RUNS / name / "model.pt").exists()


def model(name, derivs=True):
    return load_model(RUNS / name / "model.pt", use_derivs=derivs)


def meta(system, split):
    return json.loads((DATA / system / f"{split}_meta.json").read_text())


def ref_seed_rms():
    """Median seed-frame RMS per channel over all training-system train trajectories
    occupying that channel (reference units). Uses training systems only."""
    vals = {0: [], 1: [], 2: []}
    for s in TRAIN_SYSTEMS:
        f = load(s, "train")
        rms = np.sqrt((f[:, :N_IN_FRAMES] ** 2).mean(axis=(1, 3)))          # (N, 3)
        occ = np.abs(f).reshape(f.shape[0], f.shape[1], 3, -1).max(axis=(1, 3)) > 0
        for c in range(3):
            vals[c] += rms[occ[:, c], c].tolist()
    return np.array([float(np.median(vals[c])) for c in range(3)], np.float32)


def strip(summary):
    """Drop per-trajectory arrays for the compact files; keep them in the full file."""
    return {k: v for k, v in summary.items() if not k.startswith("per_traj")}


# ---------------------------------------------------------------------- exp1
def exp1():
    res = {}
    for s in TRAIN_SYSTEMS:
        f = load(s, "test")
        r = {"persistence": summarise(None, f)}
        for sd in SEEDS:
            r[f"main_s{sd}"] = summarise(model(f"main_s{sd}"), f)
        if have(f"spec_{s}"):
            r["specialist"] = summarise(model(f"spec_{s}"), f)
            r["paired_main_s0_minus_specialist"] = {
                "one_step": paired_bootstrap(r["main_s0"]["per_traj_one_step"],
                                             r["specialist"]["per_traj_one_step"]),
                **{f"roll{st}": paired_bootstrap(r["main_s0"]["per_traj_roll"][st],
                                                  r["specialist"]["per_traj_roll"][st])
                   for st in ("10", "25", "50")}}
        res[s] = r
        print("exp1", s, flush=True)
    save_json(res, OUT / "exp1_in_distribution.json")


# ---------------------------------------------------------------------- exp2
def ood_flags(system):
    """Per-trajectory flag: True if the quantity that sets the dynamics lies OUTSIDE the
    training range (audit round 1, M4). None where the primary parameter is that quantity."""
    m = meta(system, "test_ood")
    if system == "fisher":
        lo, hi = 2 * np.sqrt(0.001 * 2.0), 2 * np.sqrt(0.01 * 20.0)
        v = np.array([x["params"]["front_speed_min"] for x in m])
        return ("front speed 2 sqrt(rD)", ((v < lo) | (v > hi)).tolist())
    if system == "burgers":
        v = np.array([x["params"]["Re"] for x in m])
        return ("Reynolds number u_max L / nu", ((v < 8.3) | (v > 250.0)).tolist())
    return (None, None)


def exp2():
    ref = ref_seed_rms()
    res = {"ref_seed_rms": ref.tolist(), "held_out": {}, "boundary": {}, "out_of_range": {}}
    for s in HELD_OUT:
        f = load(s, "test")
        k = seed_factors(f, ref)
        r = {"persistence": summarise(None, f)}
        for sd in SEEDS:
            r[f"main_s{sd}_seedscaled"] = summarise(model(f"main_s{sd}"), f, k)
            r[f"main_s{sd}_refunits"] = summarise(model(f"main_s{sd}"), f)
        if have(f"oracle_{s}"):
            r["oracle"] = summarise(model(f"oracle_{s}"), f)
        if s == "telegraph":
            g = np.array([x["params"]["gamma_over_c_per_L"] for x in meta(s, "test")])
            r["strata_g"] = {"bins": ["g<0.3", "0.3<=g<3", "g>=3"],
                             "membership": np.digitize(g, [0.3, 3.0], right=False).tolist()}
        if s == "chain":
            r["per_channel_main_s0_seedscaled"] = per_channel(model("main_s0"), f, k)
            r["per_channel_persistence"] = per_channel(None, f)
            if have("oracle_chain"):
                r["per_channel_oracle"] = per_channel(model("oracle_chain"), f)
        # held-out systems under their held-out boundary and out-of-range parameters
        for split in ("test_bc", "test_ood"):
            g2 = load(s, split)
            k2 = seed_factors(g2, ref)
            rr = {"persistence": strip(summarise(None, g2))}
            for sd in SEEDS:
                rr[f"main_s{sd}_seedscaled"] = strip(summarise(model(f"main_s{sd}"), g2, k2))
            if have(f"oracle_{s}"):
                rr["oracle"] = strip(summarise(model(f"oracle_{s}"), g2))
            r[split] = rr
        res["held_out"][s] = r
        print("exp2 held-out", s, flush=True)
    for s in TRAIN_SYSTEMS:
        for split, key in (("test_bc", "boundary"), ("test_ood", "out_of_range")):
            f = load(s, split)
            r = {"persistence": summarise(None, f)}
            for sd in SEEDS:
                r[f"main_s{sd}"] = summarise(model(f"main_s{sd}"), f)
            if have(f"spec_{s}"):
                r["specialist"] = summarise(model(f"spec_{s}"), f)
            if key == "boundary" and s == "transport":
                r["excluded_from_aggregate"] = ("transport dirichlet and neumann are numerically "
                                                "identical (audit M1); not a boundary test")
            if key == "out_of_range":
                name, flags = ood_flags(s)
                if flags is not None:
                    r["dynamics_flag"] = {"quantity": name, "outside": flags,
                                          "fraction_outside": float(np.mean(flags))}
            res[key][s] = r
            print("exp2", key, s, flush=True)
    save_json(res, OUT / "exp2_zero_shot.json")


# ---------------------------------------------------------------------- exp4
def exp4():
    ref = ref_seed_rms()
    res = {}
    for s in HELD_OUT:
        f = load(s, "test")
        k = seed_factors(f, ref)
        r = {}
        for n_sys in (3, 5, 8):
            tag = "main" if n_sys == 8 else f"scale{n_sys}"
            for sd in (0, 1, 2):
                if have(f"{tag}_s{sd}"):
                    r[f"{n_sys}_s{sd}"] = strip(summarise(model(f"{tag}_s{sd}"), f, k))
        res[s] = r
        print("exp4", s, flush=True)
    res["subsets"] = SCALING_SUBSETS
    save_json(res, OUT / "exp4_scaling.json")


# ---------------------------------------------------------------------- exp5
def exp5():
    res = {}
    for s in HELD_OUT:
        f = load(s, "test")
        r = {}
        for kk in (1, 5, 25):
            for tag in ("ft", "scratch"):
                n = f"{tag}_{s}_k{kk}"
                if have(n):
                    r[n] = strip(summarise(model(n), f))
        if have(f"oracle_{s}"):
            r["oracle"] = strip(summarise(model(f"oracle_{s}"), f))
        res[s] = r
        print("exp5", s, flush=True)
    save_json(res, OUT / "exp5_finetune.json")


# ---------------------------------------------------------------------- exp6
def exp6():
    """Solver: simulate_batch for 54 frames (4 seed + 50 predicted), one thread.
    Network: 50-step rollout from 4 seed frames, one thread. Best of 3 repeats each."""
    m = model("main_s0")
    res = {"threads": 1, "frames_solver": 54, "steps_network": 50, "systems": {}}
    for s in TRAIN_SYSTEMS + HELD_OUT:
        solver = get_solver(s)
        rng = np.random.default_rng(7)
        bc = "periodic"
        out = {}
        for n in (1, 64):
            params = [solver.sample_params(rng, "train") for _ in range(n)]
            best = float("inf")
            for _ in range(3):
                t0 = time.perf_counter()
                solver.simulate_batch(params, bc, N_IN_FRAMES + 50, rng)
                best = min(best, time.perf_counter() - t0)
            out[f"solver_{n}"] = best
        res["systems"][s] = out
        print("exp6", s, out, flush=True)
    seeds = load("sound_uniform", "test")[:64, :N_IN_FRAMES]
    res["network_1"] = time_model(m, 1, seed_frames=seeds)
    res["network_64"] = time_model(m, 64, seed_frames=seeds)
    save_json(res, OUT / "exp6_timing.json")


# ---------------------------------------------------------------------- exp7
def exp7():
    if not have("noderiv_s0"):
        return
    ref = ref_seed_rms()
    res = {}
    a, b = model("main_s0"), model("noderiv_s0", derivs=False)
    for s in TRAIN_SYSTEMS:
        f = load(s, "test")
        res[s] = {"with": strip(summarise(a, f)), "without": strip(summarise(b, f))}
    for s in HELD_OUT:
        f = load(s, "test")
        k = seed_factors(f, ref)
        res[s] = {"with": strip(summarise(a, f, k)), "without": strip(summarise(b, f, k))}
    save_json(res, OUT / "exp7_ablation.json")


# ---------------------------------------------------------------------- exp8
def exp8():
    """Geometry shift. Reference units for each split are the 1D chain's training scales;
    inputs use seed-frame scaling as in exp2. Compare d=2,3 against the 1D chain with the same
    wall: pressure release (1D test split, pressure-release trajectories) and rigid (1D test_bc)."""
    ref = ref_seed_rms()
    scale = np.array(json.loads((DATA / "chain" / "norm.json").read_text())["scale"], np.float32)
    res = {}

    def run_split(fields, tag):
        k = seed_factors(fields, ref)
        r = {"persistence": strip(summarise(None, fields))}
        for sd in SEEDS:
            r[f"main_s{sd}"] = strip(summarise(model(f"main_s{sd}"), fields, k))
        res[tag] = r

    t = np.load(DATA / "chain" / "test.npz")
    sel = t["bc"] == "pressure_release"
    run_split((t["fields"][sel] / scale[None, None, :, None]).astype(np.float32), "d1_pressure_release")
    run_split(load("chain", "test_bc"), "d1_rigid")
    for d in (2, 3):
        for bc in ("pressure_release", "rigid"):
            z = np.load(DATA / "chain" / f"test_d{d}_{bc}.npz")["fields"]
            run_split((z / scale[None, None, :, None]).astype(np.float32), f"d{d}_{bc}")
        print("exp8", d, flush=True)
    save_json(res, OUT / "exp8_geometry.json")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["exp1", "exp2", "exp4", "exp5", "exp6", "exp7", "exp8"]
    for e in todo:
        t0 = time.time()
        globals()[e]()
        print(e, "done", round(time.time() - t0, 1), "s", flush=True)
