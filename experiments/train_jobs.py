"""Training job definitions and a simple queue runner.

Budgets (fixed before any results were seen):
    main model, derivative ablation, scaling runs: 20,000 steps, batch 64
    specialists and oracles (one system each):      8,000 steps, batch 64
A specialist sees 8,000 x 64 = 512,000 windows of its one system. The main model sees
20,000 x 64 / 8 = 160,000 windows per system on average, 0.3125 of a specialist's budget
(a factor of 3.2 fewer). The scaling runs see 426,667 (3 systems) and 256,000 (5 systems)
windows per system, so the scaling test changes per-system exposure together with variety;
the report states this (audit round 1, M6).

Fine-tuning (experiment 5): from main_s0, on k = 1, 5, 25 trajectories of a held-out system
(balanced over boundary conditions, fixed seed), 500 steps, learning rate 1e-4 with 50
warm-up steps, final weights kept (no validation-based selection). Control: the same from a
random initialisation ("scratch").

Usage: python experiments/train_jobs.py JOB [JOB ...]   (runs the named jobs in order)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfm.common import ROOT                                    # noqa: E402
from pfm.data import TRAIN_SYSTEMS, SCALING_SUBSETS, TRAIN_SYSTEMS_V2, TRAIN_SYSTEMS_V3  # noqa: E402
from pfm.train import train                                    # noqa: E402

MAIN_STEPS, SPEC_STEPS = 20000, 8000

JOBS = {}
for seed in (0, 1, 2):
    JOBS[f"main_s{seed}"] = dict(systems=TRAIN_SYSTEMS, steps=MAIN_STEPS, seed=seed)
    JOBS[f"scale3_s{seed}"] = dict(systems=SCALING_SUBSETS[3], steps=MAIN_STEPS, seed=seed)
    JOBS[f"scale5_s{seed}"] = dict(systems=SCALING_SUBSETS[5], steps=MAIN_STEPS, seed=seed)
JOBS["noderiv_s0"] = dict(systems=TRAIN_SYSTEMS, steps=MAIN_STEPS, seed=0, use_derivs=False)
for s in TRAIN_SYSTEMS:
    JOBS[f"spec_{s}"] = dict(systems=[s], steps=SPEC_STEPS, seed=0)
for s in ("chain", "telegraph"):
    JOBS[f"oracle_{s}"] = dict(systems=[s], steps=SPEC_STEPS, seed=0)


for s in ("chain", "telegraph"):
    for k in (1, 5, 25):
        JOBS[f"ft_{s}_k{k}"] = dict(systems=[s], steps=500, seed=0, n_traj=k, base_lr=1e-4,
                                    warm=50, select="last", init="runs/main_s0/model.pt")
        JOBS[f"scratch_{s}_k{k}"] = dict(systems=[s], steps=500, seed=0, n_traj=k,
                                         base_lr=1e-4, warm=50, select="last")


# ---------------------------------------------------------------- round 2 (21 Oct 2026)
# One change at a time against round 1, whose eight datasets are unchanged:
#   v2_*        ten systems (round 1 + photothermal + thermoacoustic), stabilised training
#               (pushforward plus input noise 0.003), round-1 network size, 20,000 steps
#   v2plain_s0  ten systems, round-1 training recipe (isolates the effect of the new systems)
#   r1stab_s0   round-1 eight systems with stabilised training (isolates the training change)
#   v2pt_s0, v2ta_s0   eight systems plus only one of the two links (which link matters)
#   v2big_s0    ten systems, stabilised, larger network (width 384, depth 8, heads 8)
#   spec_photothermal, spec_thermoacoustic   specialists for the new systems
#   ft2_*       fine-tuning from v2_s0 as in round 1
STAB = dict(noise_std=0.003, pushforward=True)
for seed in (0, 1, 2):
    JOBS[f"v2_s{seed}"] = dict(systems=TRAIN_SYSTEMS_V2, steps=MAIN_STEPS, seed=seed, **STAB)
JOBS["v2plain_s0"] = dict(systems=TRAIN_SYSTEMS_V2, steps=MAIN_STEPS, seed=0)
JOBS["r1stab_s0"] = dict(systems=TRAIN_SYSTEMS, steps=MAIN_STEPS, seed=0, **STAB)
JOBS["v2pt_s0"] = dict(systems=TRAIN_SYSTEMS + ["photothermal"], steps=MAIN_STEPS, seed=0, **STAB)
JOBS["v2ta_s0"] = dict(systems=TRAIN_SYSTEMS + ["thermoacoustic"], steps=MAIN_STEPS, seed=0, **STAB)
for seed in (1, 2):      # GPU (added 21 Oct 2026): seeds for the size test
    JOBS[f"v2big_s{seed}"] = dict(systems=TRAIN_SYSTEMS_V2, steps=MAIN_STEPS, seed=seed,
                                  width=384, depth=8, heads=8, **STAB)
JOBS["v2big_s0"] = dict(systems=TRAIN_SYSTEMS_V2, steps=MAIN_STEPS, seed=0, width=384, depth=8,
                        heads=8, **STAB)
for s in ("photothermal", "thermoacoustic"):
    JOBS[f"spec_{s}"] = dict(systems=[s], steps=SPEC_STEPS, seed=0)
for s in ("chain", "telegraph"):
    for k in (1, 5, 25):
        JOBS[f"ft2_{s}_k{k}"] = dict(systems=[s], steps=500, seed=0, n_traj=k, base_lr=1e-4,
                                     warm=50, select="last", init="runs/v2_s0/model.pt")


# ---------------------------------------------------------------- round 3 (21 Oct 2026)
# v3: the ten round-2 systems plus the seven tissue systems in cylinder and sphere geometry
# (24 systems); stabilised recipe and round-1 network size, as pre-registered in PLAN.md s.16.
for seed in (0, 1, 2):
    JOBS[f"v3_s{seed}"] = dict(systems=TRAIN_SYSTEMS_V3, steps=MAIN_STEPS, seed=seed, **STAB)
# v3L (design change on development data, PLAN.md s.18): v3 trained 60,000 steps so that each
# system receives about 160,000 training windows, matching v2's per-system exposure.
for seed in (0, 1, 2):
    JOBS[f"v3L_s{seed}"] = dict(systems=TRAIN_SYSTEMS_V3, steps=60000, seed=seed, **STAB)
# v3Lplain (PLAN.md s.22): v3L without the rollout stabilisation (round-1 recipe), three seeds.
for seed in (0, 1, 2):
    JOBS[f"v3Lplain_s{seed}"] = dict(systems=TRAIN_SYSTEMS_V3, steps=60000, seed=seed)


def run(name):
    out = ROOT / "runs" / name
    if (out / "log.json").exists() and "best_val_loss" in json.loads((out / "log.json").read_text()):
        print(name, "already done", flush=True)
        return
    t0 = time.time()
    job = dict(JOBS[name])
    if job.get("init"):
        job["init"] = ROOT / job["init"]
    log = train(out=out, threads=1, **job)
    print(name, "best val", log["best_val_loss"], "min", round((time.time() - t0) / 60, 1), flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
