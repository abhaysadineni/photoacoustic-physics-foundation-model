"""Training jobs of the grid rounds (4: d = 2, 5: d = 3), as pre-registered in PLAN.md
section 23. Every job uses the plain recipe (no pushforward, no input noise), the round-1
network size (width 192, depth 6, heads 6), derivative channels, AdamW 3e-4 with 500 warm-up
steps and cosine decay, validation every 500 steps on the fixed validation windows, best
checkpoint kept (pfm2d/train.py). Outputs: runs_grid/<name>/model.pt and log.json.

    g<d>_s0, g<d>_s1, g<d>_s2   the seven grid training systems (pfm2d.data.TRAIN_SYSTEMS), 3 seeds
    g<d>nc_s0, _s1, _s2         the same without photothermal and thermoacoustic (5 systems),
                                otherwise identical (same steps and batch; seeds 0, 1, 2)
    g<d>oracle_s0               the chain only (reference, never a zero-shot result)

Step counts and batch sizes (STEPS, BATCH, ORACLE_FRACTION) were set from the measured GPU
throughput before these jobs were queued (PLAN.md section 23, sizing entry).
Usage: python experiments/train_jobs_grid.py JOB [JOB ...]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfm2d.common import ROOT                                   # noqa: E402
from pfm2d.data import TRAIN_SYSTEMS                            # noqa: E402
from pfm2d.train import train                                   # noqa: E402

NO_COUPLING = [s for s in TRAIN_SYSTEMS if s not in ("photothermal", "thermoacoustic")]
# Set from the GPU timing job gtime1 (results/timing_grid/timing_gpu.json, gpu branch) by the
# section 23 rule: S_d = min(30,000, budget) = 30,000 in both dimensions; batch 64.
STEPS = {2: 30000, 3: 30000}
BATCH = {2: 64, 3: 64}
ORACLE_FRACTION = 0.4                 # 1D: specialists and oracles 8,000 of 20,000 steps

JOBS = {}
for d in (2, 3):
    for seed in (0, 1, 2):
        JOBS[f"g{d}_s{seed}"] = dict(dim=d, systems=TRAIN_SYSTEMS, seed=seed, kind="main")
    for seed in (0, 1, 2):
        JOBS[f"g{d}nc_s{seed}"] = dict(dim=d, systems=NO_COUPLING, seed=seed, kind="main")
    JOBS[f"g{d}oracle_s0"] = dict(dim=d, systems=["chain"], seed=0, kind="oracle")


def spec(name):
    j = dict(JOBS[name])
    d = j["dim"]
    if STEPS[d] is None:
        raise SystemExit("step counts not set (PLAN.md section 23 sizing entry missing)")
    steps = STEPS[d] if j.pop("kind") == "main" else int(round(ORACLE_FRACTION * STEPS[d]))
    return dict(j, steps=steps, batch=BATCH[d])


def run(name):
    out = ROOT / "runs_grid" / name
    lg = out / "log.json"
    if lg.exists() and "best_val_loss" in json.loads(lg.read_text()):
        print(name, "already done", flush=True)
        return
    t0 = time.time()
    log = train(out=out, threads=1, prefetch=True, **spec(name))
    print(name, "best val", log["best_val_loss"], "min", round((time.time() - t0) / 60, 1),
          flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
