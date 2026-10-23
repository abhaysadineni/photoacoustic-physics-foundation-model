"""Training throughput of the grid networks on the machine it runs on (GPU timing job).

Synthetic data with the real storage layout and sizes (pfm2d.data.SPLITS, STORED_CHANNELS,
the seven training systems) are written to a temporary root, so window drawing from the
memory-mapped files is timed with the real file sizes; values are standard normal (the
throughput does not depend on them). For each (d, batch, prefetch) a short run of the real
training loop (pfm2d.train.train, plain recipe) is made and the steady rate is taken between
the first and last validation points (validation every 100 steps, so first-step CUDA set-up is
excluded and the validation cost is included). Also: the time to draw one batch alone, and
peak GPU memory. Writes results/timing_grid/timing_<host>.json.
Usage: python experiments/time_grid.py [--steps 400]
"""
import argparse
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

from pfm2d import common as C                                       # noqa: E402
from pfm2d import data as D                                         # noqa: E402
from pfm2d import train as T                                        # noqa: E402

CONFIGS = [(2, 32, False), (2, 32, True), (2, 64, True), (2, 128, True),
           (3, 16, False), (3, 16, True), (3, 32, True), (3, 64, True)]


def make_root(dim, root):
    g = C.GEOM[dim]
    rng = np.random.default_rng(0)
    for s in D.TRAIN_SYSTEMS:
        d = Path(root) / s
        d.mkdir(parents=True, exist_ok=True)
        S = D.STORED_CHANNELS[s]
        for split in ("train", "val"):
            n = D.SPLITS[dim][split]
            f = np.lib.format.open_memmap(d / f"{split}_fields.npy", mode="w+", dtype=np.float32,
                                          shape=(n, g["n_frames"], S) + (g["n_out"],) * dim)
            for i in range(n):
                f[i] = rng.standard_normal(f.shape[1:], dtype=np.float32)
            f.flush()
            del f
            np.save(d / f"{split}_chan.npy", np.tile(np.arange(S, dtype=np.int8), (n, 1)))
            np.save(d / f"{split}_bc.npy", np.array(["periodic"] * n))
        (d / "norm.json").write_text(json.dumps({"scale": [1.0, 1.0, 1.0]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    a = ap.parse_args()
    dev = T.pick_device()
    res = {"host": socket.gethostname(), "device": str(dev), "torch": torch.__version__,
           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
           "cpu_count": os.cpu_count(), "steps_per_config": a.steps, "runs": []}
    tmp = Path(tempfile.mkdtemp(prefix="pfm_timing_"))
    try:
        for dim in (2, 3):
            root = tmp / f"d{dim}"
            t0 = time.time()
            make_root(dim, root)
            res[f"synthetic_data_seconds_d{dim}"] = time.time() - t0
            st = [D.Store(s, "train", dim, root) for s in D.TRAIN_SYSTEMS]
            pool = T.Pool(st)
            r = np.random.default_rng(1)
            for b in sorted({c[1] for c in CONFIGS if c[0] == dim}):
                T.windows(pool, r, b)
                t0 = time.time()
                for _ in range(20):
                    T.windows(pool, r, b)
                res[f"draw_seconds_per_batch_d{dim}_b{b}"] = (time.time() - t0) / 20
            for (d_, b, pf) in CONFIGS:
                if d_ != dim:
                    continue
                out = tmp / f"run_d{dim}_b{b}_pf{int(pf)}"
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                log = T.train(D.TRAIN_SYSTEMS, a.steps, 0, out, dim=dim, batch=b, threads=1,
                              val_every=100, data_root=root, prefetch=pf)
                wall = time.time() - t0
                h = log["history"]
                rate = (h[-1]["step"] - h[0]["step"]) / (h[-1]["seconds"] - h[0]["seconds"])
                row = {"dim": dim, "batch": b, "prefetch": pf, "steps": a.steps,
                       "steps_per_second": rate, "windows_per_second": rate * b,
                       "seconds_per_1000_steps": 1000.0 / rate, "wall_seconds": wall,
                       "n_params": log["n_params"],
                       "peak_gpu_memory_GB": (torch.cuda.max_memory_allocated() / 1e9
                                              if dev.type == "cuda" else None)}
                res["runs"].append(row)
                print(json.dumps(row), flush=True)
                shutil.rmtree(out, ignore_errors=True)
            shutil.rmtree(root, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    C.save_json(res, C.ROOT / "results" / "timing_grid" / "timing_gpu.json")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
