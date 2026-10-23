"""Grid datasets (rounds 4 and 5, d = 2 and d = 3): generation, reference fingerprints and the
cross-machine determinism check (pattern: tools/check_manifest.py).

Modes
  --reference   compute, on this machine, the SHA-256 of the first generation batch of every
                (system, split) through the production path (pfm2d.data.reference_subset) and
                write results/data_manifest_grid.json. Rows per batch: 16 (d = 2), 4 (d = 3).
  (default)     generate every standard split of the named systems (pfm2d.data.SPLITS) in
                parallel worker processes, one (system, split) per task, OMP threads 1 each.
                After each split is written, rows 0 .. k-1 of the stored file are compared
                with results/data_manifest_grid.json: a STRICT match (identical SHA-256) or,
                failing that, a TOLERANCE match (pattern of tools/check_manifest_tol.py:
                same shape and dtype, float64 sum, sum of squares, maximum absolute value and
                1,000 values at fixed positions within a relative tolerance of 1e-6 of the
                array's maximum absolute value, and per (trajectory, frame, slot) the L2 norm
                and maximum within 1e-6 of the largest frame norm and of the array maximum;
                channel slots identical; 1e-6 is ten times tighter than check_manifest_tol.py). A different CPU or
                vector-instruction path can change last bits with identical libraries, so the
                tolerance match is accepted and recorded as such, with the worst sampled
                relative difference. On no match the job prints "DATA MISMATCH" and exits 1,
                and no training job may use the data (tools/check_grid_data.py, run first by
                every grid training job, fails too).
                Then norm.json per system and results/datagen_grid/manifest_full_d<d>.json
                (SHA-256 of every stored file, norm scales, seconds) are written.
Never generates the locked sets (experiments/make_locked_grid.py does that, on the CPU
machine only).

Usage
  python experiments/gen_data_grid.py --reference --dim 2 3
  python experiments/gen_data_grid.py --dim 2 --workers 4
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np                                                       # noqa: E402

from pfm2d import common as C                                            # noqa: E402
from pfm2d import data as D                                              # noqa: E402

ROOT = C.ROOT
MANIFEST = ROOT / "results" / "data_manifest_grid.json"
FULL_DIR = ROOT / "results" / "datagen_grid"
SYSTEMS = D.TRAIN_SYSTEMS + D.HELD_OUT
SPLIT_NAMES = [j[0] for j in D.JOBS]


# 1e-6: 17 times one float32 unit in the last place (6e-8), the size of a last-bit difference
# after float32 storage; ten times tighter than tools/check_manifest_tol.py (1e-5), where the 1D
# cross-machine differences were at most 1.2e-13 (PLAN.md section 24).
RTOL = 1e-6


def fingerprint(a):
    """tools/check_manifest_tol.py fingerprint plus, per (trajectory, frame, stored slot), the
    L2 norm and the maximum absolute value of the field (detects any frame-level change; a
    change confined to a few cells that misses the 1,000 sampled positions and changes no frame
    norm by more than the tolerance is not detectable without the data themselves)."""
    a = np.asarray(a)
    x = a.astype(np.float64).ravel()
    n = x.size
    idx = np.linspace(0, n - 1, min(1000, n)).astype(np.int64)
    fr = a.astype(np.float64).reshape(a.shape[0], a.shape[1], a.shape[2], -1)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "sum": float(x.sum()),
            "sumsq": float((x * x).sum()), "maxabs": float(np.abs(x).max()),
            "sample": x[idx].tolist(),
            "frame_norm": np.sqrt((fr ** 2).sum(-1)).ravel().tolist(),
            "frame_maxabs": np.abs(fr).max(-1).ravel().tolist()}


def tolerance_match(got, ref):
    """(ok, worst relative difference) as tools/check_manifest_tol.py (relative tolerance RTOL =
    1e-6 of the array's maximum absolute value), plus per-frame norms and maxima within RTOL of the
    largest frame norm and of the array maximum."""
    if got["shape"] != ref["shape"] or got["dtype"] != ref["dtype"]:
        return False, float("inf")
    m = max(ref["maxabs"], 1e-30)
    n = max(1, int(np.prod(ref["shape"])))
    close = lambda a, b, sc, k=1: abs(a - b) <= RTOL * sc * k
    worst = max(abs(u - v) for u, v in zip(got["sample"], ref["sample"])) / m
    fn_r, fn_g = np.array(ref["frame_norm"]), np.array(got["frame_norm"])
    fm_r, fm_g = np.array(ref["frame_maxabs"]), np.array(got["frame_maxabs"])
    wn = float(np.abs(fn_g - fn_r).max() / max(fn_r.max(), 1e-30))
    wm = float(np.abs(fm_g - fm_r).max() / m)
    worst = max(worst, wn, wm)
    ok = (close(got["maxabs"], ref["maxabs"], m) and close(got["sum"], ref["sum"], m, n)
          and close(got["sumsq"], ref["sumsq"], m * m, n) and worst <= RTOL)
    return bool(ok), float(worst)


def compare(f, ch, ref):
    """Strict or tolerance comparison of stored rows with a manifest entry."""
    got = D.subset_hash(f, ch)
    if got == ref["sha256"]:
        return {"match": "strict", "worst_sampled_rel_diff": 0.0, "sha256_got": got}
    if np.asarray(ch).tolist() != ref["chan"]:
        return {"match": False, "reason": "channel slots differ", "sha256_got": got}
    ok, worst = tolerance_match(fingerprint(f), ref["fingerprint"])
    return {"match": "tolerance" if ok else False, "worst_sampled_rel_diff": worst,
            "sha256_got": got}


def key(dim, name, split):
    return f"d{dim}/{name}/{split}"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def reference(dims, systems, path=None):
    path = Path(path) if path else MANIFEST
    man = json.loads(path.read_text()) if path.exists() else {"reference": {}}
    import platform
    import numpy, scipy
    for dim in dims:
        for name in systems:
            for split in SPLIT_NAMES:
                k = key(dim, name, split)
                if k in man["reference"]:
                    continue
                t0 = time.time()
                f, ch = D.reference_subset(name, dim, split)
                man["reference"][k] = {"rows": int(f.shape[0]), "sha256": D.subset_hash(f, ch),
                                       "chan": ch.tolist(), "fingerprint": fingerprint(f),
                                       "seconds": round(time.time() - t0, 2)}
                print(k, man["reference"][k], flush=True)
                man["environment"] = {"python": platform.python_version(), "numpy": numpy.__version__,
                                      "scipy": scipy.__version__, "machine": platform.machine()}
                man["note"] = ("SHA-256 of rows 0 .. rows-1 (the first generation batch) of each "
                               "standard split, stored float32 fields then int8 channel slots, "
                               "computed on the CPU workspace by pfm2d.data.reference_subset; "
                               "plus a tolerance fingerprint of the same rows (shape, dtype, "
                               "float64 sum, sum of squares, max abs, 1,000 fixed-position "
                               "values, per-frame L2 norms and maxima; tools/check_manifest_tol.py "
                               "pattern, rtol 1e-6); checked by experiments/gen_data_grid.py on the GPU machine")
                C.save_json(man, path)
    return man


def _task(args):
    name, dim, split = args
    os.environ["OMP_NUM_THREADS"] = "1"
    man = json.loads(MANIFEST.read_text())["reference"]
    d_out = D.system_dir(name, dim)
    done = d_out / f"{split}_done.json"
    if done.exists():
        return json.loads(done.read_text())
    n, seed, regime, bc_set = D.split_job(name, dim, split)
    info = D.generate(name, split, n, seed, regime, bc_set, dim)
    ref = man[key(dim, name, split)]
    f = np.load(d_out / f"{split}_fields.npy", mmap_mode="r")
    ch = np.load(d_out / f"{split}_chan.npy")
    k = ref["rows"]
    cmp_ = compare(np.asarray(f[:k]), ch[:k], ref)
    info["reference_rows"] = k
    info["reference_match"] = cmp_["match"]
    info["reference_check"] = cmp_
    if not info["reference_match"]:
        info["DATA_MISMATCH"] = True
        return info
    # exactly the split's own files (a glob "test_*" would also catch test_bc_* files that a
    # parallel worker is still writing; corrected 21 Oct 2026, 17:10 PDT)
    info["files"] = {f"{split}_{x}": sha_file(d_out / f"{split}_{x}")
                     for x in ("fields.npy", "chan.npy", "bc.npy", "meta.json")}
    done.write_text(json.dumps(info, default=float))
    return info


def generate(dims, systems, workers):
    from multiprocessing import get_context
    tasks = [(name, dim, split) for dim in dims for name in systems for split in SPLIT_NAMES]
    # longest first: 3D before 2D, chain family before single systems, train before others
    cost = {"chain": 3, "thermoacoustic": 2.5, "photothermal": 2, "sound_layered": 1.2}
    tasks.sort(key=lambda t: -(10 if t[1] == 3 else 1) * cost.get(t[0], 1.0)
               * D.SPLITS[t[1]][t[2]])
    bad = []
    results = []
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        for info in pool.imap_unordered(_task, tasks):
            results.append(info)
            print(json.dumps({k: v for k, v in info.items() if k != "files"}, default=float),
                  f"elapsed {time.time() - t0:.0f} s", flush=True)
            if info.get("DATA_MISMATCH"):
                bad.append(info)
                print("DATA MISMATCH", info["system"], info["dim"], info["split"], flush=True)
                pool.terminate()
                break
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    if bad:
        for dim in dims:
            C.save_json({"DATA_MISMATCH": bad, "results": results}, FULL_DIR / f"MISMATCH_d{dim}.json")
        print("DATA MISMATCH: stopping; no grid training job may run on these data", flush=True)
        sys.exit(1)
    for dim in dims:
        full = {"dim": dim, "systems": {}}
        for name in systems:
            scale = D.compute_norm(name, dim)
            d_out = D.system_dir(name, dim)
            full["systems"][name] = {"norm_scale": scale, "splits": {
                s: json.loads((d_out / f"{s}_done.json").read_text()) for s in SPLIT_NAMES}}
        full["seconds_total"] = time.time() - t0
        full["workers"] = workers
        C.save_json(full, FULL_DIR / f"manifest_full_d{dim}.json")
        if set(D.TRAIN_SYSTEMS) <= set(systems):
            from pfm2d.evaluate import reference_seed_rms
            C.save_json({"dim": dim, "systems": D.TRAIN_SYSTEMS, "split": "train",
                         "ref_seed_rms": reference_seed_rms(dim).tolist(),
                         "definition": "per channel, median over training-system trajectories "
                                       "occupying the channel of the RMS of their four seed frames "
                                       "(normalised units), pfm2d.evaluate.reference_seed_rms"},
                        FULL_DIR / f"ref_seed_rms_d{dim}.json")
        for name in systems:
            C.save_json({"scale": D.Store(name, "train", dim).scale.tolist()},
                        FULL_DIR / f"norm_d{dim}_{name}.json")
    kinds = [r["reference_match"] for r in results]
    print(f"all reference fingerprints match ({kinds.count('strict')} strict, "
          f"{kinds.count('tolerance')} tolerance); data complete", flush=True)


def local_dev(dims, splits=("test", "test_bc", "test_ood")):
    """CPU workspace: the chain's development splits (identical seeds and code path as the GPU
    job), checked against the fingerprints; SHA-256 of every file in
    results/datagen_grid/local_dev_d<d>.json (to compare with the GPU's full manifest)."""
    for dim in dims:
        rec = {}
        for split in splits:
            info = _task(("chain", dim, split))
            rec[split] = info
            print(json.dumps({k: v for k, v in info.items() if k not in ("files",)}, default=float),
                  flush=True)
            if not info.get("reference_match"):
                print("DATA MISMATCH", dim, split)
                sys.exit(1)
        C.save_json(rec, FULL_DIR / f"local_dev_d{dim}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--systems", nargs="+", default=SYSTEMS)
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--manifest", default=None, help="--reference: write here instead (merge later)")
    ap.add_argument("--local-dev", action="store_true", help="chain development splits only")
    a = ap.parse_args()
    if a.local_dev:
        local_dev(a.dim)
    elif a.reference:
        reference(a.dim, a.systems, a.manifest)
    else:
        generate(a.dim, a.systems, a.workers)
