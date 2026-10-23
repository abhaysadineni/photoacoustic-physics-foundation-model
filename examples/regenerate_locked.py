"""Regenerate the locked test sets on a fresh clone and check them against their manifests.

Run from the repository root:

    python examples/regenerate_locked.py              # 1D, cylinder, sphere and both grids
    python examples/regenerate_locked.py --no-grid    # 1D, cylinder and sphere only

The locked sets are not in the repository. They are rebuilt here from their fixed seeds
exactly as they were first made, then compared byte for byte (SHA-256) with
results/final_locked_manifest.json (chain, telegraph, chain_r2, chain_r3; 200 trajectories each)
and results/final_locked_manifest_grid.json (grid chain, 200 trajectories for d = 2 and 100 for
d = 3). A file that already exists is not regenerated, only checked. Files are made read-only.
Exit status 1 on any mismatch.

Seeds: crc32("<system>/final_locked_2026-10-21") in 1D and radial geometry (experiments/
make_locked_set.py), crc32("chain/d<d>/final_locked_2026-10-21") on the grids (experiments/
make_locked_grid.py). Training parameter ranges and training boundary conditions throughout.
No training job reads a file named final_locked; these sets are for the locked evaluations only.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import time
import zlib
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SYSTEMS_1D = ("chain", "telegraph", "chain_r2", "chain_r3")
N_GRID = {2: 200, 3: 100}
READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def one_dimensional():
    from pfm.data import DATA, generate
    man = {m["system"]: m for m in
           json.loads((ROOT / "results" / "final_locked_manifest.json").read_text())}
    bad = 0
    for s in SYSTEMS_1D:
        path = DATA / s / "final_locked.npz"
        if not path.exists():
            t0 = time.time()
            generate(s, "final", man[s]["n"], zlib.crc32(f"{s}/final_locked_2026-10-21".encode()),
                     "train", "train", out_name="final_locked")
            for p in (path, DATA / s / "final_locked_meta.json"):
                os.chmod(p, READ_ONLY)
            print(f"{s}: generated {man[s]['n']} trajectories in {time.time() - t0:.0f} s")
        ok = sha(path) == man[s]["sha256"]
        bad += not ok
        print(f"{s}: {path.relative_to(ROOT)} SHA-256 "
              f"{'matches the manifest' if ok else 'DOES NOT MATCH the manifest'}")
    return bad


def grids(dims):
    from pfm2d import data as D
    man = json.loads((ROOT / "results" / "final_locked_manifest_grid.json").read_text())
    bad = 0
    for dim in dims:
        d = D.system_dir("chain", dim)
        if not any(d.glob("final_locked_*")):
            t0 = time.time()
            seed = zlib.crc32(f"chain/d{dim}/final_locked_2026-10-21".encode())
            D.generate("chain", "final_locked", N_GRID[dim], seed, "train", "train", dim)
            for p in d.glob("final_locked_*"):
                os.chmod(p, READ_ONLY)
            print(f"grid d = {dim}: generated {N_GRID[dim]} trajectories in "
                  f"{time.time() - t0:.0f} s")
        for rel, h in man[f"d{dim}"]["sha256"].items():
            p = ROOT / rel
            ok = p.exists() and sha(p) == h
            bad += not ok
            print(f"grid d = {dim}: {rel} SHA-256 "
                  f"{'matches the manifest' if ok else 'DOES NOT MATCH the manifest'}")
    return bad


if __name__ == "__main__":
    args = sys.argv[1:]
    n_bad = one_dimensional()
    if "--no-grid" not in args:
        dims = [int(a) for a in args if a in ("2", "3")] or [2, 3]
        n_bad += grids(dims)
    print("all locked sets match their manifests" if n_bad == 0 else f"{n_bad} MISMATCH(ES)")
    sys.exit(1 if n_bad else 0)
