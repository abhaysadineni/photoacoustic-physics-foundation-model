"""Locked final test sets of the grid chain (rounds 4 and 5), made once on the CPU workspace
before any grid network exists (PLAN.md section 23).

d = 2: N2 trajectories, d = 3: N3 trajectories of the held-out chain (pfm2d.solvers.chain.Chain),
training parameter ranges, training boundary conditions (periodic and pressure release, half
each, pfm2d.data.generate), default internal grid, seeds crc32("chain/d<d>/final_locked_2026-10-21").
Stored in the pfm2d data layout as data2d/chain/final_locked_* (d = 2) and
data2d/d3/chain/final_locked_* (d = 3), made read-only (chmod 0444), SHA-256 of every file in
results/final_locked_manifest_grid.json. If the files exist, only their hashes are checked
against the manifest (exit 1 on any mismatch).
Usage: python experiments/make_locked_grid.py [--check]
"""
import hashlib
import json
import os
import stat
import sys
import time
import zlib
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfm2d import common as C                                  # noqa: E402
from pfm2d import data as D                                    # noqa: E402

N = {2: 200, 3: 100}
MANIFEST = C.ROOT / "results" / "final_locked_manifest_grid.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def files(dim):
    d = D.system_dir("chain", dim)
    return sorted(p for p in d.glob("final_locked_*"))


def main():
    check_only = "--check" in sys.argv
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
    if man is not None:
        bad = []
        for dim in (2, 3):
            for rel, h in man[f"d{dim}"]["sha256"].items():
                if sha(C.ROOT / rel) != h:
                    bad.append(rel)
        print("locked grid sets:", "all SHA-256 match the manifest" if not bad else f"MISMATCH {bad}")
        sys.exit(1 if bad else 0)
    if check_only:
        print("no manifest")
        sys.exit(1)
    out = {"note": "locked chain test sets of the grid rounds; made before any grid network "
                   "existed; evaluated once (PLAN.md section 23)"}
    for dim in (2, 3):
        if files(dim):
            raise SystemExit(f"final_locked files exist for d = {dim} but no manifest: stop")
        seed = zlib.crc32(f"chain/d{dim}/final_locked_2026-10-21".encode())
        t0 = time.time()
        info = D.generate("chain", "final_locked", N[dim], seed, "train", "train", dim)
        info["seconds"] = time.time() - t0
        hs = {}
        for p in files(dim):
            os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            hs[str(p.relative_to(C.ROOT))] = sha(p)
        info["sha256"] = hs
        info["seed_string"] = f"chain/d{dim}/final_locked_2026-10-21"
        info["made_at_pdt"] = time.strftime("%Y-%m-%d %H:%M %Z")
        out[f"d{dim}"] = info
        print(dim, json.dumps(info, indent=1), flush=True)
    C.save_json(out, MANIFEST)
    os.chmod(MANIFEST, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


if __name__ == "__main__":
    main()
