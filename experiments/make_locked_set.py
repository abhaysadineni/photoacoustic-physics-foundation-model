"""Recreate the locked final test sets exactly as made on 21 Oct 2026, 08:36 PDT.

200 trajectories each of the chain and the telegraph equation, training parameter ranges and
training boundary conditions, seeds crc32("<system>/final_locked_2026-10-21"). The SHA-256 of
each file must match results/final_locked_manifest.json; this script checks it.
"""
import hashlib
import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfm.common import RESULTS                                   # noqa: E402
from pfm.data import DATA, generate                              # noqa: E402

if __name__ == "__main__":
    manifest = {m["system"]: m for m in json.loads((RESULTS / "final_locked_manifest.json").read_text())}
    for s in ("chain", "telegraph"):
        path = DATA / s / "final_locked.npz"
        if not path.exists():
            generate(s, "final", 200, zlib.crc32(f"{s}/final_locked_2026-10-21".encode()),
                     "train", "train", out_name="final_locked")
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        ok = h == manifest[s]["sha256"]
        print(s, "sha256 matches manifest" if ok else f"MISMATCH {h}")
        if not ok:
            sys.exit(1)
