"""Tolerance check of regenerated datasets against results/data_fingerprints.json.

Used where byte-identical regeneration is not possible (a different CPU or vector-instruction
path can change the last bits of floating-point results even with identical library versions).
For every file listed in the fingerprints and present under data/, each array must have the same
shape and dtype, and its float64 sum, sum of squares, maximum absolute value and 1,000 values at
fixed positions must agree within a relative tolerance of 1e-5 of the array's maximum absolute
value. Exit 1 on any mismatch or if no listed file is present.

    python3 tools/check_manifest_tol.py            # check
    python3 tools/check_manifest_tol.py --make     # write fingerprints (reference machine)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FP = ROOT / "results/data_fingerprints.json"
RTOL = 1e-5


def fingerprint(path):
    out = {}
    with np.load(path) as z:
        for k in z.files:
            a = z[k]
            if a.dtype.kind not in "fiu":
                continue
            x = a.astype(np.float64).ravel()
            n = x.size
            idx = np.linspace(0, n - 1, min(1000, n)).astype(np.int64) if n else np.zeros(0, np.int64)
            out[k] = {"shape": list(a.shape), "dtype": str(a.dtype), "sum": float(x.sum()),
                      "sumsq": float((x * x).sum()), "maxabs": float(np.abs(x).max()) if n else 0.0,
                      "sample": x[idx].tolist()}
    return out


def close(a, b, scale, n=1):
    return abs(a - b) <= RTOL * max(scale, 1e-30) * n


def main():
    man = json.loads((ROOT / "results/data_manifest.json").read_text())
    if "--make" in sys.argv:
        fps = {p: fingerprint(ROOT / p) for p in man if (ROOT / p).exists() and p.endswith(".npz")}
        FP.write_text(json.dumps(fps))
        print(f"wrote fingerprints for {len(fps)} files")
        return
    fps = json.loads(FP.read_text())
    bad, seen, worst = [], 0, 0.0
    for p, ref in fps.items():
        f = ROOT / p
        if not f.exists():
            continue
        seen += 1
        got = fingerprint(f)
        for k, r in ref.items():
            g = got.get(k)
            if g is None or g["shape"] != r["shape"] or g["dtype"] != r["dtype"]:
                bad.append(f"{p}:{k} shape/dtype"); continue
            m = r["maxabs"]; n = max(1, int(np.prod(r["shape"])))
            ok = (close(g["maxabs"], m, m) and close(g["sum"], r["sum"], m, n)
                  and close(g["sumsq"], r["sumsq"], m * m, n)
                  and all(close(u, v, m) for u, v in zip(g["sample"], r["sample"])))
            d = max((abs(u - v) for u, v in zip(g["sample"], r["sample"])), default=0.0) / max(m, 1e-30)
            worst = max(worst, d)
            if not ok:
                bad.append(f"{p}:{k}")
    print(f"tolerance check: {seen} files, {len(bad)} mismatches, worst sampled relative difference {worst:.2e}", bad[:5])
    if seen == 0:
        print("DATA MISSING: no file listed in the fingerprints is present")
    sys.exit(1 if bad or seen == 0 else 0)


if __name__ == "__main__":
    main()
