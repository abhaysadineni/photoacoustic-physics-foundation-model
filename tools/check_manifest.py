"""Check every dataset present under data/ against results/data_manifest.json (SHA-256).
Exit 0 if all present files listed in the manifest match, 1 otherwise; also 1 if no listed
file is present (nothing was checked)."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
man = json.loads((ROOT / "results/data_manifest.json").read_text())
bad, seen = [], 0
for p, h in man.items():
    f = ROOT / p
    if f.exists():
        seen += 1
        if hashlib.sha256(f.read_bytes()).hexdigest() != h:
            bad.append(p)
print(f"checked {seen} files, {len(bad)} mismatches", bad[:5])
if seen == 0:
    print("DATA MISSING: no file listed in the manifest is present")
sys.exit(1 if bad or seen == 0 else 0)
