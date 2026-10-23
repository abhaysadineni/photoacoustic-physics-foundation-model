"""Regression checks for the device-aware pfm2d/train.py (21 Oct 2026), CPU only:
 1. the CPU path is numerically unchanged: 60 steps with the pre-change train.py (git HEAD of
    the repository clone, given as argv[1]) and with the current one give bit-identical
    model.pt and identical logged losses;
 2. prefetch (background draw of the next batch) gives bit-identical weights;
 3. an interrupted run (stop_after 30) resumed to 60 steps gives bit-identical weights, with
    and without prefetch.
Tiny 2D datasets (light and heat, 4 training and 2 validation trajectories, internal 64^2)
are generated in a temporary root. Writes results/verification2d/train_cpu_regression.json.
Usage: python tests2d/check_train_device.py <path to the pre-change pfm2d/train.py, e.g. from git show e1e7159:pfm2d/train.py>"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["PFM_FORCE_CPU"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch                                              # noqa: E402

import pfm2d                                              # noqa: E402
from pfm2d import common as C                             # noqa: E402
from pfm2d import data as D                               # noqa: E402
from pfm2d import train as NEW                            # noqa: E402


def load_old(path):
    spec = importlib.util.spec_from_file_location("pfm2d._train_old", path,
                                                  submodule_search_locations=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "pfm2d"
    spec.loader.exec_module(mod)
    return mod


def same(a, b):
    A = torch.load(a, weights_only=True)
    B = torch.load(b, weights_only=True)
    return all(torch.equal(A[k], B[k]) for k in A) and A.keys() == B.keys()


def main():
    old = load_old(sys.argv[1])
    tmp = Path(tempfile.mkdtemp(prefix="pfm_trcheck_"))
    systems = ["light", "heat"]
    try:
        for s in systems:
            D.generate_system(s, 2, root=tmp, splits={"train": 4, "val": 2}, n_internal=64, log=False)
        kw = dict(dim=2, batch=8, threads=1, val_every=10, data_root=tmp, warm=5,
                  width=48, depth=2, heads=2)
        old.train(systems, 60, 0, tmp / "old", **kw)
        NEW.train(systems, 60, 0, tmp / "new", **kw)
        NEW.train(systems, 60, 0, tmp / "pf", prefetch=True, **kw)
        NEW.train(systems, 60, 0, tmp / "res", stop_after=30, **kw)
        NEW.train(systems, 60, 0, tmp / "res", **kw)
        NEW.train(systems, 60, 0, tmp / "respf", stop_after=30, prefetch=True, **kw)
        NEW.train(systems, 60, 0, tmp / "respf", prefetch=True, **kw)
        lo = json.loads((tmp / "old/log.json").read_text())["history"]
        ln = json.loads((tmp / "new/log.json").read_text())["history"]
        rep = {"cpu_path_unchanged": same(tmp / "old/model.pt", tmp / "new/model.pt"),
               "losses_identical": [(h["train_loss"], h["val_loss"]) for h in lo]
                                   == [(h["train_loss"], h["val_loss"]) for h in ln],
               "prefetch_identical": same(tmp / "new/model.pt", tmp / "pf/model.pt"),
               "resume_identical": same(tmp / "new/model.pt", tmp / "res/model.pt"),
               "resume_prefetch_identical": same(tmp / "new/model.pt", tmp / "respf/model.pt"),
               "old_train_py": "pre-change pfm2d/train.py (commit e1e7159)", "steps": 60, "torch": torch.__version__}
        rep["passed"] = all(v for k, v in rep.items() if isinstance(v, bool))
        C.save_json(rep, C.RESULTS / "train_cpu_regression.json")
        print(json.dumps(rep, indent=1))
        return 0 if rep["passed"] else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
