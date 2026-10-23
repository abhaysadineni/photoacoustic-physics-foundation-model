"""Generate every dataset the experiments use.

For each system:
    train (600), val (60), test (100)     training boundary conditions, training parameter ranges
    test_bc (100)                         held-out boundary conditions, training parameter ranges
    test_ood (100)                        training boundary conditions, parameters outside training ranges
For the chain only:
    test_d{2,3}_{pressure_release,rigid} (100 each)  radial profiles of the cylindrical and spherical
                                          chain, with a training boundary (pressure release)
                                          and the held-out one (rigid), so the geometry shift
                                          can be read apart from the boundary shift (audit M3)

Seeds are fixed per (system, split) so every dataset is reproducible. Usage:
    python experiments/gen_data.py transport burgers ...
"""
import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfm.common import save_json, RESULTS                      # noqa: E402
from pfm.data import generate, compute_norm, SPLITS             # noqa: E402

JOBS = [("train", SPLITS["train"], "train", "train"),
        ("val", SPLITS["val"], "train", "train"),
        ("test", SPLITS["test"], "train", "train"),
        ("test_bc", 100, "train", "test"),
        ("test_ood", 100, "ood", "train")]


def seed_for(system, split):
    return zlib.crc32(f"{system}/{split}".encode())


def run(system):
    log = []
    for split, n, regime, bc_set in JOBS:
        log.append(generate(system, split, n, seed_for(system, split), regime, bc_set))
        print(log[-1], flush=True)
    log.append({"norm_scale": compute_norm(system)})
    if system == "chain":
        from pfm.data import get_solver
        solver = get_solver("chain")
        for dim in (2, 3):
            for bc in ("pressure_release", "rigid"):
                log.append(generate_geometry(solver, dim, bc))
                print(log[-1], flush=True)
    save_json(log, RESULTS / "datagen" / f"{system}.json")


def generate_geometry(solver, dim, bc):
    """Radial-profile chain data; the solver's sample_params must accept the dim override."""
    import numpy as np
    from pfm.data import DATA, _to_arrays, N_FRAMES
    import time
    rng = np.random.default_rng(seed_for("chain", f"d{dim}_{bc}"))
    t0 = time.time()
    params = []
    for _ in range(100):
        p = solver.sample_params(rng, "train", dim=dim)
        params.append(p)
    trajs = solver.simulate_batch(params, bc, N_FRAMES, rng)
    fields, chans, meta = _to_arrays(trajs)
    np.savez(DATA / "chain" / f"test_d{dim}_{bc}.npz", fields=fields, channels=chans,
             bc=np.array([m["bc"] for m in meta]))
    (DATA / "chain" / f"test_d{dim}_{bc}_meta.json").write_text(json.dumps(meta, default=float))
    return {"system": "chain", "split": f"test_d{dim}_{bc}", "n": len(trajs),
            "seconds": time.time() - t0}


if __name__ == "__main__":
    for s in sys.argv[1:]:
        run(s)
