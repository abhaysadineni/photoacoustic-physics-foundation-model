"""Consistency checks (d = 2, 3):
 1. Photothermal and Thermoacoustic equal the chain on their shared channels for the same
    draws (train and ood, every acoustic bc); pass if max rel. difference <= 1e-6 and the
    unused channel is exactly zero.
 2. Reduction to the 1D solvers (pfm2d/solvers/reduction.py) for light, heat, sound uniform,
    sound layered and the chain, every bc; pass if every max relative difference <= 1e-6.
 3. Grid operator: the sparse steady matrix equals the time-stepping operator, is symmetric,
    and the Gershgorin bound is at least the spectral radius.
Writes results/verification2d/consistency_d<d>.json."""
import sys
import time
from _harness import main_dims, environment
from pfm2d import common as C
from pfm2d.solvers import chain as K
from pfm2d.solvers import reduction as RD


def worst(case):
    vals = []
    for k, v in case.items():
        if isinstance(v, dict):
            vals += [vv for kk, vv in v.items() if kk == "max_rel_diff"]
            vals += [worst(v)] if any(isinstance(x, dict) for x in v.values()) else []
        elif k in ("max_rel_diff", "fluence_steady_max_rel_diff") and v is not None:
            vals.append(v)
    return max(vals) if vals else 0.0


if __name__ == "__main__":
    rc = 0
    for d in main_dims(sys.argv):
        t0 = time.perf_counter()
        rep = {"dim": d}
        rep["links"] = K.consistency_check(d, n_traj=4 if d == 2 else 2)
        red = RD.run_all(d)
        for k, c in red["cases"].items():
            c["worst_rel_diff"] = worst(c)
            c["passed"] = bool(c["worst_rel_diff"] <= 1e-6)
        red["max_rel_diff"] = max(c["worst_rel_diff"] for c in red["cases"].values())
        red["passed"] = all(c["passed"] for c in red["cases"].values())
        rep["reduction_1d"] = red
        rep["grid_operator"] = K.grid_operator_check(d, n=12 if d == 2 else 7)
        rep["passed"] = bool(rep["links"]["passed"] and red["passed"] and rep["grid_operator"]["passed"])
        rep["seconds"] = time.perf_counter() - t0
        rep["environment"] = environment()
        C.save_json(rep, C.results_dir(d) / f"consistency_d{d}.json")
        print(f"d={d}: links max rel {rep['links']['max_rel_diff']:.3e} passed={rep['links']['passed']}; "
              f"reduction max rel {red['max_rel_diff']:.3e} passed={red['passed']}; "
              f"operator passed={rep['grid_operator']['passed']}")
        for k, c in red["cases"].items():
            print(f"  {k}: {c['worst_rel_diff']:.3e}")
        rc |= 0 if rep["passed"] else 1
    sys.exit(rc)
