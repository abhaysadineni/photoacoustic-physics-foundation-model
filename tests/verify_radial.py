"""Round 3: tissue systems in radial geometry (d = 2 cylinder, d = 3 sphere).

Exact or reference checks (pfm/solvers/radial_checks.py), bit-identity of Photothermal and
Thermoacoustic against the chain in d = 2, 3, smoke tests for every class, d and allowed
boundary condition (train and ood), and timing of 64 trajectories x 64 frames per class
and d at the training boundary condition. Writes results/verification/radial.json; exits
non-zero on failure."""
import sys
import time

from _tissue_harness import batch_check, environment, timing

from pfm import common
from pfm.solvers import coupled, heat, light, radial_checks as RC, sound

BCS = {"light": ("dirichlet", "neumann"), "heat": ("dirichlet", "neumann"),
       "sound_uniform": ("pressure_release", "rigid"),
       "sound_layered": ("pressure_release", "rigid"),
       "sound_lossy": ("pressure_release", "rigid"),
       "photothermal": ("pressure_release", "rigid"),
       "thermoacoustic": ("pressure_release", "rigid")}
TRAIN_BC = {"light": "dirichlet", "heat": "dirichlet"}       # others: pressure_release
CLASSES = (light.LightSolver, heat.HeatSolver, sound.SoundUniform, sound.SoundLayered,
           sound.SoundLossy, coupled.Photothermal, coupled.Thermoacoustic)


def show(k, r):
    print(f"  {k}: errors {r['error']}, orders {r['observed_order']}, passed={r['passed']}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    out = {"system": "radial", "dims": [2, 3], "allowed_bcs_d23": {k: list(v) for k, v in BCS.items()}}
    checks = {}
    for d in (2, 3):
        for bc in BCS["light"]:
            checks[f"light_d{d}_{bc}"] = RC.check_light(d, bc)
            checks[f"heat_d{d}_{bc}"] = RC.check_heat(d, bc)
        for bc in BCS["sound_uniform"]:
            checks[f"sound_uniform_d{d}_{bc}"] = RC.check_sound_uniform(d, bc)
            for kind in ("sourced", "stiff"):
                checks[f"sound_lossy_mode_d{d}_{bc}_{kind}"] = RC.check_lossy_mode(d, bc, kind)
            checks[f"sound_lossy_reference_d{d}_{bc}"] = RC.check_lossy_reference(d, bc)
        checks[f"sound_layered_d{d}_rigid"] = RC.check_sound_layered(d, "rigid")
        for k, v in RC.coupled_checks(d).items():
            checks[k] = v
    for k, r in checks.items():
        show(k, r)
    out["checks"] = checks
    cons = {}
    for d in (2, 3):
        cons[f"d{d}"] = coupled.consistency_check(dim=d)
        print(f"  consistency d{d}: max rel diff {cons[f'd{d}']['max_rel_diff']}, unused "
              f"{cons[f'd{d}']['unused_channel_max_abs']}, passed={cons[f'd{d}']['passed']}")
    out["consistency_with_chain"] = cons
    smoke, tim, smoke_ok = {}, {}, True
    for cls in CLASSES:
        s = cls()
        for d in (2, 3):
            rows = []
            for i, reg in enumerate(("train", "ood")):
                r, o = batch_check(s, BCS[s.name], seed=1000 * d + 100 * i, regime=reg,
                                   sample_kw={"dim": d})
                rows += r
                smoke_ok &= o
            smoke[f"{s.name}_d{d}"] = rows
            for r in rows:
                if r["failures"]:
                    print(f"  smoke {s.name} d{d} {r['regime']} {r['bc']}: {r['failures']}")
            bc = TRAIN_BC.get(s.name, "pressure_release")
            tim[f"{s.name}_d{d}"] = timing(s, (bc,), sample_kw={"dim": d}, seed=300 + d)
            r = tim[f"{s.name}_d{d}"][0]
            print(f"  timing {s.name} d{d} {bc}: {r['seconds']:.2f} s for 64 x 64 "
                  f"(substeps/frame {r['rk4_substeps_per_frame_min']}-"
                  f"{r['rk4_substeps_per_frame_max']})")
    out["smoke_test"] = {"passed": bool(smoke_ok), "rows": smoke}
    out["timing_64x64"] = tim
    checks_ok = all(r["passed"] for r in checks.values())
    cons_ok = all(c["passed"] for c in cons.values())
    out["checks_passed"] = bool(checks_ok)
    out["consistency_passed"] = bool(cons_ok)
    out["passed"] = bool(checks_ok and cons_ok and smoke_ok)
    out["seconds_total"] = time.perf_counter() - t0
    out["environment"] = environment()
    path = common.RESULTS / "verification" / "radial.json"
    common.save_json(out, path)
    print(f"  radial: checks {checks_ok}, consistency {cons_ok}, smoke {smoke_ok}; wrote {path}")
    sys.exit(0 if out["passed"] else 1)
