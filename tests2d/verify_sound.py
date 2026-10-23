"""Sound (d = 2, 3). SoundUniform: plane wave along the (body) diagonal and radially
symmetric Gaussian pulse (Hankel reference d = 2, exact r p formula d = 3). SoundLayered
shares the operator; its verification is the reduction to the 1D layered solver for every
acoustic bc (pfm2d/solvers/reduction.py: a band medium normal to x reproduces
pfm/solvers/sound.py SoundLayered, whose reflection-coefficient check passed in 1D), pass
if the maximum relative difference is <= 1e-6. Smoke, resolution and timing for both."""
import sys
from _harness import run, main_dims
from pfm2d import common as C
from pfm2d.solvers.sound import SoundUniform, SoundLayered
from pfm2d.solvers import reduction as RD
from resolution_sound import attach

if __name__ == "__main__":
    rc = 0
    for d in main_dims(sys.argv):
        rc |= run(SoundUniform(d))
        attach(d)   # resolution_study block from sound_resolution_d<d>.json, if present
        cases = {bc: RD.sound(d, "layered", bc, seed=30 + i) for i, bc in enumerate(C.BCS_SOUND)}
        worst = max(max(c["own_dt"]["max_rel_diff"], c["matched_dt"]["max_rel_diff"]) for c in cases.values())
        for c in cases.values():
            c["passed"] = bool(max(c["own_dt"]["max_rel_diff"], c["matched_dt"]["max_rel_diff"]) <= 1e-6)
        ver = {"system": "sound_layered", "check": "reduction to the 1D SoundLayered solver",
               "cases": cases, "max_rel_diff": worst, "passed": bool(worst <= 1e-6),
               "pass_rule": "max relative difference to the 1D solver <= 1e-6 (float32 output)"}
        rc |= run(SoundLayered(d), verify_result=ver)
    sys.exit(rc)
