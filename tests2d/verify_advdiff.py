"""Advection-diffusion (d = 2, 3): translated spreading Gaussian; smoke, resolution, timing."""
import sys
from _harness import run, main_dims
from pfm2d.solvers.advdiff import AdvDiff

if __name__ == "__main__":
    rc = 0
    for d in main_dims(sys.argv):
        rc |= run(AdvDiff(d))
    sys.exit(rc)
