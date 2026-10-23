"""Heat (d = 2, 3): Gaussian spreading on a periodic box; smoke, resolution, timing."""
import sys
from _harness import run, main_dims
from pfm2d.solvers.heat import Heat

if __name__ == "__main__":
    rc = 0
    for d in main_dims(sys.argv):
        rc |= run(Heat(d))
    sys.exit(rc)
