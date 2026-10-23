"""Light (d = 2, 3): Green's function on a periodic box and the record window through the
production path for each bc; smoke, resolution, timing. Usage: verify_light.py [2] [3]."""
import sys
from _harness import run, main_dims
from pfm2d.solvers.light import Light

if __name__ == "__main__":
    rc = 0
    for d in main_dims(sys.argv):
        rc |= run(Light(d))
    sys.exit(rc)
