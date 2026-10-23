"""Verification test for the telegraph solver: exact-solution convergence, batch sanity checks
and timing. Writes results/verification/telegraph.json; exits non-zero on failure."""
import sys

from _harness import run

from pfm.solvers.telegraph import TelegraphSolver

if __name__ == "__main__":
    sys.exit(run(TelegraphSolver(), ("periodic", "pressure_release", "rigid")))
