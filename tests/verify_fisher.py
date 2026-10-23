"""Verification test for the fisher solver: exact-solution convergence, batch sanity checks
and timing. Writes results/verification/fisher.json; exits non-zero on failure."""
import sys

from _harness import run

from pfm.solvers.fisher import FisherSolver

if __name__ == "__main__":
    sys.exit(run(FisherSolver(), ("periodic", "dirichlet", "neumann"), bounds=(-1e-6, 1 + 1e-6),
                 scale_floor=1.0))
