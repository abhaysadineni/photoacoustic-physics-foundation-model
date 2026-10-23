"""Verification test for the burgers solver: exact-solution convergence, batch sanity checks
and timing. Writes results/verification/burgers.json; exits non-zero on failure."""
import sys

from _harness import run

from pfm.solvers.burgers import BurgersSolver

if __name__ == "__main__":
    sys.exit(run(BurgersSolver(), ("periodic", "dirichlet", "neumann")))
