"""Verification test for the transport solver: exact-solution convergence, batch sanity checks
and timing. Writes results/verification/transport.json; exits non-zero on failure."""
import sys

from _harness import run

from pfm.solvers.transport import TransportSolver

if __name__ == "__main__":
    sys.exit(run(TransportSolver(), ("periodic", "dirichlet", "neumann")))
