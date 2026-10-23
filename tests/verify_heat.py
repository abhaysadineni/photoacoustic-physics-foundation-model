"""Heat conduction: Gaussian spreading convergence (d = 1, 2, 3), smoke test and timing.
Writes results/verification/heat.json; exits non-zero on failure."""
import sys

from _tissue_harness import print_rep, report_and_exit, run

from pfm.solvers import heat

if __name__ == "__main__":
    rep, ok = run(heat.HeatSolver(), ("periodic", "dirichlet", "neumann"), heat)
    print_rep(rep)
    sys.exit(report_and_exit("heat", rep, ok))
