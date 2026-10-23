"""End-to-end exact check for Fisher-KPP through simulate_batch (audit MINOR 6 and 8).
Writes results/verification/e2e_fisher.json; exits non-zero on failure.

Ablowitz-Zeppetella wave: NOT run through the production path. It needs inhomogeneous,
time-dependent wall values (u -> 1 behind the front at x = 0), and simulate_batch calls
_rhs_factory(D, r, dx, bc) without wall_values, so "dirichlet" is always u = 0 and
"neumann" zero flux at both walls. The AZ wave satisfies neither exactly.

Exact cases used ("passed"):
  Spatially uniform logistic solution u(t) = u0 e^{rt} / (1 - u0 + u0 e^{rt}), exact for the
  PDE for any D. Periodic (as specified), and also with the production "neumann" closure,
  which a uniform state satisfies exactly (zero gradient). Uniform data make the discrete
  Laplacian vanish identically, so these cases test the reaction term, RK4, per-trajectory
  substeps, downsampling and storage, but not the spatial operator.

Supplementary exact cases (reported, NOT part of "passed"): r = 0 reduces the PDE to the
heat equation; single modes cos(2 pi x) (periodic), cos(pi x) (neumann) and sin(pi x)
(dirichlet) decay as exp(-D k^2 t) and satisfy the production wall closures exactly. They
exercise the diffusion operator and every wall closure through simulate_batch, but r = 0
lies outside the sampled range r in [2, 20] (and the logistic term is then switched off),
so they are kept separate.
"""
import sys

import numpy as np

import _e2e
from pfm.solvers.fisher import DT_FRAME, FisherSolver


def logistic(u0, r):
    def u(x, t):
        e = np.exp(r * t)
        return np.broadcast_to(u0 * e / (1.0 - u0 + u0 * e), np.shape(x)).astype(float)
    return u


def logistic_case(name, D, r, u0, ch):
    _e2e.EXACT_IC[name] = lambda x: np.full_like(x, u0)
    t_end = 63 * DT_FRAME
    return {"name": name, "params": {"D": D, "r": r, "ic": {"exact_ic": name}, "channel": ch},
            "exact": logistic(u0, r), "channel": ch,
            "notes": f"uniform logistic, u0={u0}, r={r}, D={D}; u at last frame "
                     f"{logistic(u0, r)(np.zeros(1), t_end)[0]:.6f}"}


def heat_case(name, D, k, shape, ch):
    f = np.cos if shape == "cos" else np.sin
    _e2e.EXACT_IC[name] = lambda x: 0.8 * f(k * x)
    return {"name": name, "params": {"D": D, "r": 0.0, "ic": {"exact_ic": name}, "channel": ch},
            "exact": lambda x, t: 0.8 * np.exp(-D * k * k * t) * f(k * x), "channel": ch,
            "notes": f"r = 0 (heat equation), 0.8 {shape}({k / np.pi:g} pi x), D={D}"}


def main():
    s = FisherSolver()
    pi = np.pi
    groups = [
        _e2e.run_group(s, DT_FRAME, "periodic", [
            logistic_case("logistic_periodic_r6_u0.05", 0.005, 6.0, 0.05, 0),
            logistic_case("logistic_periodic_r20_u0.01", 0.001, 20.0, 0.01, 1),
            logistic_case("logistic_periodic_r2_u0.1", 0.01, 2.0, 0.1, 2),
        ]),
        {f"{k}__neumann": v for k, v in _e2e.run_group(s, DT_FRAME, "neumann", [
            logistic_case("logistic_neumann_r6_u0.05", 0.005, 6.0, 0.05, 2),
        ]).items()},
    ]
    sup = {}
    sup.update(_e2e.run_group(s, DT_FRAME, "periodic", [heat_case("heat_r0_periodic_cos2pix", 0.01, 2 * pi, "cos", 0)]))
    sup.update({f"{k}__neumann": v for k, v in _e2e.run_group(
        s, DT_FRAME, "neumann", [heat_case("heat_r0_neumann_cospix", 0.01, pi, "cos", 1)]).items()})
    sup.update({f"{k}__dirichlet": v for k, v in _e2e.run_group(
        s, DT_FRAME, "dirichlet", [heat_case("heat_r0_dirichlet_sinpix", 0.01, pi, "sin", 2)]).items()})
    rep = _e2e.summarise("fisher", DT_FRAME, groups, [
        {"case": "Ablowitz-Zeppetella travelling wave",
         "reason": "needs inhomogeneous time-dependent wall values; simulate_batch supports only "
                   "homogeneous dirichlet (u = 0) and neumann (zero flux) walls"}],
        extra={"supplementary_not_in_passed": sup,
               "note_uniform": "uniform data: the discrete Laplacian is identically zero, so the spatial "
                               "operator is not tested by the logistic cases; see the supplementary r = 0 modes"})
    print("supplementary (r = 0, outside sampled range, not in passed):")
    _e2e.print_table({"system": "fisher supplementary", "dt_frame": DT_FRAME, "n_frames": 64, "cases": sup})
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
