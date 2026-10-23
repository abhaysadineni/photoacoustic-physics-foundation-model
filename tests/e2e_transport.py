"""End-to-end exact check for transport through simulate_batch (audit MINOR 6 and 7).
Writes results/verification/e2e_transport.json; exits non-zero on failure.

Cases (Gaussian initial data through the production _mol.eval_gaussians; centres, widths
and amplitudes inside the sampled ranges, widths in [0.03, 0.15]):
  periodic : exact translation u(x, t) = u0((x - v t) mod 1) of the periodic image sum.
    - typical  : v = +1.3, three Gaussians, widths 0.06, 0.10, 0.14
    - narrowest: v = -1.5, one Gaussian of width 0.03 (training-range corner: fastest speed,
                 narrowest width), and the same Gaussian at v = +0.25 (slowest speed)
  dirichlet and neumann walls: exact translation of the tapered initial data, zero entering
  from the inflow wall, the signal leaving through the outflow wall (no reflection):
    u(x, t) = u0(x - v t), with u0 = 0 outside [0, 1] (the C2 taper is 0 there).
    - exit right: v = +1.0, a Gaussian at 0.78 crosses x = 1 from t ~ 0.1 and has left by the
                  last frame (t = 63/128); two others stay inside
    - exit left : v = -1.25, a Gaussian at 0.22 leaves through x = 0; one at 0.62 is leaving
                  at the end
  The continuum solution is the same for both wall closures (transport.py docstring), so the
  same exact solution is the reference for dirichlet and for neumann.
"""
import sys

import numpy as np

import _e2e
from pfm.solvers import _mol
from pfm.solvers.transport import DT_FRAME, TransportSolver

ORIG = _e2e._ORIG_EVAL


def periodic_case(name, v, ic, ch, notes):
    def exact(x, t):
        return ORIG(ic, np.mod(x - v * t, 1.0), True)
    return {"name": name, "params": {"v": v, "ic": ic, "channel": ch}, "exact": exact, "channel": ch,
            "notes": notes}


def wall_case(name, v, ic, ch, notes):
    def exact(x, t):
        xi = x - v * t
        return np.where((xi >= 0.0) & (xi <= 1.0), ORIG(ic, np.clip(xi, 0.0, 1.0), False), 0.0)
    return {"name": name, "params": {"v": v, "ic": ic, "channel": ch}, "exact": exact, "channel": ch,
            "notes": notes}


IC_TYP = {"centres": [0.15, 0.5, 0.8], "widths": [0.06, 0.10, 0.14], "amplitudes": [0.9, -0.5, 0.4]}
IC_NARROW = {"centres": [0.4], "widths": [0.03], "amplitudes": [0.7]}
IC_RIGHT = {"centres": [0.78, 0.3, 0.5], "widths": [0.05, 0.04, 0.12], "amplitudes": [0.8, -0.5, 0.3]}
IC_LEFT = {"centres": [0.22, 0.62, 0.78], "widths": [0.04, 0.08, 0.06], "amplitudes": [0.7, -0.6, 0.4]}


def main():
    s = TransportSolver()
    groups = []
    groups.append(_e2e.run_group(s, DT_FRAME, "periodic", [
        periodic_case("periodic_typical_v+1.3", 1.3, IC_TYP, 0, "three Gaussians, widths 0.06-0.14"),
        periodic_case("periodic_narrowest_v-1.5", -1.5, IC_NARROW, 2,
                      "training-range corner: |v| = 1.5 (max), width 0.03 (min)"),
        periodic_case("periodic_narrowest_v+0.25", 0.25, IC_NARROW, 1,
                      "width 0.03 (min) at the slowest training speed |v| = 0.25 (brackets the corner)"),
    ]))
    for bc in ("dirichlet", "neumann"):
        groups.append({f"{k}__{bc}": v for k, v in _e2e.run_group(s, DT_FRAME, bc, [
            wall_case("exit_right_v+1.0", 1.0, IC_RIGHT, 1,
                      "Gaussian at 0.78 (width 0.05) leaves through the outflow wall x = 1"),
            wall_case("exit_left_v-1.25", -1.25, IC_LEFT, 0,
                      "Gaussian at 0.22 (width 0.04) leaves through the outflow wall x = 0"),
        ]).items()})
    # how much of the exiting Gaussian has left, from the exact cell averages
    t_end = (_e2e.N_FRAMES - 1) * DT_FRAME
    left_frac = {}
    for name, v, ic, which in (("exit_right_v+1.0", 1.0, IC_RIGHT, 0), ("exit_left_v-1.25", -1.25, IC_LEFT, 0)):
        one = {k: [ic[k][which]] for k in ic}
        m0 = np.sum(_e2e.cell_average(lambda xx: ORIG(one, xx, False)))
        m1 = np.sum(_e2e.cell_average(lambda xx: np.where(
            (xx - v * t_end >= 0) & (xx - v * t_end <= 1), ORIG(one, np.clip(xx - v * t_end, 0, 1), False), 0.0)))
        left_frac[name] = float(1.0 - m1 / m0)
    rep = _e2e.summarise("transport", DT_FRAME, groups, [],
                         extra={"fraction_of_exiting_gaussian_gone_at_last_frame": left_frac,
                                "note_walls": "dirichlet and neumann share one exact solution (identical in "
                                              "the continuum for data tapered to zero at the walls)"})
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
