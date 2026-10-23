"""End-to-end exact check for viscous Burgers through simulate_batch (audit MINOR 6 and 8).
Writes results/verification/e2e_burgers.json; exits non-zero on failure.

Exact solutions (Cole-Hopf). If theta_t = nu theta_xx and theta > 0, then
u = -2 nu theta_x / theta solves u_t + u u_x = nu u_xx (the production flux form
u_t + (u^2/2)_x = nu u_xx is the same PDE). With
    theta = 1 + a exp(-nu k^2 t) cos(k x),  0 < a < 1,
    u     = 2 nu a k exp(-nu k^2 t) sin(k x) / (1 + a exp(-nu k^2 t) cos(k x)).
  periodic : k = 2 pi m (period 1).
  dirichlet: k = pi m. Then u = 0 at x = 0 and x = 1 for all t, and u is odd about both
             walls (u(1 + s) = -u(1 - s)), which is exactly the homogeneous condition the
             production odd-reflection closure imposes. This is an exact homogeneous-wall
             solution, so the production Dirichlet path can be checked too.
max|u0| = 2 nu a k / sqrt(1 - a^2) (at cos(kx) = -a); a close to 1 gives a steep,
shock-like ramp of width about sqrt(2 (1 - a)) / k.

The tanh travelling wave (verify()'s check) needs inhomogeneous, time-dependent wall values
u(0, t) = uL, u(1, t) = g(t). The production path does NOT support this: simulate_batch calls
_rhs_factory(nu, dx, bc) without wall_values, so "dirichlet" is always u = 0 at the walls.
The tanh wave is therefore not run through the production path. A supplementary near-exact
case is reported instead (not part of "passed"): the tanh wave with the production "neumann"
(zero-gradient) closure, which is not exact because the wave's gradient at the walls is
small but non-zero; the wall gradient is reported so the contamination can be bounded.
"""
import sys

import numpy as np

import _e2e
from pfm import common
from pfm.solvers.burgers import DT_FRAME, BurgersSolver, travelling_wave


def cole_hopf(nu, a, k):
    def u(x, t):
        e = a * np.exp(-nu * k * k * t)
        return 2.0 * nu * k * e * np.sin(k * x) / (1.0 + e * np.cos(k * x))
    return u


def ch_case(name, nu, a, k, ch, bc_note):
    u = cole_hopf(nu, a, k)
    _e2e.EXACT_IC[name] = lambda x: u(x, 0.0)

    def by_n(n):
        x = common.cell_centres(n, 1.0)
        return {"u_max": float(np.max(np.abs(u(x, 0.0))))}     # rescale factor exactly 1.0

    umax = 2 * nu * a * k / np.sqrt(1 - a * a)
    return {"name": name, "params": {"nu": nu, "ic": {"exact_ic": name}, "channel": ch},
            "params_by_n": by_n, "exact": u, "channel": ch,
            "notes": f"Cole-Hopf, nu={nu}, a={a}, k={k / np.pi:g} pi; max|u0| (continuum) = {umax:.4f}, "
                     f"Re = max|u0| / nu = {umax / nu:.1f}; ramp width ~ sqrt(2(1-a))/k = "
                     f"{np.sqrt(2 * (1 - a)) / k:.4f} ({np.sqrt(2 * (1 - a)) / k * 512:.1f} cells at 512); {bc_note}"}


def main():
    s = BurgersSolver()
    pi = np.pi
    groups = [
        _e2e.run_group(s, DT_FRAME, "periodic", [
            ch_case("colehopf_periodic_nu0.01_a0.97", 0.01, 0.97, 2 * pi, 0, "k = 2 pi"),
            ch_case("colehopf_periodic_nu0.004_a0.995", 0.004, 0.995, 2 * pi, 2,
                    "k = 2 pi; nu at the bottom of the training range"),
        ]),
        {f"{k}__dirichlet": v for k, v in _e2e.run_group(s, DT_FRAME, "dirichlet", [
            ch_case("colehopf_wall_nu0.01_a0.95", 0.01, 0.95, pi, 1, "k = pi, u = 0 at both walls"),
            ch_case("colehopf_wall_nu0.005_a0.98_m2", 0.005, 0.98, 2 * pi, 0,
                    "k = 2 pi on the wall domain, u = 0 at both walls"),
        ]).items()},
    ]
    # residual check of the closed form (independent of the solver)
    resid = {}
    for name, nu, a, k in (("periodic_nu0.01_a0.97", 0.01, 0.97, 2 * pi), ("wall_nu0.005_a0.98_m2", 0.005, 0.98, 2 * pi)):
        u = cole_hopf(nu, a, k)
        x = np.linspace(0.0, 1.0, 401)
        r, ut = _e2e.fd_residual(u, lambda uu, ut, ux, uxx: ut + uu * ux - nu * uxx, x, 0.3)
        resid[name] = {"max_abs_residual_fd": r, "max_abs_u_t": ut}

    # supplementary (not exact): tanh wave with the production zero-gradient closure
    c = dict(uL=1.0, uR=0.0, nu=0.01, x0=0.3)
    tw = lambda x, t: travelling_wave(x, t, c["uL"], c["uR"], c["nu"], c["x0"])
    _e2e.EXACT_IC["tanh"] = lambda x: tw(x, 0.0)

    def by_n(n):
        x = common.cell_centres(n, 1.0)
        return {"u_max": float(np.max(np.abs(tw(x, 0.0))))}

    sup = _e2e.run_group(s, DT_FRAME, "neumann", [
        {"name": "tanh_wave_neumann_SUPPLEMENTARY", "params": {"nu": c["nu"], "ic": {"exact_ic": "tanh"}, "channel": 0},
         "params_by_n": by_n, "exact": tw, "channel": 0,
         "notes": "NOT an exact check: zero-gradient walls, tanh wave uL=1, uR=0, nu=0.01, x0=0.3"}])
    tt = np.arange(64) * DT_FRAME
    g = lambda x, t: (tw(x + 1e-6, t) - tw(x - 1e-6, t)) / 2e-6
    sup["tanh_wave_neumann_SUPPLEMENTARY"]["max_abs_exact_wall_gradient"] = float(
        max(np.abs(g(0.0, tt)).max(), np.abs(g(1.0, tt)).max()))

    rep = _e2e.summarise("burgers", DT_FRAME, groups, [
        {"case": "tanh travelling wave (uL = 1, uR = 0) with exact wall values",
         "reason": "simulate_batch supports only homogeneous walls: _rhs_factory is called without "
                   "wall_values, so 'dirichlet' imposes u = 0 and cannot carry u(0, t) = 1"}],
        extra={"closed_form_residual": resid, "supplementary_not_in_passed": sup})
    a = sup["tanh_wave_neumann_SUPPLEMENTARY"]
    print(f"supplementary tanh/neumann (not exact): max512 {a['by_n']['512']['max_rel_l2']:.3e}, "
          f"max1024 {a['by_n']['1024']['max_rel_l2']:.3e}, factor {a['factor_max_512_over_1024']:.3f}, "
          f"max |exact u_x| at walls {a['max_abs_exact_wall_gradient']:.2e}")
    print("closed-form residuals:", resid)
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
