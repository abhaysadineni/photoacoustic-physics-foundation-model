"""Telegraph equation (held-out system H2), dimensionless, L = 1:

    d2p/dt2 + 2 gamma dp/dt = c^2 d2p/dx2

solved as the first-order system
    dp/dt = q ,   dq/dt = c^2 d2p/dx2 - 2 gamma q ,
with q(0) = 0. Only p is exposed, in channel 2 (common.CH_PRESSURE); channels 0 and 1 are
zero. Like the sound systems, four input frames carry the history needed to infer q.

Numerics (shared by the four generic solvers)
---------------------------------------------
- Cell-centred grid on [0, L), internal resolution n_internal (default 512), stored after
  cell-average downsampling to 128 points (common.downsample).
- Method of lines: second-order central differences in space (d2p/dx2 by the three-point
  stencil), classical RK4 in time. Stable step: the advective rule with the wave speed,
  0.4 dx / c. There is no explicit diffusion coefficient, so the 0.2 dx^2 / D rule has no
  term to apply to (the diffusion-like limit, D_eff = c^2 / (2 gamma), arises from the
  damping and does not constrain an explicit (p, q) scheme). A damping limit 1 / gamma is
  added (RK4 is stable for real eigenvalues down to about -2.78 / dt and the stiffest real
  eigenvalue is above -2 gamma); it never binds in the ranges below but guards large gamma.
  dt_stable = min(0.4 dx / c, 1 / gamma), then shrunk to dt_frame / ceil(dt_frame /
  dt_stable) so frames land exactly on multiples of dt_frame. Promised spatial order 2.
- Boundary conditions by ghost cells: "periodic" wraps; "pressure_release" is odd
  reflection of p (p_ghost = -p_edge, p = 0 at the wall); "rigid" is even reflection of p
  (dp/dx = 0 at the wall, the acoustic rigid-wall condition, since velocity u = 0 implies
  dp/dx = 0). q = dp/dt inherits the parity of p and enters no spatial derivative, so it
  needs no ghost.
- Initial condition: p0 = sum of 1 to 4 Gaussians, centres uniform (in [0.2, 0.8] with
  walls), standard deviations uniform in [0.03, 0.15], amplitudes with |a| uniform in
  [0.2, 1.0] and random sign; walls: multiplied by a C2 taper (quintic smoothstep over 0.1
  from each wall). q0 = 0.
- Frame interval rule: in the wave-like limit a pulse moving at the reference speed
  c_ref = 1 moves about one output cell per frame: dt_frame = (L / 128) / 1 = 1/128. Over
  the training range c in [0.5, 2] that is 0.5 to 2 output cells per frame. In the most
  diffusive training case (gamma = 30 c / L, D_eff = c^2 / (2 gamma) = c L / 60) the
  diffusion length per frame sqrt(2 D_eff dt_frame) is 1.46 to 2.92 output cells (c = 0.5
  to 2), so the field still changes visibly per frame across the whole range.

Exact check
-----------
Single Fourier mode p(x, 0) = cos(k x), q(x, 0) = 0, periodic, k = 2 pi m / L. Then
p = exp(-gamma t) cos(k x) [cos(w t) + (gamma / w) sin(w t)],  w = sqrt(c^2 k^2 - gamma^2)
when gamma < c k, and
p = exp(-gamma t) cos(k x) [cosh(b t) + (gamma / b) sinh(b t)],  b = sqrt(gamma^2 - c^2 k^2)
when gamma > c k. On cell centres cos(k x_i) is an exact eigenvector of the discrete
Laplacian (eigenvalue -(2 / dx)^2 sin^2(k dx / 2)), so the error is purely the O((k dx)^2)
eigenvalue error plus RK4 time error.
"""
from __future__ import annotations

import numpy as np

from pfm import common
from pfm.solvers import _mol

C_REF = 1.0
DT_FRAME = (1.0 / 128.0) / C_REF    # = 1/128

PARAM_RANGES = {
    # c log-uniform; gamma = g * c / L with g log-uniform
    "train": {"c": [0.5, 2.0], "gamma_over_c_per_L": [0.01, 30.0]},
    "ood": {"c": [[0.25, 0.5], [2.0, 4.0]], "gamma_over_c_per_L": [[0.002, 0.01], [30.0, 100.0]]},
}

BCS = ("periodic", "pressure_release", "rigid")
_KIND = {"periodic": "periodic", "pressure_release": "odd", "rigid": "even"}


def _rhs_factory(c, gamma, dx, bc):
    """State y has shape (B, 2, n): y[:, 0] = p, y[:, 1] = q. c, gamma: (B, 1)."""
    kind = _KIND.get(bc)
    if kind is None:
        raise ValueError(f"telegraph: unknown bc {bc!r}")
    c2 = c * c

    def rhs(y, t, sl):
        p, q = y[:, 0], y[:, 1]
        out = np.empty_like(y)
        out[:, 0] = q
        out[:, 1] = c2[sl] * _mol.lap(_mol.pad(p, kind), dx) - 2.0 * gamma[sl] * q
        return out
    return rhs


def dt_stable(c, gamma, dx):
    return np.minimum(_mol.CFL_ADV * dx / c, 1.0 / np.maximum(gamma, 1e-12))


def mode_exact(x, t, c, gamma, k):
    ck = c * k
    if gamma < ck:
        w = np.sqrt(ck * ck - gamma * gamma)
        br = np.cos(w * t) + (gamma / w) * np.sin(w * t)
    elif gamma > ck:
        b = np.sqrt(gamma * gamma - ck * ck)
        br = np.cosh(b * t) + (gamma / b) * np.sinh(b * t)
    else:
        br = 1.0 + gamma * t
    return np.exp(-gamma * t) * br * np.cos(k * x)


class TelegraphSolver(common.Solver):
    name = "telegraph"
    channels = (common.CH_PRESSURE,)
    order = 2.0

    def sample_params(self, rng, regime="train"):
        if regime == "train":
            c = _mol.log_uniform(rng, *PARAM_RANGES["train"]["c"])
            g = _mol.log_uniform(rng, *PARAM_RANGES["train"]["gamma_over_c_per_L"])
        elif regime == "ood":
            c = _mol.draw_outside(rng, PARAM_RANGES["ood"]["c"])
            g = _mol.draw_outside(rng, PARAM_RANGES["ood"]["gamma_over_c_per_L"])
        else:
            raise ValueError(regime)
        return {"c": c, "gamma": g * c / _mol.LENGTH, "gamma_over_c_per_L": g, "regime": regime}

    def simulate(self, params, bc, n_frames, rng, n_internal=512):
        return self.simulate_batch([params], bc, n_frames, rng, n_internal)[0]

    def simulate_batch(self, params_list, bc, n_frames, rng, n_internal=512):
        if bc not in BCS:
            raise ValueError(f"telegraph: unknown bc {bc!r}")
        periodic = bc == "periodic"
        B = len(params_list)
        dx = _mol.LENGTH / n_internal
        x = common.cell_centres(n_internal, _mol.LENGTH)
        y0 = np.zeros((B, 2, n_internal))
        ics = []
        for b, p in enumerate(params_list):
            ic = p.get("ic") or _mol.draw_gaussians(rng, periodic, _mol.signed_amplitudes)
            y0[b, 0] = _mol.eval_gaussians(ic, x, periodic)
            ics.append(ic)
        c = np.array([p["c"] for p in params_list], dtype=float).reshape(B, 1)
        gamma = np.array([p["gamma"] for p in params_list], dtype=float).reshape(B, 1)
        nsub = _mol.n_substeps(DT_FRAME, dt_stable(c[:, 0], gamma[:, 0], dx))
        out = np.zeros((B, n_frames, common.N_CH, common.N_OUT), dtype=np.float32)

        def on_frame(k, y):
            out[:, k, common.CH_PRESSURE, :] = common.downsample(y[:, 0], common.N_OUT)

        _mol.rk4_frames(_rhs_factory(c, gamma, dx, bc), y0, DT_FRAME, nsub, n_frames, on_frame)
        trajs = []
        for b, p in enumerate(params_list):
            prm = dict(p)
            prm.update({"ic": ics[b], "dt": DT_FRAME / int(nsub[b]), "n_internal": n_internal})
            trajs.append(common.Trajectory(fields=out[b], system=self.name, params=prm,
                                           dt_frame=DT_FRAME, length=_mol.LENGTH, bc=bc,
                                           channels=(common.CH_PRESSURE,)))
        return trajs

    # ------------------------------------------------------------------ verification
    CASES = {
        "underdamped": {"c": 1.0, "gamma": 1.0, "m": 2, "T": 1.0},
        "overdamped": {"c": 1.0, "gamma": 20.0, "m": 2, "T": 1.0},
    }

    def _mode_error(self, case, n):
        dx = _mol.LENGTH / n
        x = common.cell_centres(n, _mol.LENGTH)
        k = 2 * np.pi * case["m"] / _mol.LENGTH
        y0 = np.zeros((1, 2, n))
        y0[0, 0] = np.cos(k * x)
        c, g = case["c"], case["gamma"]
        rhs = _rhs_factory(np.array([[c]]), np.array([[g]]), dx, "periodic")
        y, nsteps = _mol.integrate_to(rhs, y0, case["T"], dt_stable(c, g, dx))
        exact = mode_exact(x, case["T"], c, g, k)
        return float(common.rel_l2(y[0, 0], exact)), nsteps

    def verify(self, n_list=(128, 256, 512)):
        cases = {}
        for label, case in self.CASES.items():
            errs, steps = [], []
            for n in n_list:
                e, s = self._mode_error(case, n)
                errs.append(e)
                steps.append(s)
            k = 2 * np.pi * case["m"] / _mol.LENGTH
            form = "cos/sin" if case["gamma"] < case["c"] * k else "cosh/sinh"
            rep = common.convergence_report(
                self.name,
                f"{label}: periodic single mode m={case['m']} (k={k:.4g}), c={case['c']}, "
                f"gamma={case['gamma']}, c k={case['c'] * k:.4g}, {form} exact form, q(0)=0, "
                f"final time T={case['T']}",
                n_list, errs, self.order)
            rep["rk4_steps"] = steps
            rep["exact_amplitude_at_T"] = float(mode_exact(0.0, case["T"], case["c"],
                                                           case["gamma"], k))
            cases[label] = rep
        # supplementary wall checks (reported only): sin(m pi x) for pressure release,
        # cos(m pi x) for rigid; both are exact discrete eigenvectors with these ghosts
        walls = {}
        for bc, shape in (("pressure_release", "sin"), ("rigid", "cos")):
            for label, case in self.CASES.items():
                case_w = dict(case, m=4)
                errs = [self._wall_error(case_w, n, bc, shape) for n in n_list]
                walls[f"{bc}/{label}"] = common.convergence_report(
                    self.name, f"{bc}, {shape}(4 pi x), {label} c={case['c']}, "
                               f"gamma={case['gamma']}, T={case['T']} (reported, not part of "
                               f"passed)", n_list, errs, self.order)
        main = dict(cases["underdamped"])
        main["check"] = ("single periodic Fourier mode, underdamped (top level) and "
                         "overdamped (in cases); passed requires both")
        main["cases"] = cases
        main["passed"] = bool(all(c["passed"] for c in cases.values()))
        main["error_metric"] = "relative L2 of p on the internal grid at t = T (point values)"
        main["supplementary_walls"] = walls
        return main

    def _wall_error(self, case, n, bc, shape):
        dx = _mol.LENGTH / n
        x = common.cell_centres(n, _mol.LENGTH)
        k = np.pi * case["m"] / _mol.LENGTH
        f = np.sin if shape == "sin" else np.cos
        y0 = np.zeros((1, 2, n))
        y0[0, 0] = f(k * x)
        c, g = case["c"], case["gamma"]
        rhs = _rhs_factory(np.array([[c]]), np.array([[g]]), dx, bc)
        y, _ = _mol.integrate_to(rhs, y0, case["T"], dt_stable(c, g, dx))
        amp = mode_exact(0.0, case["T"], c, g, k)     # time factor (cos(0) = 1)
        return float(common.rel_l2(y[0, 0], amp * f(k * x)))


SOLVER = TelegraphSolver()
