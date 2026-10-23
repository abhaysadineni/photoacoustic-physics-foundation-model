"""Viscous Burgers equation (shock-forming flow), dimensionless, L = 1:

    du/dt + d(u^2/2)/dx = nu d2u/dx2      (conservative flux form)

Numerics (shared by the four generic solvers)
---------------------------------------------
- Cell-centred grid on [0, L), internal resolution n_internal (default 512), stored after
  cell-average downsampling to 128 points (common.downsample).
- Method of lines: second-order central differences in space, classical RK4 in time.
  Flux term: F_i = u_i^2 / 2 and d F/dx ~ (F_i+1 - F_i-1) / (2 dx), with F evaluated on the
  ghost-padded array. Viscous term: (u_i+1 - 2 u_i + u_i-1) / dx^2.
  Stable step dt_stable = min(0.4 dx / max|u0|, 0.2 dx^2 / nu), then shrunk to
  dt_frame / ceil(dt_frame / dt_stable) so frames land exactly on multiples of dt_frame.
  max|u0| bounds |u| for all later times by the maximum principle for viscous Burgers on
  periodic, homogeneous Dirichlet and homogeneous Neumann domains. Promised spatial order 2.
- Boundary conditions by ghost cells: "periodic" wraps; "dirichlet" is odd reflection
  (u_ghost = -u_edge, u = 0 at the wall); "neumann" is even reflection (zero gradient).
  For verification the odd reflection is used with the exact, time-dependent wall value g:
  u_ghost = 2 g(t) - u_edge.
- Initial condition: sum of 1 to 4 Gaussians, centres uniform (in [0.2, 0.8] with walls),
  standard deviations uniform in [0.03, 0.15], amplitudes with random sign and magnitude
  uniform in [0.2, 1.0]; walls: multiplied by a C2 taper (quintic smoothstep over 0.1 from
  each wall). The sum is then RESCALED so that max|u0| = u_max exactly, where u_max is a
  sampled parameter. This keeps the Reynolds-type number Re = u_max L / nu controlled.
- Resolution rule (the spec's "keep |u| L / nu modest" condition, made explicit): the
  thinnest structure is a viscous shock u = s - (du/2) tanh(du (x - s t) / (4 nu)), whose
  tanh length scale is delta = 4 nu / du. The jump du cannot exceed 2 u_max, so
  delta >= 2 nu / u_max. Training ranges guarantee delta >= 4 dx at n_internal = 512
  (nu >= u_max / 256) and cell Peclet number u_max dx / nu <= 0.49 < 2 (the threshold
  above which central differences produce wiggles at a steady front). Out-of-distribution
  draws go down to delta >= 2 dx and cell Peclet <= 0.98. An empirical self-convergence
  check at the worst-case corners is in verify()["resolution_check"].
- Frame interval rule: features moving at the reference speed u_ref = 0.5 move about one
  output cell per frame: dt_frame = (L / 128) / 0.5 = 1/64.
- channels = "random": simulate draws one channel with rng and writes the field there.
"""
from __future__ import annotations

import numpy as np

from pfm import common
from pfm.solvers import _mol

U_REF = 0.5
DT_FRAME = (1.0 / 128.0) / U_REF   # = 1/64

PARAM_RANGES = {
    # nu log-uniform; u_max uniform (same range in both regimes: the OOD axis is nu, and
    # therefore also Re = u_max L / nu)
    "train": {"nu": [0.004, 0.03], "u_max": [0.25, 1.0], "Re": [8.3, 250.0]},
    "ood": {"nu": [[0.002, 0.004], [0.03, 0.06]], "u_max": [0.25, 1.0], "Re": [4.2, 500.0]},
}

BCS = ("periodic", "dirichlet", "neumann")


def _rhs_factory(nu, dx, bc, wall_values=None):
    """nu: (B, 1). wall_values: optional callable t -> (gL, gR), each (B,), for odd
    reflection with inhomogeneous wall values (verification only)."""
    if bc == "periodic":
        kind = "periodic"
    elif bc == "dirichlet":
        kind = "odd"
    elif bc == "neumann":
        kind = "even"
    else:
        raise ValueError(f"burgers: unknown bc {bc!r}")

    def rhs(u, t, sl):
        if wall_values is not None:
            gl, gr = wall_values(t)
            up = _mol.pad(u, kind, kind, gl, gr)
        else:
            up = _mol.pad(u, kind)
        flux = 0.5 * up * up
        return -_mol.ddx(flux, dx) + _mol.take(nu, sl) * _mol.lap(up, dx)
    return rhs


def dt_stable(umax, nu, dx):
    return np.minimum(_mol.CFL_ADV * dx / np.maximum(umax, 1e-12), _mol.CFL_DIFF * dx * dx / nu)


def travelling_wave(x, t, uL, uR, nu, x0):
    s = 0.5 * (uL + uR)
    return s - 0.5 * (uL - uR) * np.tanh((uL - uR) * (x - x0 - s * t) / (4.0 * nu))


class BurgersSolver(common.Solver):
    name = "burgers"
    channels = "random"
    order = 2.0

    def sample_params(self, rng, regime="train"):
        if regime == "train":
            nu = _mol.log_uniform(rng, *PARAM_RANGES["train"]["nu"])
        elif regime == "ood":
            nu = _mol.draw_outside(rng, PARAM_RANGES["ood"]["nu"])
        else:
            raise ValueError(regime)
        u_max = float(rng.uniform(*PARAM_RANGES[regime]["u_max"]))
        return {"nu": nu, "u_max": u_max, "Re": u_max / nu, "regime": regime}

    def simulate(self, params, bc, n_frames, rng, n_internal=512):
        return self.simulate_batch([params], bc, n_frames, rng, n_internal)[0]

    def simulate_batch(self, params_list, bc, n_frames, rng, n_internal=512):
        if bc not in BCS:
            raise ValueError(f"burgers: unknown bc {bc!r}")
        periodic = bc == "periodic"
        B = len(params_list)
        dx = _mol.LENGTH / n_internal
        x = common.cell_centres(n_internal, _mol.LENGTH)
        u0 = np.empty((B, n_internal))
        ics, chans = [], []
        for b, p in enumerate(params_list):
            ic = p.get("ic") or _mol.draw_gaussians(rng, periodic, _mol.signed_amplitudes)
            f = _mol.eval_gaussians(ic, x, periodic)
            u0[b] = f * (p["u_max"] / np.max(np.abs(f)))
            ics.append(ic)
            chans.append(int(p["channel"]) if "channel" in p else int(rng.integers(0, common.N_CH)))
        nu = np.array([p["nu"] for p in params_list], dtype=float).reshape(B, 1)
        umax = np.max(np.abs(u0), axis=1)
        nsub = _mol.n_substeps(DT_FRAME, dt_stable(umax, nu[:, 0], dx))
        out = np.zeros((B, n_frames, common.N_CH, common.N_OUT), dtype=np.float32)
        bidx = np.arange(B)

        def on_frame(k, u):
            out[bidx, k, chans, :] = common.downsample(u, common.N_OUT)

        _mol.rk4_frames(_rhs_factory(nu, dx, bc), u0, DT_FRAME, nsub, n_frames, on_frame)
        trajs = []
        for b, p in enumerate(params_list):
            prm = dict(p)
            prm.update({"ic": ics[b], "dt": DT_FRAME / int(nsub[b]), "n_internal": n_internal})
            trajs.append(common.Trajectory(fields=out[b], system=self.name, params=prm,
                                           dt_frame=DT_FRAME, length=_mol.LENGTH, bc=bc,
                                           channels=(chans[b],)))
        return trajs

    # ------------------------------------------------------------------ verification
    VERIFY = {"uL": 1.0, "uR": 0.0, "nu": 0.01, "x0": 0.3, "T": 0.6}

    def _tw_error(self, n):
        c = self.VERIFY
        dx = _mol.LENGTH / n
        x = common.cell_centres(n, _mol.LENGTH)
        u0 = travelling_wave(x, 0.0, c["uL"], c["uR"], c["nu"], c["x0"])[None, :]
        nu = np.array([[c["nu"]]])

        def walls(t):
            t = np.asarray(t)
            return (travelling_wave(0.0, t, c["uL"], c["uR"], c["nu"], c["x0"]),
                    travelling_wave(_mol.LENGTH, t, c["uL"], c["uR"], c["nu"], c["x0"]))

        rhs = _rhs_factory(nu, dx, "dirichlet", wall_values=walls)
        u, nsteps = _mol.integrate_to(rhs, u0, c["T"],
                                      dt_stable(max(abs(c["uL"]), abs(c["uR"])), c["nu"], dx))
        exact = travelling_wave(x, c["T"], c["uL"], c["uR"], c["nu"], c["x0"])
        return float(common.rel_l2(u[0], exact)), nsteps

    def _resolution_check(self):
        """Worst-case corners: steepest admissible initial data (u_max at the top of the
        range, nu at the bottom). Compare the 512-cell run with a 2048-cell reference after
        both are downsampled to 128, at frame 64. Reported, not pass/fail."""
        rows = []
        rng = np.random.default_rng(12345)
        ic = {"centres": [0.35, 0.65], "widths": [0.05, 0.05], "amplitudes": [1.0, -1.0]}
        for label, nu in (("train corner", PARAM_RANGES["train"]["nu"][0]),
                          ("ood corner", PARAM_RANGES["ood"]["nu"][0][0])):
            p = {"nu": nu, "u_max": 1.0, "ic": ic, "channel": 0}
            for bc in ("periodic", "dirichlet"):
                a = self.simulate(p, bc, 65, rng, n_internal=512).fields[:, 0]
                r = self.simulate(p, bc, 65, rng, n_internal=2048).fields[:, 0]
                rows.append({"case": label, "bc": bc, "nu": nu, "u_max": 1.0, "Re": 1.0 / nu,
                             "delta_over_dx_512": 2 * nu / 1.0 * 512,
                             "cell_peclet_512": 1.0 / 512 / nu,
                             "rel_l2_512_vs_2048_at_frame64": float(common.rel_l2(a[64], r[64])),
                             "max_rel_l2_512_vs_2048_all_frames": float(np.max(
                                 common.rel_l2(a, r, axis=-1))),
                             "max_abs_u_512": float(np.abs(a).max())})
        return rows

    def verify(self, n_list=(128, 256, 512)):
        errs, steps = [], []
        for n in n_list:
            e, s = self._tw_error(n)
            errs.append(e)
            steps.append(s)
        c = self.VERIFY
        rep = common.convergence_report(
            self.name,
            f"viscous travelling wave, uL={c['uL']}, uR={c['uR']}, nu={c['nu']}, "
            f"x0={c['x0']}, final time T={c['T']}, exact time-dependent Dirichlet wall values "
            f"(odd reflection about g(t))",
            n_list, errs, self.order)
        rep["rk4_steps"] = steps
        rep["front_width_4nu_over_du_cells_at_n128"] = 4 * c["nu"] / (c["uL"] - c["uR"]) * 128
        rep["error_metric"] = "relative L2 on the internal grid at t = T (point values)"
        rep["resolution_check"] = self._resolution_check()
        return rep


SOLVER = BurgersSolver()
