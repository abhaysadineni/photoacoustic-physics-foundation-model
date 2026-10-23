"""Linear transport (advection): du/dt + v du/dx = 0, dimensionless, L = 1.

Numerics (shared by the four generic solvers)
---------------------------------------------
- Cell-centred grid on [0, L), x_i = (i + 1/2) dx, internal resolution n_internal
  (default 512), stored after cell-average downsampling to 128 points (common.downsample).
- Method of lines: second-order central differences in space, classical RK4 in time.
  Stable step dt_stable = CFL_ADV * dx / |v| with CFL_ADV = 0.4 (there is no diffusion
  term here), then shrunk to dt_frame / ceil(dt_frame / dt_stable) so that frames land
  exactly on multiples of dt_frame. Promised spatial order: 2.
- Boundary conditions by ghost cells. "periodic" wraps.
  Walls: pure advection is a first-order hyperbolic equation and admits a boundary
  condition only at the INFLOW wall (x = 0 for v > 0, x = L for v < 0). Imposing a value at
  the outflow wall as well over-determines the problem; with central differences an odd or
  even reflection there acts as an artificial mirror that turns the outgoing signal into a
  grid-scale (two-cell sawtooth) mode travelling backwards. So:
    inflow wall : "dirichlet" -> odd reflection, u_ghost = -u_edge (u = 0 at the wall);
                  "neumann"   -> even reflection, u_ghost = u_edge (zero gradient).
    outflow wall: quadratic extrapolation u_ghost = 3 u_N-1 - 3 u_N-2 + u_N-3, which makes
                  the edge-cell derivative the second-order one-sided (upwind) difference,
                  so the signal leaves the domain (no condition imposed).
  Consequence, stated plainly: in the continuum the two wall conditions give the SAME
  solution for the initial data used here (tapered to zero at the walls), because at the
  inflow wall u_t = -v u_x, so u_x = 0 forces u to keep its initial wall value, which is 0.
  The numerical "dirichlet" and "neumann" trajectories differ only by discretisation error.
- Initial condition: sum of 1 to 4 Gaussians, centres uniform (in [0.2, 0.8] with walls),
  standard deviations uniform in [0.03, 0.15], amplitudes with |a| uniform in [0.2, 1.0]
  and random sign. Periodic: summed over periodic images. Walls: multiplied by a C2 taper
  (quintic smoothstep over 0.1 from each wall).
- Frame interval rule: features move about one output cell per frame at the reference
  speed |v| = 1: dt_frame = (L / 128) / 1 = 1/128. Over the training range |v| in
  [0.25, 1.5] a feature moves 0.25 to 1.5 output cells per frame.
- channels = "random": simulate draws one channel with rng (after drawing the initial
  condition) and writes the field there, zeros elsewhere.
"""
from __future__ import annotations

import numpy as np

from pfm import common
from pfm.solvers import _mol

DT_FRAME = 1.0 / 128.0
V_REF = 1.0

PARAM_RANGES = {
    # v = sign * speed, sign uniform in {-1, +1}, speed log-uniform in the band(s)
    "train": {"speed_abs_v": [0.25, 1.5], "sign": [-1, 1]},
    "ood": {"speed_abs_v": [[0.08, 0.25], [1.5, 3.0]], "sign": [-1, 1]},
}

BCS = ("periodic", "dirichlet", "neumann")


def _rhs_factory(v, dx, bc):
    """v: (B, 1) array. Returns rhs(u, t)."""
    if bc == "periodic":
        def rhs(u, t, sl):
            return -v[sl] * _mol.ddx(_mol.pad(u, "periodic"), dx)
        return rhs
    inflow = "odd" if bc == "dirichlet" else "even"
    pos = (v[:, 0] >= 0)

    def rhs(u, t, sl):
        up = np.empty(u.shape[:-1] + (u.shape[-1] + 2,))
        up[..., 1:-1] = u
        g_in_left = u[..., 0] * (-1.0 if inflow == "odd" else 1.0)
        g_in_right = u[..., -1] * (-1.0 if inflow == "odd" else 1.0)
        g_out_left = 3.0 * u[..., 0] - 3.0 * u[..., 1] + u[..., 2]
        g_out_right = 3.0 * u[..., -1] - 3.0 * u[..., -2] + u[..., -3]
        up[..., 0] = np.where(pos[sl], g_in_left, g_out_left)
        up[..., -1] = np.where(pos[sl], g_out_right, g_in_right)
        return -v[sl] * _mol.ddx(up, dx)
    return rhs


def dt_stable(v, dx):
    return _mol.CFL_ADV * dx / np.maximum(np.abs(v), 1e-12)


class TransportSolver(common.Solver):
    name = "transport"
    channels = "random"
    order = 2.0

    # ------------------------------------------------------------------ parameters
    def sample_params(self, rng, regime="train"):
        sign = float(rng.choice([-1.0, 1.0]))
        if regime == "train":
            speed = _mol.log_uniform(rng, *PARAM_RANGES["train"]["speed_abs_v"])
        elif regime == "ood":
            speed = _mol.draw_outside(rng, PARAM_RANGES["ood"]["speed_abs_v"])
        else:
            raise ValueError(regime)
        return {"v": sign * speed, "regime": regime}

    # ------------------------------------------------------------------ simulation
    def simulate(self, params, bc, n_frames, rng, n_internal=512):
        return self.simulate_batch([params], bc, n_frames, rng, n_internal)[0]

    def simulate_batch(self, params_list, bc, n_frames, rng, n_internal=512):
        if bc not in BCS:
            raise ValueError(f"transport: unknown bc {bc!r}")
        periodic = bc == "periodic"
        B = len(params_list)
        dx = _mol.LENGTH / n_internal
        x = common.cell_centres(n_internal, _mol.LENGTH)
        u0 = np.empty((B, n_internal))
        ics, chans = [], []
        for b, p in enumerate(params_list):
            ic = p.get("ic") or _mol.draw_gaussians(rng, periodic, _mol.signed_amplitudes)
            u0[b] = _mol.eval_gaussians(ic, x, periodic)
            ics.append(ic)
            chans.append(int(p["channel"]) if "channel" in p else int(rng.integers(0, common.N_CH)))
        v = np.array([p["v"] for p in params_list], dtype=float).reshape(B, 1)
        nsub = _mol.n_substeps(DT_FRAME, dt_stable(v[:, 0], dx))
        out = np.zeros((B, n_frames, common.N_CH, common.N_OUT), dtype=np.float32)
        bidx = np.arange(B)

        def on_frame(k, u):
            out[bidx, k, chans, :] = common.downsample(u, common.N_OUT)

        _mol.rk4_frames(_rhs_factory(v, dx, bc), u0, DT_FRAME, nsub, n_frames, on_frame)
        trajs = []
        for b, p in enumerate(params_list):
            prm = dict(p)
            prm.update({"ic": ics[b], "dt": DT_FRAME / int(nsub[b]), "n_internal": n_internal})
            trajs.append(common.Trajectory(fields=out[b], system=self.name, params=prm,
                                           dt_frame=DT_FRAME, length=_mol.LENGTH, bc=bc,
                                           channels=(chans[b],)))
        return trajs

    # ------------------------------------------------------------------ verification
    @staticmethod
    def _exact_ic(x):
        """sin(2 pi x) plus a smooth periodic Gaussian bump (centre 0.3, std 0.08)."""
        return np.sin(2 * np.pi * x) + _mol.eval_gaussians(
            {"centres": [0.3], "widths": [0.08], "amplitudes": [0.8]}, x, periodic=True)

    def _period_error(self, v, n):
        dx = _mol.LENGTH / n
        x = common.cell_centres(n, _mol.LENGTH)
        u0 = self._exact_ic(x)[None, :]
        T = _mol.LENGTH / abs(v)                    # one full period: exact solution = u0
        vv = np.array([[v]])
        u, nsteps = _mol.integrate_to(_rhs_factory(vv, dx, "periodic"), u0, T,
                                      dt_stable(abs(v), dx))
        return float(common.rel_l2(u[0], u0[0])), nsteps

    def verify(self, n_list=(128, 256, 512)):
        cases = {}
        for label, v in (("v=+1.0", 1.0), ("v=-0.6", -0.6)):
            errs, steps = [], []
            for n in n_list:
                e, s = self._period_error(v, n)
                errs.append(e)
                steps.append(s)
            rep = common.convergence_report(
                self.name, f"periodic, sin(2 pi x) + Gaussian bump, {label}, "
                           f"exact translation after one period T = L/|v|",
                n_list, errs, self.order)
            rep["rk4_steps"] = steps
            cases[label] = rep
        main = dict(cases["v=+1.0"])
        main["check"] = ("periodic translation after one full period, v = +1.0 (top level) "
                         "and v = -0.6 (in cases); passed requires both")
        main["cases"] = cases
        main["passed"] = bool(all(c["passed"] for c in cases.values()))
        main["error_metric"] = "relative L2 on the internal grid at t = T (point values)"
        return main


SOLVER = TransportSolver()
