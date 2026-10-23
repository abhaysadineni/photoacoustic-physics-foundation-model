"""Fisher-KPP reaction-diffusion (pattern formation), dimensionless, L = 1:

    du/dt = D d2u/dx2 + r u (1 - u)

Numerics (shared by the four generic solvers)
---------------------------------------------
- Cell-centred grid on [0, L), internal resolution n_internal (default 512), stored after
  cell-average downsampling to 128 points (common.downsample).
- Method of lines: second-order central differences in space, classical RK4 in time.
  Stable step dt_stable = min(0.2 dx^2 / D, 0.2 / r), then shrunk to
  dt_frame / ceil(dt_frame / dt_stable) so frames land exactly on multiples of dt_frame.
  There is no advective speed here. The reaction limit 0.2 / r is an addition the spec did
  not state; at n_internal = 512 it never binds in the parameter ranges below (0.2 dx^2 / D
  is smaller by a factor of at least 13 in the training range and at least 2 in the OOD
  range) but keeps the step safe if a caller passes a large r or a coarse grid.
  Promised spatial order 2.
- Boundary conditions by ghost cells: "periodic" wraps; "dirichlet" is odd reflection
  (u_ghost = -u_edge, u = 0 at the wall); "neumann" is even reflection (zero flux).
  For verification the odd reflection is used with the exact wall value g(t):
  u_ghost = 2 g(t) - u_edge.
- Initial condition: sum of 1 to 4 Gaussians, centres uniform (in [0.2, 0.8] with walls),
  standard deviations uniform in [0.03, 0.15], amplitudes uniform in [0.05, 1.0] (all
  positive); walls: multiplied by a C2 taper (quintic smoothstep over 0.1 from each wall).
  If the sum exceeds 1 anywhere it is divided by its maximum, so 0 <= u0 <= 1.
  The continuum problem then keeps 0 <= u <= 1 for all time (maximum principle). The
  discrete scheme keeps it to round-off: with dt <= 0.2 dx^2 / D, the RK4 polynomial of the
  discrete diffusion operator has non-negative entries (checked numerically), and the
  measured excursion is reported by the batch test.
- Frame interval rule: a front at the reference speed c_ref = 2 sqrt(r D) = 0.5 moves about
  one output cell per frame: dt_frame = (L / 128) / 0.5 = 1/64. Over the training range
  the minimal front speed 2 sqrt(rD) spans 0.089 to 0.89 (0.18 to 1.8 output cells per
  frame) and the logistic growth per frame r dt_frame spans 0.031 to 0.31.
- Resolution: the front decay length sqrt(D / r) is at least 3.6 internal cells at
  n_internal = 512 in the training range and at least 1.4 cells at the OOD corner
  (D = 0.0003, r = 40). verify()["resolution_check"] compares 512 with 2048 cells at both
  corners (reported, not pass/fail).
- channels = "random": simulate draws one channel with rng and writes the field there.

Exact solution used for check A (derivation)
--------------------------------------------
Ablowitz and Zeppetella (1979) give, for D = r = 1,
    U(X, tau) = 1 / (1 + C exp(X / sqrt(6) - 5 tau / 6))^2 ,
which satisfies U_tau = U_XX + U (1 - U) (a travelling wave of speed 5 / sqrt(6)).
For general D, r set  X = x / ell,  tau = r t,  ell = sqrt(D / r).  Then
    u_t = r U_tau ,  u_xx = U_XX / ell^2 = (r / D) U_XX ,
so  u_t = D u_xx + r u (1 - u)  becomes  r U_tau = r U_XX + r U (1 - U), i.e. the D = r = 1
equation. Hence
    u(x, t) = 1 / (1 + C exp(kappa x - 5 r t / 6))^2 ,   kappa = sqrt(r / (6 D)),
a wave of speed s = (5 r / 6) / kappa = (5 / sqrt(6)) sqrt(r D) ~ 2.041 sqrt(rD), faster
than the minimal speed 2 sqrt(rD) (this exact solution is not the minimal-speed front).
Check A uses D = 0.01, r = 6: kappa = 10 (profile scale 1 / kappa = 0.1 = 12.8 cells at
n = 128), s = 0.5. C = exp(-kappa x0) places the reference point
kappa (x - x0) - 5rt/6 = 0, where u = 1/4, at x = x0 = 0.3 when t = 0; final time T = 0.6 (the front moves 0.3). The wall values
u(0, t), u(L, t) are imposed exactly. The derivation is also checked numerically:
verify()["exact_residual"] is the max PDE residual of the formula by fine finite
differences.

Check B (reported, not pass/fail): minimal front speed
------------------------------------------------------
D = r = 1 on a long domain [0, 400) with Neumann walls, step initial data u = 1 for x < 10,
0 otherwise. Front position X(t) is where u = 1/2 (linear interpolation). For initial data
that decays faster than exp(-lambda x), lambda = sqrt(r / D), Bramson's result is
X(t) = 2 sqrt(rD) t - (3 / (2 lambda)) ln t + const + o(1), so the speed approaches the
minimal speed from below as v(t) ~ 2 sqrt(rD) - 3 / (2 lambda t). The measured speed is the
centred difference of X over a window of +- 1 time unit. Two grids (dx = 0.1, 0.05) show
whether the measured speed is grid-converged.
"""
from __future__ import annotations

import numpy as np

from pfm import common
from pfm.solvers import _mol

C_REF = 0.5
DT_FRAME = (1.0 / 128.0) / C_REF   # = 1/64
REACTION_CFL = 0.2

PARAM_RANGES = {
    # D and r each log-uniform. OOD: BOTH D and r drawn from bands outside their
    # training bands (each band picked with probability 1/2).
    "train": {"D": [0.001, 0.01], "r": [2.0, 20.0]},
    "ood": {"D": [[0.0003, 0.001], [0.01, 0.03]], "r": [[0.5, 2.0], [20.0, 40.0]]},
}

BCS = ("periodic", "dirichlet", "neumann")


def _rhs_factory(D, r, dx, bc, wall_values=None):
    kind = {"periodic": "periodic", "dirichlet": "odd", "neumann": "even"}.get(bc)
    if kind is None:
        raise ValueError(f"fisher: unknown bc {bc!r}")

    def rhs(u, t, sl):
        if wall_values is not None:
            gl, gr = wall_values(t)
            up = _mol.pad(u, kind, kind, gl, gr)
        else:
            up = _mol.pad(u, kind)
        return _mol.take(D, sl) * _mol.lap(up, dx) + _mol.take(r, sl) * u * (1.0 - u)
    return rhs


def dt_stable(D, r, dx):
    return np.minimum(_mol.CFL_DIFF * dx * dx / D, REACTION_CFL / np.maximum(r, 1e-12))


def az_exact(x, t, D, r, x0):
    kappa = np.sqrt(r / (6.0 * D))
    return 1.0 / (1.0 + np.exp(kappa * (x - x0) - 5.0 * r * t / 6.0)) ** 2


def _amps(rng, m):
    return rng.uniform(0.05, 1.0, m)


class FisherSolver(common.Solver):
    name = "fisher"
    channels = "random"
    order = 2.0

    def sample_params(self, rng, regime="train"):
        if regime == "train":
            D = _mol.log_uniform(rng, *PARAM_RANGES["train"]["D"])
            r = _mol.log_uniform(rng, *PARAM_RANGES["train"]["r"])
        elif regime == "ood":
            D = _mol.draw_outside(rng, PARAM_RANGES["ood"]["D"])
            r = _mol.draw_outside(rng, PARAM_RANGES["ood"]["r"])
        else:
            raise ValueError(regime)
        return {"D": D, "r": r, "front_speed_min": 2 * np.sqrt(r * D), "regime": regime}

    def simulate(self, params, bc, n_frames, rng, n_internal=512):
        return self.simulate_batch([params], bc, n_frames, rng, n_internal)[0]

    def simulate_batch(self, params_list, bc, n_frames, rng, n_internal=512):
        if bc not in BCS:
            raise ValueError(f"fisher: unknown bc {bc!r}")
        periodic = bc == "periodic"
        B = len(params_list)
        dx = _mol.LENGTH / n_internal
        x = common.cell_centres(n_internal, _mol.LENGTH)
        u0 = np.empty((B, n_internal))
        ics, chans = [], []
        for b, p in enumerate(params_list):
            ic = p.get("ic") or _mol.draw_gaussians(rng, periodic, _amps)
            f = _mol.eval_gaussians(ic, x, periodic)
            m = f.max()
            u0[b] = f / m if m > 1.0 else f
            ics.append(ic)
            chans.append(int(p["channel"]) if "channel" in p else int(rng.integers(0, common.N_CH)))
        D = np.array([p["D"] for p in params_list], dtype=float).reshape(B, 1)
        r = np.array([p["r"] for p in params_list], dtype=float).reshape(B, 1)
        nsub = _mol.n_substeps(DT_FRAME, dt_stable(D[:, 0], r[:, 0], dx))
        out = np.zeros((B, n_frames, common.N_CH, common.N_OUT), dtype=np.float32)
        bidx = np.arange(B)

        def on_frame(k, u):
            out[bidx, k, chans, :] = common.downsample(u, common.N_OUT)

        _mol.rk4_frames(_rhs_factory(D, r, dx, bc), u0, DT_FRAME, nsub, n_frames, on_frame)
        trajs = []
        for b, p in enumerate(params_list):
            prm = dict(p)
            prm.update({"ic": ics[b], "dt": DT_FRAME / int(nsub[b]), "n_internal": n_internal})
            trajs.append(common.Trajectory(fields=out[b], system=self.name, params=prm,
                                           dt_frame=DT_FRAME, length=_mol.LENGTH, bc=bc,
                                           channels=(chans[b],)))
        return trajs

    # ------------------------------------------------------------------ check A
    VERIFY = {"D": 0.01, "r": 6.0, "x0": 0.3, "T": 0.6}

    def _az_error(self, n):
        c = self.VERIFY
        dx = _mol.LENGTH / n
        x = common.cell_centres(n, _mol.LENGTH)
        u0 = az_exact(x, 0.0, c["D"], c["r"], c["x0"])[None, :]

        def walls(t):
            t = np.asarray(t)
            return (az_exact(0.0, t, c["D"], c["r"], c["x0"]),
                    az_exact(_mol.LENGTH, t, c["D"], c["r"], c["x0"]))

        rhs = _rhs_factory(c["D"], c["r"], dx, "dirichlet", wall_values=walls)
        u, nsteps = _mol.integrate_to(rhs, u0, c["T"], dt_stable(c["D"], c["r"], dx))
        exact = az_exact(x, c["T"], c["D"], c["r"], c["x0"])
        return float(common.rel_l2(u[0], exact)), nsteps

    def _exact_residual(self):
        c = self.VERIFY
        h, k = 1e-4, 1e-5
        x = np.linspace(0.05, 0.95, 181)
        t = 0.3
        f = lambda xx, tt: az_exact(xx, tt, c["D"], c["r"], c["x0"])
        ut = (f(x, t + k) - f(x, t - k)) / (2 * k)
        uxx = (f(x + h, t) - 2 * f(x, t) + f(x - h, t)) / h ** 2
        u = f(x, t)
        res = ut - c["D"] * uxx - c["r"] * u * (1 - u)
        return {"max_abs_residual": float(np.abs(res).max()),
                "max_abs_u_t": float(np.abs(ut).max()),
                "note": "finite-difference residual of the closed form; O(1e-4) or smaller "
                        "relative to |u_t| means the scaled formula solves the PDE"}

    # ------------------------------------------------------------------ check B
    @staticmethod
    def _front_position(u, x):
        i = np.nonzero(u < 0.5)[0][0]
        x1, x2, u1, u2 = x[i - 1], x[i], u[i - 1], u[i]
        return float(x1 + (0.5 - u1) * (x2 - x1) / (u2 - u1))

    def front_speed(self, dx_list=(0.1, 0.05), Lb=400.0, x_step=10.0,
                    times=(5.0, 10.0, 20.0, 40.0, 80.0, 160.0), window=1.0):
        D, r = 1.0, 1.0
        lam = np.sqrt(r / D)
        cmin = 2.0 * np.sqrt(r * D)
        sample_t = sorted({round(tt + s * window, 10) for tt in times for s in (-1, 1)}
                          | set(times))
        result = {"D": D, "r": r, "domain_length": Lb, "bc": "neumann",
                  "initial": f"u = 1 for x < {x_step}, 0 otherwise",
                  "minimal_speed_2sqrt_rD": cmin,
                  "bramson_leading_order_speed": "2 sqrt(rD) - 3 / (2 lambda t), lambda = sqrt(r/D)",
                  "speed_estimator": f"(X(t + {window}) - X(t - {window})) / {2 * window}",
                  "grids": []}
        for dxb in dx_list:
            n = int(round(Lb / dxb))
            x = common.cell_centres(n, Lb)
            u = np.where(x < x_step, 1.0, 0.0)[None, :]
            rhs = _rhs_factory(D, r, dxb, "neumann")
            dts = float(dt_stable(D, r, dxb))
            pos = {}
            t_now = 0.0
            for tt in sample_t:
                u, _ = _mol.integrate_to(rhs, u, tt - t_now, dts)
                t_now = tt
                pos[tt] = self._front_position(u[0], x)
            rows = []
            for tt in times:
                v = (pos[round(tt + window, 10)] - pos[round(tt - window, 10)]) / (2 * window)
                vb = cmin - 3.0 / (2.0 * lam * tt)
                rows.append({"t": tt, "X": pos[tt], "measured_speed": v,
                             "measured_over_minimal": v / cmin,
                             "bramson_leading_order": vb,
                             "measured_minus_bramson": v - vb})
            result["grids"].append({"dx": dxb, "n": n, "dt": dts, "rows": rows,
                                    "max_u": float(u.max()), "min_u": float(u.min())})
        result["note"] = ("Measured speed stays below 2 sqrt(rD) and approaches it slowly, "
                          "consistent with the Bramson logarithmic delay; the leading-order "
                          "speed correction -3/(2 lambda t) is an asymptotic statement with "
                          "corrections of higher order in 1/sqrt(t), so agreement at small t "
                          "is not expected to be close.")
        return result

    # ------------------------------------------------------------------ neumann supplement
    def _neumann_heat_error(self, n, D=0.01, T=2.0):
        """r = 0 reduces to the heat equation; cos(pi x) exp(-D pi^2 t) satisfies zero flux
        at both walls. Supplementary check of the even-reflection closure (reported)."""
        dx = _mol.LENGTH / n
        x = common.cell_centres(n, _mol.LENGTH)
        u0 = np.cos(np.pi * x)[None, :]
        rhs = _rhs_factory(D, 0.0, dx, "neumann")
        u, _ = _mol.integrate_to(rhs, u0, T, dt_stable(D, 0.0, dx))
        return float(common.rel_l2(u[0], np.cos(np.pi * x) * np.exp(-D * np.pi ** 2 * T)))

    def _resolution_check(self):
        """Steepest fronts: D small, r large (front decay length sqrt(D/r)). Compare the
        512-cell run with a 2048-cell reference after both are downsampled to 128, at frame
        64. Reported, not pass/fail."""
        rows = []
        rng = np.random.default_rng(54321)
        ic = {"centres": [0.5], "widths": [0.03], "amplitudes": [0.5]}
        corners = (("train corner", PARAM_RANGES["train"]["D"][0], PARAM_RANGES["train"]["r"][1]),
                   ("ood corner", PARAM_RANGES["ood"]["D"][0][0], PARAM_RANGES["ood"]["r"][1][1]))
        for label, D, r in corners:
            p = {"D": D, "r": r, "ic": ic, "channel": 0}
            a = self.simulate(p, "neumann", 65, rng, n_internal=512).fields[:, 0]
            ref = self.simulate(p, "neumann", 65, rng, n_internal=2048).fields[:, 0]
            rows.append({"case": label, "D": D, "r": r,
                         "decay_length_sqrt_D_over_r_cells_512": np.sqrt(D / r) * 512,
                         "rel_l2_512_vs_2048_at_frame64": float(common.rel_l2(a[64], ref[64])),
                         "max_rel_l2_512_vs_2048_all_frames": float(np.max(
                             common.rel_l2(a, ref, axis=-1)))})
        return rows

    def verify(self, n_list=(128, 256, 512)):
        errs, steps = [], []
        for n in n_list:
            e, s = self._az_error(n)
            errs.append(e)
            steps.append(s)
        c = self.VERIFY
        rep = common.convergence_report(
            self.name,
            f"Ablowitz-Zeppetella exact wave, scaled: D={c['D']}, r={c['r']}, "
            f"kappa=sqrt(r/(6D))={np.sqrt(c['r'] / (6 * c['D'])):.4g}, "
            f"speed (5/sqrt6) sqrt(rD)={5 / np.sqrt(6) * np.sqrt(c['r'] * c['D']):.4g}, "
            f"x0={c['x0']}, final time T={c['T']}, exact Dirichlet wall values",
            n_list, errs, self.order)
        rep["rk4_steps"] = steps
        rep["error_metric"] = "relative L2 on the internal grid at t = T (point values)"
        rep["exact_residual"] = self._exact_residual()
        nerr = [self._neumann_heat_error(n) for n in n_list]
        rep["supplementary_neumann_heat"] = common.convergence_report(
            self.name, "r = 0, cos(pi x) exp(-D pi^2 t), D = 0.01, T = 2, even reflection "
                       "(reported, not part of passed)", n_list, nerr, self.order)
        rep["resolution_check"] = self._resolution_check()
        rep["front_speed"] = self.front_speed()
        return rep


SOLVER = FisherSolver()
