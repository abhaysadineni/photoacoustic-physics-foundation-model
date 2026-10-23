"""Generic advection-diffusion (dimensionless, L = 1) on a d-dimensional grid (d = 2, 3),
written to one channel chosen at random per trajectory (as the 1D generic systems):

    du/dt + v . grad u = D lap u,   v constant.

Parameters:
  speed |v|: transport.PARAM_RANGES (imported): train log-uniform in [0.25, 1.5], ood
      log-uniform in [0.08, 0.25] or [1.5, 3.0] (probability 1/2 each, _mol.draw_outside);
      direction uniform on the circle (d = 2) or sphere (d = 3).
  D: log-uniform in [D_LO, D_HI] (modelling choice, the same for train and ood):
      D_LO = |v|_max,train h / 2 with h the default internal cell (1/128 for d = 2, 1/64
      for d = 3): the cell Peclet number |v| h / D is at most 2 for every training speed,
      so the central scheme is free of grid-scale oscillation at fronts and the diffusive
      front thickness D / |v| is at least half an internal cell. D_HI: the spreading length
      sqrt(4 D T_rec) over the record reaches L/4. Values: d = 2: [5.859e-3, 3.226e-2];
      d = 3: [1.172e-2, 2.174e-2]. Ood speeds above 1.5 give cell Peclet numbers up to 4.
Frame interval (the transport rule, about one output cell per frame at the reference speed
|v| = 1): DT_FRAME = (1 / n_out) / 1 = 1/64 (d = 2) and 1/32 (d = 3).
Initial condition: 1 to 4 isotropic Gaussians a exp(-|x - c|^2 / (2 sigma^2)), sigma in
_mol.WIDTH_RANGE = [0.03, 0.15], |a| in [0.2, 1] with random sign (_mol.signed_amplitudes);
centres uniform in [0, 1)^d (periodic, images summed over the 5^d nearest) or in
[0.2, 0.8]^d with walls, multiplied by the product of _mol wall tapers (width 0.1).
Boundary conditions: periodic, dirichlet (u_ghost = -u_edge), neumann (u_ghost = u_edge,
zero normal gradient). Finite-volume fluxes v (u_L + u_R)/2 - D (u_R - u_L)/h, RK4,
dt = 2.5 / (G_diff + sum_a |v_a| / h) (grid.py). Order 2.
Exact check: translated spreading Gaussian on the periodic box,
    u = (s0^2/(s0^2 + 4 D t))^(d/2) exp(-|x - x0 - v t|^2/(s0^2 + 4 D t)), images summed.
"""
from __future__ import annotations

import math

import numpy as np

from pfm.solvers import _mol
from pfm.solvers import transport as TM

from .. import common as C
from . import grid as G

LENGTH = 1.0
V_REF = 1.0
SPEED = TM.PARAM_RANGES


def dt_frame(d):
    return (LENGTH / C.GEOM[d]["n_out"]) / V_REF


def d_range(d):
    h = LENGTH / C.GEOM[d]["n_internal"]
    lo = SPEED["train"]["speed_abs_v"][1] * h / 2.0
    T_rec = (C.GEOM[d]["n_frames"] - 1) * dt_frame(d)
    hi = (LENGTH / 4.0) ** 2 / (4.0 * T_rec)
    return lo, hi


def _flux_div(u, vs, Dc, bcs, h):
    d = len(bcs)
    out = None
    for a in range(d):
        n = u.shape[-d + a]
        lo, hi = u[G.ix(d, a, slice(0, 1))], u[G.ix(d, a, slice(n - 1, n))]
        if bcs[a] == "periodic":
            glo, ghi = hi, lo
        elif bcs[a] == "dirichlet":
            glo, ghi = -lo, -hi
        else:
            glo, ghi = lo, hi
        up = np.concatenate([glo, u, ghi], axis=u.ndim - d + a)
        L_, R_ = up[G.ix(d, a, slice(None, -1))], up[G.ix(d, a, slice(1, None))]
        F = vs[a] * 0.5 * (L_ + R_) - Dc * (R_ - L_) / h
        div = F[G.ix(d, a, slice(1, None))] - F[G.ix(d, a, slice(None, -1))]
        out = div if out is None else out + div
    return -out / h


def run_advdiff(u0, v, Dv, bcs, h, n_frames, dtf, record, nsub=None, t_start=0.0):
    """u0 (B, *n), v (B, d), Dv (B,)."""
    d = len(bcs)
    B = u0.shape[0]
    ex = (slice(None),) + (None,) * d
    if nsub is None:
        G_diff = np.array([G.gershgorin([np.full((1,) + (1,) * d, Dv[b])] * d, bcs, h,
                                        np.zeros((1,) + (1,) * d), (1,) + u0.shape[1:])[0]
                           for b in range(B)])
        dts = G.RK4_DISC / (G_diff + np.abs(v).sum(1) / h)
        nsub = G.n_substeps(dtf, dts)
    P = {"D": Dv[ex]}
    for a in range(d):
        P[f"v{a}"] = v[:, a][ex]

    def rhs(t, y, Q):
        return (_flux_div(y[0], [Q[f"v{a}"] for a in range(d)], Q["D"], bcs, h),)
    out = G.integrate(rhs, (u0,), P, dtf, nsub, n_frames, record, t_start=t_start)
    return out, np.asarray(nsub)


def _unit(rng, d):
    if d == 2:
        t = float(rng.uniform(0, 2 * math.pi))
        return [math.cos(t), math.sin(t)]
    w = rng.normal(size=3)
    return [float(x) for x in w / np.linalg.norm(w)]


class AdvDiff(C.Solver):
    name = "advdiff"
    channels = "random"
    family = "scalar"

    def __init__(self, dim=2):
        super().__init__(dim)
        self.dt_frame = dt_frame(self.dim)
        self.D_RANGE = d_range(self.dim)

    def sample_params(self, rng, regime="train"):
        if regime == "train":
            sp = _mol.log_uniform(rng, *SPEED["train"]["speed_abs_v"])
        elif regime == "ood":
            sp = _mol.draw_outside(rng, SPEED["ood"]["speed_abs_v"])
        else:
            raise ValueError(regime)
        e = _unit(rng, self.dim)
        p = {"regime": regime, "dim": self.dim, "speed": sp, "v": [sp * x for x in e],
             "D": _mol.log_uniform(rng, *self.D_RANGE)}
        m = int(rng.integers(1, 5))
        p["ic"] = {"centres": [[float(rng.uniform(0, 1)) for _ in range(self.dim)] for _ in range(m)],
                   "widths": rng.uniform(*_mol.WIDTH_RANGE, m).tolist(),
                   "amplitudes": _mol.signed_amplitudes(rng, m).tolist()}
        return p

    def initial(self, p, X, bc):
        per = bc == "periodic"
        out = 0.0
        for c, w, a in zip(p["ic"]["centres"], p["ic"]["widths"], p["ic"]["amplitudes"]):
            cc = c if per else [_mol.WALL_CENTRE_MARGIN + (1 - 2 * _mol.WALL_CENTRE_MARGIN) * x for x in c]
            for sft in G.image_shifts(len(X), per, 2):
                out = out + a * np.exp(-0.5 * G.dist2_point(X, cc, LENGTH, sft) / w ** 2)
        if not per:
            for x in X:
                out = out * _mol.wall_taper(x)
        return out

    def simulate_batch(self, params_list, bc, n_frames=None, rng=None, n_internal=None):
        n_frames, n = self._defaults(n_frames, n_internal)
        self.check_bc(bc)
        d = self.dim
        X = C.mesh(n, LENGTH, d)
        u0 = np.stack([np.broadcast_to(self.initial(p, X, bc), (n,) * d) for p in params_list])
        v = np.array([p["v"] for p in params_list], float)
        Dv = np.array([p["D"] for p in params_list], float)
        chans = []
        for p in params_list:
            if "channel" in p:
                chans.append(int(p["channel"]))
            else:
                if rng is None:
                    raise ValueError("advdiff needs rng to draw the channel")
                chans.append(int(rng.integers(0, C.N_CH)))
        out, nsub = run_advdiff(u0, v, Dv, (bc,) * d, LENGTH / n, n_frames, self.dt_frame,
                                lambda y: C.downsample(y[0], d, self.n_out))
        trajs = []
        for b, p in enumerate(params_list):
            f = np.zeros((n_frames, C.N_CH) + (self.n_out,) * d, np.float32)
            f[:, chans[b]] = out[b]
            q = dict(p)
            q.update({"channel": chans[b], "dt": float(self.dt_frame / nsub[b]), "n_internal": n,
                      "cell_peclet": float(p["speed"] * (LENGTH / n) / p["D"])})
            trajs.append(C.Trajectory(f, self.name, q, self.dt_frame, LENGTH, bc,
                                      (chans[b],), d))
        return trajs

    def verify(self, n_list=None):
        d = self.dim
        n_list = n_list or C.default_n_list(d)
        cases = {}
        for label, sp, Dv in (("mid", 1.0, math.sqrt(self.D_RANGE[0] * self.D_RANGE[1])),
                              ("fast_low_D", SPEED["train"]["speed_abs_v"][1], self.D_RANGE[0])):
            e = np.array([0.6, 0.8, 0.0][:d]) if d == 2 else np.array([0.48, 0.6, 0.64])
            e = e / np.linalg.norm(e)
            v = sp * e
            s0 = 0.1
            t1 = 0.25
            x0 = [0.37, 0.52, 0.45][:d]

            def exact(X, t):
                s2 = s0 ** 2 + 4 * Dv * t
                g = 0.0
                for sft in G.image_shifts(d, True, 2):
                    g = g + np.exp(-G.dist2_point(X, [x0[a] + v[a] * t for a in range(d)],
                                                  LENGTH, sft) / s2)
                return (s0 ** 2 / s2) ** (d / 2) * g
            errs = []
            for n in n_list:
                X = C.mesh(n, LENGTH, d)
                u0 = np.broadcast_to(exact(X, 0.0), (n,) * d)[None].copy()
                out, _ = run_advdiff(u0, v[None], np.array([Dv]), ("periodic",) * d,
                                     LENGTH / n, 2, t1, lambda y: y[0].copy())
                num = out[0, -1]
                ex = np.broadcast_to(exact(X, t1), num.shape)
                errs.append(float(np.sqrt(np.sum((num - ex) ** 2) / np.sum(ex ** 2))))
            rep = C.convergence_report(self.name, f"translated_spreading_gaussian_{label}_d{d}",
                                       n_list, errs, 2.0)
            rep.update({"d": d, "v": v.tolist(), "D": Dv, "s0": s0, "t1": t1,
                        "norm": "relative L2 at t1 (point values)"})
            cases[label] = rep
        main = dict(cases["mid"])
        main["check"] = "translated spreading Gaussian, periodic box (two cases)"
        main["cases"] = cases
        main["passed"] = all(c["passed"] for c in cases.values())
        main["D_range"] = list(self.D_RANGE)
        main["dt_frame"] = self.dt_frame
        return main


class AdvDiff2D(AdvDiff):
    def __init__(self):
        super().__init__(2)


class AdvDiff3D(AdvDiff):
    def __init__(self):
        super().__init__(3)
