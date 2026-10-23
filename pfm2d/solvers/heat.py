"""Heat conduction with sources on a d-dimensional Cartesian grid (d = 2, 3), channel 1,
as pfm/solvers/heat.py:

    dT/dt = alpha lap T + H(x, t) / (rho C_p)      (temperature rise, K)

Parameter ranges: heat.PARAM_RANGES (train and ood), imported unchanged: alpha log-uniform
in [fat, muscle] (train) or [muscle, water 37 C] (ood), rho C_p uniform in the matching
range, 1 to 3 hot spots (amplitude 0.2 to 2 K, 1/e radius 0.2 to 1 mm), 0 to 2 heaters
(1/e radius 0.2 to 1 mm) with heating 0.02 to 0.2 K per 1D frame interval.

Frame interval, by the 1D rule (a hot spot in the reference tissue alpha_ref =
soft_tissue_alpha_th spreads to a 1/e radius L/4 over the record of F - 1 intervals):
    DT_FRAME = (L/4)^2 / (4 (F - 1) alpha_ref):  d = 2 (F = 32): 0.8258 s;  d = 3 (F = 24):
    1.1130 s;  1D (F = 64): 0.4063 s. The record, 25.6 s, is the same in every dimension.
To keep the physical distributions identical to 1D, every time-like parameter is drawn in
seconds as in 1D: the heating rate H / (rho C_p) = K_per_frame / heat.DT_FRAME (K/s, with
K_per_frame from the 1D range per 1D frame), switch-on times uniform in [0, 40]
heat.DT_FRAME, durations [5, 20] heat.DT_FRAME, tanh switching width heat.DT_FRAME
(0.406 s), each heater switched with probability 1/2.
Hot spots and heaters are isotropic Gaussians exp(-|x - c|^2 / w^2), each centre
coordinate uniform in [0.2, 0.8] L; periodic images summed for periodic boundaries.
Dirichlet walls: initial data and heaters multiplied by the product of C2 tapers of width
1 mm (heat.WALL_TAPER) in every direction.
Numerics: grid.py, RK4, dt = 2.5 / Gershgorin. Order 2.
Exact check: Gaussian spreading T = (s0^2 / (s0^2 + 4 alpha t))^(d/2)
exp(-r^2 / (s0^2 + 4 alpha t)) on a periodic box with images summed.
"""
from __future__ import annotations

import math

import numpy as np

from pfm.solvers import heat as HM

from .. import common as C
from . import grid as G

PARAM_RANGES = HM.PARAM_RANGES
LENGTH = C.LENGTH
ALPHA_REF = HM.ALPHA_REF
SWITCH_WIDTH = HM.DT_FRAME           # s, the 1D width (one 1D frame)


def dt_frame(d, n_frames=None):
    F = C.GEOM[d]["n_frames"] if n_frames is None else n_frames
    return (LENGTH / 4.0) ** 2 / (4.0 * (F - 1) * ALPHA_REF)


def run_heat(T0, alpha, shapes, ton, toff, bcs, h, n_frames, dtf, record, nsub=None):
    """T0 (B, *n); alpha (B,); shapes (B, H, *n) heating in K/s; ton, toff (B, H) s."""
    d = len(bcs)
    B = T0.shape[0]
    ex = (slice(None),) + (None,) * d
    kf = [alpha[ex]] * d
    if nsub is None:
        nsub = G.n_substeps(dtf, G.diffusion_dt(kf, bcs, h, np.zeros((B,) + (1,) * d), T0.shape))
    P = {"alpha": alpha[ex], "shape": shapes, "ton": ton, "toff": toff}

    def rhs(t, y, Q):
        tt = t[:, None]
        w = 0.25 * (1 + np.tanh((tt - Q["ton"]) / SWITCH_WIDTH)) * \
            (1 + np.tanh((Q["toff"] - tt) / SWITCH_WIDTH))
        out = G.diffusion(y[0], [Q["alpha"]] * d, bcs, h)
        out += np.einsum("bh,bh...->b...", w, Q["shape"])
        return (out,)
    out = G.integrate(rhs, (T0,), P, dtf, nsub, n_frames, record)
    return out, np.asarray(nsub)


class Heat(C.Solver):
    name = "heat"
    channels = (C.CH_HEAT,)
    family = "scalar"

    def __init__(self, dim=2):
        super().__init__(dim)
        self.dt_frame = dt_frame(self.dim)

    def sample_params(self, rng, regime="train"):
        tr = PARAM_RANGES["train"]
        rg = PARAM_RANGES[regime]
        d = self.dim
        u = lambda ab: float(rng.uniform(*ab))
        cen = lambda: [float(rng.uniform(0.2, 0.8) * LENGTH) for _ in range(d)]
        p = {"regime": regime, "dim": d,
             "alpha": float(np.exp(rng.uniform(*np.log(rg["alpha_m2_s"])))),
             "rho_cp": u(rg["rho_cp_J_m3K"])}
        p["hot_spots"] = [{"centre": cen(), "width": u(tr["width_m"]), "amp": u(tr["hot_spot_amp_K"])}
                          for _ in range(int(rng.integers(tr["n_hot_spots"][0], tr["n_hot_spots"][1] + 1)))]
        heaters = []
        for _ in range(int(rng.integers(tr["n_heaters"][0], tr["n_heaters"][1] + 1))):
            hh = {"centre": cen(), "width": u(tr["width_m"]),
                  "K_per_frame_1d": u(tr["heating_K_per_frame"])}
            hh["K_per_s"] = hh["K_per_frame_1d"] / HM.DT_FRAME
            if rng.random() < 0.5:
                t_on = float(rng.uniform(0, 40) * HM.DT_FRAME)
                hh["t_on"], hh["t_off"] = t_on, t_on + float(rng.uniform(5, 20) * HM.DT_FRAME)
            heaters.append(hh)
        p["heaters"] = heaters
        return p

    def arrays(self, params_list, bc, n):
        d = self.dim
        X = C.mesh(n, LENGTH, d)
        per = bc == "periodic"
        tap = G.taper(X, LENGTH, HM.WALL_TAPER) if bc == "dirichlet" else 1.0
        B = len(params_list)
        Hm = max(1, max(len(p["heaters"]) for p in params_list))
        T0 = np.zeros((B,) + (n,) * d)
        S = np.zeros((B, Hm) + (n,) * d)
        ton = np.full((B, Hm), -1e30); toff = np.full((B, Hm), 1e30)
        for b, p in enumerate(params_list):
            for hs in p["hot_spots"]:
                T0[b] += hs["amp"] * G.gaussian(X, hs["centre"], hs["width"], LENGTH, per)
            for j, hh in enumerate(p["heaters"]):
                S[b, j] = hh["K_per_s"] * G.gaussian(X, hh["centre"], hh["width"], LENGTH, per)
                if "t_on" in hh:
                    ton[b, j], toff[b, j] = hh["t_on"], hh["t_off"]
        T0 *= tap; S *= tap
        alpha = np.array([p["alpha"] for p in params_list])
        return T0, alpha, S, ton, toff

    def simulate_batch(self, params_list, bc, n_frames=None, rng=None, n_internal=None):
        n_frames, n = self._defaults(n_frames, n_internal)
        self.check_bc(bc)
        d = self.dim
        T0, alpha, S, ton, toff = self.arrays(params_list, bc, n)
        rec = lambda y: C.downsample(y[0], d, self.n_out)
        out, nsub = run_heat(T0, alpha, S, ton, toff, (bc,) * d, LENGTH / n, n_frames,
                             self.dt_frame, rec)
        trajs = []
        for b, p in enumerate(params_list):
            f = np.zeros((n_frames, C.N_CH) + (self.n_out,) * d, np.float32)
            f[:, C.CH_HEAT] = out[b]
            q = dict(p)
            q.update({"dt": float(self.dt_frame / nsub[b]), "n_internal": n,
                      "H_peak_W_m3": [hh["K_per_s"] * p["rho_cp"] for hh in p["heaters"]]})
            trajs.append(C.Trajectory(f, self.name, q, self.dt_frame, LENGTH, bc,
                                      self.channels, d))
        return trajs

    def gaussian_check(self, n_list=None, alpha=ALPHA_REF):
        d = self.dim
        n_list = n_list or C.default_n_list(d)
        L = LENGTH
        s0 = L / 10.0
        t1 = 3.0 * s0 ** 2 / (4.0 * alpha)           # s(t1)^2 = 4 s0^2
        x0 = [0.5 * L + 0.013 * L * (a + 1) for a in range(d)]

        def exact(X, t):
            s2 = s0 ** 2 + 4 * alpha * t
            g = 0.0
            for sft in G.image_shifts(d, True, 2):
                g = g + np.exp(-G.dist2_point(X, x0, L, sft) / s2)
            return (s0 ** 2 / s2) ** (d / 2) * g
        errs, steps = [], []
        for n in n_list:
            X = C.mesh(n, L, d)
            T0 = np.broadcast_to(exact(X, 0.0), (n,) * d)[None].copy()
            out, ns = run_heat(T0, np.array([alpha]), np.zeros((1, 1) + (n,) * d),
                               np.full((1, 1), -1e30), np.full((1, 1), 1e30),
                               ("periodic",) * d, L / n, 2, t1, lambda y: y[0].copy())
            num = out[0, -1]
            ex = np.broadcast_to(exact(X, t1), num.shape)
            errs.append(float(np.sqrt(np.sum((num - ex) ** 2) / np.sum(ex ** 2))))
            steps.append(int(ns[0]))
        rep = C.convergence_report(self.name, f"gaussian_spreading_d{d}", n_list, errs, 2.0)
        rep.update({"d": d, "norm": "relative L2 at t1 (point values)", "s0_m": s0, "t1_s": t1,
                    "alpha_m2_s": alpha, "centre": x0, "rk4_steps": steps,
                    "bc": "periodic, images summed in the exact solution"})
        return rep

    def verify(self, n_list=None):
        rep = self.gaussian_check(n_list)
        rep["dt_frame_s"] = self.dt_frame
        return rep


class Heat2D(Heat):
    def __init__(self):
        super().__init__(2)


class Heat3D(Heat):
    def __init__(self):
        super().__init__(3)
