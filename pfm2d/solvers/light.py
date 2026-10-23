"""Light transport on a d-dimensional Cartesian grid (d = 2, 3), channel 0: time-resolved
diffusion with heterogeneous absorption, as pfm/solvers/light.py.

    d phi/dt = div(D(x) grad phi) - mu_a(x) c phi,   D = c / (3 (mu_a + mu_s')),  c = c0 / n

Optical properties, parameter ranges, the mu_a / mu_s' <= 0.3 cap and the wavelengths are
those of the 1D module (imported; see tissue.py for the d-dimensional absorbers).

Time window: T_START = light.T_START = 65.81 ps (ten transport times of the slowest-
scattering corner of the training box) and T_END = light.T_END = 144.97 ps (spreading
length of the reference draw reaches L/2), imported. The rule of the 1D module, record
from T_START to T_END, is kept; with F frames DT_FRAME = (T_END - T_START) / (F - 1):
    d = 2, 32 frames: 2.5537 ps;   d = 3, 24 frames: 3.4417 ps
(1D: 64 frames, 1.2565 ps). The physical record is the same in every dimension.

Initial state: the exact homogeneous Green's-function state at T_START of 1 to 4 impulses
(n_impulses), amplitudes impulse_amp, centres with every coordinate in impulse_centre_frac
x L, in the background medium of the trajectory, with images for the boundary:
    phi_j(x, T) = A_j prod_a g_a(x_a),   g_a = sum_images exp(-(x_a - x_image)^2 / (4 D_bg T))
(periodic images in every direction; odd images for dirichlet, even for neumann, each axis
independently). The d-dimensional kernel on a box with homogeneous conditions is the
product of 1D kernels, so the state is exact for the homogeneous box. A_j is the peak of
source j (the constant c (4 pi D T)^(-d/2) exp(-mu_a c T) is absorbed into it). There is
no source during the record, as in 1D.

Exact evolution in a homogeneous medium: each image Gaussian spreads with amplitude
(T_START / t)^(d/2) exp(-mu_a c (t - T_START)); cell averages are products of erf
differences (exact_cell_average).

Numerics: grid.py finite volumes, face D = arithmetic mean, RK4 with dt = 2.5 / Gershgorin
bound per trajectory, equal substeps per frame. Order 2.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import erf

from pfm.solvers import light as LM

from .. import common as C
from . import grid as G
from . import tissue as T

T_START, T_END = LM.T_START, LM.T_END
N_IMAGES = LM.N_IMAGES
PARAM_RANGES = LM.PARAM_RANGES
LENGTH = C.LENGTH


def dt_frame(d, n_frames=None):
    F = C.GEOM[d]["n_frames"] if n_frames is None else n_frames
    return (T_END - T_START) / (F - 1)


def bcs_for(bc, d):
    return (bc,) * d


def _g1d(x, x0, s2, bc, length):
    out = 0.0
    for m in range(-N_IMAGES, N_IMAGES + 1):
        if bc == "periodic":
            out = out + np.exp(-(x - x0 - m * length) ** 2 / s2)
        else:
            sign = -1.0 if bc == "dirichlet" else 1.0
            out = out + (np.exp(-(x - x0 - 2 * m * length) ** 2 / s2)
                         + sign * np.exp(-(x + x0 - 2 * m * length) ** 2 / s2))
    return out


def background_D(p):
    return speed_of_light(p) / (3.0 * (T.mua_background(p) + T.musp(p)))


speed_of_light = T.speed_of_light


def initial_state(p, X, bcs, t=T_START, length=LENGTH):
    s2 = 4.0 * background_D(p) * t
    out = 0.0
    for g in p["impulses"]:
        term = g["amp"]
        for a, x in enumerate(X):
            term = term * _g1d(x, g["centre"][a], s2, bcs[a], length)
        out = out + term
    return out


def _avg1d(edges, x0, s, bc, length):
    xl, xr = edges[:-1], edges[1:]
    avg = lambda a: s * math.sqrt(math.pi) / 2 * (erf((xr - a) / s) - erf((xl - a) / s)) / (xr - xl)
    out = np.zeros(len(xl))
    for m in range(-N_IMAGES, N_IMAGES + 1):
        if bc == "periodic":
            out += avg(x0 + m * length)
        else:
            sign = -1.0 if bc == "dirichlet" else 1.0
            out += avg(x0 + 2 * m * length) + sign * avg(-x0 + 2 * m * length)
    return out


def exact_cell_average(p, n_cells, bcs, t, length=LENGTH):
    """Exact cell averages on an n_cells^d grid at t >= T_START (homogeneous medium)."""
    d = len(bcs)
    c = speed_of_light(p)
    mu_a = T.mua_background(p)
    D = background_D(p)
    s = math.sqrt(4.0 * D * t)
    fac = (T_START / t) ** (d / 2) * math.exp(-mu_a * c * (t - T_START))
    edges = np.linspace(0.0, length, n_cells + 1)
    out = 0.0
    for g in p["impulses"]:
        term = g["amp"] * fac
        for a in range(d):
            v = _avg1d(edges, g["centre"][a], s, bcs[a], length)
            shp = [1] * d
            shp[a] = n_cells
            term = term * v.reshape(shp)
        out = out + term
    return out


def transport_time(p):
    return LM.transport_time(p)


def run_light(mu, c, musp, phi0, bcs, h, n_frames, dtf, record, t_start=T_START, nsub=None):
    """Array-level integrator. mu (B, *n) m^-1, c, musp (B,), phi0 (B, *n). Returns
    (frames (B, F, ...), nsub, D)."""
    d = len(bcs)
    B = mu.shape[0]
    ex = (slice(None),) + (None,) * d
    D = c[ex] / (3.0 * (mu + musp[ex]))
    kf = G.face_mean(D, d, bcs)
    absorb = mu * c[ex]
    if nsub is None:
        nsub = G.n_substeps(dtf, G.diffusion_dt(kf, bcs, h, absorb, mu.shape))
    P = {f"kf{a}": kf[a] for a in range(d)}
    P["absorb"] = absorb

    def rhs(t, y, Q):
        out = G.diffusion(y[0], [Q[f"kf{a}"] for a in range(d)], bcs, h)
        out -= Q["absorb"] * y[0]
        return (out,)
    out = G.integrate(rhs, (phi0,), P, dtf, nsub, n_frames, record, t_start=t_start)
    return out, np.asarray(nsub), D


class Light(C.Solver):
    name = "light"
    channels = (C.CH_LIGHT,)
    family = "scalar"

    def __init__(self, dim=2):
        super().__init__(dim)
        self.dt_frame = dt_frame(self.dim)

    def sample_params(self, rng, regime="train"):
        p = T.sample_tissue(rng, regime, self.dim, LENGTH)
        tr = PARAM_RANGES["train"]
        n_imp = int(rng.integers(tr["n_impulses"][0], tr["n_impulses"][1] + 1))
        p["impulses"] = [{"centre": [float(rng.uniform(*tr["impulse_centre_frac"]) * LENGTH)
                                     for _ in range(self.dim)],
                          "amp": float(rng.uniform(*tr["impulse_amp"]))} for _ in range(n_imp)]
        return p

    def arrays(self, params_list, bc, n):
        d = self.dim
        X = C.mesh(n, LENGTH, d)
        per = bc == "periodic"
        mu = np.stack([np.broadcast_to(T.mua_field(p, X, LENGTH, per), (n,) * d) for p in params_list])
        c = np.array([speed_of_light(p) for p in params_list])
        ms = np.array([T.musp(p) for p in params_list])
        phi0 = np.stack([np.broadcast_to(initial_state(p, X, bcs_for(bc, d)), (n,) * d)
                         for p in params_list])
        return mu, c, ms, phi0

    def simulate_batch(self, params_list, bc, n_frames=None, rng=None, n_internal=None):
        n_frames, n = self._defaults(n_frames, n_internal)
        self.check_bc(bc)
        d = self.dim
        mu, c, ms, phi0 = self.arrays(params_list, bc, n)
        rec = lambda y: C.downsample(y[0], d, self.n_out)
        out, nsub, D = run_light(mu, c, ms, phi0, bcs_for(bc, d), LENGTH / n, n_frames,
                                 self.dt_frame, rec)
        trajs = []
        for b, p in enumerate(params_list):
            f = np.zeros((n_frames, C.N_CH) + (self.n_out,) * d, np.float32)
            f[:, C.CH_LIGHT] = out[b]
            q = dict(p)
            ttr = transport_time(p)
            ltr = 1.0 / (T.mua_background(p) + T.musp(p))
            q.update({"dt": float(self.dt_frame / nsub[b]), "n_internal": n,
                      "t_start_s": T_START, "t_end_s": T_START + (n_frames - 1) * self.dt_frame,
                      "mua_background_m-1": float(T.mua_background(p)),
                      "mua_max_m-1": float(mu[b].max()), "musp_m-1": float(ms[b]),
                      "c_m_s": float(c[b]), "D_max_m2_s": float(D[b].max()),
                      "mua_over_musp_max": float(mu[b].max() / ms[b]),
                      "mua_over_musp_cap": T.CAP, "t_tr_s": ttr,
                      "start_over_t_tr": T_START / ttr,
                      "init_width_over_l_tr": math.sqrt(4 * background_D(p) * T_START) / ltr,
                      "phi_init_max": float(phi0[b].max())})
            trajs.append(C.Trajectory(f, self.name, q, self.dt_frame, LENGTH, bc,
                                      self.channels, d))
        return trajs

    # ------------------------------------------------------------ verification
    VERIFY_PARAMS = LM.LightSolver.VERIFY_PARAMS

    def green_check(self, n_list=None):
        """Green's function of an impulse in a homogeneous medium, periodic box with all
        images summed (images at distance >= L contribute below exp(-25) at t1), from the
        exact state at t0 (1/e radius L/10) to t1 = 4 t0, point values at cell centres."""
        d = self.dim
        n_list = n_list or C.default_n_list(d)
        p = self.VERIFY_PARAMS
        mu_a = T.mua_background(p); c = speed_of_light(p)
        Dv = c / (3.0 * (mu_a + T.musp(p)))
        L = LENGTH
        w0 = L / 10.0
        t0 = w0 ** 2 / (4 * Dv); t1 = 4 * t0
        x0 = [0.5 * L + 0.013 * L * (a + 1) for a in range(d)]   # off the grid symmetry

        def exact(X, t):
            s2 = 4 * Dv * t
            g = 0.0
            for sft in G.image_shifts(d, True, 2):
                g = g + np.exp(-G.dist2_point(X, x0, L, sft) / s2)
            return c * (4 * math.pi * Dv * t) ** (-d / 2) * np.exp(-mu_a * c * t) * g
        errs, steps = [], []
        for n in n_list:
            X = C.mesh(n, L, d)
            bcs = ("periodic",) * d
            mu = np.full((1,) + (n,) * d, mu_a)
            phi0 = np.broadcast_to(exact(X, t0), (n,) * d)[None].copy()
            out, ns, _ = run_light(mu, np.array([c]), np.array([T.musp(p)]), phi0, bcs, L / n,
                                   2, t1 - t0, lambda y: y[0].copy(), t_start=t0)
            num = out[0, -1]
            ex = np.broadcast_to(exact(X, t1), num.shape)
            errs.append(float(np.sqrt(np.sum((num - ex) ** 2) / np.sum(ex ** 2))))
            steps.append(int(ns[0]))
        rep = C.convergence_report(self.name, f"green_function_d{d}", n_list, errs, self.order)
        rep.update({"d": d, "norm": "relative L2 at t1 (point values at cell centres)",
                    "t0_s": t0, "t1_s": t1, "D_m2_s": Dv, "mu_a_m-1": mu_a, "c_m_s": c,
                    "w0_m": w0, "source": x0, "rk4_steps": steps,
                    "bc": "periodic, images summed in the exact solution"})
        return rep

    def window_check(self, bc, n_list=None):
        """Production path (simulate_batch) for a homogeneous medium over the full record:
        last frame (t = T_END) against exact cell averages on the output grid."""
        d = self.dim
        n_list = n_list or C.default_n_list(d)
        p = dict(self.VERIFY_PARAMS)
        p["wavelength_nm"] = 800
        p["absorbers"] = []
        p["impulses"] = [{"centre": [0.3 * LENGTH, 0.55 * LENGTH, 0.4 * LENGTH][:d], "amp": 1.0},
                         {"centre": [0.62 * LENGTH, 0.3 * LENGTH, 0.7 * LENGTH][:d], "amp": 0.6},
                         {"centre": [0.8 * LENGTH, 0.75 * LENGTH, 0.25 * LENGTH][:d], "amp": 0.8}]
        F = self.n_frames
        t_last = T_START + (F - 1) * self.dt_frame
        ex = exact_cell_average(p, self.n_out, bcs_for(bc, d), t_last)
        errs = []
        for n in n_list:
            tr = self.simulate_batch([dict(p)], bc, F, None, n)[0]
            num = tr.fields[-1, C.CH_LIGHT].astype(float)
            errs.append(float(np.sqrt(np.sum((num - ex) ** 2) / np.sum(ex ** 2))))
        rep = C.convergence_report(self.name, f"record_window_{bc}_d{d}", n_list, errs, self.order)
        rep.update({"d": d, "bc": bc, "t_start_s": T_START, "t_last_s": t_last,
                    "norm": f"relative L2 on the {self.n_out}^{d} output cells (exact cell averages)"})
        return rep

    def verify(self, n_list=None, window_n_list=None):
        cases = {"green": self.green_check(n_list)}
        for bc in C.BCS_SCALAR:
            cases[f"window_{bc}"] = self.window_check(bc, window_n_list)
        main = dict(cases["green"])
        main["check"] = f"Green's function (d = {self.dim}) and record window per bc"
        main["cases"] = cases
        main["passed"] = all(v["passed"] for v in cases.values())
        main["time_window"] = {"T_START_s": T_START, "T_END_s": T_END, "DT_FRAME_s": self.dt_frame}
        return main


class Light2D(Light):
    def __init__(self):
        super().__init__(2)


class Light3D(Light):
    def __init__(self):
        super().__init__(3)
