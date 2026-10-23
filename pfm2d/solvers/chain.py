"""Photoacoustic chain and its two single-link systems on a d-dimensional Cartesian grid
(d = 2, 3), as pfm/solvers/chain.py and pfm/solvers/coupled.py.

    phi(x, t) = E(t) phi_s(x),   div(D_s grad phi_s) - mu_a phi_s = 0,  D_s = 1/(3(mu_a + mu_s'))
    dT/dt = alpha lap T + H / (rho C_p),      H = mu_a phi                     (K)
    dp/dt = -rho c^2 div u + Gamma H,  du/dt = -(1/rho) grad p                 (Pa)

Chain2D/3D ("chain") writes channels 0, 1, 2; Photothermal ("photothermal") channels 0, 1
(channel 2 zero); Thermoacoustic ("thermoacoustic") channels 1, 2 (channel 0 zero), with
heating h = mu_a phi_s built from a chain tissue draw exactly as the chain builds it (the
fluence is then discarded), as coupled.py does.

Quasi-static light: the slowest relaxation of the time-dependent diffusion equation is no
slower than in 1D (the lowest transverse mode is uniform), so chain.py's bound applies:
at most 0.8135 ns at the training wavelengths, under 0.01 of a 2D frame.

Illumination (fixed-fluence boundary, modelling choice inherited from chain.py), solved
per trajectory by grid.steady_solve (scipy sparse direct solver for d = 2; conjugate
gradients with Jacobi preconditioner, relative residual 1e-10, for d = 3; the achieved
residual is recorded per trajectory):
  walls (pressure_release, rigid): light enters through the face x_0 = 0, phi_s = phi0
      there; phi_s = 0 on the far face x_0 = L (zero-fluence free surface, as 1D); zero
      flux on the faces normal to the other axes (a laterally extended slab lit uniformly;
      modelling choice). For a uniform medium the solution is the 1D profile.
  periodic: light enters through BOTH faces normal to x_0 (phi_s = phi0 at x_0 = 0 and
      x_0 = L, the 1D periodic device, so heating is continuous across the wrap) and phi_s
      is periodic in the other directions.
Thermal walls: periodic for periodic, zero flux otherwise (chain.thermal_bc).

Parameters: chain.PARAM_RANGES (imported): optics as Light (tissue.py, d-dimensional
absorbers, the 0.3 cap), c, rho (c_soft, rho_soft ranges), alpha log-uniform in [fat,
muscle], C_p in [fat, muscle], Gruneisen in [0.201, 0.245] (train) and the fatty-tissue
range (ood), phi0 in [0.5, 1] x 1e7 W m^-2, sin^2 envelope of duration T_p in [0.25, 0.5]
L / c_ref = 2.05 to 4.10 us, pre_frames m in [0, 8]. The light switches on at
t_s = -m chain.DT_FRAME (the 1D frame, 64.06 ns), so the physical distribution of t_s is the
1D one; the integration starts at the whole number of frames K = ceil(max m chain.DT_FRAME /
DT_FRAME) before t = 0 (the state is exactly zero until t_s). ood kinds as 1D: chain
{wavelength, blood_fraction, gruneisen_fat}, photothermal {wavelength, blood_fraction},
thermoacoustic {blood_fraction, gruneisen_fat}, uniformly.

Frame interval: the sound value DT_FRAME = (L / n_out) / c_ref (128.12 ns for d = 2,
256.25 ns for d = 3), so a sound pulse moves about one output cell per frame, as in 1D.

Time stepping: (T, p, u) by RK4 with substeps from min(acoustic CFL 0.5, thermal limit),
chain.run_chain's rule. Photothermal integrates T alone with the same substep count, so
for equal parameters its channels equal the chain's to round-off (coupled.py's device).

Boundary conditions: periodic, pressure_release (train), rigid (held out). For
Photothermal the acoustic wall type has no effect (pressure_release and rigid give the same
trajectory), as in 1D.
"""
from __future__ import annotations

import math

import numpy as np

from pfm.solvers import chain as CH
from pfm.solvers import coupled as CP
from pfm.solvers import light as LM

from .. import common as C
from . import grid as G
from . import references as REF
from . import sound as S2
from . import tissue as T

LENGTH = C.LENGTH
PARAM_RANGES = CH.PARAM_RANGES
OOD_KINDS = {"chain": ("wavelength", "blood_fraction", "gruneisen_fat"),
             "photothermal": CP.OOD_KINDS["photothermal"],
             "thermoacoustic": CP.OOD_KINDS["thermoacoustic"]}
ALLOWED_BCS = C.BCS_SOUND


def dt_frame(d):
    return S2.dt_frame(d)


def thermal_bcs(bc, d):
    return ("periodic",) * d if bc == "periodic" else ("neumann",) * d


def light_bspec(bc, d, phi0):
    first = ("dirichlet", phi0, phi0 if bc == "periodic" else 0.0)
    rest = "periodic" if bc == "periodic" else "neumann"
    return [first] + [rest] * (d - 1)


def light_face_bcs(bc, d):
    return ["dirichlet"] + ["periodic" if bc == "periodic" else "neumann"] * (d - 1)


def sample_chain(rng, regime, d, kinds):
    tr = PARAM_RANGES["train"]
    u = lambda ab: float(rng.uniform(*ab))
    kind = None
    if regime == "ood":
        kind = kinds[int(rng.integers(len(kinds)))]
    elif regime != "train":
        raise ValueError(regime)
    opt = "ood" if kind in ("wavelength", "blood_fraction") else "train"
    p = T.sample_tissue(rng, opt, d, LENGTH, ood_kind=kind if opt == "ood" else None)
    p["regime"] = regime; p["ood_kind"] = kind; p["dim"] = d
    p["c"] = u(tr["c_m_s"]); p["rho"] = u(tr["rho_kg_m3"])
    p["alpha"] = float(np.exp(rng.uniform(*np.log(tr["alpha_m2_s"]))))
    p["cp"] = u(tr["cp_J_kgK"])
    p["gruneisen"] = u(CH.GAMMA_OOD_FAT) if kind == "gruneisen_fat" else u(tr["gruneisen"])
    p["phi0"] = u(tr["phi0_W_m2"])
    p["T_p"] = u(tr["pulse_duration_s"])
    p["pre_frames"] = int(rng.integers(tr["pre_frames"][0], tr["pre_frames"][1] + 1))
    p["t_pre_s"] = p["pre_frames"] * CH.DT_FRAME
    return p


def steady_fluence(mu, musp, bc, phi0, h, method="auto"):
    """phi_s for one trajectory: mu (*n), musp, phi0 scalars."""
    d = mu.ndim
    Ds = 1.0 / (3.0 * (mu + musp))
    kf = G.face_mean(Ds, d, light_face_bcs(bc, d))
    return G.steady_solve(kf, mu, light_bspec(bc, d, phi0), h, method=method)


def envelope(t, Q, d):
    return CH.envelope(t, Q).reshape((-1,) + (1,) * d)


def chain_substeps(P, bcs, h, dtf, cell_shape):
    d = len(bcs)
    B = cell_shape[0]
    Kmax = np.broadcast_to(P["K"], cell_shape).reshape(B, -1).max(1)
    imax = np.broadcast_to(P["irf0"], cell_shape).reshape(B, -1).max(1)
    cmax = np.sqrt(Kmax * imax)
    with np.errstate(divide="ignore"):
        dt_ac = G.CFL_ACOUSTIC * h / cmax
    tb = thermal_bcs(bcs[0], d)
    dt_th = G.diffusion_dt([P["alpha"]] * d, tb, h, np.zeros((B,) + (1,) * d), cell_shape)
    return G.n_substeps(dtf, np.minimum(dt_ac, dt_th))


def run_coupled(P, y0, bcs, h, n_frames, dtf, t_start, record, nsub=None, acoustics=True):
    """Integrate (T, p, u...) (acoustics=True) or T alone. bcs: acoustic bcs per axis."""
    d = len(bcs)
    tb = ("periodic" if bcs[0] == "periodic" else "neumann",) * d
    cell_shape = y0[0].shape
    if nsub is None:
        nsub = chain_substeps(P, bcs, h, dtf, cell_shape)

    if acoustics:
        def rhs(t, y, Q):
            Tm, p = y[0], y[1]
            H = envelope(t, Q, d) * Q["h_s"]
            dT = G.diffusion(Tm, [Q["alpha"]] * d, tb, h)
            dT += H * Q["inv_rcp"]
            dp, du = G.acoustic_rhs(p, list(y[2:]), Q["K"], [Q[f"irf{a}"] for a in range(d)],
                                    bcs, h, src=Q["gamma"] * H)
            return (dT, dp, *du)
    else:
        def rhs(t, y, Q):
            H = envelope(t, Q, d) * Q["h_s"]
            dT = G.diffusion(y[0], [Q["alpha"]] * d, tb, h)
            dT += H * Q["inv_rcp"]
            return (dT,)
    out, st = G.integrate(rhs, y0, P, dtf, nsub, n_frames, record, t_start=t_start,
                          return_state=True)
    return out, st, np.asarray(nsub)


def build_P(params_list, mu, phi_s):
    """Per-trajectory coefficient arrays (leading batch axis) for run_coupled from the
    chain parameters, mu_a (B, *n) and phi_s (B, *n)."""
    d = mu.ndim - 1
    B = len(params_list)
    ex = (slice(None),) + (None,) * d
    arr = lambda k: np.array([p[k] for p in params_list], float)
    rho, c = arr("rho"), arr("c")
    t_pre = np.array([p.get("t_pre_s", p["pre_frames"] * CH.DT_FRAME) for p in params_list])
    P = {"h_s": mu * phi_s, "inv_rcp": (1.0 / (rho * arr("cp")))[ex],
         "alpha": arr("alpha")[ex], "K": (rho * c ** 2)[ex], "gamma": arr("gruneisen")[ex],
         "t_s": -t_pre[:, None], "T_p": arr("T_p")[:, None], "const": np.zeros((B, 1))}
    for a in range(d):
        P[f"irf{a}"] = (1.0 / rho)[ex]
    return P


class _ChainBase(C.Solver):
    family = "sound"
    acoustics = True

    def __init__(self, dim=2):
        super().__init__(dim)
        self.dt_frame = dt_frame(self.dim)

    def sample_params(self, rng, regime="train"):
        return sample_chain(rng, regime, self.dim, OOD_KINDS[self.name])

    def assemble(self, params_list, bc, n, method="auto"):
        d = self.dim
        h = LENGTH / n
        X = C.mesh(n, LENGTH, d)
        per = bc == "periodic"
        B = len(params_list)
        ex = (slice(None),) + (None,) * d
        mu = np.stack([np.broadcast_to(T.mua_field(p, X, LENGTH, per), (n,) * d) for p in params_list])
        ms = np.array([T.musp(p) for p in params_list])
        phi_s = np.empty_like(mu)
        infos = []
        for b, p in enumerate(params_list):
            phi_s[b], inf = steady_fluence(mu[b], ms[b], bc, p["phi0"], h, method)
            infos.append(inf)
        P = build_P(params_list, mu, phi_s)
        return P, phi_s, mu, ms, infos

    def _run(self, params_list, bc, n_frames, n, nsub=None, record_internal=False):
        d = self.dim
        self.check_bc(bc)
        h = LENGTH / n
        P, phi_s, mu, ms, infos = self.assemble(params_list, bc, n)
        B = len(params_list)
        tmax = max(p["t_pre_s"] for p in params_list)
        K = int(math.ceil(tmax / self.dt_frame - 1e-9))
        cell = (B,) + (n,) * d
        if self.acoustics:
            y0 = (np.zeros(cell), np.zeros(cell)) + tuple(
                np.zeros((B,) + tuple(n + 1 if k == a else n for k in range(d))) for a in range(d))
        else:
            y0 = (np.zeros(cell),)
        ds = (lambda u: u.copy()) if record_internal else (lambda u: C.downsample(u, d, self.n_out))
        if self.acoustics:
            rec = lambda y: np.stack([ds(y[0]), ds(y[1])], axis=1)
        else:
            rec = lambda y: ds(y[0])[:, None]
        if nsub is None:
            nsub = chain_substeps(P, (bc,) * d, h, self.dt_frame, cell)
        out, _, nsub = run_coupled(P, y0, (bc,) * d, h, n_frames + K, self.dt_frame,
                                   -K * self.dt_frame, rec, nsub=nsub, acoustics=self.acoustics)
        out = out[:, K:]
        tf = np.arange(n_frames) * self.dt_frame
        E = np.stack([CH.envelope(tf, {"t_s": np.full((n_frames, 1), P["t_s"][b, 0]),
                                       "T_p": np.full((n_frames, 1), P["T_p"][b, 0]),
                                       "const": np.zeros((n_frames, 1))})[:, 0] for b in range(B)])
        phi_o = phi_s.copy() if record_internal else C.downsample(phi_s, d, self.n_out)
        return out, E, phi_o, P, mu, ms, infos, nsub, K

    def simulate_batch(self, params_list, bc, n_frames=None, rng=None, n_internal=None):
        n_frames, n = self._defaults(n_frames, n_internal)
        d = self.dim
        out, E, phi_o, P, mu, ms, infos, nsub, K = self._run(params_list, bc, n_frames, n)
        trajs = []
        for b, p in enumerate(params_list):
            f = np.zeros((n_frames, C.N_CH) + (self.n_out,) * d, np.float32)
            if C.CH_LIGHT in self.channels:
                f[:, C.CH_LIGHT] = E[b].reshape((-1,) + (1,) * d) * phi_o[b]
            if self.acoustics:
                f[:, C.CH_HEAT] = out[b, :, 0]
                if C.CH_PRESSURE in self.channels:
                    f[:, C.CH_PRESSURE] = out[b, :, 1]
            else:
                f[:, C.CH_HEAT] = out[b, :, 0]
            q = dict(p)
            Hmax = float(P["h_s"][b].max())
            q.update({"dt": float(self.dt_frame / nsub[b]), "n_internal": n, "bc": bc,
                      "frames_before_record": K,
                      "mua_background_m-1": float(T.mua_background(p)),
                      "mua_max_m-1": float(mu[b].max()), "musp_m-1": float(ms[b]),
                      "mua_over_musp_max": float(mu[b].max() / ms[b]),
                      "mua_over_musp_cap": T.CAP, "H_max_W_m3": Hmax,
                      "T_bound_K": float(Hmax * P["inv_rcp"].ravel()[b] * 0.5 * p["T_p"]),
                      "fluence_solve": infos[b]})
            trajs.append(C.Trajectory(f, self.name, q, self.dt_frame, LENGTH, bc,
                                      self.channels, d))
        return trajs


class Chain(_ChainBase):
    name = "chain"
    channels = (C.CH_LIGHT, C.CH_HEAT, C.CH_PRESSURE)


class Photothermal(_ChainBase):
    name = "photothermal"
    channels = (C.CH_LIGHT, C.CH_HEAT)
    acoustics = False


class Thermoacoustic(_ChainBase):
    name = "thermoacoustic"
    channels = (C.CH_HEAT, C.CH_PRESSURE)


Chain2D = type("Chain2D", (Chain,), {"__init__": lambda self: Chain.__init__(self, 2)})
Chain3D = type("Chain3D", (Chain,), {"__init__": lambda self: Chain.__init__(self, 3)})
Photothermal2D = type("Photothermal2D", (Photothermal,),
                      {"__init__": lambda self: Photothermal.__init__(self, 2)})
Photothermal3D = type("Photothermal3D", (Photothermal,),
                      {"__init__": lambda self: Photothermal.__init__(self, 3)})
Thermoacoustic2D = type("Thermoacoustic2D", (Thermoacoustic,),
                        {"__init__": lambda self: Thermoacoustic.__init__(self, 2)})
Thermoacoustic3D = type("Thermoacoustic3D", (Thermoacoustic,),
                        {"__init__": lambda self: Thermoacoustic.__init__(self, 3)})


# =============================================================================
# Verification
# =============================================================================
def _verify_medium():
    p = dict(LM.LightSolver.VERIFY_PARAMS)
    p["wavelength_nm"] = 800
    p["blood_fraction"] = 0.03                    # as chain.check_light
    p["absorbers"] = []
    mu_a = LM.mua_background(p); msp = LM.musp(p)
    return p, mu_a, msp, math.sqrt(3 * mu_a * (mu_a + msp)), 1.0 / (3 * (mu_a + msp))


def check_light_edge(d, bc, n_list=None):
    """Uniform medium through the production fluence path (steady_fluence): walls give
    sinh(mu_eff (L - x))/sinh(mu_eff L); periodic gives cosh(mu_eff (x - L/2))/cosh(mu_eff L/2)
    (both 1D profiles: the problem is one-dimensional for a uniform medium)."""
    n_list = n_list or C.default_n_list(d)
    _, mu_a, msp, mue, _ = _verify_medium()
    L = LENGTH
    f = (lambda x: np.cosh(mue * (x - L / 2)) / np.cosh(mue * L / 2)) if bc == "periodic" else \
        (lambda x: np.sinh(mue * (L - x)) / np.sinh(mue * L))
    errs, infos = [], []
    for n in n_list:
        X = C.mesh(n, L, d)
        phi, inf = steady_fluence(np.full((n,) * d, mu_a), msp, bc, 1.0, L / n)
        ex = np.broadcast_to(f(X[0]), phi.shape)
        errs.append(float(np.sqrt(np.sum((phi - ex) ** 2) / np.sum(ex ** 2))))
        infos.append(inf)
    rep = C.convergence_report("chain", f"light_steady_edge_{bc}_d{d}", n_list, errs, 2.0)
    rep.update({"d": d, "bc": bc, "mu_a_m-1": mu_a, "musp_m-1": msp, "mu_eff_m-1": mue,
                "solution": "cosh(mu_eff (x - L/2))/cosh(mu_eff L/2)" if bc == "periodic"
                else "sinh(mu_eff (L - x))/sinh(mu_eff L)", "solver": infos,
                "norm": "relative L2 (point values)"})
    return rep


def check_light_point(d, n_list=None):
    """Point-like source: unit power in the cell nearest the box centre (source density
    1/h^d), uniform medium, exact Dirichlet values on every face. Exact:
    K0(mu_eff r)/(2 pi D) (d = 2), exp(-mu_eff r)/(4 pi D r) (d = 3). Error over the cells
    with r >= L/8 (a fixed region away from the source)."""
    from scipy.special import k0
    n_list = n_list or C.default_n_list(d)
    _, mu_a, msp, mue, Dv = _verify_medium()
    L = LENGTH
    ex_f = (lambda r: k0(mue * r) / (2 * math.pi * Dv)) if d == 2 else \
           (lambda r: np.exp(-mue * r) / (4 * math.pi * Dv * r))
    errs, infos = [], []
    for n in n_list:
        h = L / n
        i0 = n // 2
        xs = [(i0 + 0.5) * h] * d
        X = C.mesh(n, L, d)
        bspec = []
        for a in range(d):
            def gval(xa, a=a):
                r2 = (xa - xs[a]) ** 2 + sum((X[k] - xs[k]) ** 2 for k in range(d) if k != a)
                r2 = np.broadcast_to(r2, tuple(1 if j == a else n for j in range(d)))
                return ex_f(np.sqrt(np.squeeze(r2, axis=a)))
            bspec.append(("dirichlet", gval(0.0), gval(L)))
        kf = [np.full(tuple(n + 1 if k == a else n for k in range(d)), Dv) for a in range(d)]
        src = np.zeros((n,) * d)
        src[(i0,) * d] = 1.0 / h ** d
        phi, inf = G.steady_solve(kf, np.full((n,) * d, mu_a), bspec, h, source=src)
        r = np.sqrt(G.dist2_point(X, xs, L, np.zeros(d)))
        r = np.broadcast_to(r, phi.shape)
        m = r >= L / 8
        ex = ex_f(r[m])
        errs.append(float(np.sqrt(np.sum((phi[m] - ex) ** 2) / np.sum(ex ** 2))))
        infos.append(inf)
    rep = C.convergence_report("chain", f"light_steady_point_source_d{d}", n_list, errs, 2.0)
    rep.update({"d": d, "mu_eff_m-1": mue, "D_m": Dv, "solver": infos,
                "solution": "K0(mu_eff r)/(2 pi D)" if d == 2 else "exp(-mu_eff r)/(4 pi D r)",
                "norm": "relative L2 over cells with r >= L/8",
                "bc": "exact Dirichlet values at the face centres of all walls"})
    return rep


def check_heating(d, n_list=None):
    """Constant heating H = H0 exp(-r^2/s^2), switched on at t = 0, through the chain's
    integrator (run_coupled), uniform medium, periodic box; p at t = 0.3 L / c against
    Gamma H0 x the Duhamel reference (chain.py quadrature, d = 2; closed form, d = 3)
    summed over the 3^d nearest images."""
    n_list = n_list or C.default_n_list(d)
    L = LENGTH
    c, rho, gam = CH.VER_C, CH.VER_RHO, CH.VER_GAMMA
    s = L / 10.0
    t1 = 0.3 * L / c
    H0 = 1.0e9
    x0 = [0.5 * L] * d
    prof, qc = REF.radial_profile(d, "duhamel", t1, s, c)
    errs = []
    for n in n_list:
        X = C.mesh(n, L, d)
        h_s = np.broadcast_to(H0 * REF.periodic_sum(lambda r: np.exp(-(r / s) ** 2), X, x0, L),
                              (n,) * d)[None].copy()
        P = {"h_s": h_s, "inv_rcp": np.ones((1,) + (1,) * d), "alpha": np.full((1,) + (1,) * d, CH.VER_ALPHA),
             "K": np.full((1,) + (1,) * d, rho * c * c), "gamma": np.full((1,) + (1,) * d, gam),
             "t_s": np.zeros((1, 1)), "T_p": np.ones((1, 1)), "const": np.ones((1, 1))}
        for a in range(d):
            P[f"irf{a}"] = np.full((1,) + (1,) * d, 1.0 / rho)
        y0 = (np.zeros((1,) + (n,) * d), np.zeros((1,) + (n,) * d)) + tuple(
            np.zeros((1,) + tuple(n + 1 if k == a else n for k in range(d))) for a in range(d))
        _, st, _ = run_coupled(P, y0, ("periodic",) * d, L / n, 2, t1, 0.0,
                               lambda y: y[1][:, :1].copy())
        num = st[1][0]
        ex = gam * H0 * np.broadcast_to(REF.periodic_sum(prof, X, x0, L), num.shape)
        errs.append(float(np.sqrt(np.sum((num - ex) ** 2) / np.sum(ex ** 2))))
    rep = C.convergence_report("chain", f"acoustic_constant_heating_d{d}", n_list, errs, 2.0)
    rep.update({"d": d, "s_m": s, "t_s": t1, "H0_W_m3": H0, "gruneisen": gam,
                "reference": REF.REFERENCE_NAME[("duhamel", d)], "reference_check": qc,
                "code_path": "chain.run_coupled (coupled T, p, u right-hand side)",
                "norm": "relative L2 (point values)"})
    return rep


def verify_all(d, n_list=None):
    cases = {}
    for bc in ("pressure_release", "periodic"):
        cases[f"light_edge_{bc}"] = check_light_edge(d, bc, n_list)
    cases["light_point_source"] = check_light_point(d, n_list)
    cases["constant_heating"] = check_heating(d, n_list)
    main = dict(cases["constant_heating"])
    main["system"] = "chain"
    main["check"] = (f"steady fluence (edge illumination, point source) and Duhamel constant "
                     f"heating, d = {d}")
    main["cases"] = cases
    qc = cases["constant_heating"]["reference_check"]
    main["passed"] = bool(all(v["passed"] for v in cases.values())
                          and qc.get("passed_1e-9", True))
    return main


def consistency_check(d, n_traj=4, seed=11, n_internal=None, regimes=("train", "ood"),
                      bcs=ALLOWED_BCS):
    """Same draws (chain sampler): Photothermal ch 0, 1 and Thermoacoustic ch 1, 2 against
    the chain's channels."""
    rows = []
    ch, pt, ta = Chain(d), Photothermal(d), Thermoacoustic(d)
    for i, reg in enumerate(regimes):
        for j, bc in enumerate(bcs):
            rng = np.random.default_rng(seed + 10 * i + j)
            params = [sample_chain(rng, reg, d, OOD_KINDS["chain"]) for _ in range(n_traj)]
            fc = np.stack([t.fields for t in ch.simulate_batch([dict(p) for p in params], bc, None, None, n_internal)])
            fp = np.stack([t.fields for t in pt.simulate_batch([dict(p) for p in params], bc, None, None, n_internal)])
            fa = np.stack([t.fields for t in ta.simulate_batch([dict(p) for p in params], bc, None, None, n_internal)])
            row = {"regime": reg, "bc": bc, "n_traj": n_traj}
            for lab, f, chans in (("photothermal", fp, (0, 1)), ("thermoacoustic", fa, (1, 2))):
                for cc in chans:
                    dd = np.abs(f[:, :, cc].astype(float) - fc[:, :, cc].astype(float))
                    row[f"{lab}_ch{cc}_max_abs_diff"] = float(dd.max())
                    row[f"{lab}_ch{cc}_max_rel_diff"] = float(dd.max() / np.abs(fc[:, :, cc]).max())
                unused = [cc for cc in range(3) if cc not in chans][0]
                row[f"{lab}_unused_ch{unused}_max_abs"] = float(np.abs(f[:, :, unused]).max())
            rows.append(row)
    worst = max(v for r in rows for k, v in r.items() if k.endswith("_max_rel_diff"))
    unused_max = max(v for r in rows for k, v in r.items() if "_unused_" in k)
    return {"dim": d, "rows": rows, "max_rel_diff": worst, "unused_channel_max_abs": unused_max,
            "passed": bool(worst <= 1e-6 and unused_max == 0.0),
            "pass_rule": "max |difference| / max |chain channel| <= 1e-6 (float32 output "
                         "resolution) and unused channel exactly zero"}


def check_uniform_heating(d, n_list=None):
    """Photothermal integrator (run_coupled, acoustics off), production frame interval and
    the chain's substep rule, full record: uniform constant heating H0 on from t = 0, T0 = 0.
    Exact T = H0 t / (rho C_p) (diffusion of a constant is zero on the grid; RK4 integrates
    a solution linear in t exactly), so the error is round-off at every resolution."""
    n_list = n_list or C.default_n_list(d)[:2]
    rcp = CP.VER_RCP
    H0 = 1.0e9
    dtf = dt_frame(d)
    F = C.GEOM[d]["n_frames"]
    errs = []
    for n in n_list:
        one = (1,) + (1,) * d
        P = {"h_s": np.full((1,) + (n,) * d, H0), "inv_rcp": np.full(one, 1.0 / rcp),
             "alpha": np.full(one, CH.VER_ALPHA), "K": np.full(one, CH.VER_RHO * CH.VER_C ** 2),
             "gamma": np.full(one, CH.VER_GAMMA), "t_s": np.zeros((1, 1)), "T_p": np.ones((1, 1)),
             "const": np.ones((1, 1))}
        for a in range(d):
            P[f"irf{a}"] = np.full(one, 1.0 / CH.VER_RHO)
        out, _, _ = run_coupled(P, (np.zeros((1,) + (n,) * d),), ("rigid",) * d, C.LENGTH / n,
                                F, dtf, 0.0, lambda y: y[0].copy(), acoustics=False)
        t = np.arange(F) * dtf
        ex = (H0 / rcp * t).reshape((-1,) + (1,) * d)
        errs.append(float(np.abs(out[0] - ex).max() / np.abs(ex).max()))
    return {"system": "photothermal", "check": f"uniform_constant_heating_exact_d{d}",
            "n": list(n_list), "error": errs, "observed_order": [], "promised_order": None,
            "passed": bool(max(errs) < 1e-12),
            "pass_rule": "max relative error over all frames < 1e-12 (exact in the scheme)",
            "rho_cp_J_m3K": rcp, "H0_W_m3": H0, "dt_frame_s": dtf}


def grid_operator_check(d, n=12, seed=0):
    """The sparse steady matrix equals the time-stepping operator (probed on random data,
    homogeneous wall values) for every boundary type, and the Gershgorin bound is at least
    the spectral radius (dense eigenvalues, small grid)."""
    rng = np.random.default_rng(seed)
    out = {}
    for bc in ("periodic", "dirichlet", "neumann"):
        k = rng.uniform(1.0, 3.0, (n,) * d)
        absorb = rng.uniform(0.0, 2.0, (n,) * d)
        bcs = (bc,) * d
        kf = G.face_mean(k, d, bcs)
        spec = [("dirichlet", 0.0, 0.0) if bc == "dirichlet" else bc] * d
        A, b = G.steady_matrix(kf, absorb, spec, 1.0)
        v = rng.normal(size=(n,) * d)
        op = G.diffusion(v[None], [x[None] for x in kf], bcs, 1.0)[0] - absorb * v
        diff = float(np.abs(A @ v.ravel() - op.ravel()).max() / np.abs(op).max())
        Gb = float(G.gershgorin([x[None] for x in kf], bcs, 1.0, absorb[None], (1,) + (n,) * d)[0])
        rad = float(np.abs(np.linalg.eigvals(A.toarray())).max())
        out[bc] = {"matrix_vs_operator_max_rel": diff, "gershgorin": Gb,
                   "spectral_radius": rad, "symmetric": bool(abs(A - A.T).max() < 1e-12)}
    out["passed"] = all(v["matrix_vs_operator_max_rel"] < 1e-12 and v["gershgorin"] >= v["spectral_radius"]
                        and v["symmetric"] for k, v in out.items() if k != "passed")
    return out
