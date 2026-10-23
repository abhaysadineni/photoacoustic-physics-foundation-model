"""Photoacoustic chain (held-out system H1): light heats tissue, heating launches sound.
Writes channel 0 (fluence rate), 1 (temperature rise) and 2 (pressure). Physical units.

Model (radially symmetric geometry of dimension d = 1 slab, 2 cylinder, 3 sphere, chosen by
the parameter "dim"; radial.py numerics)
---------------------------------------------------------------------------------------
Light is quasi-static on acoustic time scales. The slowest relaxation time of the
time-dependent diffusion equation on the domain, 1 / (D (pi / L)^2 + mu_a c), is at most
0.8135 ns over the training box at the training wavelengths and 0.852 ns at the 700 nm
out-of-range wavelength (D evaluated at the strongest scattering, mu_a ignored; recomputed
from the module constants on 21 Oct 2026), that is 0.013 frame intervals and about 4e-4 of
the shortest illumination pulse, so the fluence rate
is a steady diffusion profile scaled by the illumination envelope:
    phi(x, t) = E(t) phi_s(x),   d/dx (D_s d phi_s/dx) - mu_a phi_s = 0,
    D_s = 1 / (3 (mu_a + mu_s')),
with the radial operator for d = 2, 3. Illumination is a FIXED-FLUENCE (Dirichlet)
boundary (modelling choice; the Robin/extrapolated-boundary condition is not used):
    d = 1, walls (pressure_release, rigid): phi_s(0) = phi0 (illuminated face),
           phi_s(L) = 0 (far free surface, zero-fluence approximation);
    d = 1, periodic: phi_s(0) = phi_s(L) = phi0 (light enters through both faces, so the
           fluence and heating are continuous across the periodic wrap). This describes a
           periodic stack of slabs each lit from both sides; it is a modelling device kept
           so that the sound-family training boundary set can be used, not a single
           physical slab;
    d = 2, 3: phi_s(L) = phi0 on the outer surface (cylinder or sphere lit uniformly from
           outside); zero flux on the axis / at the centre by symmetry.
Heating rate per volume H = mu_a phi (W m^-3). Then
    dT/dt = alpha lap T + H / (rho C_p)              (temperature rise, K)
    dp/dt = -rho c^2 div u + Gamma H,   du/dt = -(1/rho) grad p
(the first-order form of (1/c^2) p_tt - lap p = (Gamma / c^2) dH/dt). Thermal walls:
zero flux (neumann) for wall boundaries, periodic for periodic; the thermal diffusion
length over the record, sqrt(4 alpha t) ~ 1 um, is below the internal cell (25 um), so the
choice does not affect the result.

Allowed acoustic boundary conditions
------------------------------------
pressure_release (tissue-air surfaces) and rigid (hard backing such as a glass window)
for every d; periodic for d = 1 only, with two-sided illumination (see above). The plain
one-sided illuminated slab with periodic acoustics would put a jump in heating at the
wrap and is not offered.

Parameters (JSON keys in PARAM_SOURCES)
----------------------------------------
Optics: exactly light.py (chromophores, scattering, 0 to 3 absorbers / vessels;
wavelengths 750, 800, 850 nm train, 700 nm ood (532 nm removed 20 Oct 2026); every draw
satisfies max_x mu_a / mu_s' <= 0.3 by rejection sampling, light.MUA_MUSP_CAP). Acoustics: c in c_soft.range, rho in
rho_soft.range (as SoundUniform). Thermal: alpha in [fat, muscle] (as heat.py); C_p in
[fat, muscle] specific heats from muscle_fat_thermal; rho shared with the acoustics.
Grueneisen parameter (revised 20 Oct 2026): a measured value for non-fatty SOLID soft
tissue at 37 C is still null in the JSON (gamma_soft_tissue_37C_measured, UNVERIFIED), but
fresh human whole blood has been measured at 37 C (gamma_blood_human_37C_measured: five
donors, 0.208 to 0.245, mean 0.226). Blood dominates the heating in the absorbers, so the
training range is [water at 37 C = 0.201 (gamma_water_NIST.37C), highest measured blood
donor at 37 C = 0.245]. The earlier range [0.129, 0.201] used a 22 C lower bound and is
superseded. Ood adds a fatty-tissue case: Gamma in [lard, subcutaneous fat] at 22 C
(0.69 to 0.81, gamma_tissue_measured).
Illumination: phi0 in [0.5, 1.0] x 1e7 W m^-2 (modelling choice; every field is linear in
phi0 and channels are normalised per system; no exposure-limit claim is made, because no
exposure limit is sourced). E(t) = sin^2(pi (t - t_s) / T_p) on [t_s, t_s + T_p], else 0,
T_p in [0.25, 0.5] x L / c_ref = 2.05 to 4.10 us (one quarter to one half of the slab's
acoustic transit time L / c_ref = 8.20 us; 32 to 64 frame intervals), t_s = -m dt_frame
with m an integer in [0, 8] (the record may start with the light already on; the solver
integrates from t_s and records from t = 0).

Domain and frame interval: L = 12.8 mm (as light, heat, sound). dt_frame = the sound
systems' value, (L / 128) / c_ref = 64.06 ns, so a sound pulse moves about one output cell
per frame, as in training. n_internal = 512 by default.

Verification (verify_all(); all three d):
 (a) steady light: plane exp(-mu_eff x), line K0(mu_eff r), point exp(-mu_eff r)/r on
     [r0, r0 + L] with exact Dirichlet values (r0 = L/8 for d = 2, 3); and the chain's own
     illumination: slab sinh(mu_eff (L - x))/sinh(mu_eff L), cylinder I0(mu_eff r)/I0,
     sphere sinh(mu_eff r)/r scaled; order 2.
 (b) heat through the coupled integrator (acoustics off): Gaussian spreading, and a
     constant Gaussian heater against the time integral of the Gaussian kernel; order 2.
 (c) free wave from p0 = exp(-r^2/s^2), u0 = 0: d'Alembert (d = 1, centre on the
     symmetry wall, so the even image is included exactly), Hankel transform by scipy
     quad (d = 2, accuracy checked against composite Gauss-Legendre and the t = 0
     identity), exact r p formula (d = 3); order 2.
 (d) constant heating H = H0 exp(-r^2/s^2) switched on at t = 0: Duhamel integral of (c)
     (exact erf form d = 1, exact d = 3, single Hankel integral for d = 2, checked against
     nested time x wavenumber quadrature); order 2.
 (e) top-hat p0 (slab, cylinder, sphere of radius L/4): L1 and L2 errors and orders at
     t = 0.15 and 0.45 L/c, report only; overlay data saved.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import integrate as sint
from scipy import special

from pfm import common
from pfm.solvers import _tissue as J
from pfm.solvers import heat as heat_mod
from pfm.solvers import light as light_mod
from pfm.solvers import radial as R
from pfm.solvers import sound as sound_mod

LENGTH = 12.8e-3
C_REF = sound_mod.C_REF
DT_FRAME = sound_mod.DT_FRAME
TRANSIT = LENGTH / C_REF

_G = J.get("gamma_tissue_measured")
GAMMA_NONFAT_22C = {k: J.leading_float(_G[k]) for k in
                    ("bovine_serum", "RBC_suspension_15.9_mg_ml", "RBC_suspension_59.3_mg_ml",
                     "blood_110_mg_ml_estimate")}
GAMMA_FAT_22C = {k: J.leading_float(_G[k]) for k in
                 ("porcine_lipid_lard", "porcine_subcutaneous_fat")}
GAMMA_WATER37 = float(J.get("gamma_water_NIST", "37C"))
# Round 2 (20 Oct 2026): measured human whole blood at 37 C (Villanueva-Palero 2017,
# Table 6, five donors 0.208 to 0.245) replaces the 22 C lower bound. The heating in the
# chain is dominated by blood in the absorbers, so the training range runs from water at
# 37 C (non-blood background) to the highest measured whole-blood donor at 37 C.
_B37 = J.get("gamma_blood_human_37C_measured", "37C_donors")
GAMMA_BLOOD37 = (min(_B37.values()), max(_B37.values()))
GAMMA_TRAIN = (GAMMA_WATER37, GAMMA_BLOOD37[1])
GAMMA_OOD_FAT = (min(GAMMA_FAT_22C.values()), max(GAMMA_FAT_22C.values()))
try:                                       # documented: this key is null and must stay unused
    J.get("gamma_soft_tissue_37C_measured")
    GAMMA_37C_SOFT_AVAILABLE = True
except J.MissingParameter:
    GAMMA_37C_SOFT_AVAILABLE = False

CP_RANGE = (float(J.get("muscle_fat_thermal", "fat", "c_J_kgK")),
            float(J.get("muscle_fat_thermal", "muscle", "c_J_kgK")))

PARAM_RANGES = {
    "train": {
        "optics": light_mod.PARAM_RANGES["train"],
        "c_m_s": list(sound_mod.C_RANGE), "rho_kg_m3": list(sound_mod.RHO_RANGE),
        "alpha_m2_s": heat_mod.PARAM_RANGES["train"]["alpha_m2_s"],
        "cp_J_kgK": list(CP_RANGE),
        "gruneisen": list(GAMMA_TRAIN),
        "phi0_W_m2": [0.5e7, 1.0e7],
        "pulse_duration_s": [0.25 * TRANSIT, 0.5 * TRANSIT],
        "pre_frames": [0, 8],
    },
    "ood": {
        "optics": light_mod.PARAM_RANGES["ood"],
        "gruneisen_fat": list(GAMMA_OOD_FAT),
        "note": "each ood trajectory shifts one thing: wavelength 700 nm, blood fractions "
                "above the training range, or Gamma in the fatty-tissue range (1/3 each)",
    },
}
PARAM_SOURCES = {
    "optics": "as light.PARAM_SOURCES",
    "c_m_s": "c_soft.range", "rho_kg_m3": "rho_soft.range",
    "alpha_m2_s": "muscle_fat_thermal (fat, muscle): k / (rho c)",
    "cp_J_kgK": "muscle_fat_thermal.fat.c_J_kgK, muscle_fat_thermal.muscle.c_J_kgK",
    "gruneisen (train low)": "gamma_water_NIST.37C (non-blood background at 37 C)",
    "gruneisen (train high)": "gamma_blood_human_37C_measured.37C_donors (max; measured "
                              "whole blood at 37 C, Villanueva-Palero 2017 Table 6)",
    "gruneisen (ood fat)": "gamma_tissue_measured: porcine_lipid_lard, "
                           "porcine_subcutaneous_fat (22 C)",
    "gruneisen 37 C soft tissue": "gamma_soft_tissue_37C_measured is null/UNVERIFIED; "
                                  "not used",
    "phi0_W_m2, envelope, pre_frames": "modelling choice",
    "pulse_duration_s": "modelling choice: 0.25 to 0.5 x L / c_soft.value",
    "dt_frame": "sound.DT_FRAME = (L / 128) / c_soft.value",
    "LENGTH": "modelling choice (12.8 mm)",
}

ALLOWED_BCS = {1: ("periodic", "pressure_release", "rigid"),
               2: ("pressure_release", "rigid"), 3: ("pressure_release", "rigid")}


def thermal_bc(bc):
    return "periodic" if bc == "periodic" else "neumann"


def steady_fluence(grid, mu_a, musp, bc, phi0):
    """phi_s for a batch: mu_a (B, n), musp (B, 1), phi0 (B,)."""
    Ds = 1.0 / (3.0 * (mu_a + musp))
    kf = R.face_mean(Ds, grid, "dirichlet")
    phi0 = np.asarray(phi0, float)
    if grid.d == 1:
        g_hi = phi0 if bc == "periodic" else np.zeros_like(phi0)
        return R.solve_steady(grid, kf, mu_a, "dirichlet", g_lo=phi0, g_hi=g_hi)
    return R.solve_steady(grid, kf, mu_a, "dirichlet", g_lo=0.0, g_hi=phi0)


def envelope(t, P):
    """E(t) per trajectory: const flag 1 -> E = 1 for t >= 0; else sin^2 pulse."""
    tt = t[:, None]
    s = np.clip((tt - P["t_s"]) / P["T_p"], 0.0, 1.0)
    pulse = np.sin(np.pi * s) ** 2
    return np.where(P["const"] > 0, 1.0, pulse)


def make_rhs(grid, bc):
    tbc = thermal_bc(bc)

    def rhs(t, y, P):
        T, p, u = y
        H = envelope(t, P) * P["h_s"]                 # W m^-3
        dT = R.diffusion(T, grid, P["alpha_f"], tbc) + H * P["inv_rcp"]
        dp, du = R.acoustic_rhs(p, u, grid, P["K"], P["irf"], bc, src=P["gamma"] * H)
        return dT, dp, du
    return rhs


def run_chain(grid, bc, P, y0, n_frames, dt_frame, t_start, record):
    """Integrate the coupled (T, p, u) system; P holds per-trajectory arrays."""
    B = y0[0].shape[0]
    cmax = np.sqrt(P["K"].max(axis=1) * P["irf"].max(axis=1))
    with np.errstate(divide="ignore"):
        dt_ac = R.CFL_ACOUSTIC * grid.dr / cmax          # inf when acoustics is off
    dt_th = R.diffusion_dt(grid, P["alpha_f"], thermal_bc(bc), np.zeros((B, 1)), B)
    nsub = R.n_substeps(dt_frame, np.minimum(dt_ac, dt_th))
    out, st = R.integrate(make_rhs(grid, bc), y0, P, dt_frame, nsub, n_frames, record,
                          t_start=t_start, return_state=True)
    return out, st, nsub


class PhotoacousticChain(common.Solver):
    name = "chain"
    channels = (common.CH_LIGHT, common.CH_HEAT, common.CH_PRESSURE)
    order = 2.0

    def __init__(self, dim: int = 1):
        self.dim = int(dim)

    def sample_params(self, rng, regime="train", dim=None):
        dim = self.dim if dim is None else int(dim)
        tr = PARAM_RANGES["train"]
        u = lambda ab: float(rng.uniform(*ab))
        kind = None
        if regime == "ood":
            kind = ["wavelength", "blood_fraction", "gruneisen_fat"][int(rng.integers(3))]
        opt_regime = "ood" if kind in ("wavelength", "blood_fraction") else "train"
        # capped draw (max_x mu_a / mu_s' <= light.MUA_MUSP_CAP), ood kind forced
        p = light_mod.sample_tissue(rng, opt_regime, LENGTH,
                                    ood_kind=kind if opt_regime == "ood" else None)
        p["regime"] = regime
        p["ood_kind"] = kind
        p["dim"] = dim
        p["c"] = u(tr["c_m_s"]); p["rho"] = u(tr["rho_kg_m3"])
        p["alpha"] = float(np.exp(rng.uniform(*np.log(tr["alpha_m2_s"]))))
        p["cp"] = u(tr["cp_J_kgK"])
        p["gruneisen"] = u(GAMMA_OOD_FAT) if kind == "gruneisen_fat" else u(tr["gruneisen"])
        p["phi0"] = u(tr["phi0_W_m2"])
        p["T_p"] = u(tr["pulse_duration_s"])
        p["pre_frames"] = int(rng.integers(tr["pre_frames"][0], tr["pre_frames"][1] + 1))
        return p

    def _assemble(self, params_list, grid, bc):
        per = bc == "periodic"
        B, n = len(params_list), grid.n
        mu = np.stack([light_mod.mua_field(p, grid.rc, grid.L, per) for p in params_list])
        musp = np.array([[light_mod.musp(p)] for p in params_list])
        phi0 = np.array([p["phi0"] for p in params_list])
        phi_s = steady_fluence(grid, mu, musp, bc, phi0)
        c = np.array([[p["c"]] for p in params_list]); rho = np.array([[p["rho"]] for p in params_list])
        P = {"h_s": mu * phi_s,
             "inv_rcp": 1.0 / (rho * np.array([[p["cp"]] for p in params_list])),
             "alpha_f": np.broadcast_to(np.array([[p["alpha"]] for p in params_list]),
                                        (B, n + 1)).copy(),
             "K": np.broadcast_to(rho * c ** 2, (B, n)).copy(),
             "irf": np.broadcast_to(1.0 / rho, (B, n + 1)).copy(),
             "gamma": np.array([[p["gruneisen"]] for p in params_list]),
             "t_s": np.array([[-p["pre_frames"] * DT_FRAME] for p in params_list]),
             "T_p": np.array([[p["T_p"]] for p in params_list]),
             "const": np.zeros((B, 1))}
        return P, phi_s, mu, musp

    def simulate_batch(self, params_list, bc, n_frames, rng=None, n_internal=512):
        dims = sorted({p.get("dim", self.dim) for p in params_list})
        if len(dims) > 1:
            out = [None] * len(params_list)
            for d in dims:
                idx = [i for i, p in enumerate(params_list) if p.get("dim", self.dim) == d]
                for i, tr in zip(idx, self.simulate_batch([params_list[i] for i in idx], bc,
                                                          n_frames, rng, n_internal)):
                    out[i] = tr
            return out
        d = dims[0]
        if bc not in ALLOWED_BCS[d]:
            raise ValueError(f"chain d = {d}: bc {bc!r} not in {ALLOWED_BCS[d]}")
        grid = R.RadialGrid(n_internal, LENGTH, d)
        B, n = len(params_list), grid.n
        P, phi_s, mu, musp = self._assemble(params_list, grid, bc)
        mpre = max(p["pre_frames"] for p in params_list)
        t_start = -mpre * DT_FRAME
        y0 = (np.zeros((B, n)), np.zeros((B, n)), np.zeros((B, n + 1)))
        rec = lambda y: np.stack([grid.downsample(y[0], common.N_OUT),
                                  grid.downsample(y[1], common.N_OUT)], axis=1)
        out, _, nsub = run_chain(grid, bc, P, y0, n_frames + mpre, DT_FRAME, t_start, rec)
        out = out[:, mpre:]                                   # frames at t = 0 .. 63 dt
        tf = np.arange(n_frames) * DT_FRAME
        trajs = []
        phi_out = grid.downsample(phi_s, common.N_OUT)
        for b, p in enumerate(params_list):
            Q = {k: v[b:b + 1] for k, v in P.items()}
            E = envelope(tf, {"t_s": np.repeat(Q["t_s"], n_frames, 0),
                              "T_p": np.repeat(Q["T_p"], n_frames, 0),
                              "const": np.zeros((n_frames, 1))})[:, 0]
            f = np.zeros((n_frames, common.N_CH, common.N_OUT), np.float32)
            f[:, common.CH_LIGHT] = E[:, None] * phi_out[b]
            f[:, common.CH_HEAT] = out[b, :, 0]
            f[:, common.CH_PRESSURE] = out[b, :, 1]
            q = dict(p)
            Hmax = float(P["h_s"][b].max())
            # 1D bound |p| <= Gamma max(H_s) int E dt (d'Alembert with images; see verify)
            q.update({"dt": float(DT_FRAME / nsub[b]), "n_internal": n_internal, "bc": bc,
                      "mua_background_m-1": float(light_mod.mua_background(p)),
                      "mua_max_m-1": float(mu[b].max()), "musp_m-1": float(musp[b, 0]),
                      "mua_over_musp_max": float(mu[b].max() / musp[b, 0]),
                      "mua_over_musp_cap": light_mod.MUA_MUSP_CAP,
                      "H_max_W_m3": Hmax,
                      "p_bound_1d_Pa": float(p["gruneisen"] * Hmax * 0.5 * p["T_p"])})
            trajs.append(common.Trajectory(f, self.name, q, DT_FRAME, LENGTH, bc,
                                           self.channels))
        return trajs

    def simulate(self, params, bc, n_frames, rng=None, n_internal=512):
        return self.simulate_batch([params], bc, n_frames, rng, n_internal)[0]

    def verify(self, n_list=(128, 256, 512)) -> dict:
        return verify_all(n_list)


# =============================================================================
# Verification
# =============================================================================
VER_C = C_REF
VER_RHO = float(J.get("rho_soft"))
VER_GAMMA = GAMMA_WATER37
VER_ALPHA = float(J.get("soft_tissue_alpha_th"))
# rho C_p in check (b): rho_soft x tissue_cp_wang (verification only)


def _rel_norms(num, ex, V):
    l2 = float(np.sqrt(np.sum(V * (num - ex) ** 2) / np.sum(V * ex ** 2)))
    l1 = float(np.sum(V * np.abs(num - ex)) / np.sum(V * np.abs(ex)))
    return l1, l2


def _uniform_P(grid, B=1, h_s=None, const=1.0):
    n = grid.n
    return {"h_s": np.zeros((B, n)) if h_s is None else h_s,
            "inv_rcp": np.full((B, 1), 1.0), "alpha_f": np.full((B, n + 1), VER_ALPHA),
            "K": np.full((B, n), VER_RHO * VER_C ** 2), "irf": np.full((B, n + 1), 1.0 / VER_RHO),
            "gamma": np.full((B, 1), VER_GAMMA), "t_s": np.zeros((B, 1)),
            "T_p": np.ones((B, 1)), "const": np.full((B, 1), const)}


# ------------------------------------------------------------------- (a) light
def check_light(d, n_list, kind):
    """kind 'source': plane/line/point source solution on [r0, r0 + L] with exact Dirichlet
    values at both walls. kind 'surface' (d = 2, 3): the chain's own outer-surface
    illumination, phi_s(L) = 1, symmetric at the centre."""
    p = dict(light_mod.LightSolver.VERIFY_PARAMS); p["wavelength_nm"] = 800; p["blood_fraction"] = 0.03
    mu_a = light_mod.mua_background(p); msp = light_mod.musp(p)
    mu_eff = math.sqrt(3 * mu_a * (mu_a + msp))
    L = LENGTH
    if kind == "source":
        r0 = 0.0 if d == 1 else L / 8
        f = {1: lambda r: np.exp(-mu_eff * r),
             2: lambda r: special.k0(mu_eff * r),
             3: lambda r: np.exp(-mu_eff * r) / r}[d]
    else:
        r0 = 0.0
        f = {1: lambda r: np.sinh(mu_eff * (L - r)) / np.sinh(mu_eff * L),
             2: lambda r: special.i0(mu_eff * r) / special.i0(mu_eff * L),
             3: lambda r: np.sinh(mu_eff * r) / r * L / np.sinh(mu_eff * L)}[d]
    errs = []
    for n in n_list:
        g = R.RadialGrid(n, L, d, r0=r0)
        mu = np.full((1, n), mu_a)
        kf = R.face_mean(1.0 / (3 * (mu + msp)), g, "dirichlet")
        lo = f(g.rf[0]) if (d == 1 or r0 > 0) else 0.0
        num = R.solve_steady(g, kf, mu, "dirichlet", g_lo=lo, g_hi=f(g.rf[-1]))[0]
        if kind == "surface":   # the chain's own code path
            num2 = steady_fluence(g, mu, np.full((1, 1), msp), "rigid", np.ones(1))[0]
            assert np.allclose(num, num2)
        ex = f(g.rc)
        errs.append(_rel_norms(num, ex, g.V)[1])
    name = {"source": {1: "plane exp(-mu_eff x)", 2: "line source K0(mu_eff r)",
                       3: "point source exp(-mu_eff r)/r"},
            "surface": {1: "slab lit at x = 0, phi(L) = 0: sinh(mu_eff (L - x))/sinh(mu_eff L)",
                        2: "outer-surface illumination I0(mu_eff r)/I0(mu_eff L)",
                        3: "outer-surface illumination sinh(mu_eff r)/r"}}[kind][d]
    rep = common.convergence_report("chain", f"light_steady_{kind}_d{d}", n_list, errs, 2.0)
    rep.update({"d": d, "solution": name, "mu_a_m-1": mu_a, "musp_m-1": msp,
                "mu_eff_m-1": mu_eff, "domain_m": [r0, r0 + L],
                "norm": "relative L2, volume weighted",
                "bc": "Dirichlet with exact values" if kind == "source" else
                      "chain code path (steady_fluence): d = 1 phi(0) = 1, phi(L) = 0; "
                      "d = 2, 3 phi(L) = 1, symmetric centre"})
    return rep


# --------------------------------------------------------------------- (b) heat
def check_heat(d, n_list, heated=False):
    """Gaussian spreading through the chain's own coupled right-hand side and integrator
    (make_rhs / run_chain), with the acoustic coupling switched off (K = 0, 1/rho = 0,
    Gamma = 0) so that the thermal stability limit governs and the thermal time scale
    can be reached. heated=False: T0 Gaussian, H = 0, exact
    T = (s0^2/(s0^2 + 4 alpha t))^(d/2) exp(-r^2/(s0^2 + 4 alpha t)).
    heated=True: T0 = 0, constant H = H0 exp(-r^2/s0^2) on from t = 0, exact
    T = (H0/(rho C_p)) int_0^t (s0^2/(s0^2 + 4 alpha tau))^(d/2)
        exp(-r^2/(s0^2 + 4 alpha tau)) dtau  (scipy quad_vec, rel. tol 1e-12)."""
    L = LENGTH
    s0 = L / 20.0
    t1 = 3.0 * s0 ** 2 / (4.0 * VER_ALPHA)
    rcp = VER_RHO * float(J.get("tissue_cp_wang"))        # rho C_p for the check
    H0 = rcp / t1                                         # 1 K over t1 if not spreading
    G = lambda r, tau: ((s0 ** 2 / (s0 ** 2 + 4 * VER_ALPHA * tau)) ** (d / 2)
                        * np.exp(-r ** 2 / (s0 ** 2 + 4 * VER_ALPHA * tau)))
    errs = []
    for n in n_list:
        g = R.RadialGrid(n, L, d)
        r = g.rc
        P = _uniform_P(g, h_s=(H0 * np.exp(-r ** 2 / s0 ** 2))[None] if heated else None)
        P["K"][:] = 0.0; P["irf"][:] = 0.0; P["gamma"][:] = 0.0
        P["inv_rcp"][:] = 1.0 / rcp
        T0 = np.zeros((1, n)) if heated else G(r, 0.0)[None]
        y0 = (T0, np.zeros((1, n)), np.zeros((1, n + 1)))
        with np.errstate(divide="ignore"):
            _, st, _ = run_chain(g, "rigid", P, y0, 2, t1, 0.0, lambda y: y[0][:, :1].copy())
        TN = st[0][0]
        if heated:
            ex = H0 / rcp * sint.quad_vec(lambda tau: G(r, tau), 0.0, t1, epsabs=0.0,
                                          epsrel=1e-12)[0]
        else:
            ex = G(r, t1)
        errs.append(_rel_norms(TN, ex, g.V)[1])
    tag = "heated" if heated else "gaussian"
    rep = common.convergence_report("chain", f"heat_{tag}_d{d}", n_list, errs, 2.0)
    rep.update({"d": d, "s0_m": s0, "t1_s": t1, "alpha_m2_s": VER_ALPHA,
                "rho_cp_J_m3K": rcp, "H0_W_m3": H0 if heated else 0.0,
                "centre": "r = 0 (symmetry wall for d = 1, axis/centre for d = 2, 3)",
                "code_path": "chain.make_rhs + chain.run_chain (coupled T, p, u state) with "
                             "acoustics switched off; thermal bc of the chain (neumann for "
                             "wall boundaries)",
                "norm": "relative L2, volume weighted"})
    return rep


# ------------------------------------------------------------- acoustic references
def hankel_gauss(r, t, s, c=VER_C, kind="free", rtol=1e-10):
    """d = 2 reference for p0 = exp(-r^2/s^2) (kind 'free', zero velocity) or for constant
    unit heating q = exp(-r^2/s^2) switched on at t = 0 (kind 'duhamel', returns
    int_0^t W(t - tau) q dtau, i.e. pressure / (Gamma H0)).
    free:    int_0^inf k P0(k) J0(k r) cos(c k t) dk
    duhamel: int_0^inf P0(k) J0(k r) sin(c k t) / c dk   (time integral done exactly)
    P0(k) = (s^2/2) exp(-k^2 s^2/4). The integrand is below 1e-17 of its scale beyond
    k_max = 2 sqrt(40) / s."""
    kmax = 2.0 * math.sqrt(40.0) / s
    P0 = lambda k: 0.5 * s * s * math.exp(-k * k * s * s / 4.0)
    # absolute floor 1e-15 of the solution scale (1 for 'free', s/c for 'duhamel') so that
    # points where the solution is ~0 do not demand an unreachable relative tolerance
    epsabs = 1e-15 * (1.0 if kind == "free" else s / c)
    out = np.empty(len(r)); err = np.empty(len(r))
    for i, ri in enumerate(r):
        if kind == "free":
            f = lambda k: k * P0(k) * special.j0(k * ri) * math.cos(c * k * t)
        else:
            f = lambda k: P0(k) * special.j0(k * ri) * math.sin(c * k * t) / c
        # break the interval at the oscillation scale for robust adaptive quadrature
        nb = int(np.ceil(kmax * (ri + c * t + s) / (2 * np.pi))) + 1
        edges = np.linspace(0.0, kmax, nb + 1)
        v = 0.0; e = 0.0
        with warnings.catch_warnings():
            # quad warns when a subinterval's contribution is at round-off level; the
            # accuracy is checked independently (_hankel_accuracy)
            warnings.simplefilter("ignore", sint.IntegrationWarning)
            for a, b in zip(edges[:-1], edges[1:]):
                vi, ei = sint.quad(f, a, b, epsabs=epsabs, epsrel=rtol, limit=200)
                v += vi; e += ei
        out[i], err[i] = v, e
    return out, err


def hankel_gauss_gl(r, t, s, c=VER_C, kind="free", nodes=20000):
    """Independent check: composite Gauss-Legendre on [0, k_max]."""
    kmax = 2.0 * math.sqrt(40.0) / s
    x, w = np.polynomial.legendre.leggauss(64)
    m = nodes // 64
    edges = np.linspace(0, kmax, m + 1)
    k = (0.5 * (edges[1:] - edges[:-1])[:, None] * x[None] + 0.5 * (edges[1:] + edges[:-1])[:, None]).ravel()
    wk = (0.5 * (edges[1:] - edges[:-1])[:, None] * w[None]).ravel()
    P0 = 0.5 * s * s * np.exp(-k * k * s * s / 4)
    J0 = special.j0(np.outer(r, k))
    if kind == "free":
        return J0 @ (wk * k * P0 * np.cos(c * k * t))
    return J0 @ (wk * P0 * np.sin(c * k * t) / c)


def free_gauss_exact(d, r, t, s, c=VER_C):
    """Free wave from p0 = exp(-r^2/s^2), u0 = 0 (d = 1 even about 0; d = 3 exact)."""
    if d == 1:
        return 0.5 * (np.exp(-((r - c * t) / s) ** 2) + np.exp(-((r + c * t) / s) ** 2))
    if d == 3:
        a, b = r - c * t, r + c * t
        return (a * np.exp(-(a / s) ** 2) + b * np.exp(-(b / s) ** 2)) / (2 * r)
    return hankel_gauss(r, t, s, c, "free")[0]


def duhamel_gauss_exact(d, r, t, s, c=VER_C):
    """int_0^t W(t - tau)[q] dtau for q = exp(-r^2/s^2) (multiply by Gamma H0)."""
    if d == 1:
        return s * math.sqrt(math.pi) / (4 * c) * (special.erf((r + c * t) / s)
                                                  - special.erf((r - c * t) / s))
    if d == 3:
        a, b = r - c * t, r + c * t
        return s * s / (4 * r * c) * (np.exp(-(a / s) ** 2) - np.exp(-(b / s) ** 2))
    return hankel_gauss(r, t, s, c, "duhamel")[0]


def _acoustic_run(d, n, p0_fun, h_fun, t_end, bc="rigid"):
    g = R.RadialGrid(n, LENGTH, d)
    h = None if h_fun is None else h_fun(g.rc)[None]
    P = _uniform_P(g, h_s=h, const=1.0)
    y0 = (np.zeros((1, n)), (p0_fun(g.rc) if p0_fun else np.zeros(n))[None],
          np.zeros((1, n + 1)))
    _, st, _ = run_chain(g, bc, P, y0, 2, t_end, 0.0, lambda y: y[1][:, :1].copy())
    return g, st[1][0]


# ------------------------------------------------------------ (c) free Gaussian
def check_free_gauss(d, n_list):
    L = LENGTH
    s = L / 16
    t1 = 0.5 * L / VER_C
    errs, ref_cache = [], {}
    for n in n_list:
        g, pN = _acoustic_run(d, n, lambda r: np.exp(-(r / s) ** 2), None, t1)
        ex = free_gauss_exact(d, g.rc, t1, s)
        ref_cache[n] = ex
        errs.append(_rel_norms(pN, ex, g.V)[1])
    rep = common.convergence_report("chain", f"acoustic_free_gaussian_d{d}", n_list, errs, 2.0)
    rep.update({"d": d, "s_m": s, "t_s": t1, "c_m_s": VER_C, "rho_kg_m3": VER_RHO,
                "reference": {1: "d'Alembert", 2: "Hankel transform, scipy quad",
                              3: "exact [(r-ct)p0(|r-ct|)+(r+ct)p0(r+ct)]/(2r), evaluated "
                                 "at cell centres (r > 0, so no limit is needed)"}[d],
                "outer_wall": "rigid at r = L; leading edge at c t + 4 s = 0.75 L, wall "
                              "value of the pulse exp(-64)",
                "norm": "relative L2, volume weighted"})
    if d == 2:
        rep["quadrature_check"] = _hankel_accuracy(s, t1, "free")
    return rep


def _hankel_accuracy(s, t1, kind):
    r = np.linspace(0.5, 127.5, 128) * LENGTH / 128
    v, e = hankel_gauss(r, t1, s, kind=kind)
    gl = hankel_gauss_gl(r, t1, s, kind=kind)
    scale = np.abs(v).max()
    out = {"max_abs_diff_vs_gauss_legendre_rel_to_max": float(np.abs(v - gl).max() / scale),
           "max_quad_error_estimate_rel_to_max": float(e.max() / scale)}
    if kind == "free":
        v0, _ = hankel_gauss(r, 0.0, s, kind="free")
        out["t0_identity_max_rel_error"] = float(np.abs(v0 - np.exp(-(r / s) ** 2)).max())
    else:
        # nested quadrature (time integral done numerically) at three radii
        rr = r[[0, 40, 80]]
        nested = []
        for ri in rr:
            val, _ = sint.quad(lambda tau: hankel_gauss(np.array([ri]), t1 - tau, s,
                                                        kind="free", rtol=1e-11)[0][0],
                               0.0, t1, epsabs=0.0, epsrel=1e-10, limit=200)
            nested.append(val)
        single = hankel_gauss(rr, t1, s, kind="duhamel")[0]
        out["nested_vs_single_max_rel_to_max"] = float(np.abs(np.array(nested) - single).max() / scale)
        out["nested_radii_m"] = rr.tolist()
    out["passed_1e-10"] = bool(max(v for k, v in out.items()
                                   if k.endswith("rel_to_max") or k.endswith("rel_error")) < 1e-10)
    return out


# --------------------------------------------------------- (d) constant heating
def check_heating(d, n_list):
    L = LENGTH
    s = L / 16
    H0 = 1.0e9                                    # W m^-3, scale only (linear problem)
    t1 = 0.5 * L / VER_C
    errs = []
    for n in n_list:
        g, pN = _acoustic_run(d, n, None, lambda r: H0 * np.exp(-(r / s) ** 2), t1)
        ex = VER_GAMMA * H0 * duhamel_gauss_exact(d, g.rc, t1, s)
        errs.append(_rel_norms(pN, ex, g.V)[1])
    rep = common.convergence_report("chain", f"acoustic_constant_heating_d{d}", n_list,
                                    errs, 2.0)
    rep.update({"d": d, "s_m": s, "t_s": t1, "H0_W_m3": H0, "gruneisen": VER_GAMMA,
                "reference": {1: "Duhamel integral, exact: (s sqrt(pi)/(4c)) [erf((r+ct)/s) "
                                 "- erf((r-ct)/s)]",
                              2: "Duhamel integral with the time integral done exactly: "
                                 "(1/c) int P0(k) J0(kr) sin(ckt) dk by scipy quad; checked "
                                 "against nested (time x wavenumber) quadrature",
                              3: "Duhamel integral, exact: (s^2/(4rc)) [exp(-(r-ct)^2/s^2) "
                                 "- exp(-(r+ct)^2/s^2)]"}[d],
                "norm": "relative L2, volume weighted"})
    if d == 2:
        rep["quadrature_check"] = _hankel_accuracy(s, t1, "duhamel")
    return rep


# ------------------------------------------------------------- (e) top-hat
def tophat_exact(d, r, t, a, c=VER_C):
    ct = c * t
    if d == 1:
        return 0.5 * ((np.abs(r - ct) < a).astype(float) + (np.abs(r + ct) < a).astype(float))
    if d == 3:
        A, B = r - ct, r + ct
        return (A * (np.abs(A) < a) + B * (np.abs(B) < a)) / (2 * r)
    return np.array([_poisson2d_tophat(ri, t, a, c) for ri in r])


def _theta(rho, r, a):
    """Angle of the circle of radius rho about a point at distance r inside the disc a."""
    if rho <= 0:
        return 2 * np.pi if r < a else 0.0
    if rho + r <= a:
        return 2 * np.pi
    if rho >= r + a or rho <= r - a:
        return 0.0
    x = (rho * rho + r * r - a * a) / (2 * rho * r)
    return 2.0 * math.acos(min(1.0, max(-1.0, x)))


def _U2d(r, t, a, c):
    """Poisson potential U = (1/(2 pi c)) int_{|y-x|<ct} p0(y) / sqrt(c^2t^2 - |y-x|^2) dy
    = (t / 2 pi) int_0^{pi/2} sin(phi) Theta(ct sin(phi)) dphi."""
    ct = c * t
    if ct <= 0:
        return 0.0
    brk = [math.asin(min(1.0, x / ct)) for x in (abs(r - a), r + a) if 0 < x < ct]
    f = lambda ph: math.sin(ph) * _theta(ct * math.sin(ph), r, a)
    v, _ = sint.quad(f, 0.0, math.pi / 2, points=brk or None, epsabs=1e-14, epsrel=1e-13,
                     limit=400)
    return t / (2 * math.pi) * v


def _poisson2d_tophat(r, t, a, c, rel=1e-4):
    """p = dU/dt by a fourth-order central difference (step rel * t)."""
    h = rel * t
    U = lambda tt: _U2d(r, tt, a, c)
    return (-U(t + 2 * h) + 8 * U(t + h) - 8 * U(t - h) + U(t - 2 * h)) / (12 * h)


def check_tophat(d, n_list, times_frac=(0.15, 0.45)):
    L = LENGTH
    a = L / 4
    rows = {}
    overlay = {}
    for tf in times_frac:
        t = tf * L / VER_C
        l1s, l2s = [], []
        for n in n_list:
            g, pN = _acoustic_run(d, n, lambda r: (r < a).astype(float), None, t)
            ex = tophat_exact(d, g.rc, t, a)
            l1, l2 = _rel_norms(pN, ex, g.V)
            l1s.append(l1); l2s.append(l2)
            if n == n_list[-1]:
                overlay[f"t={tf}L/c"] = {"t_s": t, "r_m": g.rc.tolist(),
                                         "numerical": pN.tolist(), "exact": ex.tolist()}
        o1 = [math.log(e1 / e2) / math.log(n2 / n1) for (n1, e1), (n2, e2)
              in zip(zip(n_list, l1s), zip(n_list[1:], l1s[1:]))]
        o2 = [math.log(e1 / e2) / math.log(n2 / n1) for (n1, e1), (n2, e2)
              in zip(zip(n_list, l2s), zip(n_list[1:], l2s[1:]))]
        rows[f"t={tf}L/c"] = {"t_s": t, "n": list(n_list), "L1": l1s, "L2": l2s,
                              "order_L1": o1, "order_L2": o2}
    ref = {1: "d'Alembert", 3: "exact N-wave formula",
           2: "Poisson integral: U by adaptive quadrature over the arc-angle function, "
              "p = dU/dt by fourth-order central difference (step 1e-4 t); no closed form "
              "exists in 2D"}[d]
    extra = {}
    if d == 2:     # check the Poisson reference at the centre against the closed form
        tt = 0.45 * L / VER_C
        num0 = _poisson2d_tophat(1e-9 * L, tt, a, VER_C)
        exact0 = 1 - VER_C * tt / math.sqrt((VER_C * tt) ** 2 - a * a)
        extra = {"centre_check_t=0.45L/c": {"poisson": num0, "closed_form": exact0,
                                            "abs_diff": abs(num0 - exact0)}}
    return {"d": d, "a_m": a, "reference": ref, "times": rows, "pass_fail": False,
            "norm": "relative L1 and L2, volume weighted; exact solution at cell centres",
            **extra}, overlay


def verify_all(n_list=(128, 256, 512)):
    res = {"system": "chain", "promised_order": 2.0, "n": list(n_list)}
    tests = {}
    for d in (1, 2, 3):
        tests[f"a_light_source_d{d}"] = check_light(d, n_list, "source")
    for d in (1, 2, 3):
        tests[f"a_light_surface_d{d}"] = check_light(d, n_list, "surface")
    for d in (1, 2, 3):
        tests[f"b_heat_d{d}"] = check_heat(d, n_list)
        tests[f"b_heat_heated_d{d}"] = check_heat(d, n_list, heated=True)
    for d in (1, 2, 3):
        tests[f"c_free_gaussian_d{d}"] = check_free_gauss(d, n_list)
    for d in (1, 2, 3):
        tests[f"d_constant_heating_d{d}"] = check_heating(d, n_list)
    tophat, overlay = {}, {}
    for d in (1, 2, 3):
        tophat[f"d{d}"], overlay[f"d{d}"] = check_tophat(d, n_list)
    res["pass_fail_tests"] = tests
    quad_ok = all(t.get("quadrature_check", {}).get("passed_1e-10", True) for t in tests.values())
    res["quadrature_1e-10_passed"] = quad_ok
    res["e_tophat_report_only"] = tophat
    res["e_tophat_overlay"] = overlay
    res["passed"] = bool(all(t["passed"] for t in tests.values()) and quad_ok)
    # convergence-harness fields for the d = 1 headline (acoustic constant heating)
    h = tests["d_constant_heating_d1"]
    res.update({"check": "chain links (a)-(d), d = 1, 2, 3; headline row = (d) d = 1",
                "error": h["error"], "observed_order": h["observed_order"]})
    return res
