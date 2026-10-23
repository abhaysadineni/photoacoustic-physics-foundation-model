"""Two coupled training systems, each one link of the photoacoustic chain (round 2,
21 Oct 2026). Physical units. d = 1, and since round 3 also d = 2, 3 (radial profiles,
exactly as the chain: params["dim"], sample_params(rng, regime, dim=None), allowed
boundary conditions chain.ALLOWED_BCS[d], i.e. pressure_release and rigid for d = 2, 3).

Photothermal ("photothermal", channels 0 and 1): light heats tissue.
    phi(x, t) = E(t) phi_s(x),   d/dx (D_s d phi_s/dx) - mu_a phi_s = 0      (channel 0)
    dT/dt = alpha d2T/dx2 + mu_a phi / (rho C_p)                              (channel 1)
    channel 2 identically zero (no acoustic link).
Thermoacoustic ("thermoacoustic", channels 1 and 2): prescribed heating launches sound.
    H(x, t) = E(t) h(x)
    dT/dt = alpha d2T/dx2 + H / (rho C_p)                                     (channel 1)
    dp/dt = -rho c^2 du/dx + Gamma H,  du/dt = -(1/rho) dp/dx                 (channel 2)
    channel 0 identically zero (no optical link).

Everything is inherited from pfm/solvers/chain.py, which is imported and NOT modified:
optics (light.sample_tissue with the mu_a / mu_s' <= 0.3 cap, wavelengths 750, 800,
850 nm train and 700 nm ood, 0 to 3 absorbers), illumination (fixed-fluence boundary,
sin^2 envelope of duration 0.25 to 0.5 acoustic transit times, 0 to 8 pre-frames), the
thermal and acoustic models, every parameter range and its JSON source
(chain.PARAM_RANGES, chain.PARAM_SOURCES), the domain L = 12.8 mm, the frame interval
chain.DT_FRAME = 64.06 ns and n_internal = 512. The code paths reused are
chain.PhotoacousticChain._assemble (mu_a(x), phi_s(x), h = mu_a phi_s and all
coefficient arrays), chain.envelope, chain.thermal_bc and chain.run_chain.

Heating profile of Thermoacoustic: h(x) is built exactly as the chain builds its heating,
by drawing a chain tissue sample and computing mu_a(x) phi_s(x) with chain._assemble;
the fluence is then discarded (channel 0 is zero). The statistics and spatial structure
of the heating are therefore those of the chain by construction.

Time stepping. Thermoacoustic integrates (T, p, u) with chain.run_chain, so it uses the
chain's substep rule (acoustic CFL 0.5 and the thermal limit). Photothermal integrates T
alone but with the same substep count per frame as the chain (the minimum of the acoustic
and thermal limits computed from the same sampled c, rho, alpha), so that for equal
parameters its channels 0 and 1 equal the chain's to round-off; the T equation does not
depend on p or u, so dropping them changes no arithmetic on T.

Boundary conditions (d = 1): periodic, pressure_release, rigid, with the chain's meaning.
The boundary condition also selects the illumination (periodic: light enters through
both faces; walls: one face lit at x = 0, phi_s(L) = 0) and the thermal wall (periodic or
zero flux). For Photothermal the acoustic wall type has no effect, so pressure_release
and rigid give identical trajectories for identical parameters (the held-out rigid
boundary is therefore not a distinct test for Photothermal).

Parameters. sample_params(rng, "train") is chain.PhotoacousticChain(1).sample_params
itself, so equal seeds give equal draws. "ood" follows the chain's rule (one ood kind per
trajectory, chosen uniformly) with the kinds that do not apply removed:
    Photothermal:   {wavelength, blood_fraction}   (gruneisen_fat removed: no acoustics)
    Thermoacoustic: {blood_fraction, gruneisen_fat} (wavelength removed: the optical link
                    is absent; the coordinator named this case)
The remaining kinds are drawn uniformly, so each has probability 1/2 instead of 1/3.
Unused parameters (c, rho, Gruneisen for Photothermal) are still drawn, which keeps the
draw sequence equal to the chain's and fixes Photothermal's substep count.

Verification: see verify_photothermal() and verify_thermoacoustic(), and
tests/verify_coupled.py.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate as sint

from pfm import common
from pfm.solvers import _tissue as J
from pfm.solvers import chain as CH
from pfm.solvers import light as light_mod
from pfm.solvers import radial as R

LENGTH = CH.LENGTH
DT_FRAME = CH.DT_FRAME
ALLOWED_BCS = CH.ALLOWED_BCS[1]
PARAM_RANGES = CH.PARAM_RANGES
PARAM_SOURCES = dict(CH.PARAM_SOURCES)
PARAM_SOURCES["all parameters"] = ("inherited unchanged from chain.PARAM_SOURCES "
                                   "(pfm/solvers/chain.py)")
OOD_KINDS = {"photothermal": ("wavelength", "blood_fraction"),
             "thermoacoustic": ("blood_fraction", "gruneisen_fat")}

_CHAIN = CH.PhotoacousticChain(dim=1)
# rho C_p for the verification checks: rho_soft x tissue_cp_wang, as chain.check_heat
VER_RCP = CH.VER_RHO * float(J.get("tissue_cp_wang"))


def sample_coupled(rng, regime, kinds, dim=1):
    """Chain parameter draw. train: the chain's own sample_params. ood: the chain's
    sampling code with the ood kind drawn uniformly from `kinds`. dim as in the chain."""
    if regime == "train":
        return _CHAIN.sample_params(rng, "train", dim=dim)
    if regime != "ood":
        raise ValueError(regime)
    tr = CH.PARAM_RANGES["train"]
    u = lambda ab: float(rng.uniform(*ab))
    kind = kinds[int(rng.integers(len(kinds)))]
    opt_regime = "ood" if kind in ("wavelength", "blood_fraction") else "train"
    p = light_mod.sample_tissue(rng, opt_regime, LENGTH,
                                ood_kind=kind if opt_regime == "ood" else None)
    p["regime"] = regime
    p["ood_kind"] = kind
    p["dim"] = int(dim)
    p["c"] = u(tr["c_m_s"]); p["rho"] = u(tr["rho_kg_m3"])
    p["alpha"] = float(np.exp(rng.uniform(*np.log(tr["alpha_m2_s"]))))
    p["cp"] = u(tr["cp_J_kgK"])
    p["gruneisen"] = (u(CH.GAMMA_OOD_FAT) if kind == "gruneisen_fat"
                      else u(tr["gruneisen"]))
    p["phi0"] = u(tr["phi0_W_m2"])
    p["T_p"] = u(tr["pulse_duration_s"])
    p["pre_frames"] = int(rng.integers(tr["pre_frames"][0], tr["pre_frames"][1] + 1))
    return p


def chain_substeps(grid, bc, P, dt_frame):
    """The chain's substep rule (chain.run_chain), reproduced for the T-only integrator."""
    B = P["alpha_f"].shape[0]
    cmax = np.sqrt(P["K"].max(axis=1) * P["irf"].max(axis=1))
    with np.errstate(divide="ignore"):
        dt_ac = R.CFL_ACOUSTIC * grid.dr / cmax
    dt_th = R.diffusion_dt(grid, P["alpha_f"], CH.thermal_bc(bc), np.zeros((B, 1)), B)
    return R.n_substeps(dt_frame, np.minimum(dt_ac, dt_th))


def run_thermal(grid, bc, P, T0, n_frames, dt_frame, t_start, record):
    """Photothermal integrator: T only, heating E(t) h_s(x) (chain.envelope), substeps as
    the chain."""
    tbc = CH.thermal_bc(bc)
    nsub = chain_substeps(grid, bc, P, dt_frame)

    def rhs(t, y, Q):
        H = CH.envelope(t, Q) * Q["h_s"]
        return (R.diffusion(y[0], grid, Q["alpha_f"], tbc) + H * Q["inv_rcp"],)

    out, st = R.integrate(rhs, (T0,), P, dt_frame, nsub, n_frames, record,
                          t_start=t_start, return_state=True)
    return out, st, nsub


class _CoupledBase(common.Solver):
    order = 2.0

    def __init__(self, dim: int = 1):
        self.dim = int(dim)

    def sample_params(self, rng, regime="train", dim=None):
        dim = self.dim if dim is None else int(dim)
        return sample_coupled(rng, regime, OOD_KINDS[self.name], dim)

    def simulate_batch(self, params_list, bc, n_frames, rng=None, n_internal=512):
        return R.split_by_dim(params_list, self.dim,
                              lambda ps, d: self._simulate(ps, bc, n_frames, n_internal, d))

    def _common(self, params_list, bc, n_internal, d=1):
        if bc not in CH.ALLOWED_BCS[d]:
            raise ValueError(f"{self.name} d = {d}: bc {bc!r} not in {CH.ALLOWED_BCS[d]}")
        grid = R.RadialGrid(n_internal, LENGTH, d)
        P, phi_s, mu, musp = _CHAIN._assemble(params_list, grid, bc)
        mpre = max(p["pre_frames"] for p in params_list)
        return grid, P, phi_s, mu, musp, mpre

    @staticmethod
    def _envelope_frames(P, b, n_frames):
        tf = np.arange(n_frames) * DT_FRAME
        return CH.envelope(tf, {"t_s": np.repeat(P["t_s"][b:b + 1], n_frames, 0),
                                "T_p": np.repeat(P["T_p"][b:b + 1], n_frames, 0),
                                "const": np.zeros((n_frames, 1))})[:, 0]

    def _meta(self, p, P, mu, musp, b, nsub, n_internal, bc):
        q = dict(p)
        Hmax = float(P["h_s"][b].max())
        q.update({"dt": float(DT_FRAME / nsub[b]), "n_internal": n_internal, "bc": bc,
                  "mua_background_m-1": float(light_mod.mua_background(p)),
                  "mua_max_m-1": float(mu[b].max()), "musp_m-1": float(musp[b, 0]),
                  "mua_over_musp_max": float(mu[b].max() / musp[b, 0]),
                  "mua_over_musp_cap": light_mod.MUA_MUSP_CAP, "H_max_W_m3": Hmax,
                  "T_bound_K": float(Hmax * P["inv_rcp"][b, 0] * 0.5 * p["T_p"]),
                  "p_bound_1d_Pa": float(p["gruneisen"] * Hmax * 0.5 * p["T_p"])})
        return q

    def simulate(self, params, bc, n_frames, rng=None, n_internal=512):
        return self.simulate_batch([params], bc, n_frames, rng, n_internal)[0]


class Photothermal(_CoupledBase):
    name = "photothermal"
    channels = (common.CH_LIGHT, common.CH_HEAT)

    def _simulate(self, params_list, bc, n_frames, n_internal, d):
        grid, P, phi_s, mu, musp, mpre = self._common(params_list, bc, n_internal, d)
        B, n = len(params_list), grid.n
        rec = lambda y: grid.downsample(y[0], common.N_OUT)
        out, _, nsub = run_thermal(grid, bc, P, np.zeros((B, n)), n_frames + mpre,
                                   DT_FRAME, -mpre * DT_FRAME, rec)
        out = out[:, mpre:]
        phi_out = grid.downsample(phi_s, common.N_OUT)
        trajs = []
        for b, p in enumerate(params_list):
            E = self._envelope_frames(P, b, n_frames)
            f = np.zeros((n_frames, common.N_CH, common.N_OUT), np.float32)
            f[:, common.CH_LIGHT] = E[:, None] * phi_out[b]
            f[:, common.CH_HEAT] = out[b]
            trajs.append(common.Trajectory(f, self.name,
                                           self._meta(p, P, mu, musp, b, nsub, n_internal, bc),
                                           DT_FRAME, LENGTH, bc, self.channels))
        return trajs

    def verify(self, n_list=(128, 256, 512)) -> dict:
        return verify_photothermal(n_list)


class Thermoacoustic(_CoupledBase):
    name = "thermoacoustic"
    channels = (common.CH_HEAT, common.CH_PRESSURE)

    def _simulate(self, params_list, bc, n_frames, n_internal, d):
        grid, P, phi_s, mu, musp, mpre = self._common(params_list, bc, n_internal, d)
        del phi_s                    # the fluence is discarded; only h = mu_a phi_s is kept
        B, n = len(params_list), grid.n
        y0 = (np.zeros((B, n)), np.zeros((B, n)), np.zeros((B, n + 1)))
        rec = lambda y: np.stack([grid.downsample(y[0], common.N_OUT),
                                  grid.downsample(y[1], common.N_OUT)], axis=1)
        out, _, nsub = CH.run_chain(grid, bc, P, y0, n_frames + mpre, DT_FRAME,
                                    -mpre * DT_FRAME, rec)
        out = out[:, mpre:]
        trajs = []
        for b, p in enumerate(params_list):
            f = np.zeros((n_frames, common.N_CH, common.N_OUT), np.float32)
            f[:, common.CH_HEAT] = out[b, :, 0]
            f[:, common.CH_PRESSURE] = out[b, :, 1]
            trajs.append(common.Trajectory(f, self.name,
                                           self._meta(p, P, mu, musp, b, nsub, n_internal, bc),
                                           DT_FRAME, LENGTH, bc, self.channels))
        return trajs

    def verify(self, n_list=(128, 256, 512)) -> dict:
        return verify_thermoacoustic(n_list)


# =============================================================================
# Verification
# =============================================================================
def check_uniform_heating(n_list=(128, 256, 512), bc="neumann", d=1):
    """Photothermal integrator (run_thermal), production frame interval and substep rule,
    64 frames: uniform constant heating H0 switched on at t = 0, T0 = 0. Exact:
    T(t) = H0 t / (rho C_p) everywhere (diffusion of a constant is exactly zero on the
    grid, and RK4 integrates a solution linear in t exactly), so the error must be at
    round-off level at every resolution; no order is defined."""
    errs = []
    for n in n_list:
        g = R.RadialGrid(n, LENGTH, d)
        P = CH._uniform_P(g, const=1.0)           # chain verification coefficients
        rcp = VER_RCP
        P["inv_rcp"][:] = 1.0 / rcp
        H0 = 1.0e9                                 # W m^-3, scale only
        P["h_s"] = np.full((1, n), H0)
        rec = lambda y: y[0].copy()
        out, _, _ = run_thermal(g, "rigid", P, np.zeros((1, n)), 64, DT_FRAME, 0.0, rec)
        t = np.arange(64) * DT_FRAME
        ex = (H0 / rcp * t)[:, None] * np.ones(n)
        errs.append(float(np.abs(out[0] - ex).max() / np.abs(ex).max()))
    passed = max(errs) < 1e-12
    return {"system": "photothermal", "check": "uniform_constant_heating_exact"
            + ("" if d == 1 else f"_d{d}"), **({} if d == 1 else {"d": d}),
            "n": list(n_list), "error": errs, "observed_order": None,
            "promised_order": None, "passed": bool(passed),
            "pass_rule": "max relative error over all 64 frames < 1e-12 (exact in the "
                         "scheme; round-off only)",
            "rho_cp_J_m3K": rcp, "H0_W_m3": 1.0e9, "dt_frame_s": DT_FRAME,
            "code_path": "coupled.run_thermal (Photothermal's integrator), chain substep rule",
            "norm": "max abs error / max |T| over all frames and cells"}


def check_gaussian_heater(n_list=(128, 256, 512), d=1):
    """Photothermal integrator on the thermal time scale: constant Gaussian heater
    H0 exp(-x^2/s0^2) on from t = 0 at the symmetry wall x = 0 (zero-flux thermal wall),
    against T = (H0/(rho C_p)) int_0^t (s0^2/(s0^2 + 4 alpha tau))^(1/2)
    exp(-x^2/(s0^2 + 4 alpha tau)) dtau (scipy quad_vec, rel. tol 1e-12). Acoustics is
    switched off (K = 0, 1/rho = 0), which makes the chain's substep rule fall back to the
    thermal limit, exactly as chain.check_heat does."""
    L = LENGTH
    s0 = L / 20.0
    alpha = CH.VER_ALPHA
    t1 = 3.0 * s0 ** 2 / (4.0 * alpha)
    errs = []
    for n in n_list:
        g = R.RadialGrid(n, L, d)
        P = CH._uniform_P(g)
        P["K"][:] = 0.0; P["irf"][:] = 0.0; P["gamma"][:] = 0.0
        rcp = VER_RCP
        P["inv_rcp"][:] = 1.0 / rcp
        H0 = rcp / t1
        P["h_s"] = (H0 * np.exp(-g.rc ** 2 / s0 ** 2))[None]
        _, st, _ = run_thermal(g, "rigid", P, np.zeros((1, n)), 2, t1, 0.0,
                               lambda y: y[0][:, :1].copy())
        G = lambda tau: ((s0 ** 2 / (s0 ** 2 + 4 * alpha * tau)) ** (d / 2)
                         * np.exp(-g.rc ** 2 / (s0 ** 2 + 4 * alpha * tau)))
        ex = H0 / rcp * sint.quad_vec(G, 0.0, t1, epsabs=0.0, epsrel=1e-12)[0]
        errs.append(CH._rel_norms(st[0][0], ex, g.V)[1])
    rep = common.convergence_report("photothermal", f"gaussian_heater_d{d}", n_list, errs, 2.0)
    if d != 1:
        rep["d"] = d
    rep.update({"s0_m": s0, "t1_s": t1, "alpha_m2_s": alpha,
                "code_path": "coupled.run_thermal with acoustics off (thermal substeps)",
                "norm": "relative L2"})
    return rep


def verify_photothermal(n_list=(128, 256, 512)) -> dict:
    cases = {"uniform_heating": check_uniform_heating(n_list),
             "gaussian_heater": check_gaussian_heater(n_list),
             # steady fluence: chain.check_light calls chain.steady_fluence, which is the
             # function Photothermal uses (through chain._assemble)
             "light_steady_source_d1": CH.check_light(1, n_list, "source"),
             "light_steady_surface_d1": CH.check_light(1, n_list, "surface")}
    main = dict(cases["gaussian_heater"])
    main["system"] = "photothermal"
    main["check"] = "uniform heating (exact), Gaussian heater (order 2), steady fluence"
    main["cases"] = cases
    main["passed"] = all(c["passed"] for c in cases.values())
    return main


def verify_thermoacoustic(n_list=(128, 256, 512)) -> dict:
    """Constant Gaussian heating switched on at t = 0 against the exact Duhamel solution
    (chain check (d), d = 1): Thermoacoustic integrates with chain.run_chain, which is the
    code path chain.check_heating exercises; the Gaussian-heater temperature check
    (chain check (b), heated) runs on the same integrator."""
    cases = {"constant_heating_duhamel_d1": CH.check_heating(1, n_list),
             "gaussian_heater_temperature_d1": CH.check_heat(1, n_list, heated=True)}
    main = dict(cases["constant_heating_duhamel_d1"])
    main["system"] = "thermoacoustic"
    main["check"] = "constant Gaussian heating, Duhamel (order 2); Gaussian heater T"
    main["cases"] = cases
    main["passed"] = all(c["passed"] for c in cases.values())
    return main


def consistency_check(n_traj=8, seed=11, regimes=("train", "ood"), bcs=None, dim=1):
    """Same parameters and seed: Photothermal ch 0, 1 and Thermoacoustic ch 1, 2 against the
    chain's channels. Thermoacoustic's heating is the chain's mu_a phi_s by construction.
    ood draws come from the chain's sampler (all three kinds), so the comparison also
    covers kinds a class would not draw itself."""
    rows = []
    bcs = CH.ALLOWED_BCS[dim] if bcs is None else bcs
    chain_d = _CHAIN if dim == 1 else CH.PhotoacousticChain(dim=dim)
    pt, ta = Photothermal(dim), Thermoacoustic(dim)
    for i, reg in enumerate(regimes):
        for j, bc in enumerate(bcs):
            rng = np.random.default_rng(seed + 10 * i + j)
            params = [chain_d.sample_params(rng, reg) for _ in range(n_traj)]
            fc = np.stack([t.fields for t in chain_d.simulate_batch(
                [dict(p) for p in params], bc, 64)])
            fp = np.stack([t.fields for t in pt.simulate_batch([dict(p) for p in params], bc, 64)])
            fa = np.stack([t.fields for t in ta.simulate_batch([dict(p) for p in params], bc, 64)])
            row = {"regime": reg, "bc": bc, "n_traj": n_traj, **({} if dim == 1 else {"dim": dim})}
            for lab, f, chans in (("photothermal", fp, (0, 1)), ("thermoacoustic", fa, (1, 2))):
                for c in chans:
                    d = np.abs(f[:, :, c].astype(float) - fc[:, :, c].astype(float))
                    row[f"{lab}_ch{c}_max_abs_diff"] = float(d.max())
                    row[f"{lab}_ch{c}_max_rel_diff"] = float(d.max() / np.abs(fc[:, :, c]).max())
                unused = [c for c in range(3) if c not in chans]
                row[f"{lab}_unused_ch{unused[0]}_max_abs"] = float(np.abs(f[:, :, unused[0]]).max())
            rows.append(row)
    worst = max(v for r in rows for k, v in r.items() if k.endswith("_max_rel_diff"))
    unused_max = max(v for r in rows for k, v in r.items() if "_unused_" in k)
    return {"rows": rows, "max_rel_diff": worst, "unused_channel_max_abs": unused_max,
            "passed": bool(worst <= 1e-6 and unused_max == 0.0),
            "pass_rule": "max |difference| / max |chain channel| <= 1e-6 (float32 output "
                         "resolution) and unused channel exactly zero"}
