"""Heat conduction with a source (system 2, channel 1), in physical units.

    dT/dt = alpha d2T/dx2 + H(x, t) / (rho C_p),

T is the temperature rise above baseline in kelvin. alpha, rho and C_p are uniform per
trajectory.

Parameter ranges (all from params/tissue_params.json, see PARAM_SOURCES):
    alpha = k / (rho C_p) for each sourced material:
        fat     k, c, rho from muscle_fat_thermal.fat            -> 9.41e-8 m^2/s
        muscle  k, c, rho from muscle_fat_thermal.muscle         -> 1.364e-7 m^2/s
        water   water37_thermal.alpha_th_m2_s (37 C, IAPWS-95)   -> 1.504e-7 m^2/s
    train: alpha in [fat, muscle]; rho C_p in [fat, muscle] (2.231e6 to 3.593e6 J/(m^3 K)).
        The generic tissue value soft_tissue_alpha_th = 1.0e-7 m^2/s lies inside.
    ood:   alpha in [muscle, water 37 C] and rho C_p in [muscle, water 37 C]
        (3.593e6 to 4.151e6). This shift is narrow (alpha up to 10 per cent beyond the
        training maximum) because no sourced tissue value lies farther out; the ood test
        for heat therefore probes only a mild extrapolation, and the report must say so.

Domain and frame interval: L = 12.8 mm (modelling choice, the same physical window as
light.py). dt_frame is chosen so that a hot spot in the reference tissue
(alpha_ref = soft_tissue_alpha_th) spreads to a 1/e radius sqrt(4 alpha_ref T) = L / 4 over
T = 63 frame intervals:  dt_frame = (L/4)^2 / (4 * 63 * alpha_ref) = 0.406 s, a 25.6 s
record.

Initial condition: 1 to 3 Gaussian hot spots (amplitude 0.2 to 2 K, 1/e radius 0.2 to
1 mm). Source: 0 to 2 Gaussian heaters (1/e radius 0.2 to 1 mm), amplitude set so that
H / (rho C_p) = 0.02 to 0.2 K per frame interval, each either on throughout or switched on
and off with smooth tanh edges (width one frame) at random times inside the record.
All amplitudes are modelling choices. Dirichlet walls: initial data and sources multiplied
by a C2 taper of width 1 mm.

Boundary conditions: periodic, dirichlet (T = baseline), neumann (insulated).
Radial geometry (round 3): params["dim"] = 2 or 3, HeatSolver(dim), sample_params(...,
dim=None); outer wall dirichlet or neumann only. Hot spots and heaters are even shells
exp(-((r-R)/w)^2) + exp(-((r+R)/w)^2), peak-normalised to the sampled amplitude, smooth at
r = 0, at R uniform in [0, 0.8] L (the d = 1 centre draw mapped linearly); dirichlet
taper at the outer wall only. Parameters and DT_FRAME as in d = 1.
Numerics: radial.py finite volumes, RK4, dt = 2.5 / Gershgorin bound. Order 2.

Exact check (d = 1, 2, 3, H = 0): T = T0 (s0^2 / (s0^2 + 4 alpha t))^(d/2)
exp(-r^2 / (s0^2 + 4 alpha t)), from t = 0 (s0 = L/20) to s(t)^2 = 4 s0^2.
"""
from __future__ import annotations

import math

import numpy as np

from pfm import common
from pfm.solvers import _tissue as J
from pfm.solvers import radial as R


def _material(name: str):
    k = float(J.get("muscle_fat_thermal", name, "k_W_mK"))
    c = float(J.get("muscle_fat_thermal", name, "c_J_kgK"))
    rho = float(J.get("muscle_fat_thermal", name, "rho_kg_m3"))
    return k / (rho * c), rho * c


ALPHA_FAT, RHOCP_FAT = _material("fat")
ALPHA_MUSCLE, RHOCP_MUSCLE = _material("muscle")
ALPHA_WATER37 = float(J.get("water37_thermal", "alpha_th_m2_s"))
RHOCP_WATER37 = (float(J.get("water37_thermal", "rho_kg_m3"))
                 * float(J.get("water37_thermal", "cp_J_kgK")))
ALPHA_REF = float(J.get("soft_tissue_alpha_th"))

LENGTH = 12.8e-3                                   # m, modelling choice
DT_FRAME = (LENGTH / 4.0) ** 2 / (4.0 * 63 * ALPHA_REF)
WALL_TAPER = 1.0e-3

PARAM_RANGES = {
    "train": {"alpha_m2_s": [ALPHA_FAT, ALPHA_MUSCLE],
              "rho_cp_J_m3K": [RHOCP_FAT, RHOCP_MUSCLE],
              "n_hot_spots": [1, 3], "hot_spot_amp_K": [0.2, 2.0],
              "width_m": [0.2e-3, 1.0e-3], "n_heaters": [0, 2],
              "heating_K_per_frame": [0.02, 0.2]},
    "ood": {"alpha_m2_s": [ALPHA_MUSCLE, ALPHA_WATER37],
            "rho_cp_J_m3K": [RHOCP_MUSCLE, RHOCP_WATER37]},
}
PARAM_SOURCES = {
    "alpha_m2_s (train low)": "muscle_fat_thermal.fat: k_W_mK / (rho_kg_m3 * c_J_kgK)",
    "alpha_m2_s (train high, ood low)": "muscle_fat_thermal.muscle: k / (rho c)",
    "alpha_m2_s (ood high)": "water37_thermal.alpha_th_m2_s",
    "rho_cp (train)": "muscle_fat_thermal.fat and .muscle: rho_kg_m3 * c_J_kgK",
    "rho_cp (ood high)": "water37_thermal: rho_kg_m3 * cp_J_kgK",
    "alpha_ref for dt_frame": "soft_tissue_alpha_th.value",
    "LENGTH": "modelling choice (12.8 mm)",
    "hot spots, heaters, amplitudes, widths, switching": "modelling choice",
}


class HeatSolver(common.Solver):
    name = "heat"
    channels = (common.CH_HEAT,)
    order = 2.0

    def __init__(self, dim: int = 1):
        self.dim = int(dim)

    def sample_params(self, rng, regime="train", dim=None):
        dim = self.dim if dim is None else int(dim)
        tr = PARAM_RANGES["train"]
        rg = PARAM_RANGES[regime]
        u = lambda ab: float(rng.uniform(*ab))
        p = {"regime": regime,
             "alpha": float(np.exp(rng.uniform(*np.log(rg["alpha_m2_s"])))),
             "rho_cp": u(rg["rho_cp_J_m3K"])}
        p["hot_spots"] = [{"centre": float(rng.uniform(0.2, 0.8) * LENGTH),
                           "width": u(tr["width_m"]), "amp": u(tr["hot_spot_amp_K"])}
                          for _ in range(int(rng.integers(1, 4)))]
        heaters = []
        for _ in range(int(rng.integers(0, 3))):
            h = {"centre": float(rng.uniform(0.2, 0.8) * LENGTH), "width": u(tr["width_m"]),
                 "K_per_frame": u(tr["heating_K_per_frame"])}
            if rng.random() < 0.5:
                t_on = float(rng.uniform(0, 40) * DT_FRAME)
                h["t_on"], h["t_off"] = t_on, t_on + float(rng.uniform(5, 20) * DT_FRAME)
            heaters.append(h)
        p["heaters"] = heaters
        if dim > 1:
            # radial geometry: hot spots and heaters become even shells (smooth at r = 0);
            # the d = 1 centre draw (uniform in [0.2, 0.8] L) maps to radius in [0, 0.8] L
            for h in p["hot_spots"] + p["heaters"]:
                h["radius"] = R.radius_from_frac((h.pop("centre") / LENGTH - 0.2) / 0.6,
                                                 LENGTH)
        p["dim"] = dim
        return p

    def simulate_batch(self, params_list, bc, n_frames, rng=None, n_internal=512):
        return R.split_by_dim(params_list, self.dim,
                              lambda ps, d: self._simulate(ps, bc, n_frames, n_internal, d))

    def _simulate(self, params_list, bc, n_frames, n_internal, d):
        if d > 1:
            return self._simulate_radial(params_list, bc, n_frames, n_internal, d)
        grid = R.RadialGrid(n_internal, LENGTH, 1)
        grid.check_bc(bc, "scalar")
        x, per, B = grid.rc, bc == "periodic", len(params_list)
        taper = R.smoothstep_taper(x, LENGTH, WALL_TAPER) if bc == "dirichlet" else 1.0
        T0 = np.zeros((B, grid.n))
        alpha = np.array([p["alpha"] for p in params_list])[:, None]
        Hmax = max(1, max(len(p["heaters"]) for p in params_list))
        shape = np.zeros((B, Hmax, grid.n)); ton = np.full((B, Hmax), -1e30)
        toff = np.full((B, Hmax), 1e30)
        for b, p in enumerate(params_list):
            for h in p["hot_spots"]:
                T0[b] += h["amp"] * R.gaussian(x, h["centre"], h["width"], LENGTH, per)
            for j, h in enumerate(p["heaters"]):
                # H / (rho C_p) in K/s; H itself = rho_cp * this (recorded in params)
                shape[b, j] = (h["K_per_frame"] / DT_FRAME
                               * R.gaussian(x, h["centre"], h["width"], LENGTH, per))
                if "t_on" in h:
                    ton[b, j], toff[b, j] = h["t_on"], h["t_off"]
        T0 *= taper; shape *= taper
        kf = np.broadcast_to(alpha, (B, grid.n + 1)).copy()
        dts = R.diffusion_dt(grid, kf, bc, np.zeros((B, 1)), B)
        nsub = R.n_substeps(DT_FRAME, dts)
        P = {"kf": kf, "shape": shape, "ton": ton, "toff": toff}

        def rhs(t, y, Q):
            tt = t[:, None]
            w = 0.25 * (1 + np.tanh((tt - Q["ton"]) / DT_FRAME)) * \
                (1 + np.tanh((Q["toff"] - tt) / DT_FRAME))
            return (R.diffusion(y[0], grid, Q["kf"], bc)
                    + np.einsum("bh,bhn->bn", w, Q["shape"]),)

        out = R.integrate(rhs, (T0,), P, DT_FRAME, nsub, n_frames,
                          lambda y: grid.downsample(y[0], common.N_OUT))
        trajs = []
        for b, p in enumerate(params_list):
            f = np.zeros((n_frames, common.N_CH, common.N_OUT), np.float32)
            f[:, common.CH_HEAT] = out[b]
            q = dict(p); q.update({"dt": float(DT_FRAME / nsub[b]), "n_internal": n_internal,
                                   "H_peak_W_m3": [h["K_per_frame"] / DT_FRAME * p["rho_cp"]
                                                   for h in p["heaters"]]})
            trajs.append(common.Trajectory(f, self.name, q, DT_FRAME, LENGTH, bc,
                                           self.channels))
        return trajs

    def _simulate_radial(self, params_list, bc, n_frames, n_internal, d):
        """d = 2, 3: same equation, parameters, heaters and frame interval as d = 1; hot
        spots and heaters are even shells amp * even_shell(r, R, w) / peak (peak = amp);
        dirichlet: initial data and heaters tapered at the outer wall only."""
        grid = R.RadialGrid(n_internal, LENGTH, d)
        grid.check_bc(bc, "scalar")
        x, B = grid.rc, len(params_list)
        taper = R.outer_taper(x, LENGTH, WALL_TAPER) if bc == "dirichlet" else 1.0
        shell = lambda h: (R.even_shell(x, h["radius"], h["width"])
                           / R.shell_peak(h["radius"], h["width"], LENGTH))
        T0 = np.zeros((B, grid.n))
        alpha = np.array([p["alpha"] for p in params_list])[:, None]
        Hmax = max(1, max(len(p["heaters"]) for p in params_list))
        shape = np.zeros((B, Hmax, grid.n)); ton = np.full((B, Hmax), -1e30)
        toff = np.full((B, Hmax), 1e30)
        for b, p in enumerate(params_list):
            for h in p["hot_spots"]:
                T0[b] += h["amp"] * shell(h)
            for j, h in enumerate(p["heaters"]):
                shape[b, j] = h["K_per_frame"] / DT_FRAME * shell(h)
                if "t_on" in h:
                    ton[b, j], toff[b, j] = h["t_on"], h["t_off"]
        T0 *= taper; shape *= taper
        kf = np.broadcast_to(alpha, (B, grid.n + 1)).copy()
        dts = R.diffusion_dt(grid, kf, bc, np.zeros((B, 1)), B)
        nsub = R.n_substeps(DT_FRAME, dts)
        P = {"kf": kf, "shape": shape, "ton": ton, "toff": toff}

        def rhs(t, y, Q):
            tt = t[:, None]
            w = 0.25 * (1 + np.tanh((tt - Q["ton"]) / DT_FRAME)) * \
                (1 + np.tanh((Q["toff"] - tt) / DT_FRAME))
            return (R.diffusion(y[0], grid, Q["kf"], bc)
                    + np.einsum("bh,bhn->bn", w, Q["shape"]),)

        out = R.integrate(rhs, (T0,), P, DT_FRAME, nsub, n_frames,
                          lambda y: grid.downsample(y[0], common.N_OUT))
        trajs = []
        for b, p in enumerate(params_list):
            f = np.zeros((n_frames, common.N_CH, common.N_OUT), np.float32)
            f[:, common.CH_HEAT] = out[b]
            q = dict(p); q.update({"dt": float(DT_FRAME / nsub[b]), "n_internal": n_internal,
                                   "H_peak_W_m3": [h["K_per_frame"] / DT_FRAME * p["rho_cp"]
                                                   for h in p["heaters"]]})
            trajs.append(common.Trajectory(f, self.name, q, DT_FRAME, LENGTH, bc,
                                           self.channels))
        return trajs

    def simulate(self, params, bc, n_frames, rng=None, n_internal=512):
        return self.simulate_batch([params], bc, n_frames, rng, n_internal)[0]

    # ---------------------------------------------------------------- verification
    @staticmethod
    def gaussian_check(d, n_list=(128, 256, 512), alpha=ALPHA_REF, L=LENGTH, name="heat"):
        s0 = L / 20.0
        t1 = 3.0 * s0 ** 2 / (4.0 * alpha)        # s(t1)^2 = 4 s0^2
        x0 = L / 2 if d == 1 else 0.0

        def exact(r, t):
            s2 = s0 ** 2 + 4 * alpha * t
            return (s0 ** 2 / s2) ** (d / 2) * np.exp(-r ** 2 / s2)
        errs = []
        for n in n_list:
            g = R.RadialGrid(n, L, d)
            r = np.abs(g.rc - x0)
            kf = np.full((1, n + 1), alpha)
            ns = R.n_substeps(t1, R.diffusion_dt(g, kf, "neumann", np.zeros((1, 1)), 1))
            rhs = lambda t, y, Q: (R.diffusion(y[0], g, kf, "neumann"),)
            out = R.integrate(rhs, (exact(r, 0.0)[None],), {}, t1, ns, 2,
                              lambda y: y[0].copy())
            num, ex = out[0, -1], exact(r, t1)
            errs.append(float(np.sqrt(np.sum(g.V * (num - ex) ** 2) / np.sum(g.V * ex ** 2))))
        rep = common.convergence_report(name, f"gaussian_spreading_d{d}", n_list, errs, 2.0)
        rep.update({"d": d, "norm": "relative L2, volume weighted, at t1", "s0_m": s0,
                    "t1_s": t1, "alpha_m2_s": alpha, "domain_m": L,
                    "bc": "neumann (wall at least 5 final 1/e radii from the centre)"})
        return rep

    def verify(self, n_list=(128, 256, 512)) -> dict:
        cases = {f"d{d}": self.gaussian_check(d, n_list) for d in (1, 2, 3)}
        main = dict(cases["d1"])
        main["check"] = "Gaussian hot spot spreading, d = 1, 2, 3"
        main["cases"] = cases
        main["passed"] = all(c["passed"] for c in cases.values())
        return main
