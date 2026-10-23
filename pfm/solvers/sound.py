"""Linear acoustics in tissue (systems 3, 4, 5, channel 2), in physical units.

    dp/dt = -rho c^2 du/dx,     du/dt = -(1/rho) dp/dx  [+ nu d2u/dx2 for SoundLossy]

solved on the staggered grid of radial.py (d = 1), RK4, dt = 0.5 dx / (c_max + 2 nu / dx).
Only the pressure is exposed (channel 2); the velocity is internal.

Common choices (modelling choices unless a JSON key is named)
-------------------------------------------------------------
- Domain L = 12.8 mm, the same physical window as light.py and heat.py.
- Frame interval: a pulse in the reference medium (c_ref = c_soft.value = 1561 m/s) moves
  one output cell per frame, dt_frame = (L / 128) / c_ref = 6.406e-8 s (64.1 ns); the
  64-frame record is 4.04 us, in which a pulse crosses 63 output cells (about half the
  domain). Over the training c range this is 0.92 to 1.02 cells per frame.
- Initial condition: 1 to 3 Gaussian pressure pulses, 1/e half-width 0.3 to 1.0 mm,
  amplitude 0.2 to 1 (arbitrary pressure unit; the equations are linear, so the scale is
  immaterial), random sign, each launched to the left, to the right or at rest
  (u0 = direction * p0 / Z, Z = rho c). With walls, centres lie in [0.2, 0.8] L and p0, u0
  are multiplied by a C2 taper of width 1 mm.
- Boundary conditions: periodic, pressure_release (p = 0 on the walls), rigid (u = 0).
- Each trajectory records "energy_ratio": discrete acoustic energy at the last frame over
  the first (radial.acoustic_energy). It must be <= 1 (RK4 dissipates weakly; the loss
  term strictly).

Radial geometry (round 3, 21 Oct 2026): params["dim"] = 2 or 3, Sound*(dim),
sample_params(rng, regime, dim=None); outer wall pressure_release or rigid (no
periodic); output is the radial pressure profile on r in [0, L). Pulses are even
pressure shells at radius uniform in [0, 0.8] L with odd velocity shells (zero on the
axis) times direction / Z. Layers become concentric shells (layer 0 the core) with
interfaces at the sampled radii. The loss term becomes nu grad(div u) (radial.py),
which equals nu u_xx in d = 1 and is dissipative by summation by parts. Parameters and
DT_FRAME as in d = 1.

SoundUniform
------------
Train: c in c_soft.range = [1430, 1595] m/s, rho in rho_soft.range = [928, 1060] kg/m^3.
Water at 37 C (water37_acoustic: c 1523.66 m/s, rho 993.33 kg/m^3) lies INSIDE both
training ranges, so it cannot serve as an out-of-distribution medium. No sourced material
lies outside the ranges, so the ood ranges are a modelling choice: 5 to 15 per cent beyond
each end of the sourced range (c in [1215.5, 1358.5] or [1674.75, 1834.25] m/s, rho in
[788.8, 881.6] or [1113, 1219] kg/m^3, side chosen at random for each quantity). They
correspond to no sourced soft tissue.
Exact check: a smooth right-moving pulse on a periodic domain returns to its initial
shape after one period t = L / c. Order 2.

SoundLayered
------------
2 to 4 layers, each a sourced material, adjacent layers different. Train materials:
water at 37 C (water37_acoustic NIST c and rho), fat and muscle (Mast 2000 via Culjat,
fat_acoustic / muscle_acoustic), soft tissue average (acoustic_table_culjat). Ood
materials (unseen): fat and skeletal muscle (Szabo via Stanford slides, the alternative
values of fat_acoustic / muscle_acoustic), blood and liver (acoustic_table_culjat).
Interfaces at random positions in [0.15, 0.85] L, at least 1 mm apart, rounded to an
output-cell face (so a sharp interface sits on a face of every internal grid); each is
sharp (probability 1/2) or a tanh blend of width 0.1 to 0.5 mm. K = rho c^2 at cells,
face density = arithmetic mean of the neighbouring cells.
Exact check: reflection coefficient R = (Z2 - Z1) / (Z2 + Z1) for a smooth right-moving
pulse hitting one sharp interface (the training pair with the largest |R|). The primary
measure is the energy ratio: |R_num| = sqrt(sum (w-)^2 over medium 1 after reflection /
sum (w+)^2 at t = 0), w- = (p - Z1 u) / 2 the left-going and w+ = (p + Z1 u) / 2 the
right-going characteristic. The peak ratio (three-point parabolic interpolation) and the
integral ratio of w- to w+ are reported as secondary measures; the scheme conserves the
integral ratio to round-off, so it cannot show refinement. Pass:
the relative error falls with refinement and is below 1 per cent at n = 512. A sharp
interface limits formal order, so no order is promised.

SoundLossy
----------
du/dt gains nu d2u/dx2. A Fourier mode exp(i k x + s t) gives
    s^2 + nu k^2 s + c^2 k^2 = 0,   s = -nu k^2 / 2 +- i sqrt(c^2 k^2 - nu^2 k^4 / 4),
so the temporal amplitude decay is exp(-nu k^2 t / 2) and, for nu k << c, the spatial
attenuation is alpha(omega) = nu omega^2 / (2 c^3) Np/m. nu is chosen to reproduce the
sourced tissue attenuation at one reference frequency f_ref:
    nu = 2 c^3 alpha_ref / omega_ref^2,
    alpha_ref [Np/m] = alpha0 [dB cm^-1 MHz^-1] * f_ref [MHz] * 100 * ln(10) / 20.
f_ref (modelling choice) is the frequency whose wavelength in the reference medium spans
4 output cells, f_ref = c_ref / (4 L / 128) = 3.90 MHz, close to the highest frequency the
128-point grid resolves. THE FREQUENCY-SQUARED LAW MATCHES THE SOURCED NEAR-LINEAR TISSUE
LAW (alpha = alpha0 f^y, y about 1; alpha_freq_dependence) ONLY AT f_ref: below f_ref it
under-attenuates by the factor f / f_ref, above it over-attenuates by the same factor.
Train: alpha0 in alpha_soft.range = [0.2, 1.09] dB cm^-1 MHz^-1, c and rho as SoundUniform.
Ood: water at 37 C, whose absorption genuinely scales as f^2 (alpha_water.37C_1MHz_dB_cm =
0.001394 dB/cm at 1 MHz), giving alpha_ref = 0.001394 * 3.9025^2 = 0.0212 dB/cm, 2.72 per cent of the training
minimum (0.2 * 3.9025 = 0.7805 dB/cm); c and rho as in training.
Exact check: single Fourier mode cos(k x) with u = 0 on a periodic domain, compared with
the exact damped-mode solution of the linear system including dispersion,
    p = e^{-g t} cos(k x) [cos(w t) + (g / w) sin(w t)],
    u = e^{-g t} sin(k x) sin(w t) k / (rho w),     g = nu k^2 / 2, w = sqrt(c^2 k^2 - g^2),
error in the energy norm over (p, u). Two cases: the sourced muscle-end nu, and a
stiff-loss case with nu scaled so that the amplitude falls by e^-1 over the test (checks
the loss term itself, whose effect at the sourced value is a few per cent). Order 2.
"""
from __future__ import annotations

import math

import numpy as np

from pfm import common
from pfm.solvers import _tissue as J
from pfm.solvers import radial as R

LENGTH = 12.8e-3
C_REF = float(J.get("c_soft"))
DT_FRAME = (LENGTH / common.N_OUT) / C_REF
F_REF = C_REF / (4.0 * LENGTH / common.N_OUT)          # Hz
WALL_TAPER = 1.0e-3

C_RANGE = tuple(float(x) for x in J.get("c_soft", field="range"))
RHO_RANGE = tuple(float(x) for x in J.get("rho_soft", field="range"))
OOD_BAND = (0.05, 0.15)       # modelling choice: fractional distance beyond the range ends
ALPHA0_RANGE = tuple(float(x) for x in J.get("alpha_soft", field="range"))   # dB/cm/MHz
ALPHA_WATER37_1MHZ = float(J.get("alpha_water", "37C_1MHz_dB_cm"))           # dB/cm at 1 MHz


def _ood_band(lo, hi):
    return {"low": [lo * (1 - OOD_BAND[1]), lo * (1 - OOD_BAND[0])],
            "high": [hi * (1 + OOD_BAND[0]), hi * (1 + OOD_BAND[1])]}


def _row(tissue):
    return (float(J.get("acoustic_table_culjat", tissue, "c_m_s")),
            float(J.get("acoustic_table_culjat", tissue, "rho_kg_m3")))


MATERIALS_TRAIN = {
    "water37": (float(J.get("water37_acoustic", "c_m_s_NIST_IAPWS95")),
                float(J.get("water37_acoustic", "rho_kg_m3_NIST"))),
    "fat": (float(J.get("fat_acoustic", "Mast2000_via_Culjat", "c")),
            float(J.get("fat_acoustic", "Mast2000_via_Culjat", "rho"))),
    "muscle": (float(J.get("muscle_acoustic", "Mast2000_via_Culjat", "c")),
               float(J.get("muscle_acoustic", "Mast2000_via_Culjat", "rho"))),
    "soft_tissue": _row("Soft tissue (average)"),
}
MATERIALS_OOD = {
    "fat_szabo": (float(J.get("fat_acoustic", "Szabo_via_Stanford", "c")),
                  float(J.get("fat_acoustic", "Szabo_via_Stanford", "rho"))),
    "muscle_szabo": (float(J.get("muscle_acoustic", "Szabo_via_Stanford_skeletal", "c")),
                     float(J.get("muscle_acoustic", "Szabo_via_Stanford_skeletal", "rho"))),
    "blood": _row("Blood"),
    "liver": _row("Liver"),
}

PARAM_RANGES_UNIFORM = {
    "train": {"c_m_s": list(C_RANGE), "rho_kg_m3": list(RHO_RANGE)},
    "ood": {"c_m_s": _ood_band(*C_RANGE), "rho_kg_m3": _ood_band(*RHO_RANGE)},
}
PARAM_RANGES_LAYERED = {
    "train": {"materials": {k: {"c_m_s": v[0], "rho_kg_m3": v[1]}
                            for k, v in MATERIALS_TRAIN.items()}, "n_layers": [2, 4],
              "interface_width_m": [1e-4, 5e-4]},
    "ood": {"materials": {k: {"c_m_s": v[0], "rho_kg_m3": v[1]}
                          for k, v in MATERIALS_OOD.items()}, "n_layers": [2, 4]},
}
PARAM_RANGES_LOSSY = {
    "train": {"c_m_s": list(C_RANGE), "rho_kg_m3": list(RHO_RANGE),
              "alpha0_dB_cm_MHz": list(ALPHA0_RANGE), "f_ref_Hz": F_REF},
    "ood": {"alpha_ref_dB_cm": ALPHA_WATER37_1MHZ * (F_REF / 1e6) ** 2,
            "c_m_s": list(C_RANGE), "rho_kg_m3": list(RHO_RANGE), "f_ref_Hz": F_REF},
}
PARAM_RANGES = {"uniform": PARAM_RANGES_UNIFORM, "layered": PARAM_RANGES_LAYERED,
                "lossy": PARAM_RANGES_LOSSY,
                "train": PARAM_RANGES_UNIFORM["train"], "ood": PARAM_RANGES_UNIFORM["ood"]}
PARAM_SOURCES = {
    "c_m_s (train)": "c_soft.range",
    "rho_kg_m3 (train)": "rho_soft.range",
    "c_m_s, rho_kg_m3 (ood, uniform)": "modelling choice: 5 to 15 per cent beyond the "
                                       "ends of c_soft.range and rho_soft.range",
    "c_ref (dt_frame, f_ref)": "c_soft.value",
    "water37 layer": "water37_acoustic (c_m_s_NIST_IAPWS95, rho_kg_m3_NIST)",
    "fat layer": "fat_acoustic.Mast2000_via_Culjat (c, rho)",
    "muscle layer": "muscle_acoustic.Mast2000_via_Culjat (c, rho)",
    "soft_tissue layer": "acoustic_table_culjat 'Soft tissue (average)'",
    "ood layers": "fat_acoustic.Szabo_via_Stanford, muscle_acoustic."
                  "Szabo_via_Stanford_skeletal, acoustic_table_culjat 'Blood', 'Liver'",
    "alpha0 (train)": "alpha_soft.range (dB cm^-1 MHz^-1)",
    "alpha_ref (ood)": "alpha_water.37C_1MHz_dB_cm x (f_ref / 1 MHz)^2",
    "f_ref": "modelling choice: c_ref / (4 output cells)",
    "dB to Np": "definition ln(10)/20 (JSON conventions.attenuation)",
    "LENGTH, pulses, layer geometry, OOD_BAND": "modelling choice",
}


def nu_from_alpha_ref(alpha_ref_dB_cm: float, c: float, f_ref: float = F_REF) -> float:
    """nu = 2 c^3 alpha_ref / omega_ref^2 with alpha_ref converted dB/cm -> Np/m."""
    a_np_m = alpha_ref_dB_cm * J.CM_TO_M * J.DB_TO_NP
    return 2.0 * c ** 3 * a_np_m / (2.0 * math.pi * f_ref) ** 2


def sample_pulses(rng):
    """1 to 3 Gaussian pulses. The centre is stored as a fraction u in [0, 1] and mapped at
    simulation time: x = u L (periodic) or x = (0.2 + 0.6 u) L (walls), so the parameters
    do not depend on the boundary condition."""
    out = []
    for _ in range(int(rng.integers(1, 4))):
        out.append({"u_centre": float(rng.uniform(0.0, 1.0)),
                    "width": float(rng.uniform(0.3e-3, 1.0e-3)),
                    "amp": float(rng.uniform(0.2, 1.0) * rng.choice([-1.0, 1.0])),
                    "direction": int(rng.integers(-1, 2))})
    return out


def pulse_centre(g, bc, length=LENGTH):
    u = g["u_centre"]
    return u * length if bc == "periodic" else (0.2 + 0.6 * u) * length


class _SoundBase(common.Solver):
    channels = (common.CH_PRESSURE,)
    order = 2.0
    lossy = False

    # media: returns per-cell c, rho arrays (B, n) and nu (B, 1) or None
    def _media(self, params_list, grid, bc):
        raise NotImplementedError

    def __init__(self, dim: int = 1):
        self.dim = int(dim)

    def _sample(self, rng, regime):
        raise NotImplementedError

    def sample_params(self, rng, regime="train", dim=None):
        """The d = 1 draw (unchanged) plus a "dim" key. d = 2, 3: each pulse becomes an
        even pressure shell at radius = u_centre x 0.8 L (uniform in [0, 0.8] L, r = 0
        included; radial.RADIAL_MAX_FRAC); its velocity is the odd shell (zero on the axis) times direction / Z,
        so direction +1 / -1 launches the shell outward / inward and 0 leaves it at rest."""
        dim = self.dim if dim is None else int(dim)
        p = self._sample(rng, regime)
        if dim > 1:
            for g in p["pulses"]:
                g["radius"] = R.radius_from_frac(g.pop("u_centre"), LENGTH)
        p["dim"] = dim
        return p

    def simulate_batch(self, params_list, bc, n_frames, rng=None, n_internal=512):
        return R.split_by_dim(params_list, self.dim,
                              lambda ps, d: self._simulate(ps, bc, n_frames, n_internal, d))

    def _simulate(self, params_list, bc, n_frames, n_internal, d):
        grid = R.RadialGrid(n_internal, LENGTH, d)
        grid.check_bc(bc, "acoustic")
        c, rho, nu = self._media(params_list, grid, bc)
        B, n = c.shape
        K = rho * c ** 2
        rho_f = R.face_density(rho, grid, bc)
        inv_rho_f = 1.0 / rho_f
        Z_f = R.face_mean(rho * c, grid, bc)
        per = bc == "periodic"
        tap_c = R.smoothstep_taper(grid.rc, LENGTH, WALL_TAPER) if not per else 1.0
        tap_f = R.smoothstep_taper(grid.rf, LENGTH, WALL_TAPER) if not per else 1.0
        p0 = np.zeros((B, n)); u0 = np.zeros((B, n + 1))
        if d > 1:                                   # radial: outer-wall taper only
            tap_c = R.outer_taper(grid.rc, LENGTH, WALL_TAPER)
            tap_f = R.outer_taper(grid.rf, LENGTH, WALL_TAPER)
        for b, prm in enumerate(params_list):
            for g in prm["pulses"]:
                if d > 1:
                    pk = R.shell_peak(g["radius"], g["width"], LENGTH)
                    p0[b] += g["amp"] * R.even_shell(grid.rc, g["radius"], g["width"]) / pk
                    u0[b] += (g["direction"] * g["amp"]
                              * R.odd_shell(grid.rf, g["radius"], g["width"]) / pk) / Z_f[b]
                    continue
                x0 = pulse_centre(g, bc)
                p0[b] += g["amp"] * R.gaussian(grid.rc, x0, g["width"], LENGTH, per)
                u0[b] += (g["direction"] * g["amp"]
                          * R.gaussian(grid.rf, x0, g["width"], LENGTH, per)) / Z_f[b]
        p0 *= tap_c; u0 *= tap_f
        if d > 1:
            u0[:, 0] = 0.0                          # axis / centre
        if per:
            u0[:, -1] = u0[:, 0]
        if bc == "rigid":
            u0[:, 0] = u0[:, -1] = 0.0
        cmax = c.max(axis=1)
        nuv = np.zeros(B) if nu is None else nu[:, 0]
        dts = R.CFL_ACOUSTIC * grid.dr / (cmax + 2.0 * nuv / grid.dr)
        nsub = R.n_substeps(DT_FRAME, dts)
        P = {"K": K, "irf": inv_rho_f, "nu": nu}

        def rhs(t, y, Q):
            return R.acoustic_rhs(y[0], y[1], grid, Q["K"], Q["irf"], bc, nu=Q["nu"])

        rec = lambda y: grid.downsample(y[0], common.N_OUT)
        out, (pN, uN) = R.integrate(rhs, (p0, u0), P, DT_FRAME, nsub, n_frames, rec,
                                    return_state=True)
        E0 = R.acoustic_energy(p0, u0, grid, K, rho_f, bc)
        EN = R.acoustic_energy(pN, uN, grid, K, rho_f, bc)
        trajs = []
        for b, prm in enumerate(params_list):
            f = np.zeros((n_frames, common.N_CH, common.N_OUT), np.float32)
            f[:, common.CH_PRESSURE] = out[b]
            q = dict(prm)
            q.update({"dt": float(DT_FRAME / nsub[b]), "n_internal": n_internal,
                      "nu_m2_s": float(nuv[b]), "energy_ratio": float(EN[b] / E0[b])})
            trajs.append(common.Trajectory(f, self.name, q, DT_FRAME, LENGTH, bc,
                                           self.channels))
        return trajs

    def simulate(self, params, bc, n_frames, rng=None, n_internal=512):
        return self.simulate_batch([params], bc, n_frames, rng, n_internal)[0]


def _draw_ood(rng, band):
    side = "low" if rng.random() < 0.5 else "high"
    return float(rng.uniform(*band[side])), side


def run_uniform_medium(p0, u0, grid, c, rho, bc, t_end, nu=None, cfl=R.CFL_ACOUSTIC):
    """Advance one uniform-medium state (verification helper). Returns final (p, u)."""
    K = np.full((1, grid.n), rho * c ** 2)
    irf = np.full((1, grid.n + 1), 1.0 / rho)
    nuv = None if nu is None else np.full((1, 1), nu)
    dt = cfl * grid.dr / (c + (0 if nu is None else 2 * nu / grid.dr))
    ns = R.n_substeps(t_end, dt)
    rhs = lambda t, y, Q: R.acoustic_rhs(y[0], y[1], grid, K, irf, bc, nu=nuv)
    _, st = R.integrate(rhs, (p0[None], u0[None]), {}, t_end, ns, 2,
                        lambda y: y[0][:, :1].copy(), return_state=True)
    return st[0][0], st[1][0]


class SoundUniform(_SoundBase):
    name = "sound_uniform"

    def _sample(self, rng, regime="train"):
        if regime == "train":
            c = float(rng.uniform(*C_RANGE)); rho = float(rng.uniform(*RHO_RANGE))
            return {"regime": regime, "c": c, "rho": rho, "pulses": sample_pulses(rng)}
        c, cs = _draw_ood(rng, PARAM_RANGES_UNIFORM["ood"]["c_m_s"])
        rho, rs = _draw_ood(rng, PARAM_RANGES_UNIFORM["ood"]["rho_kg_m3"])
        return {"regime": regime, "c": c, "rho": rho, "ood_side": [cs, rs],
                "pulses": sample_pulses(rng)}

    def _media(self, params_list, grid, bc):
        c = np.array([[p["c"]] for p in params_list]) * np.ones(grid.n)
        rho = np.array([[p["rho"]] for p in params_list]) * np.ones(grid.n)
        return c, rho, None

    def verify(self, n_list=(128, 256, 512)) -> dict:
        c, rho = C_REF, float(J.get("rho_soft"))
        w = LENGTH / 16.0
        errs = []
        for n in n_list:
            g = R.RadialGrid(n, LENGTH, 1)
            p0 = R.gaussian(g.rc, LENGTH / 2, w, LENGTH, True)
            u0 = R.gaussian(g.rf, LENGTH / 2, w, LENGTH, True) / (rho * c)
            pN, _ = run_uniform_medium(p0, u0, g, c, rho, "periodic", LENGTH / c)
            errs.append(float(np.sqrt(np.sum((pN - p0) ** 2) / np.sum(p0 ** 2))))
        rep = common.convergence_report(self.name, "travelling_wave_one_period", n_list,
                                        errs, self.order)
        rep.update({"norm": "relative L2 of p after one period", "c_m_s": c,
                    "rho_kg_m3": rho, "pulse_1e_halfwidth_m": w, "t_end_s": LENGTH / c,
                    "water37_inside_train_range": bool(
                        C_RANGE[0] <= MATERIALS_TRAIN["water37"][0] <= C_RANGE[1]
                        and RHO_RANGE[0] <= MATERIALS_TRAIN["water37"][1] <= RHO_RANGE[1])})
        return rep


class SoundLayered(_SoundBase):
    name = "sound_layered"

    def _sample(self, rng, regime="train"):
        mats = MATERIALS_TRAIN if regime == "train" else MATERIALS_OOD
        names = list(mats)
        nl = int(rng.integers(2, 5))
        layers = [str(rng.choice(names))]
        while len(layers) < nl:
            m = str(rng.choice([x for x in names if x != layers[-1]]))
            layers.append(m)
        # interfaces on output-cell faces in [0.15, 0.85] L, >= 1 mm apart
        cell = LENGTH / common.N_OUT
        for _ in range(1000):
            xs = np.sort(rng.uniform(0.15, 0.85, nl - 1) * LENGTH)
            xs = np.round(xs / cell) * cell
            if nl == 2 or np.min(np.diff(xs)) >= 1e-3:
                break
        ifaces = []
        for x in xs:
            sharp = bool(rng.random() < 0.5)
            ifaces.append({"x": float(x), "sharp": sharp,
                           "width": 0.0 if sharp else float(rng.uniform(1e-4, 5e-4))})
        return {"regime": regime, "layers": layers, "interfaces": ifaces,
                "media": {m: {"c": mats[m][0], "rho": mats[m][1]} for m in sorted(set(layers))},
                "pulses": sample_pulses(rng)}

    @staticmethod
    def profile(prm, x):
        layers, ifs = prm["layers"], prm["interfaces"]
        c = np.full_like(x, prm["media"][layers[0]]["c"], dtype=float)
        rho = np.full_like(x, prm["media"][layers[0]]["rho"], dtype=float)
        for j, itf in enumerate(ifs):
            a, b = prm["media"][layers[j]], prm["media"][layers[j + 1]]
            if itf["sharp"]:
                h = (x > itf["x"]).astype(float)
            else:
                h = 0.5 * (1 + np.tanh((x - itf["x"]) / itf["width"]))
            c += h * (b["c"] - a["c"]); rho += h * (b["rho"] - a["rho"])
        return c, rho

    def _media(self, params_list, grid, bc):
        prof = [self.profile(p, grid.rc) for p in params_list]
        return np.stack([q[0] for q in prof]), np.stack([q[1] for q in prof]), None

    def verify(self, n_list=(128, 256, 512)) -> dict:
        # the training pair with the largest |R|
        names = list(MATERIALS_TRAIN)
        best = None
        for a in names:
            for b in names:
                if a == b:
                    continue
                Z1 = MATERIALS_TRAIN[a][0] * MATERIALS_TRAIN[a][1]
                Z2 = MATERIALS_TRAIN[b][0] * MATERIALS_TRAIN[b][1]
                Rx = (Z2 - Z1) / (Z2 + Z1)
                if best is None or abs(Rx) > abs(best[2]):
                    best = (a, b, Rx)
        m1, m2, R_exact = best
        (c1, r1), (c2, r2) = MATERIALS_TRAIN[m1], MATERIALS_TRAIN[m2]
        Z1 = r1 * c1
        L = 2 * LENGTH
        xi, x0, w = L / 2, L / 4, LENGTH / 32
        t_end = (xi - x0) / c1 * 2.0       # incident peak travels to the interface and back
        errs, Rn, Rint, Ren = [], [], [], []
        for n in n_list:
            nn = 2 * n                      # same dx as an n-cell grid on LENGTH
            g = R.RadialGrid(nn, L, 1)
            c = np.where(g.rc < xi, c1, c2); rho = np.where(g.rc < xi, r1, r2)
            K = (rho * c ** 2)[None]
            irf = 1.0 / R.face_density(rho[None], g, "rigid")
            p0 = np.exp(-((g.rc - x0) / w) ** 2)
            u0 = np.exp(-((g.rf - x0) / w) ** 2) / Z1
            u0[0] = u0[-1] = 0.0
            dt = R.CFL_ACOUSTIC * g.dr / max(c1, c2)
            ns = R.n_substeps(t_end, dt)
            rhs = lambda t, y, Q: R.acoustic_rhs(y[0], y[1], g, K, irf, "rigid")
            _, (pN, uN) = R.integrate(rhs, (p0[None], u0[None]), {}, t_end, ns, 2,
                                      lambda y: y[0][:, :1].copy(), return_state=True)
            pN, uN = pN[0], uN[0]
            uc = 0.5 * (uN[1:] + uN[:-1])
            left = g.rc < xi
            wm = 0.5 * (pN - Z1 * uc)[left]
            wp0 = 0.5 * (p0 + Z1 * 0.5 * (u0[1:] + u0[:-1]))
            # peak of the reflected pulse by parabolic interpolation through the extreme
            # sample and its neighbours (removes the O(dx^2) sampling bias of a raw max)
            i = int(np.argmax(np.abs(wm)))
            a, b_, cc = wm[i - 1], wm[i], wm[i + 1]
            peak = b_ - 0.125 * (cc - a) ** 2 / (cc - 2 * b_ + a)
            j = int(np.argmax(wp0))
            a, b_, cc = wp0[j - 1], wp0[j], wp0[j + 1]
            peak0 = b_ - 0.125 * (cc - a) ** 2 / (cc - 2 * b_ + a)
            Rpk = float(peak / peak0)
            Rn.append(Rpk)
            Rint.append(float(np.sum(wm) / np.sum(wp0)))
            Ren.append(float(np.sqrt(np.sum(wm ** 2) / np.sum(wp0 ** 2))))
            errs.append(abs(Ren[-1] - abs(R_exact)) / abs(R_exact))
        # order table from the common harness; promised order 0 (none promised for a sharp
        # interface), pass/fail replaced by the spec's rule
        rep = common.convergence_report(self.name, "reflection_coefficient_sharp_interface",
                                        n_list, errs, 0.0)
        passed = all(b < a for a, b in zip(errs, errs[1:])) and errs[-1] < 0.01
        rep.update({"promised_order": None, "passed": bool(passed),
                "pass_rule": "relative error falls with refinement and < 1 per cent at the "
                             "finest grid; no formal order promised (sharp interface)",
                "media": [m1, m2], "Z1": Z1, "Z2": r2 * c2, "R_exact": R_exact,
                "measure": "primary: sqrt(energy of the left-going characteristic "
                           "(p - Z1 u)/2 in medium 1 after reflection / energy of the incident "
                           "(p + Z1 u)/2 at t = 0), compared with |R|",
                "R_numerical_peak": Rn,
                "peak_error": [abs(x - R_exact) / abs(R_exact) for x in Rn],
                "peak_note": "secondary: reflected / incident peak (three-point parabolic "
                             "interpolation); converges, but not monotonically in rate, "
                             "because dispersive trailing ripples move the sampled extreme",
                "R_numerical_integral": Rint,
                "integral_note": "the integral ratio is conserved exactly by the discrete "
                                 "scheme (round-off agreement), so it cannot show refinement "
                                 "and is reported only",
                "R_numerical_energy_sqrt": Ren,
                "domain_m": L, "dx_m": [L / (2 * n) for n in n_list]})
        return rep


class SoundLossy(_SoundBase):
    name = "sound_lossy"

    def _sample(self, rng, regime="train"):
        c = float(rng.uniform(*C_RANGE)); rho = float(rng.uniform(*RHO_RANGE))
        if regime == "train":
            a0 = float(rng.uniform(*ALPHA0_RANGE))
            a_ref = a0 * F_REF / 1e6
            p = {"alpha0_dB_cm_MHz": a0}
        else:
            a_ref = PARAM_RANGES_LOSSY["ood"]["alpha_ref_dB_cm"]
            p = {"medium": "water37 absorption law"}
        p.update({"regime": regime, "c": c, "rho": rho, "alpha_ref_dB_cm": a_ref,
                  "f_ref_Hz": F_REF, "nu": nu_from_alpha_ref(a_ref, c),
                  "pulses": sample_pulses(rng)})
        return p

    def _media(self, params_list, grid, bc):
        c = np.array([[p["c"]] for p in params_list]) * np.ones(grid.n)
        rho = np.array([[p["rho"]] for p in params_list]) * np.ones(grid.n)
        nu = np.array([[p["nu"]] for p in params_list])
        return c, rho, nu

    @staticmethod
    def mode_check(nu, c, rho, n_list, m=3, label=""):
        L = LENGTH
        k = 2 * math.pi * m / L
        gam = nu * k ** 2 / 2
        om = math.sqrt(c ** 2 * k ** 2 - gam ** 2)
        t_end = 2 * L / c
        errs = []
        for n in n_list:
            g = R.RadialGrid(n, L, 1)
            p0 = np.cos(k * g.rc); u0 = np.zeros(n + 1)
            pN, uN = run_uniform_medium(p0, u0, g, c, rho, "periodic", t_end, nu=nu)
            e = math.exp(-gam * t_end)
            pe = e * np.cos(k * g.rc) * (math.cos(om * t_end) + gam / om * math.sin(om * t_end))
            ue = e * np.sin(k * g.rf) * math.sin(om * t_end) * k / (rho * om)
            num = np.sum((pN - pe) ** 2) / (rho * c ** 2) + rho * np.sum((uN[:-1] - ue[:-1]) ** 2)
            den = np.sum(pe ** 2) / (rho * c ** 2) + rho * np.sum(ue[:-1] ** 2)
            errs.append(float(np.sqrt(num / den)))
        rep = common.convergence_report("sound_lossy", f"damped_fourier_mode{label}",
                                        n_list, errs, 2.0)
        rep.update({"nu_m2_s": nu, "c_m_s": c, "rho_kg_m3": rho, "mode": m, "k_rad_m": k,
                    "t_end_s": t_end, "amplitude_factor_exact": math.exp(-gam * t_end),
                    "omega_rad_s": om, "norm": "relative energy norm over (p, u)"})
        return rep

    def verify(self, n_list=(128, 256, 512)) -> dict:
        c, rho = C_REF, float(J.get("rho_soft"))
        nu_src = nu_from_alpha_ref(ALPHA0_RANGE[1] * F_REF / 1e6, c)
        m = 3
        k = 2 * math.pi * m / LENGTH
        nu_stiff = 2.0 / (k ** 2 * 2 * LENGTH / c)     # amplitude e^-1 over t_end
        cases = {"sourced_nu_alpha0_max": self.mode_check(nu_src, c, rho, n_list, m,
                                                          "_sourced"),
                 "stiff_loss": self.mode_check(nu_stiff, c, rho, n_list, m, "_stiff")}
        main = dict(cases["sourced_nu_alpha0_max"])
        main["check"] = "damped Fourier mode (two loss levels)"
        main["cases"] = cases
        main["passed"] = all(v["passed"] for v in cases.values())
        main["f_ref_Hz"] = F_REF
        main["nu_range_train_m2_s"] = [nu_from_alpha_ref(ALPHA0_RANGE[0] * F_REF / 1e6, C_RANGE[0]),
                                       nu_from_alpha_ref(ALPHA0_RANGE[1] * F_REF / 1e6, C_RANGE[1])]
        return main
