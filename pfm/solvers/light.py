"""Light transport (system 1, channel 0): time-resolved diffusion with absorption, in
physical units.

    d phi/dt = d/dx (D(x) d phi/dx) - mu_a(x) c phi + S(x, t),
    D = c / (3 (mu_a + mu_s')),  c = c0 / n.

phi is the fluence rate in arbitrary units (peak of each initial source 0.5 to 1). With heterogeneous
mu_a the conservative form d/dx (D dphi/dx) is used (D at faces = arithmetic mean of the
neighbouring cells); for uniform mu_a it is D d2phi/dx2.

Optical properties from chromophores at wavelength lambda (all values read from
params/tissue_params.json by key):
    mu_a = B [S mu_a,blood,HbO2 + (1 - S) mu_a,blood,Hb]       (Jacques 2013 Eq. 12)
         + W mu_a,water + F mu_a,fat + M mu_a,melanosome
    mu_a,blood,X(lambda) = ln(10) eps_X(lambda) C_Hb / MW_Hb    (Prahl; Jacques Eq. 6)
        eps in cm^-1 M^-1, C_Hb / MW_Hb in mol L^-1, result in cm^-1, x 100 for m^-1
    mu_a,melanosome = a_mel (lambda / lambda_ref)^-m             (Jacques 2013 Eq. 8)
    mu_s' = a (lambda / 500 nm)^-b                               (Jacques 2013 Eq. 1)
    n = n_dry - (n_dry - n_water) W                              (Jacques 2013 Eq. 3)
B blood volume fraction, S oxygen saturation, W water, F fat and M melanosome volume
fractions. Water: Hale and Querry 1973 column. Fat: van Veen 2004 pig lard (m^-1).

Domain, time window and frame interval (revised 20 Oct 2026 after the validity audit)
-------------------------------------------------------------------------------------
L = 12.8 mm (modelling choice, the same physical window as heat, sound and chain).

The diffusion approximation holds only some transport times t_tr = 1/((mu_a + mu_s') c)
after an impulse. The record therefore starts at a fixed time T_START after the impulse,
ten transport times of the slowest-scattering corner of the training box:
    T_START = 10 / (mu_s'_min c_min),
    mu_s'_min = a_min (850 / 500)^(-b_max)   (smallest Table 2 tissue mean a, largest b,
                                              longest training wavelength),
    c_min = c0 / n(W = 0.6)                  (lowest water fraction gives the largest n),
with mu_a set to 0, which makes t_tr as long as possible; every draw (train and ood)
therefore has T_START >= 10 of its own t_tr. Values: mu_s'_min = 708.6 m^-1, c_min =
2.1444e8 m/s, t_tr,max = 6.581 ps, T_START = 65.81 ps.
Initial condition: the exact homogeneous Green's-function state at T_START of 1 to 4
impulse sources released at t = 0, in the background medium of the trajectory
(D_bg = c / (3 (mu_a,bg + mu_s')), mu_a,bg the background absorption), with the method
of images for the boundary (periodic images; odd images for dirichlet, which vanish on
the walls; even images for neumann), so the state is exact for the homogeneous slab:
    phi_j(x, T) = A_j sum_images exp(-(x - x_image)^2 / (4 D_bg T_START)).
A_j in [0.5, 1] is the peak of source j alone (the constant c (4 pi D T)^(-1/2)
exp(-mu_a c T) is absorbed into A_j; phi is in arbitrary units). The state then evolves
with the heterogeneous mu_a(x) (background plus absorbers). There is no source term
during the record (a new impulse would put light into the record that is younger than
ten transport times). Source centres lie in [0.2, 0.8] L (modelling choice).
The 1/e width at the start, sqrt(4 D_bg T_START), is sqrt(4 T_START / (3 t_tr)) transport
mean free paths l_tr = 1/(mu_a + mu_s'), so at least sqrt(40/3) = 3.65 l_tr for every draw.
Frame interval: the record ends when the spreading length sqrt(4 D_ref t) of the reference
draw reaches L / 2,
    T_END = (L / 2)^2 / (4 D_ref),   DT_FRAME = (T_END - T_START) / 63,
D_ref = c_ref / (3 mu_s'_ref), mu_s'_ref the 'other soft tissues' Table 2 mean at 800 nm,
c_ref = c_tissue, mu_a neglected (as in l_tr_800_soft). Values: D_ref = 7.0634e4 m^2/s,
T_END = 144.97 ps, DT_FRAME = 1.2565 ps; the 64 frames span 65.81 ps to 144.97 ps after
the impulse (79.2 ps of record). For the reference draw the spreading length grows from
4.31 mm to 6.40 mm over the record.

Boundary conditions: periodic, dirichlet (phi = 0 on the wall, the zero-fluence
approximation of a free surface), neumann (zero flux).

Heterogeneous mu_a: a background plus 0 to 3 absorbers (vessels), each a Gaussian (1/e
half-width 0.2 to 1 mm) or a smoothed box (full width 0.2 to 1 mm, tanh edges 50 to
150 um) of blood fraction B_v and saturation S_v, centres in [0.1, 0.9] L.

Validity cap (revised 20 Oct 2026): every trajectory, train and ood, light and chain,
satisfies max_x mu_a(x) / mu_s' <= MUA_MUSP_CAP = 0.3, imposed by rejection sampling in
sample_tissue (the whole tissue draw is redrawn). max_x mu_a is evaluated on a 4096-point
grid with periodic images summed, which bounds the field for every boundary condition.
Each draw records n_rejected; the rejection rates are reported by tests/verify_light.py.
The ood wavelength is 700 nm (replacing 532 nm, at which blood absorption exceeds the
reduced scattering and diffusion theory fails).

Radial geometry (round 3, 21 Oct 2026): params["dim"] = 2 (cylinder) or 3 (sphere),
sample_params(rng, regime, dim=None), LightSolver(dim). Output is the radial profile on
r in [0, L) (128 volume-weighted cells), symmetric at r = 0. Allowed boundary conditions
in d = 2, 3: dirichlet and neumann at the outer wall (no periodic). Every parameter, the
time window and DT_FRAME are as in d = 1. Impulses become shells at a radius uniform in
[0, 0.8] L (the d = 1 centre draw mapped linearly; r = 0 included). The initial state is
the exact Green's function of the bounded homogeneous cylinder or sphere at T_START,
built from the eigenfunction series (radial_initial_state, 200 modes), so it is exact for
the homogeneous domain as the image sum is in d = 1. Absorbers are the chain's: shells
mu_a(r) at the sampled centre radii (as in chain.py for d = 2, 3); a Gaussian absorber
whose centre is within about two widths of r = 0 has a kink in mu_a at the axis, which
the chain shares and which is left as is so that the tissue model matches the chain.

Stable step: RK4 with dt = 2.5 / (Gershgorin bound of the assembled operator), per
trajectory, equal substeps per frame. Promised order 2 (space; dt is tied to dr^2 so
the RK4 error is O(dr^8)).

Exact checks.
 1. (radial.py, d = 1, 2, 3) Green's function of an instantaneous plane, line or point
    source in an infinite homogeneous medium,
        phi = c (4 pi D t)^(-d/2) exp(-r^2 / (4 D t) - mu_a c t),
    started from the exact solution at t0 > 0 and compared at t1 = 4 t0. This is a check
    of the numerics of the diffusion equation; t0 is chosen for the numerics (1/e radius
    L/20 at t0), not for optical validity.
 2. (record window, d = 1, each boundary condition) the production path simulate_batch
    on a homogeneous medium: last frame (t = T_END) against the exact image-sum Green's
    state, as exact cell averages on the 128 output cells, internal n = 128, 256, 512.
"""
from __future__ import annotations

import math

import numpy as np

from pfm import common
from pfm.solvers import _tissue as J
from pfm.solvers import radial as R

# ------------------------------------------------------------------ sourced constants
C0 = float(J.get("c0"))                                       # m/s, exact
N_WATER = float(J.get("n_tissue_model", "n_water"))
N_DRY = float(J.get("n_tissue_model", "n_dry"))
# C_Hb / MW_Hb in mol per litre; ln(10) for decadic -> Napierian; x100 cm^-1 -> m^-1
HB_TO_MUA = (math.log(10.0) * float(J.get("C_Hb_blood")) / float(J.get("MW_Hb"))
             * J.CM_TO_M)                                     # m^-1 per (cm^-1 M^-1)
MEL_A = float(J.get("melanin_formula_jacques", "a_cm-1")) * J.CM_TO_M   # m^-1
MEL_M = float(J.get("melanin_formula_jacques", "m"))
MEL_REF_NM = J.reference_wavelength_nm("melanin_formula_jacques", "formula")
SCAT_REF_NM = J.reference_wavelength_nm("musp_powerlaw_soft", "symbol")
SCAT_A_RANGE = tuple(x * J.CM_TO_M for x in J.get("musp_powerlaw_by_tissue", "a_cm-1",
                                                     field="range"))      # m^-1
SCAT_B_RANGE = tuple(J.get("musp_powerlaw_by_tissue", "b", field="range"))
SCAT_A_SOFT = float(J.get("musp_powerlaw_soft", "a_m-1", field="value_SI"))
SCAT_B_SOFT = float(J.get("musp_powerlaw_soft", "b"))
C_REF = float(J.get("c_tissue"))                              # m/s

WAVELENGTHS_TRAIN = (750, 800, 850)     # nm, design choice (PLAN section 2)
WAVELENGTHS_OOD = (700,)    # replaces 532 nm (20 Oct 2026 audit)
MUA_MUSP_CAP = 0.3          # max_x mu_a / mu_s' for every trajectory (modelling choice set after the audit; the audit suggested about 0.1, the 0.3 cap keeps every training draw, max 0.28, and is stated as a limitation)


def _chromophores(lam: int) -> dict:
    return {
        "mua_HbO2": HB_TO_MUA * float(J.get(f"eps_HbO2_{lam}")),
        "mua_Hb": HB_TO_MUA * float(J.get(f"eps_Hb_{lam}")),
        "mua_water": J.first_key(f"mua_water_{lam}", "Hale1973")[0],   # m^-1
        "mua_fat": float(J.get(f"mua_fat_{lam}", field="value_SI")),    # m^-1
        "mua_mel": MEL_A * (lam / MEL_REF_NM) ** (-MEL_M),              # m^-1
    }


CHROMOPHORES = {lam: _chromophores(lam) for lam in WAVELENGTHS_TRAIN + WAVELENGTHS_OOD}

LENGTH = 12.8e-3                         # m, modelling choice
WATER_FRACTION_RANGE = (0.6, 0.8)        # modelling choice (see PARAM_SOURCES)
MUSP_REF = SCAT_A_SOFT * (800.0 / SCAT_REF_NM) ** (-SCAT_B_SOFT)
D_REF = C_REF / (3.0 * MUSP_REF)
# slowest-scattering corner of the training box (mu_a = 0): longest transport time
MUSP_MIN = min(SCAT_A_RANGE[0] * (lam / SCAT_REF_NM) ** (-SCAT_B_RANGE[1])
               for lam in WAVELENGTHS_TRAIN)
C_MIN = C0 / (N_DRY - (N_DRY - N_WATER) * WATER_FRACTION_RANGE[0])
T_TR_MAX = 1.0 / (MUSP_MIN * C_MIN)
T_START = 10.0 * T_TR_MAX
T_END = (LENGTH / 2.0) ** 2 / (4.0 * D_REF)
DT_FRAME = (T_END - T_START) / 63.0
N_IMAGES = 4                             # image copies on each side (see initial_state)

PARAM_RANGES = {
    "train": {
        "wavelength_nm": list(WAVELENGTHS_TRAIN),
        "blood_fraction": [0.002, 0.03],
        "saturation": [0.6, 1.0],
        "melanosome_fraction": [0.0, 5e-4],
        "water_fraction": list(WATER_FRACTION_RANGE),
        "fat_fraction": [0.0, 0.2],
        "scat_a_m-1": list(SCAT_A_RANGE),
        "scat_b": list(SCAT_B_RANGE),
        "vessel_blood_fraction": [0.05, 0.3],
        "n_absorbers": [0, 3],
        "absorber_width_m": [0.2e-3, 1.0e-3],
        "absorber_edge_m": [5e-5, 1.5e-4],
        "n_impulses": [1, 4],
        "impulse_amp": [0.5, 1.0],
        "impulse_centre_frac": [0.2, 0.8],
        "mua_over_musp_cap": MUA_MUSP_CAP,
    },
    "ood": {
        # each ood trajectory shifts ONE thing: either the wavelength (700 nm, blood in
        # the train range) or the blood fractions (train wavelengths), 50/50; both subject
        # to the mu_a / mu_s' cap
        "wavelength_nm": list(WAVELENGTHS_OOD),
        "blood_fraction": [0.04, 0.08],
        "vessel_blood_fraction": [0.4, 1.0],
    },
}
PARAM_SOURCES = {
    "wavelength_nm": "design choice (PLAN section 2); eps_HbO2_<l>, eps_Hb_<l>, "
                     "mua_water_<l>, mua_fat_<l> must exist for each l",
    "c0": "c0",
    "n (refractive index)": "n_tissue_model (n_dry, n_water) with the drawn water_fraction",
    "mu_a,blood": "eps_HbO2_<l>, eps_Hb_<l>, C_Hb_blood, MW_Hb (ln10 eps C/MW, x100)",
    "mu_a,water": "mua_water_<l> (Hale1973 column, value_SI)",
    "mu_a,fat": "mua_fat_<l> (value_SI)",
    "mu_a,melanosome": "melanin_formula_jacques (a_cm-1 x100, m, reference wavelength "
                       "parsed from the formula)",
    "scat_a_m-1": "musp_powerlaw_by_tissue.range.a_cm-1 x100 (Table 2 tissue-type means)",
    "scat_b": "musp_powerlaw_by_tissue.range.b",
    "scattering reference wavelength": "musp_powerlaw_soft.symbol ('lambda/500 nm')",
    "D_ref for dt_frame": "musp_powerlaw_soft (a_m-1, b) at 800 nm, c_tissue.value",
    "T_START": "derived: 10 / (mu_s'_min c_min); mu_s'_min from musp_powerlaw_by_tissue."
               "range (a min, b max) at 850 nm; c_min = c0 / n_tissue_model(W = 0.6)",
    "T_END, DT_FRAME": "derived: T_END = (L/2)^2 / (4 D_ref), DT_FRAME = (T_END - T_START)/63",
    "700 nm chromophores": "eps_HbO2_700, eps_Hb_700, mua_water_700 (Hale1973), "
                           "mua_fat_700, melanin_formula_jacques (added 20 Oct 2026)",
    "MUA_MUSP_CAP": "modelling choice (0.3) after audit round 1, which suggested about 0.1; applied by rejection sampling",
    "blood_fraction": "modelling choice (Jacques 2013 Sec. 5 worked example B = 0.01 is "
                      "quoted in l_tr_800_soft.condition; bounds not sourced)",
    "saturation": "modelling choice",
    "melanosome_fraction": "modelling choice (uniform background; melanin is sourced "
                           "only as a melanosome-interior coefficient)",
    "water_fraction": "modelling choice, matched to the illustrative W = 0.6 to 0.8 in "
                      "n_tissue_model.condition (not sourced)",
    "fat_fraction": "modelling choice",
    "vessel_blood_fraction": "modelling choice",
    "absorber geometry, impulse number, amplitudes and centres": "modelling choice",
    "LENGTH": "modelling choice (12.8 mm)",
}


# ---------------------------------------------------------------- optical properties
def refractive_index(W):
    return N_DRY - (N_DRY - N_WATER) * np.asarray(W, float)


def mua_background(p: dict) -> float:
    ch = CHROMOPHORES[p["wavelength_nm"]]
    S = p["saturation"]
    blood = S * ch["mua_HbO2"] + (1 - S) * ch["mua_Hb"]
    return (p["blood_fraction"] * blood + p["water_fraction"] * ch["mua_water"]
            + p["fat_fraction"] * ch["mua_fat"] + p["melanosome_fraction"] * ch["mua_mel"])


def mua_field(p: dict, x: np.ndarray, length: float, periodic: bool) -> np.ndarray:
    """mu_a(x) in m^-1: background plus absorbers (whole-blood fraction B_v replaces the
    background blood fraction inside each absorber)."""
    ch = CHROMOPHORES[p["wavelength_nm"]]
    mu = np.full_like(x, mua_background(p), dtype=float)
    for a in p.get("absorbers", []):
        blood = a["saturation"] * ch["mua_HbO2"] + (1 - a["saturation"]) * ch["mua_Hb"]
        if a["shape"] == "gaussian":
            s = R.gaussian(x, a["centre"], a["width"], length, periodic)
        else:
            s = R.smoothed_box(x, a["centre"], a["width"], a["edge"], length, periodic)
        mu += np.clip(s, 0.0, 1.0) * max(a["blood_fraction"] - p["blood_fraction"], 0.0) * blood
    return mu


def musp(p: dict) -> float:
    return p["scat_a"] * (p["wavelength_nm"] / SCAT_REF_NM) ** (-p["scat_b"])


def speed_of_light(p: dict) -> float:
    return C0 / float(refractive_index(p["water_fraction"]))


# ---------------------------------------------------------------------------- sampling
_CAP_GRID = (np.arange(4096) + 0.5) / 4096


def mua_over_musp_max(p: dict, length: float = LENGTH) -> float:
    """max_x mu_a(x) / mu_s' on a 4096-point grid with periodic images summed (an upper
    bound for every boundary condition, since images only add absorption)."""
    return float(mua_field(p, _CAP_GRID * length, length, True).max() / musp(p))


def _draw_tissue(rng, regime, length, centre_frac, kind=None):
    tr = PARAM_RANGES["train"]
    u = lambda lo_hi: float(rng.uniform(*lo_hi))
    p = {"regime": regime}
    if regime == "train":
        p["wavelength_nm"] = int(rng.choice(tr["wavelength_nm"]))
        bg, vb = tr["blood_fraction"], tr["vessel_blood_fraction"]
    elif regime == "ood":
        p["ood_kind"] = kind
        if kind == "wavelength":
            p["wavelength_nm"] = int(rng.choice(PARAM_RANGES["ood"]["wavelength_nm"]))
            bg, vb = tr["blood_fraction"], tr["vessel_blood_fraction"]
        else:
            p["wavelength_nm"] = int(rng.choice(tr["wavelength_nm"]))
            bg = PARAM_RANGES["ood"]["blood_fraction"]
            vb = PARAM_RANGES["ood"]["vessel_blood_fraction"]
    else:
        raise ValueError(regime)
    p["blood_fraction"] = u(bg)
    p["saturation"] = u(tr["saturation"])
    p["melanosome_fraction"] = u(tr["melanosome_fraction"])
    p["water_fraction"] = u(tr["water_fraction"])
    p["fat_fraction"] = u(tr["fat_fraction"])
    p["scat_a"] = u(tr["scat_a_m-1"])
    p["scat_b"] = u(tr["scat_b"])
    absorbers = []
    for _ in range(int(rng.integers(tr["n_absorbers"][0], tr["n_absorbers"][1] + 1))):
        w = u(tr["absorber_width_m"])
        a = {"shape": "gaussian" if rng.random() < 0.5 else "box",
             "centre": float(rng.uniform(*centre_frac) * length),
             "width": w,
             "blood_fraction": u(vb), "saturation": u(tr["saturation"])}
        if a["shape"] == "box":
            a["width"] = 0.5 * w                      # half-width
            # tanh edge 50 to 150 um: at least two internal cells (25 um at n = 512)
            a["edge"] = float(rng.uniform(*tr["absorber_edge_m"]))
        absorbers.append(a)
    p["absorbers"] = absorbers
    return p


def sample_tissue(rng: np.random.Generator, regime: str = "train", length: float = LENGTH,
                  centre_frac=(0.1, 0.9), ood_kind: str | None = None,
                  max_tries: int = 100000) -> dict:
    """Chromophores, scattering and 0 to 3 absorbers (shared with the chain), redrawn
    until max_x mu_a / mu_s' <= MUA_MUSP_CAP. For regime 'ood' the shift ('wavelength' or
    'blood_fraction', 50/50 unless ood_kind is given) is fixed BEFORE the rejection loop,
    so the cap does not change the mix of ood kinds; it only trims each kind's own
    distribution. The returned dict records n_rejected (cap rejections before acceptance)
    and the accepted ratio."""
    kind = None
    if regime == "ood":
        kind = ood_kind or ("wavelength" if rng.random() < 0.5 else "blood_fraction")
        if kind not in ("wavelength", "blood_fraction"):
            raise ValueError(kind)
    rejected = 0
    for _ in range(max_tries):
        p = _draw_tissue(rng, regime, length, centre_frac, kind)
        ratio = mua_over_musp_max(p, length)
        if ratio <= MUA_MUSP_CAP:
            p["n_rejected"] = rejected
            p["mua_over_musp_max_sampled"] = ratio
            return p
        rejected += 1
    raise RuntimeError(f"no tissue draw met the mu_a/mu_s' cap in {max_tries} tries")


def transport_time(p: dict) -> float:
    """t_tr = 1 / ((mu_a,bg + mu_s') c) with the background mu_a (the largest t_tr in the
    domain, since absorbers only add mu_a)."""
    return 1.0 / ((mua_background(p) + musp(p)) * speed_of_light(p))


def initial_state(p: dict, x: np.ndarray, bc: str, t: float = T_START,
                  length: float = LENGTH) -> np.ndarray:
    """Exact homogeneous-background Green's-function state at time t of the impulses in
    p["impulses"], with images for the boundary condition."""
    c = speed_of_light(p)
    D = c / (3.0 * (mua_background(p) + musp(p)))
    s2 = 4.0 * D * t
    out = np.zeros_like(x, dtype=float)
    for g in p["impulses"]:
        x0, A = g["centre"], g["amp"]
        for m in range(-N_IMAGES, N_IMAGES + 1):
            if bc == "periodic":
                out += A * np.exp(-(x - x0 - m * length) ** 2 / s2)
            else:
                sign = -1.0 if bc == "dirichlet" else 1.0
                out += A * (np.exp(-(x - x0 - 2 * m * length) ** 2 / s2)
                            + sign * np.exp(-(x + x0 - 2 * m * length) ** 2 / s2))
    return out


def exact_cell_average(p: dict, edges: np.ndarray, bc: str, t: float,
                       length: float = LENGTH) -> np.ndarray:
    """Exact cell averages at time t >= T_START of the homogeneous evolution of
    initial_state (verification): each image Gaussian spreads as the 1D Green's function,
    amplitude x (T_START / t)^(1/2) exp(-mu_a c (t - T_START))."""
    from scipy.special import erf
    c = speed_of_light(p)
    mu_a = mua_background(p)
    D = c / (3.0 * (mu_a + musp(p)))
    s = math.sqrt(4.0 * D * t)
    fac = math.sqrt(T_START / t) * math.exp(-mu_a * c * (t - T_START))
    xl, xr = edges[:-1], edges[1:]
    avg = lambda a: s * math.sqrt(math.pi) / 2 * (erf((xr - a) / s) - erf((xl - a) / s)) / (xr - xl)
    out = np.zeros(len(xl))
    for g in p["impulses"]:
        x0, A = g["centre"], g["amp"] * fac
        for m in range(-N_IMAGES, N_IMAGES + 1):
            if bc == "periodic":
                out += A * avg(x0 + m * length)
            else:
                sign = -1.0 if bc == "dirichlet" else 1.0
                out += A * (avg(x0 + 2 * m * length) + sign * avg(-x0 + 2 * m * length))
    return out


# ------------------------------------------------- radial geometry (d = 2, 3), round 3
N_MODES = 200      # eigenfunctions in the bounded-domain Green's function (see below)


def _tan_roots(m_count):
    """Positive roots of tan x = x, one in (m pi, m pi + pi/2) for m = 1 .. m_count."""
    from scipy.optimize import brentq
    f = lambda x: x * math.cos(x) - math.sin(x)
    return np.array([brentq(f, m * math.pi, m * math.pi + math.pi / 2 - 1e-12,
                            xtol=1e-15, rtol=1e-15, maxiter=200)
                     for m in range(1, m_count + 1)])


def radial_modes(d: int, bc: str, R_: float, M: int = N_MODES) -> np.ndarray:
    """Eigenvalues k_m of the radial Laplacian on [0, R) with the outer-wall condition:
    d = 2: J0(kR) = 0 (dirichlet), J1(kR) = 0 plus k = 0 (neumann);
    d = 3: kR = m pi (dirichlet), tan(kR) = kR plus k = 0 (neumann)."""
    from scipy.special import jn_zeros
    if d == 2:
        return (jn_zeros(0, M) if bc == "dirichlet"
                else np.concatenate([[0.0], jn_zeros(1, M - 1)])) / R_
    if d == 3:
        return ((np.arange(1, M + 1) * math.pi) if bc == "dirichlet"
                else np.concatenate([[0.0], _tan_roots(M - 1)])) / R_
    raise ValueError(d)


def radial_psi(d, k, r):
    """Eigenfunctions with psi(0) = 1: J0(kr) (d = 2), sin(kr)/(kr) (d = 3)."""
    from scipy.special import j0
    kr = np.multiply.outer(k, r)
    return j0(kr) if d == 2 else np.sinc(kr / math.pi)


def radial_norm(d, k, R_):
    """N = int_0^R psi^2 r^(d-1) dr."""
    from scipy.special import j0, j1
    k = np.asarray(k, float)
    if d == 2:
        return 0.5 * R_ ** 2 * (j0(k * R_) ** 2 + j1(k * R_) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (R_ / 2 - np.sin(2 * k * R_) / (4 * k)) / k ** 2
    return np.where(k == 0, R_ ** 3 / 3.0, v)


def radial_psi_integral(d, k, r):
    """int_0^r psi rho^(d-1) d rho, shape (len(k), len(r))."""
    from scipy.special import j1
    k = np.asarray(k, float)[:, None]
    r = np.asarray(r, float)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        if d == 2:
            v = r * j1(k * r) / k
            return np.where(k == 0, r ** 2 / 2.0, v)
        v = (np.sin(k * r) - k * r * np.cos(k * r)) / k ** 3
        return np.where(k == 0, r ** 3 / 3.0, v)


from functools import lru_cache


@lru_cache(maxsize=16)
def _mode_table(d, bc, length, M=None):
    """k, N and psi on the fixed 8193-point peak grid for one geometry and wall (cached;
    the values do not depend on the trajectory). Arrays are read-only."""
    k = radial_modes(d, bc, length, N_MODES if M is None else M)
    Nm = radial_norm(d, k, length)
    psi_fix = radial_psi(d, k, np.linspace(0.0, length, 8193))
    for a in (k, Nm, psi_fix):
        a.setflags(write=False)
    return k, Nm, psi_fix


def _radial_coeffs(p, bc, d, t, length):
    """Per-impulse mode weights at time t after release, normalised so each source alone
    peaks at its amplitude A at T_START. Returns (k, weights (n_imp, M))."""
    c = speed_of_light(p)
    D = c / (3.0 * (mua_background(p) + musp(p)))
    k, Nm, psi_fix = _mode_table(d, bc, float(length))
    W = []
    for g in p["impulses"]:
        a = radial_psi(d, k, np.array([g["radius"]]))[:, 0] / Nm
        a0 = a * np.exp(-D * k ** 2 * T_START)
        peak = float(np.abs(a0 @ psi_fix).max())
        W.append(g["amp"] / peak * a * np.exp(-D * k ** 2 * t))
    return k, np.array(W)


def radial_initial_state(p, r, bc, d, t=T_START, length=LENGTH):
    """d = 2, 3: exact state at time t of shell impulses (radius g["radius"], released at
    t = 0) in the bounded homogeneous background cylinder or sphere with the outer-wall
    condition bc, by the eigenfunction expansion
        phi(r, t) = sum_j s_j sum_m psi_m(r_j) psi_m(r) exp(-D k_m^2 t) / N_m
    (the common factor exp(-mu_a c t) is absorbed into s_j, as in d = 1), s_j chosen so
    each source alone peaks at A_j at T_START (peak over a fixed 8193-point grid).
    N_MODES = 200 modes: the dropped terms carry exp(-D k^2 T_START) < exp(-6e3) for the
    smallest D T_START in the training box (1.2e-6 m^2), i.e. none."""
    k, W = _radial_coeffs(p, bc, d, t, length)
    return W.sum(0) @ radial_psi(d, k, r)


def radial_exact_cell_average(p, edges, bc, d, t, length=LENGTH):
    """Exact volume averages over [edges_i, edges_i+1] at time t of the homogeneous
    evolution of radial_initial_state, including exp(-mu_a c (t - T_START))."""
    k, W = _radial_coeffs(p, bc, d, t, length)
    I = radial_psi_integral(d, k, edges)
    vol = (edges[1:] ** d - edges[:-1] ** d) / d
    fac = math.exp(-mua_background(p) * speed_of_light(p) * (t - T_START))
    return fac * (W.sum(0) @ (I[:, 1:] - I[:, :-1])) / vol


class LightSolver(common.Solver):
    name = "light"
    channels = (common.CH_LIGHT,)
    order = 2.0

    def __init__(self, dim: int = 1):
        self.dim = int(dim)

    def sample_params(self, rng, regime="train", dim=None):
        dim = self.dim if dim is None else int(dim)
        p = sample_tissue(rng, regime)
        tr = PARAM_RANGES["train"]
        n_imp = int(rng.integers(tr["n_impulses"][0], tr["n_impulses"][1] + 1))
        p["impulses"] = [{"centre": float(rng.uniform(*tr["impulse_centre_frac"]) * LENGTH),
                          "amp": float(rng.uniform(*tr["impulse_amp"]))}
                         for _ in range(n_imp)]
        if dim > 1:
            # radial geometry: impulse shells; the d = 1 centre draw (uniform in
            # [0.2, 0.8] L) is mapped linearly to a radius uniform in [0, 0.8] L
            lo, hi = tr["impulse_centre_frac"]
            for g in p["impulses"]:
                g["radius"] = R.radius_from_frac((g.pop("centre") / LENGTH - lo) / (hi - lo),
                                                 LENGTH)
        p["dim"] = dim
        return p

    # ------------------------------------------------------------------ assembly
    def _assemble(self, params_list, grid, bc):
        x, per = grid.rc, bc == "periodic"
        mu = np.stack([mua_field(p, x, grid.L, per) for p in params_list])
        c = np.array([speed_of_light(p) for p in params_list])[:, None]
        msp = np.array([musp(p) for p in params_list])[:, None]
        D = c / (3.0 * (mu + msp))
        if grid.d == 1:
            phi0 = np.stack([initial_state(p, x, bc, T_START, grid.L) for p in params_list])
        else:
            phi0 = np.stack([radial_initial_state(p, x, bc, grid.d, T_START, grid.L)
                             for p in params_list])
        return dict(kf=R.face_mean(D, grid, bc), absorb=mu * c), phi0, mu, D, c

    def simulate_batch(self, params_list, bc, n_frames, rng=None, n_internal=512):
        return R.split_by_dim(params_list, self.dim,
                              lambda ps, d: self._simulate(ps, bc, n_frames, n_internal, d))

    def _simulate(self, params_list, bc, n_frames, n_internal, d):
        grid = R.RadialGrid(n_internal, LENGTH, d)
        grid.check_bc(bc, "scalar")
        P, phi0, mu, D, c = self._assemble(params_list, grid, bc)
        dts = R.diffusion_dt(grid, P["kf"], bc, P["absorb"], len(params_list))
        nsub = R.n_substeps(DT_FRAME, dts)

        def rhs(t, y, Q):
            return (R.diffusion(y[0], grid, Q["kf"], bc) - Q["absorb"] * y[0],)

        rec = lambda y: grid.downsample(y[0], common.N_OUT)
        out = R.integrate(rhs, (phi0,), P, DT_FRAME, nsub, n_frames, rec, t_start=T_START)
        trajs = []
        for b, p in enumerate(params_list):
            f = np.zeros((n_frames, common.N_CH, common.N_OUT), np.float32)
            f[:, common.CH_LIGHT] = out[b]
            q = dict(p)
            ttr = transport_time(p)
            ltr = 1.0 / (mua_background(p) + musp(p))
            Dbg = speed_of_light(p) * ltr / 3.0
            q.update({"dt": float(DT_FRAME / nsub[b]), "n_internal": n_internal,
                      "t_start_s": T_START, "t_end_s": T_START + (n_frames - 1) * DT_FRAME,
                      "mua_background_m-1": float(mua_background(p)),
                      "mua_max_m-1": float(mu[b].max()), "musp_m-1": float(musp(p)),
                      "c_m_s": float(c[b, 0]), "D_max_m2_s": float(D[b].max()),
                      "mua_over_musp_max": float(mu[b].max() / musp(p)),
                      "mua_over_musp_cap": MUA_MUSP_CAP,
                      "t_tr_s": ttr, "start_over_t_tr": T_START / ttr,
                      "init_width_over_l_tr": math.sqrt(4 * Dbg * T_START) / ltr,
                      "phi_init_max": float(phi0[b].max())})
            trajs.append(common.Trajectory(f, self.name, q, DT_FRAME, LENGTH, bc,
                                           self.channels))
        return trajs

    def simulate(self, params, bc, n_frames, rng=None, n_internal=512):
        return self.simulate_batch([params], bc, n_frames, rng, n_internal)[0]

    # ---------------------------------------------------------------- verification
    VERIFY_PARAMS = {"wavelength_nm": 700, "blood_fraction": 0.02, "saturation": 0.8,
                     "melanosome_fraction": 0.0, "water_fraction": 0.7,
                     "fat_fraction": 0.1, "scat_a": SCAT_A_SOFT, "scat_b": SCAT_B_SOFT,
                     "absorbers": []}
    # scat_a, scat_b: musp_powerlaw_soft (JSON); fractions: modelling choice

    def green_check(self, d: int, n_list=(128, 256, 512)) -> dict:
        p = self.VERIFY_PARAMS
        mu_a = mua_background(p); c = speed_of_light(p)
        D = c / (3.0 * (mu_a + musp(p)))
        L = LENGTH
        w0 = L / 20.0                         # 1/e radius sqrt(4 D t0) at the start
        t0 = w0 ** 2 / (4 * D); t1 = 4 * t0
        x0 = L / 2 if d == 1 else 0.0

        def exact(r, t):
            return c * (4 * math.pi * D * t) ** (-d / 2) * np.exp(-r ** 2 / (4 * D * t)
                                                                   - mu_a * c * t)
        errs = []
        for n in n_list:
            g = R.RadialGrid(n, L, d)
            r = np.abs(g.rc - x0)
            kf = np.full((1, n + 1), D); absorb = np.full((1, n), mu_a * c)
            dt = R.diffusion_dt(g, kf, "neumann", absorb, 1)
            ns = R.n_substeps(t1 - t0, dt)
            rhs = lambda t, y, Q: (R.diffusion(y[0], g, kf, "neumann") - absorb * y[0],)
            out = R.integrate(rhs, (exact(r, t0)[None],), {}, t1 - t0, ns, 2,
                              lambda y: y[0].copy(), t_start=t0)
            num, ex = out[0, -1], exact(r, t1)
            errs.append(float(np.sqrt(np.sum(g.V * (num - ex) ** 2) / np.sum(g.V * ex ** 2))))
        rep = common.convergence_report(self.name, f"green_function_d{d}", n_list, errs,
                                        self.order)
        rep.update({"d": d, "norm": "relative L2, volume weighted, at t1",
                    "t0_s": t0, "t1_s": t1, "D_m2_s": D, "mu_a_m-1": mu_a,
                    "c_m_s": c, "mu_a_c_(t1-t0)": mu_a * c * (t1 - t0),
                    "domain_m": L, "bc": "neumann (walls at least 5 final 1/e radii from the source)"})
        return rep

    def window_check(self, bc: str, n_list=(128, 256, 512)) -> dict:
        """Production path (simulate_batch, homogeneous medium) over the full record:
        last frame against exact cell averages of the image-sum Green's state."""
        p = dict(self.VERIFY_PARAMS)
        p["wavelength_nm"] = 800
        p["impulses"] = [{"centre": 0.3 * LENGTH, "amp": 1.0},
                         {"centre": 0.62 * LENGTH, "amp": 0.6},
                         {"centre": 0.8 * LENGTH, "amp": 0.8}]
        t_last = T_START + 63 * DT_FRAME
        edges = np.linspace(0.0, LENGTH, common.N_OUT + 1)
        ex = exact_cell_average(p, edges, bc, t_last)
        errs = []
        for n in n_list:
            tr = self.simulate_batch([dict(p)], bc, 64, None, n)[0]
            num = tr.fields[-1, common.CH_LIGHT].astype(float)
            errs.append(float(np.sqrt(np.sum((num - ex) ** 2) / np.sum(ex ** 2))))
        rep = common.convergence_report(self.name, f"record_window_{bc}", n_list, errs,
                                        self.order)
        rep.update({"bc": bc, "t_start_s": T_START, "t_last_s": t_last,
                    "norm": "relative L2 on the 128 output cells (exact cell averages)",
                    "note": "float32 output limits the floor to about 1e-7"})
        return rep

    def verify(self, n_list=(128, 256, 512)) -> dict:
        cases = {f"d{d}": self.green_check(d, n_list) for d in (1, 2, 3)}
        for bc in ("periodic", "dirichlet", "neumann"):
            cases[f"window_{bc}"] = self.window_check(bc, n_list)
        main = dict(cases["d1"])
        main["check"] = ("Green's function, plane/line/point source, d = 1, 2, 3; record "
                         "window (production path) for each bc")
        main["cases"] = cases
        main["passed"] = all(c["passed"] for c in cases.values())
        main["time_window"] = {"T_START_s": T_START, "T_END_s": T_END, "DT_FRAME_s": DT_FRAME,
                               "t_tr_max_s": T_TR_MAX, "musp_min_m-1": MUSP_MIN,
                               "c_min_m_s": C_MIN, "D_ref_m2_s": D_REF,
                               "spread_ref_start_m": math.sqrt(4 * D_REF * T_START),
                               "spread_ref_end_m": math.sqrt(4 * D_REF * T_END)}
        return main
