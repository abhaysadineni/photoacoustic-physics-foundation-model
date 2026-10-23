"""Optical tissue draws on a d-dimensional box (d = 2, 3), shared by Light and the chain.

Every number comes from pfm/solvers/light.py, which reads params/tissue_params.json
through pfm/solvers/_tissue.py: the chromophore model (light.CHROMOPHORES, light.musp,
light.speed_of_light, light.mua_background), the parameter ranges (light.PARAM_RANGES,
train and ood, unchanged) and the validity cap light.MUA_MUSP_CAP = 0.3. The background
draw is the 1D draw (light._draw_tissue) term for term; only the absorber geometry is new.

Absorbers (0 to 3, light.PARAM_RANGES n_absorbers), each with whole-blood fraction B_v
(vessel_blood_fraction) and saturation S_v, replacing the background blood inside it as in
1D (light.mua_field: mu += clip(s, 0, 1) max(B_v - B, 0) mu_a,blood(S_v)):
  profile   "gaussian" exp(-r^2 / w^2), w the 1/e half-width in absorber_width_m, or "box"
            0.5 [tanh((r + w/2)/e) - tanh((r - w/2)/e)], full width w, edge e in
            absorber_edge_m: exactly the 1D profiles with r the distance to the absorber's
            core (probability 1/2 each, as 1D);
  geometry  (modelling choice, probability 1/2 each) "blob": a disc (d = 2) or ball
            (d = 3), r the distance to the centre; "vessel": a straight line-like vessel,
            r the distance to a segment of length l through the centre, l uniform in
            VESSEL_LENGTH_FRAC x L = [0.25, 0.5] L (modelling choice), direction uniform on
            the circle (d = 2) or sphere (d = 3). A finite segment rather than an infinite
            line keeps the field smooth across a periodic wrap at any orientation.
  centre    each coordinate uniform in [0.1, 0.9] L (the 1D centre_frac).
Periodic boundaries: the profile is summed over the 3^d nearest periodic images (the 1D
code sums 3 images).

Validity cap: max_x mu_a / mu_s' <= MUA_MUSP_CAP for every draw, imposed by rejecting the
whole draw (as light.sample_tissue), with periodic images summed (an upper bound for every
boundary condition). The maximum is decided exactly where possible: reject if background +
the largest single absorber peak exceeds the cap (each contribution is non-negative, so the
field reaches at least that value at that absorber's core); accept if background + the sum
of all peaks is within the cap. Otherwise (overlapping absorbers) the field is evaluated on
a 512^2 (d = 2) or 128^3 (d = 3) grid and at every absorber core point, and the largest of
these points and the grid maximum are each refined by a local Nelder-Mead maximisation
(position tolerance 1e-8 m); the largest value found is used. This finds the true maximum
unless two separate local maxima differ by less than the grid can resolve (not observed;
stated as the remaining approximation). The field actually used on the simulation grid is recorded per trajectory
("mua_over_musp_max").
"""
from __future__ import annotations

import math

import numpy as np

from pfm.solvers import light as LM

from . import grid as G

TR = LM.PARAM_RANGES["train"]
OOD = LM.PARAM_RANGES["ood"]
CAP = LM.MUA_MUSP_CAP
VESSEL_LENGTH_FRAC = (0.25, 0.5)          # modelling choice
CAP_GRID = {2: 512, 3: 128}
mua_background = LM.mua_background
musp = LM.musp
speed_of_light = LM.speed_of_light


def _unit_vector(rng, d):
    if d == 2:
        t = float(rng.uniform(0.0, 2 * math.pi))
        return [math.cos(t), math.sin(t)]
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    return [float(x) for x in v]


def _draw(rng, regime, d, length, centre_frac, kind):
    u = lambda ab: float(rng.uniform(*ab))
    p = {"regime": regime, "dim": d}
    if regime == "train":
        p["wavelength_nm"] = int(rng.choice(TR["wavelength_nm"]))
        bg, vb = TR["blood_fraction"], TR["vessel_blood_fraction"]
    elif regime == "ood":
        p["ood_kind"] = kind
        if kind == "wavelength":
            p["wavelength_nm"] = int(rng.choice(OOD["wavelength_nm"]))
            bg, vb = TR["blood_fraction"], TR["vessel_blood_fraction"]
        else:
            p["wavelength_nm"] = int(rng.choice(TR["wavelength_nm"]))
            bg, vb = OOD["blood_fraction"], OOD["vessel_blood_fraction"]
    else:
        raise ValueError(regime)
    p["blood_fraction"] = u(bg)
    p["saturation"] = u(TR["saturation"])
    p["melanosome_fraction"] = u(TR["melanosome_fraction"])
    p["water_fraction"] = u(TR["water_fraction"])
    p["fat_fraction"] = u(TR["fat_fraction"])
    p["scat_a"] = u(TR["scat_a_m-1"])
    p["scat_b"] = u(TR["scat_b"])
    absorbers = []
    for _ in range(int(rng.integers(TR["n_absorbers"][0], TR["n_absorbers"][1] + 1))):
        w = u(TR["absorber_width_m"])
        a = {"shape": "gaussian" if rng.random() < 0.5 else "box",
             "geometry": "blob" if rng.random() < 0.5 else "vessel",
             "centre": [float(rng.uniform(*centre_frac) * length) for _ in range(d)],
             "width": w, "blood_fraction": u(vb), "saturation": u(TR["saturation"])}
        if a["shape"] == "box":
            a["width"] = 0.5 * w                         # half-width, as 1D
            a["edge"] = float(rng.uniform(*TR["absorber_edge_m"]))
        if a["geometry"] == "vessel":
            a["direction"] = _unit_vector(rng, d)
            a["length"] = float(rng.uniform(*VESSEL_LENGTH_FRAC) * length)
        absorbers.append(a)
    p["absorbers"] = absorbers
    return p


def _dist2(a, X, length, shift):
    d = len(X)
    rel = [X[k] - a["centre"][k] - shift[k] * length for k in range(d)]
    if a["geometry"] == "blob":
        return sum(r * r for r in rel)
    e = a["direction"]
    s = sum(rel[k] * e[k] for k in range(d))
    s = np.clip(s, -0.5 * a["length"], 0.5 * a["length"])
    return sum((rel[k] - s * e[k]) ** 2 for k in range(d))


def _profile_r2(a, r2):
    if a["shape"] == "gaussian":
        return np.exp(-r2 / a["width"] ** 2)
    r = np.sqrt(r2)
    return 0.5 * (np.tanh((r + a["width"]) / a["edge"]) - np.tanh((r - a["width"]) / a["edge"]))


def absorber_shape(a, X, length, periodic):
    out = 0.0
    for s in G.image_shifts(len(X), periodic):
        out = out + _profile_r2(a, _dist2(a, X, length, s))
    return out


def _blood(ch, S):
    return S * ch["mua_HbO2"] + (1 - S) * ch["mua_Hb"]


def absorber_gain(p, a):
    ch = LM.CHROMOPHORES[p["wavelength_nm"]]
    return max(a["blood_fraction"] - p["blood_fraction"], 0.0) * _blood(ch, a["saturation"])


def mua_field(p, X, length, periodic):
    """mu_a on the coordinate arrays X (list of d broadcastable arrays), m^-1."""
    shape = np.broadcast_shapes(*[x.shape for x in X])
    mu = np.full(shape, mua_background(p), dtype=float)
    for a in p.get("absorbers", []):
        mu += np.clip(absorber_shape(a, X, length, periodic), 0.0, 1.0) * absorber_gain(p, a)
    return mu


def _peak(a):
    return 1.0 if a["shape"] == "gaussian" else math.tanh(a["width"] / a["edge"])


def mua_over_musp_max(p, d, length):
    bg = mua_background(p)
    peaks = [absorber_gain(p, a) * _peak(a) for a in p["absorbers"]]
    ms = musp(p)
    if (bg + (max(peaks) if peaks else 0.0)) / ms > CAP:
        return (bg + max(peaks)) / ms, "lower_bound"
    if (bg + sum(peaks)) / ms <= CAP:
        return (bg + sum(peaks)) / ms, "upper_bound"
    n = CAP_GRID[d]
    from ..common import mesh
    X = mesh(n, length, d)
    from scipy.optimize import minimize
    F = mua_field(p, X, length, True)
    i = np.unravel_index(int(np.argmax(F)), F.shape)
    starts = [[(k + 0.5) * length / n for k in i]] + [list(a["centre"]) for a in p["absorbers"]]
    f = lambda z: -float(mua_field(p, [np.array([c]) for c in z], length, True).ravel()[0])
    m = float(F.max())
    for z0 in starts:
        r = minimize(f, np.array(z0), method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-12 * abs(f(z0)), "maxiter": 4000})
        m = max(m, -float(r.fun), -f(z0))
    return m / ms, "grid_refined"


def sample_tissue(rng, regime, d, length, centre_frac=(0.1, 0.9), ood_kind=None,
                  max_tries=100000):
    """As light.sample_tissue (the ood kind is fixed before the rejection loop)."""
    kind = None
    if regime == "ood":
        kind = ood_kind or ("wavelength" if rng.random() < 0.5 else "blood_fraction")
        if kind not in ("wavelength", "blood_fraction"):
            raise ValueError(kind)
    rejected = 0
    for _ in range(max_tries):
        p = _draw(rng, regime, d, length, centre_frac, kind)
        ratio, how = mua_over_musp_max(p, d, length)
        if ratio <= CAP:
            p["n_rejected"] = rejected
            p["mua_over_musp_max_sampled"] = ratio
            p["cap_decided_by"] = how
            return p
        rejected += 1
    raise RuntimeError("no tissue draw met the mu_a/mu_s' cap")
