"""Exact references for the radially symmetric acoustic checks (d = 2, 3), built only from
pfm/solvers/chain.py's reference functions (imported, unchanged).

d = 3: chain.free_gauss_exact(3, ...) and chain.duhamel_gauss_exact(3, ...), closed forms.
d = 2: chain.hankel_gauss_gl (composite Gauss-Legendre, 20000 nodes, on [0, k_max]) on a
dense radial grid of N_R = 4001 points on [0, R_MAX], R_MAX = c t + 6 s, interpolated by a
cubic spline (scipy CubicSpline, not-a-knot). Accuracy is checked at 40 radii against
chain.hankel_gauss (adaptive scipy quad, rel. tol 1e-10): the spline-versus-quad and the
Gauss-Legendre-versus-quad maximum differences, relative to the maximum of |profile|, are
returned and must be below 1e-9. Beyond R_MAX the profile is set to zero; the value at
R_MAX relative to the maximum is reported (domain of dependence: the data are below
exp(-36) of their peak there).
The periodic test boxes use the sum over the 3^d nearest periodic images of the
free-space profile; with c t + 6 s < 1.5 L farther images cannot contribute.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from pfm.solvers import chain as CH

from . import grid as G

N_R = 4001
REFERENCE_NAME = {
    ("free", 2): "Hankel transform int k P0(k) J0(kr) cos(ckt) dk (chain.hankel_gauss_gl on a "
                 "4001-point radial grid, cubic spline; checked against chain.hankel_gauss)",
    ("free", 3): "exact [(r-ct) p0(|r-ct|) + (r+ct) p0(r+ct)] / (2r) (chain.free_gauss_exact)",
    ("duhamel", 2): "Duhamel integral with the time integral done exactly, (1/c) int P0(k) "
                    "J0(kr) sin(ckt) dk (chain.hankel_gauss_gl, spline; checked against "
                    "chain.hankel_gauss)",
    ("duhamel", 3): "exact (s^2/(4rc)) [exp(-(r-ct)^2/s^2) - exp(-(r+ct)^2/s^2)] "
                    "(chain.duhamel_gauss_exact)",
}
_CACHE = {}


def radial_profile(d, kind, t, s, c):
    """Returns (f(r), check dict). f gives the free-space solution at radius r for p0 =
    exp(-r^2/s^2), u0 = 0 ('free') or per unit Gamma H0 for constant heating exp(-r^2/s^2)
    switched on at t = 0 ('duhamel')."""
    key = (d, kind, float(t), float(s), float(c))
    if key in _CACHE:
        return _CACHE[key]
    if d == 3:
        fn = (lambda r: CH.free_gauss_exact(3, r, t, s, c)) if kind == "free" else \
             (lambda r: CH.duhamel_gauss_exact(3, r, t, s, c))
        res = (fn, {"closed_form": True})
        _CACHE[key] = res
        return res
    rmax = c * t + 6.0 * s
    r = np.linspace(0.0, rmax, N_R)
    vals = np.concatenate([CH.hankel_gauss_gl(r[i:i + 250], t, s, c, kind)
                           for i in range(0, N_R, 250)])
    sp = CubicSpline(r, vals)
    rng = np.random.default_rng(7)
    rr = np.sort(rng.uniform(0.0, rmax, 40))
    q, qerr = CH.hankel_gauss(rr, t, s, c, kind)
    scale = float(np.abs(vals).max())
    chk = {"closed_form": False, "n_radii": N_R, "r_max_m": rmax,
           "spline_vs_quad_max_rel": float(np.abs(sp(rr) - q).max() / scale),
           "gauss_legendre_vs_quad_max_rel": float(
               np.abs(CH.hankel_gauss_gl(rr, t, s, c, kind) - q).max() / scale),
           "quad_error_estimate_max_rel": float(qerr.max() / scale),
           "value_at_r_max_rel": float(abs(vals[-1]) / scale)}
    chk["passed_1e-9"] = bool(max(chk["spline_vs_quad_max_rel"],
                                  chk["gauss_legendre_vs_quad_max_rel"]) < 1e-9)

    def fn(rq):
        rq = np.asarray(rq, float)
        out = np.zeros(rq.shape)
        m = rq <= rmax
        out[m] = sp(rq[m])
        return out
    res = (fn, chk)
    _CACHE[key] = res
    return res


def periodic_sum(fn, X, x0, length):
    out = 0.0
    for sft in G.image_shifts(len(X), True, 1):
        out = out + fn(np.sqrt(G.dist2_point(X, x0, length, sft)))
    return out
