"""Consistency of the d-dimensional solvers with the 1D solvers (pfm/solvers, imported and
run unchanged): a problem uniform in every direction but x_0 must reproduce the 1D result.

Set-up per check: one training draw of the 1D solver, simulated by the 1D solver with
n_internal = 128 (cell 0.1 mm), and the same physics built from the 1D formulas on a
d-dimensional grid with 128 cells along x_0 (the same 0.1 mm cell) and uniform data in the
other directions:
    d = 2: a full 128 x 128 square (12.8 mm x 12.8 mm);
    d = 3: a 128 x 8 x 8 box (12.8 mm x 0.8 mm x 0.8 mm). The extent in the uniform
           directions does not enter a uniform solution; the short box keeps the 3D run
           cheap and uses the same code path.
Boundary condition along x_0: the 1D one. Transverse directions: periodic (the 1D walls
pressure_release and dirichlet would force the field to zero on transverse walls, which a
transverse-uniform field cannot satisfy); for the chain the light is zero-flux or periodic
transversally as in production. Fields are compared at the internal resolution on every
transverse line, at the frames where the 1D and d-dimensional records coincide in time:
    light, heat  frames 0 and last (the d-dimensional DT_FRAME is not a multiple of the 1D
                 one; both records end at the same physical time);
    sound, chain every d-dimensional frame k with 1D frame m k <= 63, m = DT_FRAME / 1D
                 DT_FRAME (m = 2 for d = 2, 4 for d = 3).
Two runs where the time step can differ: "own_dt" uses the d-dimensional substep rule
(the diffusion bound is about d times stricter, the acoustic rule may round differently),
so differences are RK4 time-discretisation differences; "matched_dt" (sound, chain) passes
the 1D step, so the arithmetic is the same and differences are round-off.
"""
from __future__ import annotations

import numpy as np

from pfm.solvers import chain as CH1
from pfm.solvers import heat as HM
from pfm.solvers import light as LM
from pfm.solvers import radial as R
from pfm.solvers import sound as SM

from .. import common as C
from . import chain as K2
from . import grid as G
from . import heat as H2
from . import light as L2
from . import sound as S2

N1 = 128
LENGTH = C.LENGTH


def box(d):
    return (N1, N1) if d == 2 else (N1, 8, 8)


def _bcast(v, shape):
    """1D array along x_0 (length n0 or n0 + 1) -> (1, *shape) uniform transversally."""
    s = [1] * len(shape)
    s[0] = v.shape[-1]
    return np.broadcast_to(v.reshape(s), (v.shape[-1],) + tuple(shape[1:]))[None].copy()


def _compare(f2, f1, frames2, frames1):
    """f2 (F2, *shape) internal fields; f1 (F1, 128) 1D. Max abs and rel difference over the
    compared frames and every transverse line, plus transverse non-uniformity."""
    a = f2[frames2]
    b = f1[frames1]
    d = a.ndim - 2
    lines = a.reshape(a.shape[0], a.shape[1], -1)
    diff = np.abs(lines - b[:, :, None])
    nonu = np.abs(lines - lines[:, :, :1]).max()
    scale = np.abs(b).max()
    return {"max_abs_diff": float(diff.max()), "max_rel_diff": float(diff.max() / scale),
            "transverse_nonuniformity_rel": float(nonu / scale), "scale": float(scale),
            "frames_compared_d": [int(x) for x in frames2], "frames_compared_1d": [int(x) for x in frames1],
            "dims_transverse": d}


def light(d, bc, seed=1):
    s1 = LM.LightSolver()
    rng = np.random.default_rng(seed)
    p = s1.sample_params(rng, "train")
    f1 = s1.simulate_batch([dict(p)], bc, 64, None, N1)[0].fields[:, C.CH_LIGHT].astype(float)
    x = C.centres(N1, LENGTH)
    per = bc == "periodic"
    shape = box(d)
    mu = _bcast(LM.mua_field(p, x, LENGTH, per), shape)
    phi0 = _bcast(LM.initial_state(p, x, bc), shape)
    bcs = (bc,) + ("periodic",) * (d - 1)
    F = C.GEOM[d]["n_frames"]
    out, nsub, _ = L2.run_light(mu, np.array([LM.speed_of_light(p)]), np.array([LM.musp(p)]),
                                phi0, bcs, LENGTH / N1, F, L2.dt_frame(d), lambda y: y[0].copy())
    rep = _compare(out[0], f1, [0, F - 1], [0, 63])
    rep.update({"system": "light", "bc": bc, "nsub_d": int(nsub[0])})
    return rep


def heat(d, bc, seed=2):
    s1 = HM.HeatSolver()
    rng = np.random.default_rng(seed)
    p = s1.sample_params(rng, "train")
    f1 = s1.simulate_batch([dict(p)], bc, 64, None, N1)[0].fields[:, C.CH_HEAT].astype(float)
    x = C.centres(N1, LENGTH)
    per = bc == "periodic"
    tap = R.smoothstep_taper(x, LENGTH, HM.WALL_TAPER) if bc == "dirichlet" else 1.0
    shape = box(d)
    T0 = sum(h["amp"] * R.gaussian(x, h["centre"], h["width"], LENGTH, per) for h in p["hot_spots"]) * tap
    Hm = max(1, len(p["heaters"]))
    S = np.zeros((1, Hm) + shape); ton = np.full((1, Hm), -1e30); toff = np.full((1, Hm), 1e30)
    for j, h in enumerate(p["heaters"]):
        S[0, j] = _bcast(h["K_per_frame"] / HM.DT_FRAME * R.gaussian(x, h["centre"], h["width"], LENGTH, per) * tap, shape)[0]
        if "t_on" in h:
            ton[0, j], toff[0, j] = h["t_on"], h["t_off"]
    bcs = (bc,) + ("periodic",) * (d - 1)
    F = C.GEOM[d]["n_frames"]
    out, nsub = H2.run_heat(_bcast(T0, shape), np.array([p["alpha"]]), S, ton, toff, bcs,
                            LENGTH / N1, F, H2.dt_frame(d), lambda y: y[0].copy())
    rep = _compare(out[0], f1, [0, F - 1], [0, 63])
    rep.update({"system": "heat", "bc": bc, "nsub_d": int(nsub[0]),
                "n_heaters": len(p["heaters"]),
                "switched": [("t_on" in h) for h in p["heaters"]]})
    return rep


def _sound_1d_initial(s1, p, bc):
    grid = R.RadialGrid(N1, LENGTH, 1)
    c, rho, _ = s1._media([p], grid, bc)
    Zf = R.face_mean(rho * c, grid, bc)
    per = bc == "periodic"
    p0 = np.zeros(N1); u0 = np.zeros(N1 + 1)
    for g in p["pulses"]:
        x0 = SM.pulse_centre(g, bc)
        p0 += g["amp"] * R.gaussian(grid.rc, x0, g["width"], LENGTH, per)
        u0 += g["direction"] * g["amp"] * R.gaussian(grid.rf, x0, g["width"], LENGTH, per) / Zf[0]
    if not per:
        p0 *= R.smoothstep_taper(grid.rc, LENGTH, SM.WALL_TAPER)
        u0 *= R.smoothstep_taper(grid.rf, LENGTH, SM.WALL_TAPER)
    if per:
        u0[-1] = u0[0]
    if bc == "rigid":
        u0[0] = u0[-1] = 0.0
    return c[0], rho[0], p0, u0


def sound(d, kind, bc, seed=3):
    s1 = SM.SoundUniform() if kind == "uniform" else SM.SoundLayered()
    rng = np.random.default_rng(seed)
    p = s1.sample_params(rng, "train")
    tr1 = s1.simulate_batch([dict(p)], bc, 64, None, N1)[0]
    f1 = tr1.fields[:, C.CH_PRESSURE].astype(float)
    c, rho, p0, u0 = _sound_1d_initial(s1, p, bc)
    shape = box(d)
    bcs = (bc,) + ("periodic",) * (d - 1)
    m = int(round(S2.dt_frame(d) / SM.DT_FRAME))
    F = min(C.GEOM[d]["n_frames"], 63 // m + 1)
    ui = [_bcast(u0, shape)] + [np.zeros((1,) + tuple(n + 1 if k == a else n for k, n in enumerate(shape)))
                                for a in range(1, d)]
    rep = {"system": f"sound_{kind}", "bc": bc, "frame_ratio": m}
    for tag, ns in (("own_dt", None), ("matched_dt", np.array([int(round(S2.dt_frame(d) / tr1.params["dt"]))]))):
        out, nsub, E0, EN = S2.run_sound(_bcast(c, shape), _bcast(rho, shape), _bcast(p0, shape),
                                         [u.copy() for u in ui], bcs, LENGTH / N1, F,
                                         S2.dt_frame(d), lambda y: y[0].copy(), nsub=ns)
        r = _compare(out[0], f1, list(range(F)), [m * k for k in range(F)])
        r["nsub_d"] = int(nsub[0]); r["dt_d_s"] = S2.dt_frame(d) / int(nsub[0])
        rep[tag] = r
    rep["dt_1d_s"] = tr1.params["dt"]
    if kind == "layered":
        rep["layers"] = p["layers"]; rep["interfaces"] = p["interfaces"]
    return rep


def chain(d, bc, seed=4):
    s1 = CH1.PhotoacousticChain(1)
    rng = np.random.default_rng(seed)
    p = s1.sample_params(rng, "train")
    tr1 = s1.simulate_batch([dict(p)], bc, 64, None, N1)[0]
    f1 = tr1.fields.astype(float)
    x = C.centres(N1, LENGTH)
    shape = box(d)
    mu = _bcast(LM.mua_field(p, x, LENGTH, bc == "periodic"), shape)
    phi_s, info = K2.steady_fluence(mu[0], LM.musp(p), bc, p["phi0"], LENGTH / N1)
    q = dict(p); q["t_pre_s"] = p["pre_frames"] * CH1.DT_FRAME
    P = K2.build_P([q], mu, phi_s[None])
    dtf = K2.dt_frame(d)
    m = int(round(dtf / CH1.DT_FRAME))
    F = min(C.GEOM[d]["n_frames"], 63 // m + 1)
    Kpre = int(np.ceil(q["t_pre_s"] / dtf - 1e-9))
    y0 = (np.zeros((1,) + shape), np.zeros((1,) + shape)) + tuple(
        np.zeros((1,) + tuple(n + 1 if k == a else n for k, n in enumerate(shape))) for a in range(d))
    bcs = (bc,) + ("periodic",) * (d - 1)
    rep = {"system": "chain", "bc": bc, "frame_ratio": m, "fluence_solve": info,
           "pre_frames_1d": p["pre_frames"], "frames_before_record_d": Kpre}
    phi_line = phi_s.reshape(N1, -1)
    rep["fluence_steady_max_rel_diff"] = None
    for tag, ns in (("own_dt", None), ("matched_dt", np.array([int(round(dtf / tr1.params["dt"]))]))):
        out, _, nsub = K2.run_coupled(P, y0, bcs, LENGTH / N1, F + Kpre, dtf, -Kpre * dtf,
                                      lambda y: np.stack([y[0], y[1]], 1), nsub=ns)
        out = out[0, Kpre:]
        tf = np.arange(F) * dtf
        E = CH1.envelope(tf, {"t_s": np.full((F, 1), P["t_s"][0, 0]),
                              "T_p": np.full((F, 1), P["T_p"][0, 0]), "const": np.zeros((F, 1))})[:, 0]
        light_f = E.reshape((-1,) + (1,) * d) * phi_s[None]
        r = {"ch0": _compare(light_f, f1[:, 0], list(range(F)), [m * k for k in range(F)]),
             "ch1": _compare(out[:, 0], f1[:, 1], list(range(F)), [m * k for k in range(F)]),
             "ch2": _compare(out[:, 1], f1[:, 2], list(range(F)), [m * k for k in range(F)]),
             "nsub_d": int(nsub[0])}
        r["max_rel_diff"] = max(r[f"ch{c}"]["max_rel_diff"] for c in range(3))
        rep[tag] = r
    rep["dt_1d_s"] = tr1.params["dt"]
    # steady fluence alone: 1D tridiagonal solve (chain.steady_fluence) against the sparse solve
    g1 = R.RadialGrid(N1, LENGTH, 1)
    ph1 = CH1.steady_fluence(g1, LM.mua_field(p, x, LENGTH, bc == "periodic")[None],
                             np.array([[LM.musp(p)]]), bc, np.array([p["phi0"]]))[0]
    rep["fluence_steady_max_rel_diff"] = float(np.abs(phi_line - ph1[:, None]).max() / np.abs(ph1).max())
    return rep


def run_all(d):
    out = {"dim": d, "box": list(box(d)), "n_1d": N1, "cases": {}}
    for bc in C.BCS_SCALAR:
        out["cases"][f"light_{bc}"] = light(d, bc)
        out["cases"][f"heat_{bc}"] = heat(d, bc)
    for bc in C.BCS_SOUND:
        out["cases"][f"sound_uniform_{bc}"] = sound(d, "uniform", bc)
        out["cases"][f"sound_layered_{bc}"] = sound(d, "layered", bc)
        out["cases"][f"chain_{bc}"] = chain(d, bc)
    return out
