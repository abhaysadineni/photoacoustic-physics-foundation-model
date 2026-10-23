"""Linear acoustics on a d-dimensional Cartesian grid (d = 2, 3), channel 2, as
pfm/solvers/sound.py (SoundUniform and SoundLayered; no lossy variant in round 4).

    dp/dt = -rho c^2 div u,   du/dt = -(1/rho) grad p

staggered grid (grid.py), RK4, dt = 0.5 h / c_max. Only p is exposed. Default internal grid
SOUND_INTERNAL: 256^2 (d = 2), 96^3 (d = 3), not the common 128^2 / 64^3 (see below).

Frame interval, by the 1D rule (a pulse in the reference medium, c_ref = c_soft.value =
1561 m/s, moves about one output cell per frame): DT_FRAME = (L / n_out) / c_ref
    d = 2 (0.2 mm cells): 128.12 ns, 32 frames = 3.97 us;
    d = 3 (0.4 mm cells): 256.25 ns, 24 frames = 5.89 us   (1D: 64.06 ns, 4.04 us).

Initial condition (sound.sample_pulses, generalised): 1 to 3 Gaussian pressure pulses
exp(-|x - c|^2 / w^2), amplitude 0.2 to 1 with random sign (as 1D). Pulse width (changed 21 Oct
2026, before any grid data existed, PLAN.md section 23): w uniform in PULSE_WIDTH_CELLS = [3, 10]
OUTPUT cells, i.e. the 1D range [0.3, 1.0] mm = [3, 10] cells of 0.1 mm expressed in cells:
d = 2 (0.2 mm cells) w in [0.6, 2.0] mm, d = 3 (0.4 mm cells) w in [1.2, 4.0] mm. Reason
(tests2d/resolution_sound.py, results/verification<d>d/sound_resolution_d<d>.json): with the 1D
millimetre range the stored pressure of a launched 0.3 mm pulse carries a discretisation error
of 79 per cent at the default 64^3 internal grid and still 15 per cent at 192^3 (d = 3), against
about 1 per cent in 1D; the error scales roughly as h^2 X / w^3 (X the distance travelled, one
output cell per frame), so the 1D accuracy is recovered by keeping w in output cells as in 1D.
The 3D and 2D sound claims therefore cover pulses of 3 to 10 output cells, not the 1D
millimetre range; each pulse is at rest (u0 = 0) with probability 1/3, otherwise launched along a
random unit vector n (uniform on the circle or sphere; the 1D left/right choice becomes a
direction, modelling choice): u0 = n p0 / Z, Z = rho c at the faces. Centre fraction u_a
uniform in [0, 1] per axis, mapped as sound.pulse_centre: u L (periodic) or (0.2 + 0.6 u) L
(walls). With walls p0 and u0 are multiplied by the product of C2 tapers of width 1 mm
(sound.WALL_TAPER). A launched Gaussian blob is not an exact plane wave; it travels mostly
along n and diffracts.

SoundUniform: c, rho from sound.PARAM_RANGES_UNIFORM (train: c_soft.range, rho_soft.range;
ood: the 1D 5 to 15 per cent bands beyond each end, side random per quantity).

SoundLayered: 2 to 4 sourced materials (sound.MATERIALS_TRAIN / MATERIALS_OOD, adjacent
materials different), arranged either as BANDS or as INCLUSIONS (probability 1/2 each,
modelling choice):
  bands       planar layers normal to a random grid axis; interfaces drawn as in 1D
              (uniform in [0.15, 0.85] L, at least 1 mm apart, rounded to an output-cell
              face so a sharp interface lies on a face of every internal grid), each sharp
              (probability 1/2) or a tanh blend of width interface_width_m = 0.1 to 0.5 mm.
              For periodic boundaries the wrap adds one more interface, as in 1D.
  inclusions  a background of the first material and 1 to 3 inclusions (the remaining
              materials, each different from the one before it) shaped as discs (d = 2) or
              balls (d = 3) of radius uniform in [1, 3] mm, centre coordinates uniform in
              [0.25, 0.75] L (modelling choices), with a tanh edge of width uniform in
              interface_width_m; later inclusions overwrite earlier ones where they
              overlap. Inclusion edges are always blended: a sharp curved interface would
              be a staircase on a Cartesian grid. Periodic boundaries use the distance to
              the nearest periodic image.
K = rho c^2 at cells; face density the arithmetic mean of neighbouring cells.

Each trajectory records the discrete energy ratio E(last) / E(0), which must be <= 1.

Exact checks: (a) plane wave along the grid diagonal (1, 1)/sqrt 2 or body diagonal
(1, 1, 1)/sqrt 3 on a periodic box, which returns to its initial state after
T = L / (sqrt(d) c); (b) radially symmetric Gaussian pulse p0 = exp(-r^2/s^2), u0 = 0,
against the Hankel-transform reference (d = 2, pfm/solvers/chain.py quadrature) or the
exact r p formula (d = 3, chain.free_gauss_exact). Order 2.
"""
from __future__ import annotations

import math

import numpy as np

from pfm.solvers import sound as SM

from .. import common as C
from . import grid as G
from . import references as REF

LENGTH = C.LENGTH
C_REF = SM.C_REF
PARAM_RANGES_UNIFORM = SM.PARAM_RANGES_UNIFORM
PARAM_RANGES_LAYERED = SM.PARAM_RANGES_LAYERED
INCLUSION_RADIUS_M = (1.0e-3, 3.0e-3)       # modelling choice
INCLUSION_CENTRE_FRAC = (0.25, 0.75)        # modelling choice
PULSE_WIDTH_CELLS = (3.0, 10.0)             # the 1D [0.3, 1.0] mm in 0.1 mm output cells
# Internal grid of the two sound systems (21 Oct 2026, before any grid data; PLAN.md section 23):
# finer than the common default (128^2, 64^3) so that the stored pressure meets about the 1D
# accuracy (results/verification<d>d/sound_resolution_d<d>.json; worst case, a launched pulse of
# the minimum width 3 output cells: d = 2 at 256^2 and d = 3 at 96^3, see section 23).
SOUND_INTERNAL = {2: 256, 3: 96}


def dt_frame(d):
    return (LENGTH / C.GEOM[d]["n_out"]) / C_REF


def _unit(rng, d):
    if d == 2:
        t = float(rng.uniform(0, 2 * math.pi))
        return [math.cos(t), math.sin(t)]
    v = rng.normal(size=3)
    return [float(x) for x in v / np.linalg.norm(v)]


def sample_pulses(rng, d):
    out = []
    for _ in range(int(rng.integers(1, 4))):
        cell = LENGTH / C.GEOM[d]["n_out"]
        g = {"u_centre": [float(rng.uniform(0.0, 1.0)) for _ in range(d)],
             "width": float(rng.uniform(*PULSE_WIDTH_CELLS) * cell),
             "amp": float(rng.uniform(0.2, 1.0) * rng.choice([-1.0, 1.0]))}
        g["direction"] = None if int(rng.integers(-1, 2)) == 0 else _unit(rng, d)
        out.append(g)
    return out


def pulse_centre(g, bc):
    return [u * LENGTH if bc == "periodic" else (0.2 + 0.6 * u) * LENGTH for u in g["u_centre"]]


def face_values(cell, d, bcs):
    return G.face_mean(cell, d, bcs)


def run_sound(c, rho, p0, u0, bcs, h, n_frames, dtf, record, nsub=None, src=None):
    """c, rho (B, *n); p0 (B, *n); u0 list of face arrays. Returns (frames, nsub, E0, EN)."""
    d = len(bcs)
    K = rho * c ** 2
    rho_f = G.face_mean(rho, d, bcs)
    irf = [1.0 / r for r in rho_f]
    if nsub is None:
        nsub = G.n_substeps(dtf, G.acoustic_dt(c.reshape(c.shape[0], -1).max(1), h))
    P = {"K": K}
    for a in range(d):
        P[f"irf{a}"] = irf[a]

    def rhs(t, y, Q):
        dp, du = G.acoustic_rhs(y[0], list(y[1:]), Q["K"], [Q[f"irf{a}"] for a in range(d)], bcs, h)
        return (dp, *du)
    y0 = (p0, *u0)
    out, st = G.integrate(rhs, y0, P, dtf, nsub, n_frames, record, return_state=True)
    E0 = G.acoustic_energy(p0, list(u0), K, rho_f, bcs, h)
    EN = G.acoustic_energy(st[0], list(st[1:]), K, rho_f, bcs, h)
    return out, np.asarray(nsub), E0, EN


class _SoundBase(C.Solver):
    channels = (C.CH_PRESSURE,)
    family = "sound"

    def __init__(self, dim=2):
        super().__init__(dim)
        self.dt_frame = dt_frame(self.dim)
        self.n_internal = SOUND_INTERNAL[self.dim]

    def media(self, params_list, X, bc):
        raise NotImplementedError

    def initial(self, params_list, bc, n, c, rho):
        d = self.dim
        bcs = (bc,) * d
        X = C.mesh(n, LENGTH, d)
        per = bc == "periodic"
        B = len(params_list)
        Zf = G.face_mean(rho * c, d, bcs)
        p0 = np.zeros((B,) + (n,) * d)
        u0 = [np.zeros(z.shape) for z in Zf]
        Xf = [C.mesh(n, LENGTH, d, stagger=a) for a in range(d)]
        for b, prm in enumerate(params_list):
            for g in prm["pulses"]:
                x0 = pulse_centre(g, bc)
                p0[b] += g["amp"] * G.gaussian(X, x0, g["width"], LENGTH, per)
                if g["direction"] is not None:
                    for a in range(d):
                        u0[a][b] += (g["direction"][a] * g["amp"]
                                     * G.gaussian(Xf[a], x0, g["width"], LENGTH, per)) / Zf[a][b]
        if not per:
            p0 *= G.taper(X, LENGTH, SM.WALL_TAPER)
            for a in range(d):
                u0[a] *= G.taper(Xf[a], LENGTH, SM.WALL_TAPER)
        for a in range(d):
            if per:
                u0[a][G.ix(d, a, n)] = u0[a][G.ix(d, a, 0)]
            if bc == "rigid":
                u0[a][G.ix(d, a, 0)] = 0.0
                u0[a][G.ix(d, a, n)] = 0.0
        return p0, u0

    def simulate_batch(self, params_list, bc, n_frames=None, rng=None, n_internal=None):
        n_frames, n = self._defaults(n_frames, n_internal)
        self.check_bc(bc)
        d = self.dim
        X = C.mesh(n, LENGTH, d)
        c, rho = self.media(params_list, X, bc, n)
        p0, u0 = self.initial(params_list, bc, n, c, rho)
        rec = lambda y: C.downsample(y[0], d, self.n_out)
        out, nsub, E0, EN = run_sound(c, rho, p0, u0, (bc,) * d, LENGTH / n, n_frames,
                                      self.dt_frame, rec)
        trajs = []
        for b, prm in enumerate(params_list):
            f = np.zeros((n_frames, C.N_CH) + (self.n_out,) * d, np.float32)
            f[:, C.CH_PRESSURE] = out[b]
            q = dict(prm)
            q.update({"dt": float(self.dt_frame / nsub[b]), "n_internal": n,
                      "energy_ratio": float(EN[b] / E0[b]),
                      "c_min_m_s": float(c[b].min()), "c_max_m_s": float(c[b].max())})
            trajs.append(C.Trajectory(f, self.name, q, self.dt_frame, LENGTH, bc,
                                      self.channels, d))
        return trajs

    # ----------------------------------------------------------- verification
    def plane_wave_check(self, n_list=None, kappa=2.0):
        """p0 = exp(kappa (cos(2 pi (s - s0)/L) - 1)), s = sum_a x_a, travelling along the
        (body) diagonal n = (1, ..., 1)/sqrt(d) at c; u0 = n p0 / Z at the faces. After
        T = L / (sqrt(d) c) the exact solution equals the initial state."""
        d = self.dim
        n_list = n_list or C.default_n_list(d)
        c, rho = C_REF, float(SM.J.get("rho_soft"))
        L = LENGTH
        s0 = 0.3 * L
        F = lambda s: np.exp(kappa * (np.cos(2 * math.pi * (s - s0) / L) - 1.0))
        T_end = L / (math.sqrt(d) * c)
        errs, steps, eratio = [], [], []
        for n in n_list:
            X = C.mesh(n, L, d)
            bcs = ("periodic",) * d
            p0 = np.broadcast_to(F(sum(X)), (n,) * d)[None].copy()
            u0 = []
            for a in range(d):
                Xf = C.mesh(n, L, d, stagger=a)
                shp = tuple(n + 1 if k == a else n for k in range(d))
                u0.append(np.broadcast_to(F(sum(Xf)), shp)[None] / (math.sqrt(d) * rho * c))
            cc = np.full((1,) + (n,) * d, c); rr = np.full((1,) + (n,) * d, rho)
            out, ns, E0, EN = run_sound(cc, rr, p0, [x.copy() for x in u0], bcs, L / n, 2,
                                        T_end, lambda y: y[0].copy())
            pN = out[0, -1]
            errs.append(float(np.sqrt(np.sum((pN - p0[0]) ** 2) / np.sum(p0[0] ** 2))))
            steps.append(int(ns[0])); eratio.append(float(EN[0] / E0[0]))
        rep = C.convergence_report(self.name, f"plane_wave_diagonal_d{d}", n_list, errs, 2.0)
        rep.update({"d": d, "direction": [1 / math.sqrt(d)] * d, "kappa": kappa,
                    "c_m_s": c, "rho_kg_m3": rho, "t_end_s": T_end, "rk4_steps": steps,
                    "energy_ratio": eratio,
                    "norm": "relative L2 of p after one period (exact: initial state)"})
        return rep

    def radial_pulse_check(self, n_list=None):
        d = self.dim
        n_list = n_list or C.default_n_list(d)
        c, rho = C_REF, float(SM.J.get("rho_soft"))
        L = LENGTH
        s = L / 10.0
        t1 = 0.3 * L / c
        x0 = [0.5 * L] * d
        prof, qc = REF.radial_profile(d, "free", t1, s, c)
        errs = []
        for n in n_list:
            X = C.mesh(n, L, d)
            bcs = ("periodic",) * d
            p0 = np.broadcast_to(REF.periodic_sum(lambda r: np.exp(-(r / s) ** 2), X, x0, L),
                                 (n,) * d)[None].copy()
            u0 = [np.zeros((1,) + tuple(n + 1 if k == a else n for k in range(d))) for a in range(d)]
            cc = np.full((1,) + (n,) * d, c); rr = np.full((1,) + (n,) * d, rho)
            out, _, _, _ = run_sound(cc, rr, p0, u0, bcs, L / n, 2, t1, lambda y: y[0].copy())
            ex = np.broadcast_to(REF.periodic_sum(prof, X, x0, L), (n,) * d)
            num = out[0, -1]
            errs.append(float(np.sqrt(np.sum((num - ex) ** 2) / np.sum(ex ** 2))))
        rep = C.convergence_report(self.name, f"radial_gaussian_pulse_d{d}", n_list, errs, 2.0)
        rep.update({"d": d, "s_m": s, "t_s": t1, "c_t_over_L": 0.3, "c_m_s": c,
                    "reference": REF.REFERENCE_NAME[("free", d)], "reference_check": qc,
                    "bc": "periodic; the exact solution is summed over the 3^d nearest images",
                    "norm": "relative L2 at t (point values)"})
        return rep

    def verify(self, n_list=None):
        cases = {"plane_wave": self.plane_wave_check(n_list),
                 "radial_pulse": self.radial_pulse_check(n_list)}
        main = dict(cases["plane_wave"])
        main["system"] = self.name
        main["check"] = "plane wave along the diagonal; radially symmetric Gaussian pulse"
        main["cases"] = cases
        main["passed"] = all(v["passed"] for v in cases.values())
        main["dt_frame_s"] = self.dt_frame
        return main


class SoundUniform(_SoundBase):
    name = "sound_uniform"

    def sample_params(self, rng, regime="train"):
        d = self.dim
        if regime == "train":
            c = float(rng.uniform(*SM.C_RANGE)); rho = float(rng.uniform(*SM.RHO_RANGE))
            return {"regime": regime, "dim": d, "c": c, "rho": rho, "pulses": sample_pulses(rng, d)}
        c, cs = SM._draw_ood(rng, PARAM_RANGES_UNIFORM["ood"]["c_m_s"])
        rho, rs = SM._draw_ood(rng, PARAM_RANGES_UNIFORM["ood"]["rho_kg_m3"])
        return {"regime": regime, "dim": d, "c": c, "rho": rho, "ood_side": [cs, rs],
                "pulses": sample_pulses(rng, d)}

    def media(self, params_list, X, bc, n):
        shp = (len(params_list),) + (n,) * self.dim
        c = np.array([p["c"] for p in params_list]).reshape((-1,) + (1,) * self.dim) * np.ones(shp)
        rho = np.array([p["rho"] for p in params_list]).reshape((-1,) + (1,) * self.dim) * np.ones(shp)
        return c, rho


class SoundLayered(_SoundBase):
    name = "sound_layered"

    def sample_params(self, rng, regime="train"):
        d = self.dim
        mats = SM.MATERIALS_TRAIN if regime == "train" else SM.MATERIALS_OOD
        rg = PARAM_RANGES_LAYERED["train"]
        names = list(mats)
        nl = int(rng.integers(rg["n_layers"][0], rg["n_layers"][1] + 1))
        layers = [str(rng.choice(names))]
        while len(layers) < nl:
            layers.append(str(rng.choice([x for x in names if x != layers[-1]])))
        p = {"regime": regime, "dim": d, "layers": layers,
             "media": {m: {"c": mats[m][0], "rho": mats[m][1]} for m in sorted(set(layers))},
             "pulses": None}
        if rng.random() < 0.5:
            p["arrangement"] = "bands"
            p["axis"] = int(rng.integers(d))
            cell = LENGTH / self.n_out
            for _ in range(1000):
                xs = np.sort(rng.uniform(0.15, 0.85, nl - 1) * LENGTH)
                xs = np.round(xs / cell) * cell
                if nl == 2 or np.min(np.diff(xs)) >= 1e-3:
                    break
            ifs = []
            for x in xs:
                sharp = bool(rng.random() < 0.5)
                ifs.append({"x": float(x), "sharp": sharp,
                            "width": 0.0 if sharp else float(rng.uniform(*rg["interface_width_m"]))})
            p["interfaces"] = ifs
        else:
            p["arrangement"] = "inclusions"
            p["inclusions"] = [{"centre": [float(rng.uniform(*INCLUSION_CENTRE_FRAC) * LENGTH)
                                           for _ in range(d)],
                                "radius": float(rng.uniform(*INCLUSION_RADIUS_M)),
                                "width": float(rng.uniform(*rg["interface_width_m"]))}
                               for _ in range(nl - 1)]
        p["pulses"] = sample_pulses(rng, d)
        return p

    @staticmethod
    def profile(prm, X, length, periodic):
        d = len(X)
        shape = np.broadcast_shapes(*[x.shape for x in X])
        m0 = prm["media"][prm["layers"][0]]
        c = np.full(shape, m0["c"], float); rho = np.full(shape, m0["rho"], float)
        if prm["arrangement"] == "bands":
            x = np.broadcast_to(X[prm["axis"]], shape)
            for j, itf in enumerate(prm["interfaces"]):
                a, b = prm["media"][prm["layers"][j]], prm["media"][prm["layers"][j + 1]]
                hh = (x > itf["x"]).astype(float) if itf["sharp"] else \
                    0.5 * (1 + np.tanh((x - itf["x"]) / itf["width"]))
                c += hh * (b["c"] - a["c"]); rho += hh * (b["rho"] - a["rho"])
            return c, rho
        for j, inc in enumerate(prm["inclusions"]):
            m = prm["media"][prm["layers"][j + 1]]
            r2 = None
            for sft in G.image_shifts(d, periodic):
                q = G.dist2_point(X, inc["centre"], length, sft)
                r2 = q if r2 is None else np.minimum(r2, q)
            hh = 0.5 * (1 - np.tanh((np.sqrt(r2) - inc["radius"]) / inc["width"]))
            c = c * (1 - hh) + m["c"] * hh
            rho = rho * (1 - hh) + m["rho"] * hh
        return c, rho

    def media(self, params_list, X, bc, n):
        per = bc == "periodic"
        shp = (n,) * self.dim
        prof = [self.profile(p, X, LENGTH, per) for p in params_list]
        return (np.stack([np.broadcast_to(q[0], shp) for q in prof]),
                np.stack([np.broadcast_to(q[1], shp) for q in prof]))


class Sound2DUniform(SoundUniform):
    def __init__(self):
        super().__init__(2)


class Sound2DLayered(SoundLayered):
    def __init__(self):
        super().__init__(2)


class Sound3DUniform(SoundUniform):
    def __init__(self):
        super().__init__(3)


class Sound3DLayered(SoundLayered):
    def __init__(self):
        super().__init__(3)
