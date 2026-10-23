"""Shared numerics for the tissue solvers (light, heat, sound, chain).

Geometry
--------
Cell-centred finite volumes on r in [0, L) for d in {1, 2, 3}:
    n cells, dr = L / n, centres r_i = (i + 1/2) dr, faces r_{j} = j dr (j = 0..n).
    Face weights  A_j = r_j^(d-1)            (A = 1 for d = 1)
    Cell volumes  V_i = (r_{i+1}^d - r_i^d) / d   (exact; the solid-angle factor 2 pi or
                  4 pi cancels from every equation and is left out)
d = 1 is a slab with walls at 0 and L. d = 2 is a cylinder and d = 3 a sphere of radius L;
for them the face at r = 0 has A_0 = 0, so the flux through the axis or centre is zero and
the scheme is symmetric about r = 0 by construction.

Scalar operator
---------------
    (1/r^(d-1)) d/dr (r^(d-1) k dr u)  ->  (F_{i+1} - F_i) / V_i,
    F_j = A_j k_j (u_j - u_{j-1}) / dr  (two-point flux, second order),
with k_j the face coefficient. Boundary conditions by ghost cells at the outer wall (and,
for d = 1, also at the inner wall):
    periodic  (d = 1 only)   u_ghost = value on the other side
    dirichlet (odd)          u_ghost = 2 g - u_edge  (g = wall value, 0 unless given)
    neumann   (even)         u_ghost = u_edge        (zero flux)
The ghost value enters only through the wall flux, which is written out directly.

Acoustics
---------
Staggered grid: pressure p at cell centres, velocity u at faces.
    dp/dt = -K (1/r^(d-1)) d/dr (r^(d-1) u) + s(r, t),   K = rho c^2 at cells
    du/dt = -(1/rho_f) dp/dr  [+ nu d2u/dr2, d = 1 only]
rho_f is the arithmetic mean of the two neighbouring cell densities (the mass of the
face-centred control volume); this is the standard treatment of a sharp interface that
sits on a face. Wall conditions:
    pressure_release   p odd (p_ghost = -p_edge, p = 0 on the wall), u even; the wall
                       face velocity is a degree of freedom
    rigid              u odd (u = 0 on the wall face, held fixed), p even
    periodic (d = 1)   face n is face 0
For d = 2, 3 the face at r = 0 carries u = 0 (symmetry).
The semi-discrete system conserves the discrete energy
    E = sum_i p_i^2 V_i / (2 K_i) + sum_j rho_j u_j^2 w_j / 2,
w_j = A_j dr on interior faces and A_j dr / 2 on a pressure-release wall face (checked
analytically: the wall terms cancel with this weight), so RK4, which is weakly dissipative
on the imaginary axis, makes E non-increasing. acoustic_energy() returns it.

Time stepping
-------------
Classical four-stage Runge-Kutta. integrate() advances a batch in which every trajectory
takes its own number of equal substeps per output frame (so frames land exactly on
multiples of dt_frame and a trajectory's result does not depend on which others share the
batch). The batch is sorted by substep count so every substep updates a contiguous
leading block with one vectorised operation.

Stable step sizes
-----------------
Diffusion: RK4 is stable for real eigenvalues in [-2.785, 0]. The step is
2.5 / G, with G the Gershgorin bound of the assembled operator (row sums of absolute
values including absorption), computed per trajectory from the actual coefficients, so
it holds for every geometry and coefficient field.
Acoustics: dt = CFL_ACOUSTIC dr / c_max with CFL_ACOUSTIC = 0.5. The spectral radius of the
discrete operator is measured in wave_spectral_radius() (about 2 c / dr in all three
geometries, see the verification output); RK4 is stable on the imaginary axis up to
2.828, so the margin is about 2.8.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

BCS_SCALAR = ("periodic", "dirichlet", "neumann")
BCS_ACOUSTIC = ("periodic", "pressure_release", "rigid")
RK4_REAL_LIMIT = 2.5        # used fraction of RK4's real-axis stability interval (2.785)
CFL_ACOUSTIC = 0.5          # dt = CFL_ACOUSTIC * dr / c_max


class RadialGrid:
    def __init__(self, n: int, length: float, d: int, r0: float = 0.0):
        """r0 > 0 gives an annulus or shell [r0, r0 + length) with a second wall at r0
        (used only by the verification of line- and point-source light solutions)."""
        if d not in (1, 2, 3):
            raise ValueError("d must be 1, 2 or 3")
        self.n, self.L, self.d, self.r0 = int(n), float(length), int(d), float(r0)
        self.dr = self.L / self.n
        self.rc = self.r0 + (np.arange(self.n) + 0.5) * self.dr
        self.rf = self.r0 + np.arange(self.n + 1) * self.dr
        self.A = self.rf ** (d - 1) if d > 1 else np.ones(self.n + 1)
        self.V = (self.rf[1:] ** d - self.rf[:-1] ** d) / d
        # inner face is a wall unless the grid starts on the axis/centre of d = 2, 3
        self.axis = (d > 1 and self.r0 == 0.0)

    def check_bc(self, bc: str, family: str):
        allowed = BCS_SCALAR if family == "scalar" else BCS_ACOUSTIC
        if bc not in allowed:
            raise ValueError(f"bc {bc!r} not in {allowed}")
        if bc == "periodic" and self.d != 1:
            raise ValueError("periodic boundaries exist only for d = 1")

    def downsample(self, u: np.ndarray, n_out: int) -> np.ndarray:
        """Average n cells down to n_out. d = 1: plain mean (common.downsample). d = 2, 3:
        volume-weighted mean, the exact cell average of a finite-volume field."""
        f = self.n // n_out
        if f * n_out != self.n:
            raise ValueError(f"{self.n} is not a multiple of {n_out}")
        if self.d == 1:
            return u.reshape(*u.shape[:-1], n_out, f).mean(-1)
        V = self.V.reshape(n_out, f)
        return (u.reshape(*u.shape[:-1], n_out, f) * V).sum(-1) / V.sum(-1)


# ----------------------------------------------------------------------------- scalar
def face_mean(k_cells: np.ndarray, grid: RadialGrid, bc: str) -> np.ndarray:
    """Arithmetic mean of a cell coefficient at the n + 1 faces. Wall faces take the edge
    cell value; periodic wraps."""
    k = np.asarray(k_cells, float)
    kf = np.empty(k.shape[:-1] + (grid.n + 1,))
    kf[..., 1:-1] = 0.5 * (k[..., 1:] + k[..., :-1])
    if bc == "periodic":
        kf[..., 0] = kf[..., -1] = 0.5 * (k[..., 0] + k[..., -1])
    else:
        kf[..., 0], kf[..., -1] = k[..., 0], k[..., -1]
    return kf


def diffusion(u, grid: RadialGrid, kf, bc: str, g_lo=0.0, g_hi=0.0):
    """(1/r^(d-1)) d/dr (r^(d-1) k du/dr) on the batch u (..., n). kf: face coefficient
    (..., n + 1) or scalar. g_lo, g_hi: Dirichlet wall values (g_lo only for d = 1)."""
    n, dr = grid.n, grid.dr
    F = np.empty(u.shape[:-1] + (n + 1,))
    F[..., 1:n] = (u[..., 1:] - u[..., :-1]) / dr
    if bc == "periodic":
        F[..., n] = (u[..., 0] - u[..., -1]) / dr
        F[..., 0] = F[..., n]
    elif bc == "dirichlet":
        F[..., n] = 2.0 * (g_hi - u[..., -1]) / dr
        F[..., 0] = 2.0 * (u[..., 0] - g_lo) / dr
    elif bc == "neumann":
        F[..., n] = 0.0
        F[..., 0] = 0.0
    else:
        raise ValueError(bc)
    if grid.axis:
        F[..., 0] = 0.0
    F *= kf
    F *= grid.A
    return (F[..., 1:] - F[..., :-1]) / grid.V


def tridiag(grid: RadialGrid, kf, bc: str, batch: int):
    """Lower, main and upper diagonals of the homogeneous diffusion operator, by probing
    diffusion() with three comb vectors (so the matrix is exactly the time-stepping
    operator). For bc 'periodic' the two corner entries are not returned. Shapes (B, n)."""
    n = grid.n
    idx = np.arange(n)
    ys = [diffusion(np.broadcast_to((idx % 3 == m).astype(float), (batch, n)).copy(),
                    grid, kf, bc) for m in range(3)]
    diag = np.empty((batch, n)); lo = np.zeros((batch, n)); up = np.zeros((batch, n))
    for i_mod in range(3):
        rows = idx[idx % 3 == i_mod]
        diag[:, rows] = ys[i_mod][:, rows]
        r_up = rows[rows < n - 1]
        up[:, r_up] = ys[(i_mod + 1) % 3][:, r_up]
        r_lo = rows[rows > 0]
        lo[:, r_lo] = ys[(i_mod - 1) % 3][:, r_lo]
    return lo, diag, up


def diffusion_dt(grid: RadialGrid, kf, bc: str, absorb, batch: int) -> np.ndarray:
    """Per-trajectory RK4-stable step: RK4_REAL_LIMIT / Gershgorin bound of
    (diffusion - absorb). absorb: (B, n) or (B, 1) rate in s^-1."""
    lo, diag, up = tridiag(grid, kf, bc, batch)
    if bc == "periodic":   # corner entries equal the wrapped face couplings
        kfa = np.broadcast_to(np.asarray(kf, float), (batch, grid.n + 1))
        corner = kfa[:, 0] * grid.A[0] / (grid.dr ** 2)
        lo[:, 0] += corner; up[:, -1] += corner
    G = np.abs(lo) + np.abs(diag - np.broadcast_to(absorb, diag.shape)) + np.abs(up)
    return RK4_REAL_LIMIT / G.max(axis=1)


def solve_steady(grid: RadialGrid, kf, absorb, bc: str, g_lo=0.0, g_hi=0.0, source=None):
    """Solve diffusion(phi) - absorb * phi + source = 0 for each trajectory (non-periodic
    bc). kf (B, n+1), absorb (B, n), g_lo/g_hi scalars or (B,), source (B, n) or None."""
    if bc == "periodic":
        raise ValueError("steady solve uses dirichlet or neumann walls")
    absorb = np.atleast_2d(absorb)
    B, n = absorb.shape
    kf = np.broadcast_to(np.asarray(kf, float), (B, n + 1))
    lo, diag, up = tridiag(grid, kf, bc, B)
    g_lo = np.broadcast_to(np.asarray(g_lo, float), (B,))
    g_hi = np.broadcast_to(np.asarray(g_hi, float), (B,))
    const = diffusion(np.zeros((B, n)), grid, kf, bc, g_lo, g_hi)
    rhs = -const if source is None else -(const + source)
    out = np.empty((B, n))
    ab = np.zeros((3, n))
    for b in range(B):
        ab[0, 1:] = up[b, :-1]
        ab[1] = diag[b] - absorb[b]
        ab[2, :-1] = lo[b, 1:]
        out[b] = solve_banded((1, 1), ab, rhs[b])
    return out


# --------------------------------------------------------------------------- acoustics
def face_density(rho_cells: np.ndarray, grid: RadialGrid, bc: str) -> np.ndarray:
    return face_mean(rho_cells, grid, bc)


def acoustic_rhs(p, u, grid: RadialGrid, K, inv_rho_f, bc: str, nu=None, src=None):
    """Right-hand side of the staggered acoustic system. p (..., n), u (..., n + 1),
    K (..., n) or broadcastable, inv_rho_f (..., n + 1) or broadcastable, nu (..., 1) or
    None, src (..., n) or None (added to dp/dt)."""
    n, dr = grid.n, grid.dr
    dp = (grid.A[1:] * u[..., 1:] - grid.A[:-1] * u[..., :-1]) / grid.V
    dp *= -K
    if src is not None:
        dp += src
    g = np.empty(u.shape)
    g[..., 1:n] = (p[..., 1:] - p[..., :-1]) / dr
    if bc == "periodic":
        g[..., 0] = (p[..., 0] - p[..., -1]) / dr
        g[..., n] = g[..., 0]
    elif bc == "pressure_release":
        g[..., n] = -2.0 * p[..., -1] / dr
        g[..., 0] = 2.0 * p[..., 0] / dr
    elif bc == "rigid":
        g[..., n] = 0.0
        g[..., 0] = 0.0
    else:
        raise ValueError(bc)
    du = g
    du *= -inv_rho_f
    if grid.axis:
        du[..., 0] = 0.0
    if bc == "rigid":
        du[..., n] = 0.0
        du[..., 0] = 0.0
    if nu is not None and grid.d != 1:
        # Radial thermoviscous term nu grad(div u) (equal to nu lap u for irrotational
        # flow; reduces to nu u_xx in d = 1). div at cells is the same discrete divergence
        # as in dp/dt, grad at faces the same two-point difference, so by summation by
        # parts the term contributes -nu rho sum_i V_i div_i^2 <= 0 to the energy rate.
        # Walls: axis face u = 0 (fixed); pressure_release: div is odd about the wall
        # (p = 0 there), ghost div = -div_edge; rigid: u = 0 on the wall face (fixed).
        if not grid.axis:
            raise ValueError("radial viscous term requires a grid that starts on the axis")
        div = (grid.A[1:] * u[..., 1:] - grid.A[:-1] * u[..., :-1]) / grid.V
        gd = np.zeros(u.shape)
        gd[..., 1:n] = (div[..., 1:] - div[..., :-1]) / dr
        if bc == "pressure_release":
            gd[..., n] = -2.0 * div[..., -1] / dr
        elif bc != "rigid":
            raise ValueError(f"bc {bc!r} not allowed for d = {grid.d}")
        du += nu * gd
    elif nu is not None:
        lap = np.empty(u.shape)
        lap[..., 1:n] = (u[..., 2:] - 2.0 * u[..., 1:n] + u[..., :n - 1]) / dr ** 2
        if bc == "periodic":
            lap[..., 0] = (u[..., 1] - 2.0 * u[..., 0] + u[..., n - 1]) / dr ** 2
            lap[..., n] = lap[..., 0]
        elif bc == "pressure_release":
            lap[..., 0] = 2.0 * (u[..., 1] - u[..., 0]) / dr ** 2
            lap[..., n] = 2.0 * (u[..., n - 1] - u[..., n]) / dr ** 2
        else:
            lap[..., 0] = 0.0
            lap[..., n] = 0.0
        du += nu * lap
    return dp, du


def acoustic_energy(p, u, grid: RadialGrid, K, rho_f, bc: str):
    """Discrete acoustic energy (per unit solid angle), conserved by the semi-discrete
    scheme; shape (...,)."""
    w = grid.A * grid.dr
    w = w.copy()
    if bc == "periodic":
        w[-1] = 0.0                       # face n duplicates face 0
    elif bc == "pressure_release":
        w[-1] *= 0.5
        if grid.d == 1:
            w[0] *= 0.5
    return (np.sum(p ** 2 * grid.V / (2.0 * K), axis=-1)
            + np.sum(rho_f * u ** 2 * w, axis=-1) / 2.0)


def wave_spectral_radius(n: int, d: int, bc: str) -> float:
    """max |eigenvalue| * dr / c of the uniform-medium acoustic operator (c = rho = 1),
    assembled by probing acoustic_rhs on its independent degrees of freedom."""
    grid = RadialGrid(n, 1.0, d)
    faces = list(range(n + 1))
    if bc == "periodic":
        faces = list(range(n))            # face n duplicates face 0
    if d > 1 or bc == "rigid":
        faces = [j for j in faces if j != 0]
    if bc == "rigid":
        faces = [j for j in faces if j != n]
    m = n + len(faces)
    M = np.zeros((m, m))
    for k in range(m):
        p = np.zeros(n); u = np.zeros(n + 1)
        if k < n:
            p[k] = 1.0
        else:
            j = faces[k - n]; u[j] = 1.0
            if bc == "periodic" and j == 0:
                u[n] = 1.0
        dp, du = acoustic_rhs(p, u, grid, 1.0, 1.0, bc)
        M[:n, k] = dp
        M[n:, k] = du[faces]
    return float(np.max(np.abs(np.linalg.eigvals(M)))) * grid.dr


# ---------------------------------------------------------------------------- stepping
def _bcast(a, x):
    return x.reshape(x.shape + (1,) * (a.ndim - 1))


def rk4_step(rhs, t, y, dt, P):
    """One classical RK4 step for a tuple of (k, ...) arrays y; dt (k,) per trajectory."""
    k1 = rhs(t, y, P)
    k2 = rhs(t + 0.5 * dt, tuple(a + _bcast(a, 0.5 * dt) * q for a, q in zip(y, k1)), P)
    k3 = rhs(t + 0.5 * dt, tuple(a + _bcast(a, 0.5 * dt) * q for a, q in zip(y, k2)), P)
    k4 = rhs(t + dt, tuple(a + _bcast(a, dt) * q for a, q in zip(y, k3)), P)
    return tuple(a + _bcast(a, dt / 6.0) * (q1 + 2.0 * q2 + 2.0 * q3 + q4)
                 for a, q1, q2, q3, q4 in zip(y, k1, k2, k3, k4))


def integrate(rhs, y0, params: dict, dt_frame: float, n_sub, n_frames: int, record,
              t_start: float = 0.0, return_state: bool = False):
    """Advance the batch y0 (tuple of (B, ...) arrays) and record n_frames frames at
    t_start + f dt_frame, f = 0 .. n_frames - 1.

    params: dict of per-trajectory arrays with leading axis B (or scalars / None).
    rhs(t, y, P): t (k,) per-trajectory time, y the leading block of the state, P the
    matching leading block of params. Trajectory b takes n_sub[b] equal RK4 substeps per
    frame. record(y) returns an array (B, ...) for one frame. Returns (B, n_frames, ...)
    in the original order; with return_state also the final state tuple (original order).
    """
    n_sub = np.atleast_1d(np.asarray(n_sub, int))
    order = np.argsort(-n_sub, kind="stable")
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    ns = n_sub[order]
    y = tuple(np.array(np.asarray(a, float)[order], copy=True) for a in y0)
    Ps = {k: (v[order] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == len(order)
              else v) for k, v in params.items()}
    dt = dt_frame / ns
    frames = [record(y)]
    smax = int(ns.max()) if len(ns) else 0
    counts = [(ns > s).sum() for s in range(smax)]
    blocks = {}
    for f in range(n_frames - 1):
        tf = t_start + f * dt_frame
        for s in range(smax):
            k = int(counts[s])
            if k not in blocks:
                blocks[k] = {kk: (v[:k] if isinstance(v, np.ndarray) and v.ndim
                                  and v.shape[0] == len(order) else v)
                             for kk, v in Ps.items()}
            ys = y if k == len(ns) else tuple(a[:k] for a in y)
            yn = rk4_step(rhs, tf + s * dt[:k], ys, dt[:k], blocks[k])
            if k == len(ns):
                y = yn
            else:
                for a, b in zip(y, yn):
                    a[:k] = b
        frames.append(record(y))
    out = np.stack(frames, axis=1)[inv]
    if return_state:
        return out, tuple(a[inv] for a in y)
    return out


def n_substeps(dt_frame: float, dt_stable) -> np.ndarray:
    """Equal substeps per frame not exceeding the stable step."""
    return np.maximum(1, np.ceil(dt_frame / np.asarray(dt_stable, float) - 1e-9)).astype(int)


# ------------------------------------------------------------------------ initial data
def smoothstep_taper(x, length, width):
    """C2 quintic taper: 0 at both walls, 1 farther than width from them."""
    def s(z):
        z = np.clip(z, 0.0, 1.0)
        return z ** 3 * (10 - 15 * z + 6 * z ** 2)
    return s(x / width) * s((length - x) / width)


def gaussian(x, centre, width, length=None, periodic=False):
    """exp(-(x - centre)^2 / width^2); periodic images summed when periodic."""
    if not periodic:
        return np.exp(-((x - centre) / width) ** 2)
    out = np.zeros_like(x, dtype=float)
    for m in (-1, 0, 1):
        out += np.exp(-((x - centre + m * length) / width) ** 2)
    return out


def smoothed_box(x, centre, half_width, edge, length=None, periodic=False):
    """Box of half-width half_width with tanh edges of width edge."""
    def one(xx):
        return 0.5 * (np.tanh((xx - centre + half_width) / edge)
                      - np.tanh((xx - centre - half_width) / edge))
    if not periodic:
        return one(x)
    return sum(one(x + m * length) for m in (-1, 0, 1))


# ------------------------------------------------------- radial initial data (d = 2, 3)
# Shell radii of initial conditions and sources in d = 2, 3 are uniform in
# [0, RADIAL_MAX_FRAC L] (modelling choice): r = 0 is included, and the outermost shells
# start 0.2 L from the wall, the same smallest wall distance as the d = 1 walled cases.
RADIAL_MAX_FRAC = 0.8


def radius_from_frac(u, length):
    """Map a uniform draw u in [0, 1] to a shell radius in [0, RADIAL_MAX_FRAC L]."""
    return u * RADIAL_MAX_FRAC * length

def outer_taper(r, length, width):
    """C2 quintic taper that vanishes at the outer wall r = length only (d = 2, 3)."""
    z = np.clip((length - r) / width, 0.0, 1.0)
    return z ** 3 * (10 - 15 * z + 6 * z ** 2)


def even_shell(r, radius, width):
    """Shell exp(-((r - R)/w)^2) + exp(-((r + R)/w)^2): the even extension about r = 0,
    so the profile is smooth at the axis or centre for every shell radius R >= 0
    (R = 0 gives a centred Gaussian of peak 2). Callers normalise the peak."""
    return np.exp(-((r - radius) / width) ** 2) + np.exp(-((r + radius) / width) ** 2)


def odd_shell(r, radius, width):
    """Odd counterpart, exp(-((r - R)/w)^2) - exp(-((r + R)/w)^2): zero at r = 0 (used
    for radial velocity, which must vanish on the axis)."""
    return np.exp(-((r - radius) / width) ** 2) - np.exp(-((r + radius) / width) ** 2)


def shell_peak(radius, width, length, fn=even_shell):
    """max over r in [0, length] of fn on a fixed 8193-point grid plus r = radius
    (independent of the simulation resolution)."""
    r = np.concatenate([np.linspace(0.0, length, 8193), [min(radius, length)]])
    return float(np.abs(fn(r, radius, width)).max())


def split_by_dim(params_list, default_dim, run):
    """Run run(sub_list, d) for each geometry d present in params_list ("dim" key, default
    default_dim) and return the trajectories in the original order. With a single d the
    call is run(params_list, d) on the unchanged list."""
    dims = sorted({int(p.get("dim", default_dim)) for p in params_list})
    if len(dims) == 1:
        return run(params_list, dims[0])
    out = [None] * len(params_list)
    for d in dims:
        idx = [i for i, p in enumerate(params_list) if int(p.get("dim", default_dim)) == d]
        for i, tr in zip(idx, run([params_list[i] for i in idx], d)):
            out[i] = tr
    return out
