"""Cartesian finite-volume numerics for d = 2 and d = 3 (shared by every solver).

Grid: n cells per axis on [0, L)^d, spacing h = L / n (the same in every direction),
cell centres (i + 1/2) h. Arrays carry any number of leading axes (batch) followed by the
d spatial axes; a face array of axis a has n + 1 entries along that axis (faces j h,
j = 0 .. n) and n along the others. The operators accept any spatial shape (n_0, ..., n_{d-1})
with a common spacing h, so a problem uniform in some directions can be run on a box that
is short in those directions.

Scalar operator (the d-dimensional version of pfm/solvers/radial.py, d = 1 slab):
    div(k grad u) -> sum_a (F_{a, j+1} - F_{a, j}) / h,  F_{a, j} = k_{a, j} (u_j - u_{j-1}) / h
two-point fluxes, second order; face coefficient = arithmetic mean of the two neighbouring
cells (wall faces take the edge cell; periodic faces the mean of the two wrapped cells).
Walls by ghost cells, entering only through the wall flux:
    periodic   wrap;   dirichlet  u_ghost = 2 g - u_edge (g = 0 in time stepping);
    neumann    zero flux.
Stable RK4 step: RK4_REAL_LIMIT / G with G the Gershgorin bound of the assembled operator
(including absorption), per trajectory, as radial.diffusion_dt. For this operator the
absolute row sum is G_i = sum over the 2d faces of cell i of 2 k_f m_f / h^2 + |absorb_i|,
m_f = 0 on a neumann wall face and 1 otherwise (an interior or periodic face contributes k
to the diagonal and k off the diagonal; a dirichlet wall face 2k to the diagonal).

Acoustics: staggered grid, p at cell centres, u_a at the faces of axis a.
    dp/dt = -K sum_a d u_a / d x_a + s,      du_a/dt = -(1/rho_f) dp/dx_a,
K = rho c^2 at cells, rho_f the arithmetic mean of the neighbouring cell densities.
Walls (as radial.acoustic_rhs): pressure_release p_ghost = -p_edge (the wall-face velocity
is a degree of freedom); rigid u = 0 on the wall faces, held fixed; periodic, face n is
face 0. The semi-discrete energy
    E = sum_i p_i^2 h^d / (2 K_i) + sum_a sum_j rho_j u_j^2 w_j / 2,
w_j = h^d on interior faces, h^d / 2 on pressure-release wall faces, 0 on the duplicated
periodic face, is conserved; RK4 is weakly dissipative on the imaginary axis, so E does not
grow. Step dt = CFL_ACOUSTIC h / c_max with CFL_ACOUSTIC = 0.5 (radial.CFL_ACOUSTIC). For a
uniform periodic medium the eigenvalues are i (2c/h) (sum_a sin^2(k_a h/2))^(1/2), so the
spectral radius is at most 2 sqrt(d) c / h; RK4 is stable on the imaginary axis up to
2.828, giving a margin of 2.828 / (0.5 * 2 sqrt(d)) = 2.00 for d = 2 and 1.63 for d = 3.

Advection-diffusion step: dt = RK4_DISC / (G_diff + sum_a |v_a| / h) with RK4_DISC = 2.5;
the RK4 stability region contains the closed left half-disc of radius 2.6156 (computed
numerically on 21 Oct 2026, minimum at 122.7 degrees), and every eigenvalue of the central
advection-diffusion operator lies in the left half-plane with modulus at most
G_diff + sum_a |v_a| / h.

Steady diffusion (light): sparse assembly of the same two-point operator, with per-axis
boundary specification "periodic", "neumann" or ("dirichlet", g_lo, g_hi) where g are
scalars or arrays over the wall cross-section. Solved by scipy.sparse.linalg.spsolve
(SuperLU, direct) up to DIRECT_MAX unknowns, otherwise by conjugate gradients on -A
(symmetric positive definite) with a Jacobi preconditioner, relative residual tolerance
CG_RTOL = 1e-10; the achieved relative residual ||b - A x|| / ||b|| is returned.

Time stepping uses pfm/solvers/radial.py's batched RK4 integrator (radial.integrate,
radial.n_substeps), imported unchanged: every trajectory takes its own number of equal
substeps per frame.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pfm.solvers import radial as R

RK4_REAL_LIMIT = R.RK4_REAL_LIMIT      # 2.5 of RK4's real-axis interval 2.785
CFL_ACOUSTIC = R.CFL_ACOUSTIC          # 0.5
RK4_DISC = 2.5                         # inside the half-disc of radius 2.6156
DIRECT_MAX = 70000                     # unknowns; above this, conjugate gradients
CG_RTOL = 1e-10
integrate = R.integrate
n_substeps = R.n_substeps


def ix(d: int, a: int, s):
    """Index selecting s along spatial axis a of an array whose last d axes are spatial."""
    t = [slice(None)] * d
    t[a] = s
    return (Ellipsis, *t)


def _face_shape(shape, d, a):
    s = list(shape)
    s[len(s) - d + a] += 1
    return s


# ---------------------------------------------------------------------------- scalar
def face_mean(k: np.ndarray, d: int, bcs) -> list:
    """Face coefficients (list over axes) from cell coefficients k (..., *n)."""
    out = []
    for a in range(d):
        n = k.shape[-d + a]
        kf = np.empty(_face_shape(k.shape, d, a))
        kf[ix(d, a, slice(1, n))] = 0.5 * (k[ix(d, a, slice(1, None))] + k[ix(d, a, slice(None, -1))])
        if bcs[a] == "periodic":
            v = 0.5 * (k[ix(d, a, 0)] + k[ix(d, a, -1)])
            kf[ix(d, a, 0)] = v
            kf[ix(d, a, n)] = v
        else:
            kf[ix(d, a, 0)] = k[ix(d, a, 0)]
            kf[ix(d, a, n)] = k[ix(d, a, -1)]
        out.append(kf)
    return out


def diffusion(u: np.ndarray, kf, bcs, h: float) -> np.ndarray:
    """div(k grad u) with homogeneous wall data. kf: list over axes of face coefficients
    (full face arrays or anything broadcastable to them)."""
    d = len(bcs)
    out = None
    for a in range(d):
        n = u.shape[-d + a]
        F = np.empty(_face_shape(u.shape, d, a))
        np.subtract(u[ix(d, a, slice(1, None))], u[ix(d, a, slice(None, -1))],
                    out=F[ix(d, a, slice(1, n))])
        bc = bcs[a]
        if bc == "periodic":
            w = u[ix(d, a, 0)] - u[ix(d, a, -1)]
            F[ix(d, a, 0)] = w
            F[ix(d, a, n)] = w
        elif bc == "dirichlet":
            F[ix(d, a, n)] = -2.0 * u[ix(d, a, -1)]
            F[ix(d, a, 0)] = 2.0 * u[ix(d, a, 0)]
        elif bc == "neumann":
            F[ix(d, a, 0)] = 0.0
            F[ix(d, a, n)] = 0.0
        else:
            raise ValueError(bc)
        F *= kf[a]
        div = F[ix(d, a, slice(1, None))] - F[ix(d, a, slice(None, -1))]
        if out is None:
            out = div
        else:
            out += div
    out *= 1.0 / (h * h)
    return out


def gershgorin(kf, bcs, h, absorb, cell_shape) -> np.ndarray:
    """Per-trajectory Gershgorin bound of div(k grad .) - absorb. cell_shape = (B, *n)."""
    d = len(bcs)
    G = np.zeros(cell_shape)
    for a in range(d):
        k = np.broadcast_to(kf[a], _face_shape(cell_shape, d, a)).copy()
        if bcs[a] == "neumann":
            k[ix(d, a, 0)] = 0.0
            k[ix(d, a, -1)] = 0.0
        G += 2.0 * (np.abs(k[ix(d, a, slice(1, None))]) + np.abs(k[ix(d, a, slice(None, -1))]))
    G /= h * h
    G += np.abs(np.broadcast_to(absorb, cell_shape))
    return G.reshape(cell_shape[0], -1).max(axis=1)


def diffusion_dt(kf, bcs, h, absorb, cell_shape) -> np.ndarray:
    return RK4_REAL_LIMIT / gershgorin(kf, bcs, h, absorb, cell_shape)


# ------------------------------------------------------------------ steady (sparse)
def steady_matrix(kf, absorb: np.ndarray, bspec, h: float):
    """Sparse A and right-hand side b of div(k grad phi) - absorb phi = 0 with boundary
    data moved to b (A phi = b) for ONE trajectory. kf: list of full face arrays (no batch
    axis); absorb: (*n); bspec: per axis "periodic" | "neumann" | ("dirichlet", g_lo, g_hi)."""
    shape = absorb.shape
    d = len(shape)
    N = absorb.size
    idx = np.arange(N).reshape(shape)
    rows, cols, vals = [], [], []
    diag = -np.asarray(absorb, float).ravel().copy()
    b = np.zeros(N)
    h2 = h * h
    for a in range(d):
        n = shape[a]
        k = np.asarray(kf[a], float)
        c = (k[ix(d, a, slice(1, n))] / h2).ravel()
        lo = idx[ix(d, a, slice(0, n - 1))].ravel()
        hi = idx[ix(d, a, slice(1, n))].ravel()
        rows += [lo, hi]; cols += [hi, lo]; vals += [c, c]
        np.subtract.at(diag, lo, c)
        np.subtract.at(diag, hi, c)
        first = idx[ix(d, a, 0)].ravel()
        last = idx[ix(d, a, -1)].ravel()
        spec = bspec[a]
        if spec == "periodic":
            cw = (k[ix(d, a, 0)] / h2).ravel()
            rows += [first, last]; cols += [last, first]; vals += [cw, cw]
            np.subtract.at(diag, first, cw)
            np.subtract.at(diag, last, cw)
        elif spec == "neumann":
            pass
        elif isinstance(spec, tuple) and spec[0] == "dirichlet":
            g_lo = np.broadcast_to(np.asarray(spec[1], float), k[ix(d, a, 0)].shape).ravel()
            g_hi = np.broadcast_to(np.asarray(spec[2], float), k[ix(d, a, 0)].shape).ravel()
            c0 = 2.0 * (k[ix(d, a, 0)] / h2).ravel()
            cn = 2.0 * (k[ix(d, a, n)] / h2).ravel()
            np.subtract.at(diag, first, c0)
            np.subtract.at(diag, last, cn)
            np.subtract.at(b, first, c0 * g_lo)
            np.subtract.at(b, last, cn * g_hi)
        else:
            raise ValueError(spec)
    rows.append(np.arange(N)); cols.append(np.arange(N)); vals.append(diag)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N, N))
    return A, b


def steady_solve(kf, absorb, bspec, h, source=None, method="auto"):
    """Solve div(k grad phi) - absorb phi + source = 0. Returns (phi, info) with info the
    method and the relative residual ||b - A phi|| / ||b||."""
    A, b = steady_matrix(kf, absorb, bspec, h)
    if source is not None:
        b = b - np.asarray(source, float).ravel()
    N = b.size
    if method == "auto":
        method = "direct" if N <= DIRECT_MAX else "cg"
    if method == "direct":
        x = spla.spsolve(A.tocsc(), b)
        it = None
    else:
        M = -A
        dinv = 1.0 / M.diagonal()
        pre = spla.LinearOperator(M.shape, matvec=lambda v: dinv * v)
        count = [0]

        def cb(_):
            count[0] += 1
        x, flag = spla.cg(M, -b, rtol=CG_RTOL, atol=0.0, maxiter=20000, M=pre, callback=cb)
        if flag != 0:
            raise RuntimeError(f"CG did not converge (flag {flag})")
        it = count[0]
    res = float(np.linalg.norm(b - A @ x) / max(np.linalg.norm(b), 1e-300))
    return x.reshape(absorb.shape), {"method": method, "unknowns": int(N),
                                     "relative_residual": res, "cg_iterations": it,
                                     "cg_rtol": CG_RTOL if method == "cg" else None}


# ----------------------------------------------------------------------- acoustics
def acoustic_rhs(p, u, K, irf, bcs, h, src=None):
    """Staggered acoustic right-hand side. p (..., *n), u list over axes of face arrays,
    K (..., *n) or broadcastable, irf list of 1/rho at faces, src added to dp/dt."""
    d = len(bcs)
    div = None
    for a in range(d):
        t = u[a][ix(d, a, slice(1, None))] - u[a][ix(d, a, slice(None, -1))]
        if div is None:
            div = t
        else:
            div += t
    dp = div
    dp *= -1.0 / h
    dp *= K
    if src is not None:
        dp += src
    du = []
    for a in range(d):
        n = p.shape[-d + a]
        g = np.empty(u[a].shape)
        np.subtract(p[ix(d, a, slice(1, None))], p[ix(d, a, slice(None, -1))],
                    out=g[ix(d, a, slice(1, n))])
        bc = bcs[a]
        if bc == "periodic":
            w = p[ix(d, a, 0)] - p[ix(d, a, -1)]
            g[ix(d, a, 0)] = w
            g[ix(d, a, n)] = w
        elif bc == "pressure_release":
            g[ix(d, a, n)] = -2.0 * p[ix(d, a, -1)]
            g[ix(d, a, 0)] = 2.0 * p[ix(d, a, 0)]
        elif bc == "rigid":
            g[ix(d, a, 0)] = 0.0
            g[ix(d, a, n)] = 0.0
        else:
            raise ValueError(bc)
        g *= -1.0 / h
        g *= irf[a]
        du.append(g)
    return dp, du


def acoustic_energy(p, u, K, rho_f, bcs, h):
    """Discrete acoustic energy, shape p.shape[:-d]."""
    d = len(bcs)
    hv = h ** d
    E = np.sum((p ** 2 / (2.0 * K)).reshape(*p.shape[:-d], -1), axis=-1) * hv
    for a in range(d):
        w = np.ones(u[a].shape[-d + a])
        if bcs[a] == "periodic":
            w[-1] = 0.0
        elif bcs[a] == "pressure_release":
            w[0] = w[-1] = 0.5
        shp = [1] * d
        shp[a] = w.size
        e = rho_f[a] * u[a] ** 2 * w.reshape(shp)
        E = E + 0.5 * hv * np.sum(e.reshape(*e.shape[:-d], -1), axis=-1)
    return E


def acoustic_dt(c_max, h):
    return CFL_ACOUSTIC * h / np.asarray(c_max, float)


# ------------------------------------------------------------------ initial data
def taper(X, length, width):
    """Product over axes of radial.smoothstep_taper (C2, zero at every wall)."""
    out = 1.0
    for x in X:
        out = out * R.smoothstep_taper(x, length, width)
    return out


def image_shifts(d, periodic, m=1):
    if not periodic:
        return [np.zeros(d)]
    g = np.stack(np.meshgrid(*([np.arange(-m, m + 1)] * d), indexing="ij"), -1).reshape(-1, d)
    return [s.astype(float) for s in g]


def dist2_point(X, centre, length, shift):
    r2 = 0.0
    for a, x in enumerate(X):
        r2 = r2 + (x - centre[a] - shift[a] * length) ** 2
    return r2


def gaussian(X, centre, width, length, periodic):
    """exp(-|x - c|^2 / width^2), periodic images (3^d nearest) summed when periodic."""
    out = 0.0
    for s in image_shifts(len(X), periodic):
        out = out + np.exp(-dist2_point(X, centre, length, s) / width ** 2)
    return out
