"""Private method-of-lines machinery shared by the generic solvers (transport, burgers,
fisher, telegraph). Not a system module; downstream code should not import it directly.

Contents
--------
- Ghost-cell padding for periodic, odd-reflection ("odd"), even-reflection ("even") and
  quadratic-extrapolation ("extrap", used only as a non-reflecting outflow closure for
  pure advection) boundaries. Odd reflection with wall value g sets u_ghost = 2 g - u_edge,
  so the linear interpolant between ghost and edge cell centres equals g at the wall;
  g = 0 gives the homogeneous Dirichlet closure u_ghost = -u_edge. Even reflection sets
  u_ghost = u_edge (zero gradient at the wall).
- Classical four-stage Runge-Kutta (RK4) time stepping over a leading batch axis, with a
  per-trajectory number of equal substeps per output frame, so frames land exactly on
  multiples of dt_frame and each trajectory's time step depends only on its own
  parameters (never on which other trajectories share the batch). The batch is sorted by
  substep count so each substep updates a contiguous leading block with one vectorised
  operation.
- The random initial condition: a sum of 1 to 4 Gaussians.
"""
from __future__ import annotations

import numpy as np

LENGTH = 1.0             # every generic system is dimensionless on [0, 1)
CFL_ADV = 0.4            # dt <= CFL_ADV * dx / |advective speed|
CFL_DIFF = 0.2           # dt <= CFL_DIFF * dx**2 / D
WALL_TAPER = 0.1         # width of the C2 taper applied at walls to the initial condition
WALL_CENTRE_MARGIN = 0.2 # Gaussian centres lie in [0.2, 0.8] when the domain has walls
WIDTH_RANGE = (0.03, 0.15)


# --------------------------------------------------------------------------- ghosts
def _ghost(e0, e1, e2, kind, g):
    """Ghost value next to edge cell e0 (e1, e2 are the next cells inward)."""
    if kind == "odd":
        return 2.0 * g - e0
    if kind == "even":
        return e0
    if kind == "extrap":          # quadratic extrapolation, third-order accurate ghost
        return 3.0 * e0 - 3.0 * e1 + e2
    raise ValueError(f"unknown ghost kind {kind!r}")


def pad(u, bc_left, bc_right=None, g_left=0.0, g_right=0.0):
    """Return u padded with one ghost cell on each side along the last axis.

    bc_left / bc_right: "periodic" (both sides), "odd", "even" or "extrap".
    g_left / g_right: wall values for "odd" (scalars or arrays broadcastable to u[..., 0]).
    """
    if bc_right is None:
        bc_right = bc_left
    out = np.empty(u.shape[:-1] + (u.shape[-1] + 2,), dtype=u.dtype)
    out[..., 1:-1] = u
    if bc_left == "periodic":
        out[..., 0] = u[..., -1]
        out[..., -1] = u[..., 0]
        return out
    out[..., 0] = _ghost(u[..., 0], u[..., 1], u[..., 2], bc_left, g_left)
    out[..., -1] = _ghost(u[..., -1], u[..., -2], u[..., -3], bc_right, g_right)
    return out


def ddx(up, dx):
    """Second-order central first derivative from a padded array."""
    return (up[..., 2:] - up[..., :-2]) / (2.0 * dx)


def lap(up, dx):
    """Second-order central second derivative from a padded array."""
    return (up[..., 2:] - 2.0 * up[..., 1:-1] + up[..., :-2]) / (dx * dx)


# --------------------------------------------------------------------------- time stepping
def take(a, sl):
    """Per-trajectory parameter a ((B, 1) array or scalar) restricted to trajectories sl."""
    return a if np.ndim(a) == 0 else a[sl]


def n_substeps(interval, dt_stable):
    """Smallest integer number of equal substeps covering `interval` with dt <= dt_stable."""
    dt_stable = np.asarray(dt_stable, dtype=float)
    return np.maximum(1, np.ceil(interval / dt_stable * (1.0 - 1e-12))).astype(int)


def rk4_frames(rhs, u0, dt_frame, nsub, n_frames, on_frame, t0=0.0):
    """Integrate du/dt = rhs(u, t, sl) with classical RK4 and report n_frames frames.

    u0       : array (B, ..., n), float64
    nsub     : (B,) integer substeps per frame; trajectory b uses dt_b = dt_frame / nsub[b]
    rhs      : rhs(u, t, sl) -> array like u. u holds the trajectories listed in sl, t is
               the matching array of current times.
    on_frame : on_frame(k, u) for k = 0 .. n_frames - 1, at t = t0 + k * dt_frame
               (k = 0 is the initial condition), u in the original batch order.

    Implementation: the batch is permuted so nsub is descending; at substep s of a frame
    only the trajectories with nsub > s are advanced, and these form a contiguous leading
    slice, so the update is a vectorised operation on a view (no Python loop over
    trajectories). The total work is therefore sum_b nsub_b, not B * max_b nsub_b, and each
    trajectory's result is independent of its batch mates. `sl` is an index array into the
    original batch order (perm[:m]); every rhs factory indexes its (B, 1) parameter arrays
    with it.
    """
    B = u0.shape[0]
    nsub = np.asarray(nsub, dtype=int).reshape(B)
    perm = np.argsort(-nsub, kind="stable")
    inv = np.argsort(perm)
    ns = nsub[perm]
    bshape = (B,) + (1,) * (u0.ndim - 1)
    dt = dt_frame / ns.astype(float)
    dtb = dt.reshape(bshape)
    nmax = int(ns[0])
    # active count for each substep index s: number of trajectories with ns > s
    m_of_s = (ns[None, :] > np.arange(nmax)[:, None]).sum(axis=1)
    u = np.array(u0, dtype=float, copy=True)[perm]
    on_frame(0, u[inv])
    for k in range(1, n_frames):
        t_start = t0 + (k - 1) * dt_frame
        for s in range(nmax):
            m = int(m_of_s[s])
            idx = perm[:m]
            v = u[:m]
            h = dtb[:m]
            t = t_start + s * dt[:m]
            th = t + 0.5 * dt[:m]
            k1 = rhs(v, t, idx)
            k2 = rhs(v + (0.5 * h) * k1, th, idx)
            k3 = rhs(v + (0.5 * h) * k2, th, idx)
            k4 = rhs(v + h * k3, t + dt[:m], idx)
            v += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        on_frame(k, u[inv])
    return u[inv]


def integrate_to(rhs, u0, T, dt_stable):
    """Integrate a single (B = 1 or shared-step) problem from t = 0 to exactly T."""
    nsub = n_substeps(T, dt_stable)
    nsub = np.full(u0.shape[0], int(np.max(nsub)))
    return rk4_frames(rhs, u0, T, nsub, 2, lambda k, u: None), int(nsub[0])


# --------------------------------------------------------------------------- initial data
def _smoothstep5(xi):
    xi = np.clip(xi, 0.0, 1.0)
    return xi ** 3 * (10.0 - 15.0 * xi + 6.0 * xi ** 2)


def wall_taper(x, length=LENGTH, width=WALL_TAPER):
    """C2 taper: 0 with zero first and second derivative at both walls, 1 beyond `width`."""
    return _smoothstep5(x / width) * _smoothstep5((length - x) / width)


def draw_gaussians(rng, periodic, amp_sampler):
    """Draw 1-4 Gaussians: centres, widths (std dev in [0.03, 0.15]) and amplitudes."""
    m = int(rng.integers(1, 5))
    if periodic:
        centres = rng.uniform(0.0, LENGTH, m)
    else:
        centres = rng.uniform(WALL_CENTRE_MARGIN, LENGTH - WALL_CENTRE_MARGIN, m)
    widths = rng.uniform(*WIDTH_RANGE, m)
    amps = amp_sampler(rng, m)
    return {"centres": centres.tolist(), "widths": widths.tolist(), "amplitudes": amps.tolist()}


def eval_gaussians(ic, x, periodic):
    """Evaluate the Gaussian sum on grid x. Periodic: sum of images (exactly periodic).
    Walls: multiplied by the C2 wall taper."""
    s = np.zeros_like(x)
    shifts = (-2, -1, 0, 1, 2) if periodic else (0,)
    for c, w, a in zip(ic["centres"], ic["widths"], ic["amplitudes"]):
        for j in shifts:
            s += a * np.exp(-0.5 * ((x - c - j * LENGTH) / w) ** 2)
    if not periodic:
        s *= wall_taper(x)
    return s


def signed_amplitudes(rng, m, lo=0.2, hi=1.0):
    """Amplitudes with |a| in [lo, hi] and random sign."""
    return rng.uniform(lo, hi, m) * rng.choice([-1.0, 1.0], m)


def log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def draw_outside(rng, bands):
    """Draw log-uniformly from one of two bands (below / above the training band), each with
    probability 1/2."""
    lo, hi = bands[int(rng.integers(0, 2))]
    return log_uniform(rng, lo, hi)
