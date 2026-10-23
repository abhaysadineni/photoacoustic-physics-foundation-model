"""End-to-end exact check for the telegraph equation through simulate_batch (audit MINOR 6).
Writes results/verification/e2e_telegraph.json; exits non-zero on failure.

Exact solution: standing mode p(x, 0) = A cos(k x), q(x, 0) = 0 (the production q0 = 0),
p(x, t) = A T(t) cos(k x) with T(t) from telegraph.mode_exact (cos/sin form when
gamma < c k, cosh/sinh when gamma > c k). Stored in channel 2 only.

Periodic cases (as specified), k = 2 pi m:
  - underdamped_m2_c1_g1   : c = 1, gamma = 1 (g = gamma L / c = 1), m = 2. Oscillates with
                             period about 0.50, so T(t) passes through zero twice in the record
                             (near frames 17 and 49).
  - underdamped_m1_c0.5_g1 : c = 0.5, gamma = 1 (g = 2), m = 1. No zero crossing in the record.
  - fast_m3_c2_g0.04       : c = 2, gamma = 0.04 (g = 0.02), m = 3: highest training speed,
                             weakly damped, several oscillations.
  - overdamped_m2_c1_g20   : c = 1, gamma = 20 (g = 20), m = 2: cosh/sinh form.
Wall cases (exact for the production closures, included in "passed"):
  - pressure_release: p = A T(t) sin(m pi x) (odd about both walls, p = 0 there);
  - rigid           : p = A T(t) cos(m pi x) (even about both walls, dp/dx = 0 there).
Underdamped modes pass through zero amplitude; at such frames the per-frame relative L2
error divides by a near-zero norm, so the secondary metric (error over the largest frame
norm) is the meaningful one there. Both are reported; the per-frame metric is the pass rule,
and any case that fails it only because of a near-node frame is stated as such.
"""
import sys

import numpy as np

import _e2e
from pfm.solvers.telegraph import DT_FRAME, TelegraphSolver, mode_exact

CH = 2


def mode_case(name, c, gamma, k, shape="cos", amp=0.7):
    f = np.cos if shape == "cos" else np.sin
    _e2e.EXACT_IC[name] = lambda x: amp * f(k * x)
    tfac = lambda t: mode_exact(0.0, t, c, gamma, k)       # cos(0) = 1: the time factor
    t = np.arange(64) * DT_FRAME
    T = np.array([tfac(tt) for tt in t])
    return {"name": name, "params": {"c": c, "gamma": gamma, "ic": {"exact_ic": name}},
            "exact": lambda x, tt: amp * tfac(tt) * f(k * x), "channel": CH,
            "notes": f"{shape}({k / np.pi:g} pi x), c={c}, gamma={gamma}, c k={c * k:.4g}; "
                     f"min |T(t_k)| over frames = {np.abs(T).min():.3e} (frame {int(np.abs(T).argmin())}), "
                     f"T(t_63) = {T[-1]:.4f}"}


def main():
    s = TelegraphSolver()
    pi = np.pi
    groups = [
        _e2e.run_group(s, DT_FRAME, "periodic", [
            mode_case("underdamped_m2_c1_g1", 1.0, 1.0, 4 * pi),
            mode_case("underdamped_m1_c0.5_g1", 0.5, 1.0, 2 * pi),
            mode_case("fast_m3_c2_g0.04", 2.0, 0.04, 6 * pi),
            mode_case("overdamped_m2_c1_g20", 1.0, 20.0, 4 * pi),
        ]),
        {f"{k}__pressure_release": v for k, v in _e2e.run_group(s, DT_FRAME, "pressure_release", [
            mode_case("wall_sin3pix_c1_g1", 1.0, 1.0, 3 * pi, "sin"),
            mode_case("wall_sin2pix_c0.5_g10", 0.5, 10.0, 2 * pi, "sin"),
        ]).items()},
        {f"{k}__rigid": v for k, v in _e2e.run_group(s, DT_FRAME, "rigid", [
            mode_case("wall_cos3pix_c1_g1", 1.0, 1.0, 3 * pi, "cos"),
            mode_case("wall_cos2pix_c0.5_g10", 0.5, 10.0, 2 * pi, "cos"),
        ]).items()},
    ]
    sec = {}
    for g in groups:
        for n, r in g.items():
            a, b = r["by_n"]["512"], r["by_n"]["1024"]
            ok = a["max_err_over_max_frame_norm"] < _e2e.TOL_512 and \
                b["max_err_over_max_frame_norm"] < a["max_err_over_max_frame_norm"] and \
                b["mean_err_over_max_frame_norm"] < a["mean_err_over_max_frame_norm"]
            sec[n] = {"max_512": a["max_err_over_max_frame_norm"], "max_1024": b["max_err_over_max_frame_norm"],
                      "factor": a["max_err_over_max_frame_norm"] / b["max_err_over_max_frame_norm"],
                      "would_pass_on_this_metric": bool(ok)}
    rep = _e2e.summarise("telegraph", DT_FRAME, groups, [], extra={
        "secondary_metric_error_over_max_frame_norm": sec,
        "secondary_note": "NOT the pass rule. Reported because underdamped modes pass through zero amplitude, "
                          "where the per-frame relative L2 error divides by a near-zero norm; 'passed' uses "
                          "the per-frame rule as specified."})
    print("secondary metric (max over frames of ||err|| / max frame norm):")
    for n, r in rep["cases"].items():
        print(f"  {n:40s} 512 {r['by_n']['512']['max_err_over_max_frame_norm']:.3e}  "
              f"1024 {r['by_n']['1024']['max_err_over_max_frame_norm']:.3e}  "
              f"per-frame max at frame {r['by_n']['512']['frame_of_max']}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
