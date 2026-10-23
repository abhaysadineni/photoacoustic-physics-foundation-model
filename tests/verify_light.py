"""Light transport: Green's function convergence (d = 1, 2, 3), smoke test and timing.
Writes results/verification/light.json; exits non-zero on failure."""
import sys

from _tissue_harness import print_rep, report_and_exit, run, sampling_stats

from pfm.solvers import light

if __name__ == "__main__":
    rep, ok = run(light.LightSolver(), ("periodic", "dirichlet", "neumann"), light)
    rep["chromophores_m-1"] = light.CHROMOPHORES
    solver = light.LightSolver()
    rep["sampling_stats"] = sampling_stats(solver.sample_params)
    for reg, st in rep["sampling_stats"].items():
        print(f"  sampling {reg}: rejection {st['rejection_rate']:.4f} "
              f"{ {k: round(v['rejection_rate'], 4) for k, v in st['by_ood_kind'].items()} }, "
              f"min start/t_tr {st['light_start_over_t_tr_min']:.3f}, "
              f"min width/l_tr {st['light_init_width_over_l_tr_min']:.3f}, "
              f"max mu_a/mu_s' {st['mua_over_musp_max_accepted']:.4f}")
    print_rep(rep)
    sys.exit(report_and_exit("light", rep, ok))
