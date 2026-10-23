"""Photoacoustic chain: links (a) light, (b) heat, (c) free acoustics, (d) acoustics from
constant heating, each in d = 1, 2, 3 (pass/fail order tests), (e) top-hat slab, cylinder,
sphere (reported only, with overlay data); smoke tests for d = 1 (periodic,
pressure_release, rigid) and d = 2, 3 (pressure_release, rigid); timing.
Writes results/verification/chain.json; exits non-zero on failure."""
import sys

from _tissue_harness import batch_check, environment, sampling_stats, timing

from pfm import common
from pfm.solvers import chain

if __name__ == "__main__":
    rep = chain.verify_all()
    for k, t in rep["pass_fail_tests"].items():
        print(f"  {k}: errors {t['error']}, orders {t['observed_order']}, passed={t['passed']}")
        if "quadrature_check" in t:
            print(f"    quadrature: {t['quadrature_check']}")
    for k, t in rep["e_tophat_report_only"].items():
        for tt, row in t["times"].items():
            print(f"  e_tophat_{k} {tt}: L1 {row['L1']} order {row['order_L1']}; "
                  f"L2 {row['L2']} order {row['order_L2']}")
    verify_ok = rep["passed"]
    smoke, ok, tim = {}, True, {}
    for d in (1, 2, 3):
        s = chain.PhotoacousticChain(dim=d)
        rows, o = batch_check(s, chain.ALLOWED_BCS[d], seed=10 * d)
        smoke[f"d{d}"] = {"passed": o, "rows": rows}
        ok &= o
        for r in rows:
            if r["failures"]:
                print(f"  smoke d{d} {r['bc']}: {r['failures']}")
        tim[f"d{d}"] = timing(s, chain.ALLOWED_BCS[d], seed=100 + d)
        for r in tim[f"d{d}"]:
            print(f"  timing d{d} {r['bc']}: {r['seconds']:.2f} s for 64 x 64")
    # item 4: ood draws through sample_params(rng, regime, dim=d) on a default (dim = 1)
    # instance, d = 2 and 3, pressure_release and rigid
    base = chain.PhotoacousticChain()
    for d in (2, 3):
        rows, o = batch_check(base, chain.ALLOWED_BCS[d], seed=700 + d, regime="ood",
                              sample_kw={"dim": d})
        smoke[f"d{d}_ood_dim_kwarg"] = {"passed": o, "rows": rows}
        ok &= o
        for r in rows:
            if r["failures"]:
                print(f"  smoke d{d} ood {r['bc']}: {r['failures']}")
    rep["smoke_test"] = smoke
    rep["smoke_passed"] = bool(ok)
    rep["sampling_stats"] = sampling_stats(chain.PhotoacousticChain(1).sample_params, n=3000,
                                           seed=7)
    for reg, st in rep["sampling_stats"].items():
        print(f"  sampling {reg}: rejection {st['rejection_rate']:.4f} "
              f"{ {k: round(v['rejection_rate'], 4) for k, v in st['by_ood_kind'].items()} }, "
              f"max mu_a/mu_s' {st['mua_over_musp_max_accepted']:.4f}")
    rep["timing_64x64"] = tim
    rep["allowed_bcs"] = {f"d{d}": list(v) for d, v in chain.ALLOWED_BCS.items()}
    rep["dt_frame_s"] = chain.DT_FRAME
    rep["length_m"] = chain.LENGTH
    rep["param_ranges"] = chain.PARAM_RANGES
    rep["param_sources"] = chain.PARAM_SOURCES
    rep["gruneisen_37C_soft_tissue_available"] = chain.GAMMA_37C_SOFT_AVAILABLE
    rep["environment"] = environment()
    # "passed" is the order-test verdict required by the spec; smoke is reported beside it
    path = common.RESULTS / "verification" / "chain.json"
    common.save_json(rep, path)
    print(f"  chain: order tests passed={verify_ok}, smoke passed={ok}; wrote {path}")
    sys.exit(0 if (verify_ok and ok) else 1)
