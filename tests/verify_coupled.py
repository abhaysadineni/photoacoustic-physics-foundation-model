"""Round-2 coupled training systems (pfm/solvers/coupled.py): Photothermal and
Thermoacoustic. Exact checks (order tables via common.convergence_report), the consistency
check against the chain, smoke tests (train and ood, every allowed boundary condition)
and timing. Writes results/verification/coupled.json; exits non-zero on failure."""
import hashlib
import sys

from _tissue_harness import batch_check, environment, print_rep, sampling_stats, timing

from pfm import common
from pfm.solvers import coupled

if __name__ == "__main__":
    out, ok = {"system": "coupled"}, True
    for cls in (coupled.Photothermal, coupled.Thermoacoustic):
        s = cls()
        rep = s.verify()
        print_rep(rep)
        rows, o1 = batch_check(s, coupled.ALLOWED_BCS, seed=0)
        rows_o, o2 = batch_check(s, coupled.ALLOWED_BCS, seed=500, regime="ood")
        rep["smoke_test"] = {"passed": bool(o1 and o2), "rows": rows + rows_o}
        for r in rows + rows_o:
            if r["failures"]:
                print(f"  {s.name} smoke {r['regime']} {r['bc']}: {r['failures']}")
        rep["timing_64x64"] = timing(s, coupled.ALLOWED_BCS)
        for r in rep["timing_64x64"]:
            print(f"  {s.name} timing {r['bc']}: {r['seconds']:.2f} s for 64 x 64 "
                  f"(substeps/frame {r['rk4_substeps_per_frame_min']}-"
                  f"{r['rk4_substeps_per_frame_max']})")
        rep["sampling_stats"] = sampling_stats(s.sample_params, n=3000, seed=7)
        for reg, st in rep["sampling_stats"].items():
            print(f"  {s.name} sampling {reg}: rejection {st['rejection_rate']:.4f} "
                  f"{ {k: (v['accepted'], round(v['rejection_rate'], 4)) for k, v in st['by_ood_kind'].items()} }")
        out[s.name] = rep
        ok &= bool(rep["passed"] and rep["smoke_test"]["passed"])
    cc = coupled.consistency_check()
    out["consistency_with_chain"] = cc
    print(f"  consistency: max rel diff {cc['max_rel_diff']}, unused channel max "
          f"{cc['unused_channel_max_abs']}, passed={cc['passed']}")
    ok &= cc["passed"]
    out["ood_kinds"] = {k: list(v) for k, v in coupled.OOD_KINDS.items()}
    out["dt_frame_s"] = coupled.DT_FRAME
    out["length_m"] = coupled.LENGTH
    out["allowed_bcs"] = list(coupled.ALLOWED_BCS)
    out["param_sources"] = coupled.PARAM_SOURCES
    out["chain_py_md5"] = hashlib.md5((common.ROOT / "pfm/solvers/chain.py").read_bytes()).hexdigest()
    out["passed"] = bool(ok)
    out["environment"] = environment()
    path = common.RESULTS / "verification" / "coupled.json"
    common.save_json(out, path)
    print(f"  coupled: passed={ok}; wrote {path}")
    sys.exit(0 if ok else 1)
