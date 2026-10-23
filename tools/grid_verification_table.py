"""Print the grid verification table (markdown) from results/verification2d and verification3d."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(dim):
    d = ROOT / "results" / f"verification{dim}d"
    out = []
    for f in sorted(d.glob(f"*_d{dim}.json")):
        r = json.loads(f.read_text())
        name = f.stem
        if name.startswith("consistency"):
            red = r["reduction_1d"]
            out.append((name, "reduction to 1D solvers (light, heat, sound x2, chain; every bc)",
                        f"max rel diff {red['max_rel_diff']:.2e}", r["passed"]))
            out.append((name, "photothermal/thermoacoustic = chain channels",
                        f"max rel diff {r['links']['max_rel_diff']:.2e}", r["links"]["passed"]))
            out.append((name, "sparse steady matrix = time-stepping operator; Gershgorin bound",
                        "", r["grid_operator"]["passed"]))
            continue
        if name.startswith("sound_resolution"):
            for k, c in r["cases"].items():
                e = c["rel_L2_error_vs_richardson"]
                out.append((name, f"stored-data error {k}", ", ".join(f"{n}: {v:.4f}" for n, v in e.items()), None))
            continue
        v = r.get("verify", r)
        cases = v.get("cases", {"main": v})
        for k, c in cases.items():
            if "observed_order" in c and c.get("observed_order"):
                val = (f"order {c['observed_order'][-1]:.3f}, finest error {c['error'][-1]:.2e}")
            elif "max_rel_diff" in c:
                val = f"max rel diff {c['max_rel_diff']:.2e}"
            elif "own_dt" in c:
                val = f"max rel diff {max(c['own_dt']['max_rel_diff'], c['matched_dt']['max_rel_diff']):.2e}"
            elif "error" in c:
                val = f"max error {max(c['error']):.2e}"
            else:
                val = ""
            out.append((name, k, val, c.get("passed")))
        if "smoke" in r:
            out.append((name, "smoke (shape, finite, bounds, every bc, train and ood)", "", r["smoke"]["passed"]))
        if "resolution" in r:
            res = r["resolution"]
            out.append((name, f"stored-data difference internal {res['n'][0]} vs {res['n'][1]}",
                        f"max {res['max']:.2e} (est. error {max(res['estimated_error_default_grid']):.2e})", None))
    return out


if __name__ == "__main__":
    for dim in (2, 3):
        print(f"\n### d = {dim}\n\n| file | check | measured | pass |\n|---|---|---|---|")
        for n, k, v, p in rows(dim):
            print(f"| {n} | {k} | {v} | {'' if p is None else ('pass' if p else 'FAIL')} |")
