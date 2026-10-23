"""Markdown summary of an experiments/run_eval_grid.py output: per model and condition, the
one-step and rollout medians, ratio to persistence with the paired 95 per cent interval of the
difference of medians, fraction of trajectories better than persistence, per-channel one-step,
and the pre-registered comparisons (S1-g, S2-g, S3-g style rows; development data unless the
file says final_locked). Usage: python tools/summarise_grid_eval.py results/eval_grid/dev_d2.json"""
import json
import sys

r = json.load(open(sys.argv[1]))
d = r["dim"]
steps = r["roll_steps"]
res, cmp_ = r["results"], r["comparisons"]
p = res["persistence"]
print(f"## d = {d}, split {r['split']}, n = {r['n']} ({r['bc_counts']})\n")
print(f"persistence: one-step median {p['one_step']['median']:.4g}; "
      + ", ".join(f"{s}-step {p['rollout'][str(s)]['median']:.4g}" for s in steps) + "\n")
print("| model, condition | one-step median | ratio to persistence | paired diff of medians [95% CI] | frac better | "
      + " | ".join(f"{s}-step" for s in steps) + " | ch0 | ch1 | ch2 |")
print("|---" * (8 + len(steps)) + "|")
for k, v in res.items():
    if k == "persistence":
        continue
    c = cmp_[f"{k}_vs_persistence"]
    o = c["one_step"]
    pc = v.get("per_channel_rollout_from_seed", {})
    chans = [f"{pc[str(ch)]['1']['median']:.3g}" if str(ch) in pc else "" for ch in range(3)]
    print(f"| {k} | {v['one_step']['median']:.4g} | {c['ratio_one_step_median']:.3f} | "
          f"{o['median_diff']:.4g} [{o['median_diff_ci'][0]:.4g}, {o['median_diff_ci'][1]:.4g}] | "
          f"{o['frac_a_better']:.2f} | "
          + " | ".join(f"{v['rollout'][str(s)]['median']:.3g}" for s in steps)
          + " | " + " | ".join(chans) + " |")
print()
for k, c in cmp_.items():
    if "_minus_" in k:
        o = c["one_step"]
        r10 = c["roll10"]
        print(f"- {k}: one-step diff of medians {o['median_diff']:.4g} [{o['median_diff_ci'][0]:.4g}, "
              f"{o['median_diff_ci'][1]:.4g}], first better on {o['frac_a_better']:.2f}; 10-step "
              f"{r10['median_diff']:.4g} [{r10['median_diff_ci'][0]:.4g}, {r10['median_diff_ci'][1]:.4g}]")
if "per_bc" in r:
    print()
    for bc, v in r["per_bc"].items():
        row = ", ".join(f"{k} {x['ratio_to_persistence']:.3f}" for k, x in v.items()
                        if isinstance(x, dict) and k.endswith("_phys"))
        print(f"- {bc} (n = {v['n']}): ratio to persistence: {row}")
