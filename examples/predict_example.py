"""Load a trained one-dimensional or radial network, forecast the photoacoustic chain, plot it.

Run from the repository root:

    python examples/predict_example.py --run v2_s0 --geom chain --traj 0 --steps 10

What it needs:
  runs/<run>/model.pt and runs/<run>/log.json   a trained network (release asset or training)
  data/<geom>/<split>.npz                       the chain split, from
                                                python experiments/gen_data.py <geom>
  results/datagen/<system>.json                 normalisation scales logged at generation
                                                (in the repository)

Input scaling is the physical-unit condition of PLAN.md section 14. Each channel of the chain,
in its physical unit (fluence rate W m^-2, temperature rise K, pressure Pa), is divided by a
constant taken from TRAINING systems of the same geometry: channel 0 by the photothermal
training RMS, channel 1 by the mean of the photothermal and thermoacoustic training RMS,
channel 2 by the thermoacoustic training RMS. No statistic of the chain enters the input.
The network's output is multiplied by the same constants to return to physical units.

Errors are scored as experiments/run_eval_r23.py scores them: fields in the chain's own
training-RMS units, the L2 norm of the frame error divided by the trajectory's RMS frame norm.
When a stored evaluation file holds the same run and split, the stored per-trajectory values
are printed beside the recomputed ones.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pfm.evaluate import load_model, one_step, traj_scale, _err      # noqa: E402
from pfm.model import N_IN_FRAMES                                    # noqa: E402

SUFFIX = {"chain": "", "chain_r2": "_r2", "chain_r3": "_r3"}
STORED = {"test": ["results/eval_r2/dev.json", "results/eval_r23/dev_r3_final.json",
                   "results/eval_r23/dev_r3.json", "results/eval_r23/dev_r3_plain.json"],
          "final_locked": ["results/eval_r2/locked.json", "results/eval_r23/locked_r3.json",
                           "results/eval_r2/locked_v2big_s0.json",
                           "results/eval_r2/locked_v2big_s1.json",
                           "results/eval_r2/locked_v2big_s2.json"]}
LABEL = ["fluence rate (W m$^{-2}$)", "temperature rise (K)", "pressure (Pa)"]


def logged_scale(system):
    """Training-split RMS per channel, as logged by experiments/gen_data.py (last entry)."""
    log = json.loads((ROOT / "results" / "datagen" / f"{system}.json").read_text())
    return np.array(log[-1]["norm_scale"], np.float32)


def stored_values(run, geom, split, traj):
    for name in STORED.get(split, []):
        p = ROOT / name
        if not p.exists():
            continue
        g = json.loads(p.read_text()).get("geoms", {}).get(geom, {})
        s = g.get(f"{run}_phys")
        if s and "per_traj_one_step" in s:
            return name, s["per_traj_one_step"][traj], s["per_traj_roll"]["10"][traj]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2_s0", help="folder name under runs/")
    ap.add_argument("--geom", default="chain", choices=list(SUFFIX),
                    help="chain (slab), chain_r2 (cylinder) or chain_r3 (sphere)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--plot", default="examples/predict_example.png")
    a = ap.parse_args()
    torch.set_num_threads(1)

    model = load_model(ROOT / "runs" / a.run / "model.pt")      # architecture from log.json
    z = np.load(ROOT / "data" / a.geom / f"{a.split}.npz")
    true = z["fields"][a.traj:a.traj + 1].astype(np.float32)    # (1, 64, 3, 128), physical units
    meta = json.loads((ROOT / "data" / a.geom / f"{a.split}_meta.json").read_text())[a.traj]
    print(f"{a.geom}/{a.split} trajectory {a.traj}: boundary {z['bc'][a.traj]}, "
          f"frame interval {meta['dt_frame']:.4g} s, length {meta['length'] * 1e3:.1f} mm")

    # Physical-unit input scaling (PLAN.md section 14), constants from training systems only.
    suf = SUFFIX[a.geom]
    pt, ta = logged_scale("photothermal" + suf), logged_scale("thermoacoustic" + suf)
    phys = np.array([pt[0], 0.5 * (pt[1] + ta[1]), ta[2]], np.float32)
    print("input constants per channel (W m^-2, K, Pa):", phys)

    x0 = np.ascontiguousarray(true[:, :N_IN_FRAMES] / phys[None, None, :, None])
    with torch.no_grad():
        pred = model.rollout(torch.from_numpy(x0), a.steps).numpy()
    pred = pred * phys[None, None, :, None]                      # back to physical units
    last = N_IN_FRAMES - 1 + a.steps                             # frame compared with the last step
    print(f"pressure after {a.steps} steps (frame {last}), first 4 cells (Pa)")
    print("   predicted:", pred[0, -1, 2, :4])
    print("   true:     ", true[0, last, 2, :4])

    # Scores in the chain's own training-RMS units, as experiments/run_eval_r23.py.
    own = np.array(json.loads((ROOT / "data" / a.geom / "norm.json").read_text())["scale"],
                   np.float32)
    t_own = true / own[None, None, :, None]
    k = (own / phys)[None, :].astype(np.float32)                 # the same scaling, in own units
    e1 = float(one_step(model, t_own, k)[0])
    p1 = float(one_step(None, t_own)[0])
    S = traj_scale(t_own)
    eN = float(_err(pred[:, -1] / own[None, :, None], t_own[:, last], S)[0])
    pN = float(_err(t_own[:, N_IN_FRAMES - 1], t_own[:, last], S)[0])
    print(f"one-step error (mean over the {t_own.shape[1] - N_IN_FRAMES} starts): "
          f"network {e1:.6f}, persistence {p1:.6f}")
    print(f"{a.steps}-step error: network {eN:.6f}, persistence {pN:.6f}")
    st = stored_values(a.run, a.geom, a.split, a.traj) if a.steps == 10 else None
    if st:
        print(f"stored in {st[0]}: one-step {st[1]:.6f}, 10-step {st[2]:.6f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "font.size": 9})
    n = true.shape[-1]
    x = (np.arange(n) + 0.5) * meta["length"] / n * 1e3
    fig, ax = plt.subplots(1, 3, figsize=(9.0, 2.8), constrained_layout=True)
    for c in range(3):
        ax[c].plot(x, true[0, N_IN_FRAMES - 1, c], color="#8a8984", lw=1, ls=":",
                   label=f"last seed frame ({N_IN_FRAMES - 1})")
        ax[c].plot(x, true[0, last, c], color="#0b0b0b", lw=1.2, label=f"true, frame {last}")
        ax[c].plot(x, pred[0, -1, c], color="#2a78d6", lw=1.2, ls="--",
                   label=f"network, step {a.steps}")
        ax[c].set_xlabel("radius (mm)" if suf else "position (mm)")
        ax[c].set_ylabel(LABEL[c])
    ax[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{a.run} on {a.geom}/{a.split} trajectory {a.traj}", fontsize=9)
    out = ROOT / a.plot
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("wrote", a.plot)


if __name__ == "__main__":
    main()
