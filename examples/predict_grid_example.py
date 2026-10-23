"""Load a trained two- or three-dimensional grid network, forecast the photoacoustic chain, plot.

Run from the repository root:

    python examples/predict_grid_example.py --dim 2 --run g2_s0 --traj 0 --steps 10
    python examples/predict_grid_example.py --dim 3 --run g3_s0 --traj 0 --steps 10

What it needs:
  runs_grid/<run>/model.pt and log.json        a trained grid network (release asset or training)
  results/datagen_grid/norm_d<d>_<system>.json training-split scales of the grid systems
  results/data_manifest_grid.json              reference fingerprints of the grid data
All three are in the repository or the release asset. No grid data have to be generated: if
the chain development split is not under data2d/, the script recomputes the first generation
batch of that split (16 trajectories in 2D, 4 in 3D) through the production path
(pfm2d.data.reference_subset) and checks it against the reference SHA-256, so --traj must
then be below 16 (2D) or 4 (3D). With the split present (python experiments/gen_data_grid.py
--local-dev --dim <d>), any trajectory of it can be used.

Input scaling is the physical-unit condition of PLAN.md sections 14 and 23: chain channel 0 in
W m^-2 divided by the photothermal training RMS, channel 1 in K by the mean of the
photothermal and thermoacoustic training RMS, channel 2 in Pa by the thermoacoustic training
RMS, all of the same dimension. Scores follow experiments/run_eval_grid.py (the chain's own
training-RMS units, L2 over all channels and cells divided by the trajectory's RMS frame norm),
and the stored per-trajectory values of results/eval_grid/dev_d<d>.json are printed beside them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pfm2d import common as C                                        # noqa: E402
from pfm2d import data as D                                          # noqa: E402
from pfm2d import evaluate as E                                      # noqa: E402

LABEL = ["fluence rate (W m$^{-2}$)", "temperature rise (K)", "pressure (Pa)"]


def scale(system, dim):
    p = ROOT / "results" / "datagen_grid" / f"norm_d{dim}_{system}.json"
    return np.array(json.loads(p.read_text())["scale"], np.float32)


def chain_trajectory(dim, split, traj):
    """(1, frames, 3, *grid) in physical units and the boundary condition of one trajectory."""
    d = D.system_dir("chain", dim)
    if (d / f"{split}_fields.npy").exists() and (d / "norm.json").exists():
        st = D.Store("chain", split, dim, normalise=False)
        print(f"read {split} trajectory {traj} from {d.relative_to(ROOT)}")
        return st.full([traj]), str(st.bc[traj])
    t0 = time.time()
    fields, chan = D.reference_subset("chain", dim, split)
    ref = json.loads((ROOT / "results" / "data_manifest_grid.json").read_text())["reference"]
    want = ref[f"d{dim}/chain/{split}"]["sha256"]
    got = D.subset_hash(fields, chan)
    print(f"recomputed the first {len(fields)} trajectories of chain/d{dim}/{split} in "
          f"{time.time() - t0:.0f} s; SHA-256 {'matches' if got == want else 'DIFFERS FROM'} "
          "the reference fingerprint")
    if traj >= len(fields):
        raise SystemExit(f"--traj must be below {len(fields)} without the stored split")
    grid = fields.shape[3:]
    out = np.zeros((1, fields.shape[1], C.N_CH) + grid, np.float32)
    for s in range(fields.shape[2]):
        out[0, :, int(chan[traj, s])] = fields[traj, :, s]
    bc_set = D.split_job("chain", dim, split)[3]
    return out, D.bcs_for("chain", bc_set)[0]          # the first batch has the first condition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2, choices=[2, 3])
    ap.add_argument("--run", default=None, help="folder under runs_grid/ (default g<d>_s0)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    dim, run = a.dim, a.run or f"g{a.dim}_s0"
    torch.set_num_threads(1)

    model = E.load_model(ROOT / "runs_grid" / run / "model.pt", dim=dim)
    true, bc = chain_trajectory(dim, a.split, a.traj)            # physical units
    g = C.GEOM[dim]
    print(f"chain d = {dim}: {g['n_out']}^{dim} cells, {true.shape[1]} frames, boundary {bc}")

    pt, ta = scale("photothermal", dim), scale("thermoacoustic", dim)
    phys = np.array([pt[0], 0.5 * (pt[1] + ta[1]), ta[2]], np.float32)
    bshape = (1, 1, 3) + (1,) * dim
    print("input constants per channel (W m^-2, K, Pa):", phys)
    n_in = E.N_IN_FRAMES
    x0 = np.ascontiguousarray(true[:, :n_in] / phys.reshape(bshape))
    with torch.no_grad():
        pred = model.rollout(torch.from_numpy(x0), a.steps).numpy() * phys.reshape(bshape)
    last = n_in - 1 + a.steps

    own = scale("chain", dim)
    t_own = true / own.reshape(bshape)
    k = (own / phys)[None, :].astype(np.float32)
    e1 = float(E.one_step(model, t_own, k)[0])
    p1 = float(E.one_step(None, t_own)[0])
    S = E.traj_scale(t_own)
    eN = float(E._err(pred[:, -1] / own.reshape((1, 3) + (1,) * dim), t_own[:, last], S)[0])
    pN = float(E._err(t_own[:, n_in - 1], t_own[:, last], S)[0])
    print(f"one-step error (mean over the {t_own.shape[1] - n_in} starts): network {e1:.6f}, "
          f"persistence {p1:.6f}")
    print(f"{a.steps}-step error: network {eN:.6f}, persistence {pN:.6f}")
    dev = ROOT / "results" / "eval_grid" / f"dev_d{dim}.json"
    if a.split == "test" and a.steps == 10 and dev.exists():
        pt_ = json.loads(dev.read_text())["per_trajectory"].get(f"{run}_phys")
        if pt_:
            print(f"stored in {dev.relative_to(ROOT)}: one-step {pt_['one_step'][a.traj]:.6f}, "
                  f"10-step {pt_['roll']['10'][a.traj]:.6f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "font.size": 8})
    mid = g["n_out"] // 2
    cut = (lambda f: f) if dim == 2 else (lambda f: f[mid])      # 3D: the central plane
    ext = [0, C.LENGTH * 1e3, 0, C.LENGTH * 1e3]
    fig, ax = plt.subplots(3, 3, figsize=(7.2, 6.6), constrained_layout=True)
    for c in range(3):
        tr, pr = cut(true[0, last, c]), cut(pred[0, -1, c])
        lim = float(max(np.abs(tr).max(), np.abs(pr).max())) or 1.0
        for j, (img, title) in enumerate([(tr, f"true, frame {last}"),
                                          (pr, f"network, step {a.steps}"),
                                          (pr - tr, "network minus true")]):
            im = ax[c, j].imshow(img.T, origin="lower", extent=ext, cmap="RdBu_r",
                                 vmin=-lim, vmax=lim)
            ax[c, j].set_title(title, fontsize=8)
            ax[c, j].set_xlabel("x (mm)")
            ax[c, j].set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax[c, :], shrink=0.85, label=LABEL[c])
    where = "" if dim == 2 else f", central plane z = {mid}"
    fig.suptitle(f"{run} on chain d = {dim}, {a.split} trajectory {a.traj}{where}", fontsize=9)
    out = ROOT / (a.plot or f"examples/predict_grid_example_d{dim}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    try:
        shown = out.resolve().relative_to(ROOT.resolve())    # inside the repository: relative path
    except ValueError:
        shown = out                                          # elsewhere: the path as given
    print("wrote", shown)


if __name__ == "__main__":
    main()
