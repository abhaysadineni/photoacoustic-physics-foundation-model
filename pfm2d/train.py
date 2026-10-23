"""Training loop for the d-dimensional surrogate (d = 2, 3), the loop of pfm/train.py.

Each step draws a batch of windows: a system chosen uniformly from the training list, a
trajectory chosen uniformly from that system, a start frame chosen uniformly. Windows are
read from the memory-mapped arrays of pfm2d/data.py (Store.window), so no split is held in
memory. Input frames t..t+3, target t+4, MSE on the next frame in normalised units. AdamW,
lr 3e-4 with 500 warm-up steps and cosine decay to zero, weight decay 0.01, gradient norm
clipped at 1. Validation every val_every steps on a fixed set of windows (512 for d = 2,
128 for d = 3, evaluated in chunks of 32; the smaller d = 3 set keeps it under 210 MB);
select="best" keeps the checkpoint with the lowest validation loss, select="last" the final
weights (fine-tuning). Resume from out/state.pt (model, optimiser, both generators, best
score, log). Options as pfm/train.py: noise_std (input noise, normalised units),
pushforward (half of each batch trained on inputs containing the network's own one-step
prediction, no gradient through that step), n_traj (fixed-seed subset balanced over the two
halves of each split), width, depth, heads.
Device (added 21 Oct 2026, as pfm/train.py): CUDA when a GPU is present and PFM_FORCE_CPU is
unset (pfm.train.pick_device), else CPU. Windows are drawn on the CPU from the same generator
in the same order on every device; on the CPU path every tensor stays where it was, so CPU
results are numerically unchanged. GPU runs are not bit-for-bit reproducible across devices.
Checkpoints (model.pt) are always saved as CPU tensors.
Default batch: 32 for d = 2 and 16 for d = 3 (1D used 64; the per-sample cost is 1.9 and
about 8 times larger here; modelling choice).

Usage:
    python -m pfm2d.train --dim 2 --systems light heat ... --steps 20000 --out runs2d/main_s0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from pfm.train import pick_device

from .common import ROOT
from .data import Store
from .model import Surrogate, n_params, N_IN_FRAMES

DEFAULT_BATCH = {2: 32, 3: 16}
N_VAL = {2: 512, 3: 128}


class Pool:
    """Training windows from a list of Stores, each restricted to an index list."""

    def __init__(self, stores, idx=None):
        self.stores = stores
        self.idx = idx or [list(range(len(s))) for s in stores]

    def draw(self, rng, n, k):
        """n windows of k consecutive frames: array (n, k, 3, *grid)."""
        out = []
        sys_idx = rng.integers(len(self.stores), size=n)
        for s in sys_idx:
            st, ids = self.stores[s], self.idx[s]
            i = ids[int(rng.integers(len(ids)))]
            t = int(rng.integers(st.n_frames - k + 1))
            out.append(st.window(i, t, k))
        return np.stack(out)


def windows(pool, rng, n):
    w = torch.from_numpy(pool.draw(rng, n, N_IN_FRAMES + 1))
    return w[:, :N_IN_FRAMES], w[:, N_IN_FRAMES]


def windows_pf(pool, rng, n):
    w = torch.from_numpy(pool.draw(rng, n, N_IN_FRAMES + 2))
    return w[:, :N_IN_FRAMES], w[:, N_IN_FRAMES], w[:, N_IN_FRAMES + 1]


def lr_at(step, total, base=3e-4, warm=500):
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return base * 0.5 * (1 + math.cos(math.pi * p))


def _val_loss(model, vx, vy, chunk=32):
    tot = 0.0
    for j in range(0, len(vx), chunk):
        tot += float(torch.sum((model(vx[j:j + chunk]) - vy[j:j + chunk]) ** 2))
    return tot / vy.numel()


def train(systems, steps, seed, out, dim=2, use_derivs=True, batch=None, threads=1,
          init=None, base_lr=3e-4, warm=500, train_split="train", n_traj=None,
          val_every=500, select="best", noise_std=0.0, pushforward=False,
          width=192, depth=6, heads=6, data_root=None, stop_after=None, prefetch=False):
    """stop_after: stop (state saved, as an interrupted job) once this many steps are done
    and a validation point has been written; a later call with the same arguments resumes.
    prefetch: draw the next batch in one background thread while the current step runs. The
    draws use the same generator in the same order (one draw at a time), so the batches are
    identical with and without prefetch; the checkpoint stores the generator state from
    before the pending draw, so a resumed run continues the same sequence."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    batch = batch or DEFAULT_BATCH[dim]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    tr_st = [Store(s, train_split, dim, data_root) for s in systems]
    log_idx = None
    idx = None
    if n_traj is not None:
        idx = []
        for st in tr_st:
            half = st.n // 2
            prng = np.random.default_rng(1000 + n_traj)
            blocks = [prng.permutation(half), half + prng.permutation(st.n - half)]
            idx.append([int(blocks[i % 2][i // 2]) for i in range(n_traj)])
        log_idx = idx
    pool = Pool(tr_st, idx)
    vpool = Pool([Store(s, "val", dim, data_root) for s in systems])
    vx, vy = windows(vpool, np.random.default_rng(12345), N_VAL[dim])

    dev = pick_device()
    vx, vy = vx.to(dev), vy.to(dev)
    model = Surrogate(dim, use_derivs=use_derivs, width=width, depth=depth, heads=heads)
    if init:
        model.load_state_dict(torch.load(init, weights_only=True, map_location="cpu"))
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    log = {"systems": systems, "dim": dim, "steps": steps, "seed": seed, "use_derivs": use_derivs,
           "batch": batch, "n_params": n_params(model), "init": str(init) if init else None,
           "n_traj": n_traj, "traj_idx": log_idx, "base_lr": base_lr, "select": select,
           "noise_std": noise_std, "pushforward": pushforward, "n_val_windows": N_VAL[dim],
           "arch": {"width": width, "depth": depth, "heads": heads},
           "data_root": str(data_root) if data_root else None, "device": str(dev),
           "prefetch": bool(prefetch), "history": []}
    best, t0, start = float("inf"), time.time(), 0
    state_path = out / "state.pt"
    if state_path.exists():
        st = torch.load(state_path, weights_only=False, map_location=dev)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        rng.bit_generator.state = st["np_rng"]
        torch.set_rng_state(st["torch_rng"])
        if dev.type == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(st["cuda_rng"].cpu())
        best, start, log = st["best"], st["step"], st["log"]
        t0 = time.time() - st["elapsed"]
        log.setdefault("resumed_at", []).append(start)
    def fetch():
        if pushforward:
            n_pf = batch // 2
            return windows(pool, rng, batch - n_pf) + windows_pf(pool, rng, n_pf)
        return windows(pool, rng, batch)

    ex = None
    if prefetch:
        from concurrent.futures import ThreadPoolExecutor
        ex = ThreadPoolExecutor(1)
        pending_state = rng.bit_generator.state
        fut = ex.submit(fetch)
    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, base_lr, warm)
        if ex is None:
            got = fetch()
        else:
            got = fut.result()
            pending_state = rng.bit_generator.state
            fut = ex.submit(fetch) if step + 1 < steps else None
        if pushforward:
            x, y, xp, yp1, yp2 = got
            x, y, xp, yp2 = x.to(dev), y.to(dev), xp.to(dev), yp2.to(dev)
            if noise_std > 0:
                x = x + noise_std * torch.randn_like(x)
                xp = xp + noise_std * torch.randn_like(xp)
            with torch.no_grad():
                p1 = model(xp)
            xp2 = torch.cat([xp[:, 1:], p1[:, None]], dim=1)
            loss = (torch.sum((model(x) - y) ** 2) + torch.sum((model(xp2) - yp2) ** 2)) / (
                batch * y[0].numel())
        else:
            x, y = got
            x, y = x.to(dev), y.to(dev)
            if noise_std > 0:
                x = x + noise_std * torch.randn_like(x)
            loss = torch.mean((model(x) - y) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % val_every == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vl = _val_loss(model, vx, vy)
            model.train()
            log["history"].append({"step": step + 1, "train_loss": loss.item(),
                                   "val_loss": vl, "seconds": time.time() - t0})
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if select == "best" and vl < best:
                best = vl
                torch.save(cpu_state, out / "model.pt")
            elif select == "last":
                best = vl
                torch.save(cpu_state, out / "model.pt")
            (out / "log.json").write_text(json.dumps(log, indent=1))
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "np_rng": rng.bit_generator.state if ex is None else pending_state,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state() if dev.type == "cuda" else None,
                        "best": best, "step": step + 1, "log": log,
                        "elapsed": time.time() - t0}, state_path)
            if stop_after is not None and step + 1 >= stop_after and step + 1 < steps:
                if ex is not None:
                    ex.shutdown(wait=True)
                return {"stopped_at": step + 1}
    if ex is not None:
        ex.shutdown(wait=True)
    log["best_val_loss"] = best
    log["seconds"] = time.time() - t0
    (out / "log.json").write_text(json.dumps(log, indent=1))
    state_path.unlink(missing_ok=True)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-derivs", action="store_true")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--init", default=None)
    ap.add_argument("--n-traj", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warm", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--select", choices=["best", "last"], default="best")
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--pushforward", action="store_true")
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--prefetch", action="store_true")
    a = ap.parse_args()
    train(a.systems, a.steps, a.seed, ROOT / a.out, dim=a.dim, use_derivs=not a.no_derivs,
          batch=a.batch, threads=a.threads, init=a.init, n_traj=a.n_traj, base_lr=a.lr,
          warm=a.warm, val_every=a.val_every, select=a.select, noise_std=a.noise_std,
          pushforward=a.pushforward, width=a.width, depth=a.depth, heads=a.heads,
          data_root=a.data_root, prefetch=a.prefetch)
