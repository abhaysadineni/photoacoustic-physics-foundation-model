"""Training loop for the surrogate and every comparison network.

Each step draws a batch of windows: a system chosen uniformly from the training list, a
trajectory chosen uniformly from that system, and a start frame chosen uniformly. Input is
frames t..t+3, target is frame t+4. Loss is mean squared error on the next frame in
normalised units. AdamW, learning rate 3e-4 with 500 warm-up steps and cosine decay to zero,
weight decay 0.01, batch 64, gradient norm clipped at 1. Validation every 500 steps on a
fixed set of windows; the checkpoint with the lowest validation loss is kept.

Round 2 options (defaults reproduce round 1 exactly, including the random-number sequence):
  noise_std    Gaussian noise of this standard deviation (normalised units) added to the four
               input frames; the target stays clean, so the network learns to correct small
               input errors of the kind a rollout produces (input-noise training).
  pushforward  with probability 1/2 per sample, the window is extended by one frame: the
               network first predicts frame t+4 from t..t+3 with gradients switched off, that
               prediction replaces the true frame in the window t+1..t+4, and the loss is taken
               on frame t+5. The network is thereby trained on inputs that contain its own
               errors (the pushforward trick). Cost about 1.25 times a plain step.
  width, depth, heads   network size (round 1: 192, 6, 6).

Usage:
    python -m pfm.train --systems light heat ... --steps 20000 --seed 0 --out runs/main_s0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from .common import ROOT
from .data import load
from .model import Surrogate, n_params, N_IN_FRAMES


def windows(arrs, rng, n):
    """Sample n (input, target) windows from a list of (N, F, 3, 128) arrays."""
    xs, ys = [], []
    sys_idx = rng.integers(len(arrs), size=n)
    for s in sys_idx:
        a = arrs[s]
        i = rng.integers(a.shape[0])
        t = rng.integers(a.shape[1] - N_IN_FRAMES)
        xs.append(a[i, t:t + N_IN_FRAMES])
        ys.append(a[i, t + N_IN_FRAMES])
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def windows_pf(arrs, rng, n):
    """Windows of 4 input frames plus TWO following frames (t+4, t+5)."""
    xs, y1, y2 = [], [], []
    sys_idx = rng.integers(len(arrs), size=n)
    for s in sys_idx:
        a = arrs[s]
        i = rng.integers(a.shape[0])
        t = rng.integers(a.shape[1] - N_IN_FRAMES - 1)
        xs.append(a[i, t:t + N_IN_FRAMES])
        y1.append(a[i, t + N_IN_FRAMES])
        y2.append(a[i, t + N_IN_FRAMES + 1])
    f = lambda v: torch.from_numpy(np.stack(v))
    return f(xs), f(y1), f(y2)


def pick_device():
    """CUDA when a GPU is present (and PFM_FORCE_CPU is unset), else CPU. The CPU path is
    unchanged from round 1; GPU runs are not bit-for-bit reproducible across devices."""
    if torch.cuda.is_available() and not os.environ.get("PFM_FORCE_CPU"):
        return torch.device("cuda")
    return torch.device("cpu")


def lr_at(step, total, base=3e-4, warm=500):
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return base * 0.5 * (1 + math.cos(math.pi * p))


def train(systems, steps, seed, out, use_derivs=True, batch=64, threads=1,
          init=None, base_lr=3e-4, warm=500, train_split="train", n_traj=None,
          val_every=500, select="best", noise_std=0.0, pushforward=False,
          width=192, depth=6, heads=6):
    """select="best" keeps the checkpoint with the lowest validation loss; select="last"
    keeps the final weights and uses validation only for logging (fine-tuning, so the
    target system's 60 validation trajectories never steer a few-shot result)."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    log_idx = None
    tr = [load(s, train_split) for s in systems]
    if n_traj is not None:
        # Fine-tuning on a few trajectories (audit round 1, M2): stored splits hold one
        # boundary condition per contiguous block, so draw the subset with a fixed seed,
        # alternating between the two halves so boundary conditions are balanced.
        picked = []
        for a in tr:
            half = a.shape[0] // 2
            prng = np.random.default_rng(1000 + n_traj)
            blocks = [prng.permutation(half), half + prng.permutation(a.shape[0] - half)]
            idx = [int(blocks[i % 2][i // 2]) for i in range(n_traj)]
            picked.append(a[idx])
        tr = picked
        log_idx = idx
    va = [load(s, "val") for s in systems]
    vrng = np.random.default_rng(12345)
    vx, vy = windows(va, vrng, 512)

    dev = pick_device()
    vx, vy = vx.to(dev), vy.to(dev)
    model = Surrogate(use_derivs=use_derivs, width=width, depth=depth, heads=heads)
    if init:
        model.load_state_dict(torch.load(init, weights_only=True, map_location="cpu"))
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    log = {"systems": systems, "steps": steps, "seed": seed, "use_derivs": use_derivs,
           "batch": batch, "n_params": n_params(model), "init": str(init) if init else None,
           "n_traj": n_traj, "traj_idx": log_idx, "base_lr": base_lr, "select": select,
           "noise_std": noise_std, "pushforward": pushforward,
           "arch": {"width": width, "depth": depth, "heads": heads}, "device": str(dev),
           "history": []}
    best, t0, start = float("inf"), time.time(), 0
    state_path = out / "state.pt"
    if state_path.exists():
        # Resume after an interrupted run (each background command is time-limited).
        # Model, optimiser, both random generators, the best score and the log are restored,
        # so the resumed run follows the same sequence of batches as an uninterrupted one.
        st = torch.load(state_path, weights_only=False, map_location=dev)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        rng.bit_generator.state = st["np_rng"]
        torch.set_rng_state(st["torch_rng"])
        best, start, log = st["best"], st["step"], st["log"]
        t0 = time.time() - st["elapsed"]
        log.setdefault("resumed_at", []).append(start)
    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, base_lr, warm)
        if pushforward:
            n_pf = batch // 2
            x, y = windows(tr, rng, batch - n_pf)
            xp, yp1, yp2 = windows_pf(tr, rng, n_pf)
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
            x, y = windows(tr, rng, batch)
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
                vl = float(torch.mean((model(vx) - vy) ** 2))
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
                        "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "best": best, "step": step + 1, "log": log,
                        "elapsed": time.time() - t0}, state_path)
    log["best_val_loss"] = best
    log["seconds"] = time.time() - t0
    (out / "log.json").write_text(json.dumps(log, indent=1))
    state_path.unlink(missing_ok=True)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-derivs", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--init", default=None, help="checkpoint to start from (fine-tuning)")
    ap.add_argument("--n-traj", type=int, default=None, help="train on this many trajectories")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warm", type=int, default=500)
    ap.add_argument("--select", choices=["best", "last"], default="best")
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--pushforward", action="store_true")
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    a = ap.parse_args()
    train(a.systems, a.steps, a.seed, ROOT / a.out, use_derivs=not a.no_derivs,
          threads=a.threads, init=a.init, n_traj=a.n_traj, base_lr=a.lr, warm=a.warm,
          select=a.select, noise_std=a.noise_std, pushforward=a.pushforward,
          width=a.width, depth=a.depth, heads=a.heads)
