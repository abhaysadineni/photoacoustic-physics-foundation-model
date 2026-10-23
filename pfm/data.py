"""Dataset generation, storage and normalisation.

Layout: data/<system>/<split>.npz with arrays
    fields   (N, F, 3, 128) float32, physical units (unnormalised)
    bc       (N,) str
    channels (N, 3) bool, which channels the trajectory occupies
plus data/<system>/<split>_meta.json (parameters per trajectory, dt_frame, length).

Normalisation: data/<system>/norm.json holds one scale per channel, the root mean square of
that channel over the training split, counting only trajectories that occupy the channel.
No mean is subtracted, so zero stays zero (unused channels, Dirichlet walls). For the two
held-out systems the "train" split exists only for the oracle and fine-tuning experiments, and
their scales come from that split, never from the test split.
"""
from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import numpy as np

from .common import ROOT, N_CH, TRAIN_BCS_SCALAR, TEST_BCS_SCALAR, TRAIN_BCS_SOUND, TEST_BCS_SOUND

DATA = ROOT / "data"

# name -> (module, class name, boundary family)
REGISTRY = {
    "light":         ("pfm.solvers.light",     "LightSolver",        "scalar"),
    "heat":          ("pfm.solvers.heat",      "HeatSolver",         "scalar"),
    "sound_uniform": ("pfm.solvers.sound",     "SoundUniform",       "sound"),
    "sound_layered": ("pfm.solvers.sound",     "SoundLayered",       "sound"),
    "sound_lossy":   ("pfm.solvers.sound",     "SoundLossy",         "sound"),
    "transport":     ("pfm.solvers.transport", None,                 "scalar"),
    "burgers":       ("pfm.solvers.burgers",   None,                 "scalar"),
    "fisher":        ("pfm.solvers.fisher",    None,                 "scalar"),
    "chain":         ("pfm.solvers.chain",     "PhotoacousticChain", "sound"),
    "telegraph":     ("pfm.solvers.telegraph", None,                 "sound"),
    # round 2: one link of the chain each (added 21 Oct 2026)
    "photothermal":   ("pfm.solvers.coupled",  "Photothermal",       "sound"),
    "thermoacoustic": ("pfm.solvers.coupled",  "Thermoacoustic",     "sound"),
}
# Round 3 (21 Oct 2026): radial profiles of the tissue systems in a cylinder (r2) and a sphere
# (r3), and the chain in both, registered as separate systems "<name>_r2" and "<name>_r3".
RADIAL_BASE = ["light", "heat", "sound_uniform", "sound_layered", "sound_lossy",
               "photothermal", "thermoacoustic"]
for _b in RADIAL_BASE + ["chain"]:
    for _d in (2, 3):
        _m, _c, _f = REGISTRY[_b]
        REGISTRY[f"{_b}_r{_d}"] = (_m, _c, _f, _d)
TRAIN_SYSTEMS = ["light", "heat", "sound_uniform", "sound_layered", "sound_lossy",
                 "transport", "burgers", "fisher"]
HELD_OUT = ["chain", "telegraph"]
# Round 2: the eight round-1 systems (data unchanged) plus the two single-link coupled systems.
COUPLED = ["photothermal", "thermoacoustic"]
TRAIN_SYSTEMS_V2 = TRAIN_SYSTEMS + COUPLED
RADIAL_SYSTEMS = [f"{b}_r{d}" for d in (2, 3) for b in RADIAL_BASE]
TRAIN_SYSTEMS_V3 = TRAIN_SYSTEMS_V2 + RADIAL_SYSTEMS
HELD_OUT_V3 = ["chain", "chain_r2", "chain_r3", "telegraph"]
SCALING_SUBSETS = {
    3: ["light", "heat", "sound_uniform"],
    5: ["light", "heat", "sound_uniform", "sound_layered", "sound_lossy"],
    8: TRAIN_SYSTEMS,
}

N_FRAMES = 64
SPLITS = {"train": 600, "val": 60, "test": 100}


def get_solver(name: str):
    from .common import Solver
    entry = REGISTRY[name]
    mod_name, cls_name = entry[0], entry[1]
    if len(entry) == 4:                       # radial system: construct with its geometry
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls_name)(dim=entry[3])
    mod = importlib.import_module(mod_name)
    if cls_name is None:
        cands = [v for v in vars(mod).values()
                 if isinstance(v, type) and issubclass(v, Solver) and v is not Solver
                 and v.__module__ == mod.__name__]
        if len(cands) != 1:
            raise RuntimeError(f"{mod_name}: expected one Solver subclass, found {cands}")
        return cands[0]()
    return getattr(mod, cls_name)()


def bcs_for(name: str, which: str):
    family = REGISTRY[name][2]
    if len(REGISTRY[name]) == 4:              # radial: outer wall only, no periodic
        if family == "scalar":
            return ("dirichlet",) if which == "train" else ("neumann",)
        return ("pressure_release",) if which == "train" else ("rigid",)
    if family == "scalar":
        return TRAIN_BCS_SCALAR if which == "train" else TEST_BCS_SCALAR
    return TRAIN_BCS_SOUND if which == "train" else TEST_BCS_SOUND


def _to_arrays(trajs):
    fields = np.stack([t.fields for t in trajs]).astype(np.float32)
    chans = np.zeros((len(trajs), N_CH), bool)
    for i, t in enumerate(trajs):
        for c in t.channels:
            chans[i, c] = True
    meta = [{"params": t.params, "dt_frame": t.dt_frame, "length": t.length, "bc": t.bc}
            for t in trajs]
    return fields, chans, meta


def generate(name: str, split: str, n: int, seed: int, regime: str = "train",
             bc_set: str = "train", batch: int = 32, out_name: str | None = None):
    """Generate n trajectories, boundary conditions cycled evenly over the bc set."""
    solver = get_solver(name)
    rng = np.random.default_rng(seed)
    bcs = bcs_for(name, bc_set)
    trajs, t0 = [], time.time()
    per_bc = [n // len(bcs) + (1 if i < n % len(bcs) else 0) for i in range(len(bcs))]
    for bc, k in zip(bcs, per_bc):
        done = 0
        while done < k:
            m = min(batch, k - done)
            params = [solver.sample_params(rng, regime) for _ in range(m)]
            trajs += solver.simulate_batch(params, bc, N_FRAMES, rng)
            done += m
    fields, chans, meta = _to_arrays(trajs)
    if not np.isfinite(fields).all():
        raise RuntimeError(f"{name}/{split}: non-finite values")
    d = DATA / name
    d.mkdir(parents=True, exist_ok=True)
    tag = out_name or split
    np.savez(d / f"{tag}.npz", fields=fields, channels=chans,
             bc=np.array([m["bc"] for m in meta]))
    (d / f"{tag}_meta.json").write_text(json.dumps(meta, default=float))
    return {"system": name, "split": tag, "n": len(trajs), "seconds": time.time() - t0,
            "regime": regime, "bc_set": bc_set}


def compute_norm(name: str):
    z = np.load(DATA / name / "train.npz")
    f, ch = z["fields"], z["channels"]
    scale = []
    for c in range(N_CH):
        sel = f[ch[:, c], :, c]
        scale.append(float(np.sqrt(np.mean(sel ** 2))) if sel.size else 1.0)
    scale = [s if s > 0 else 1.0 for s in scale]
    (DATA / name / "norm.json").write_text(json.dumps({"scale": scale}))
    return scale


def load(name: str, split: str, normalise: bool = True) -> np.ndarray:
    f = np.load(DATA / name / f"{split}.npz")["fields"]
    if normalise:
        s = np.array(json.loads((DATA / name / "norm.json").read_text())["scale"], np.float32)
        f = f / s[None, None, :, None]
    return f.astype(np.float32)
