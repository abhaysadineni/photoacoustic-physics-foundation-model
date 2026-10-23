"""Dataset generation, storage and normalisation for the Cartesian rounds (d = 2, 3),
mirroring pfm/data.py.

Layout (one directory per system): data2d/<system>/ for d = 2 and data2d/d3/<system>/ for
d = 3, with, for each split,
    <split>_fields.npy   (N, F, S, *grid) float32, physical units, the S STORED channels
    <split>_chan.npy     (N, S) int8, the output channel (0, 1, 2) of each stored slot
    <split>_bc.npy       (N,) unicode, boundary condition
    <split>_meta.json    parameters per trajectory, dt_frame, length, bc
and norm.json (one scale per channel, the RMS over the training split of trajectories
occupying the channel; no mean subtracted, as 1D). Every .npy file loads with
np.load(..., mmap_mode="r").

Storage decision (stated): only occupied channels are stored (S = 1 for light, heat, the
two sound systems and the random-channel advection-diffusion; 2 for photothermal and
thermoacoustic; 3 for the chain). The common format (frames, 3, *grid), with zeros in
unused channels, is rebuilt exactly on reading (Store.window, Store.full). This halves the
d = 2 disk use (estimate_disk gives the totals; d = 3 with train 200: about 15 GB).

Splits (training parameter ranges unless stated), boundary conditions cycled evenly over
the set as in pfm/data.py:
    d = 2: train 600, val 60, test 100 (training bcs); test_bc 100 (held-out bc);
           test_ood 100 (training bcs, ood parameters)
           (d = 2 train and val raised from 300 and 30 to the 1D sizes 600 and 60 on 21 Oct 2026,
           before any grid data existed, PLAN.md section 23)
    d = 3: train 200, val 16, test 50, test_bc 50, test_ood 50   (train raised from 100 on
           21 Oct 2026 before any grid data existed, PLAN.md section 23: 100 trajectories give
           only 2,000 distinct windows per system against 36,000 in 1D)
For the held-out chain the train split exists only for oracle and fine-tuning experiments
(as 1D). Seeds: crc32("<system>/d<d>/<split>"), new seeds never used in 1D.
"""
from __future__ import annotations

import importlib
import json
import time
import zlib
from pathlib import Path

import numpy as np

from . import common as C

REGISTRY = {
    "light":          ("pfm2d.solvers.light",   "Light",          "scalar"),
    "heat":           ("pfm2d.solvers.heat",    "Heat",           "scalar"),
    "sound_uniform":  ("pfm2d.solvers.sound",   "SoundUniform",   "sound"),
    "sound_layered":  ("pfm2d.solvers.sound",   "SoundLayered",   "sound"),
    "photothermal":   ("pfm2d.solvers.chain",   "Photothermal",   "sound"),
    "thermoacoustic": ("pfm2d.solvers.chain",   "Thermoacoustic", "sound"),
    "advdiff":        ("pfm2d.solvers.advdiff", "AdvDiff",        "scalar"),
    "chain":          ("pfm2d.solvers.chain",   "Chain",          "sound"),
}
TRAIN_SYSTEMS = ["light", "heat", "sound_uniform", "sound_layered", "photothermal",
                 "thermoacoustic", "advdiff"]
HELD_OUT = ["chain"]
STORED_CHANNELS = {"light": 1, "heat": 1, "sound_uniform": 1, "sound_layered": 1,
                   "advdiff": 1, "photothermal": 2, "thermoacoustic": 2, "chain": 3}
SPLITS = {2: {"train": 600, "val": 60, "test": 100, "test_bc": 100, "test_ood": 100},
          3: {"train": 200, "val": 16, "test": 50, "test_bc": 50, "test_ood": 50}}
JOBS = [("train", "train", "train"), ("val", "train", "train"), ("test", "train", "train"),
        ("test_bc", "train", "test"), ("test_ood", "ood", "train")]
GEN_BATCH = {2: 16, 3: 4}


def get_solver(name: str, dim: int):
    mod, cls, _ = REGISTRY[name]
    return getattr(importlib.import_module(mod), cls)(dim)


def bcs_for(name: str, which: str):
    fam = REGISTRY[name][2]
    if fam == "scalar":
        return C.TRAIN_BCS_SCALAR if which == "train" else C.TEST_BCS_SCALAR
    return C.TRAIN_BCS_SOUND if which == "train" else C.TEST_BCS_SOUND


def seed_for(system, dim, split):
    return zlib.crc32(f"{system}/d{dim}/{split}".encode())


def system_dir(name, dim, root=None):
    return (Path(root) if root else C.data_dir(dim)) / name


def _batches(name, n, seed, regime, bc_set, dim, batch, n_internal):
    """The generation sequence: yields (bc, list of trajectories) batch by batch, in the
    order generate() stores them. generate() and reference_subset() both use it, so a
    reference subset is computed by exactly the code path of the full split."""
    solver = get_solver(name, dim)
    rng = np.random.default_rng(seed)
    bcs = bcs_for(name, bc_set)
    batch = batch or GEN_BATCH[dim]
    per_bc = [n // len(bcs) + (1 if i < n % len(bcs) else 0) for i in range(len(bcs))]
    for bc, k in zip(bcs, per_bc):
        done = 0
        while done < k:
            m = min(batch, k - done)
            params = [solver.sample_params(rng, regime) for _ in range(m)]
            yield bc, solver.simulate_batch(params, bc, None, rng, n_internal)
            done += m


def generate(name: str, split: str, n: int, seed: int, regime: str = "train",
             bc_set: str = "train", dim: int = 2, batch: int | None = None,
             root=None, n_internal=None):
    """Generate n trajectories into <root>/<name>/<split>_*.npy (written incrementally)."""
    solver = get_solver(name, dim)
    S = STORED_CHANNELS[name]
    d_out = system_dir(name, dim, root)
    d_out.mkdir(parents=True, exist_ok=True)
    shape = (n, solver.n_frames, S) + (solver.n_out,) * dim
    fields = np.lib.format.open_memmap(d_out / f"{split}_fields.npy", mode="w+",
                                       dtype=np.float32, shape=shape)
    chan = np.zeros((n, S), np.int8)
    bc_arr, meta = [], []
    t0, i = time.time(), 0
    for bc, trajs in _batches(name, n, seed, regime, bc_set, dim, batch, n_internal):
        for tr in trajs:
            if not np.isfinite(tr.fields).all():
                raise RuntimeError(f"{name}/{split}: non-finite values")
            ch = list(tr.channels)
            if len(ch) != S:
                raise RuntimeError(f"{name}: {len(ch)} channels, expected {S}")
            fields[i] = tr.fields[:, ch]
            chan[i] = ch
            bc_arr.append(bc)
            meta.append({"params": tr.params, "dt_frame": tr.dt_frame, "length": tr.length,
                         "bc": tr.bc, "dim": dim, "channels": ch})
            i += 1
    fields.flush()
    del fields
    np.save(d_out / f"{split}_chan.npy", chan)
    np.save(d_out / f"{split}_bc.npy", np.array(bc_arr))
    (d_out / f"{split}_meta.json").write_text(json.dumps(meta, default=float))
    return {"system": name, "dim": dim, "split": split, "n": n, "seconds": time.time() - t0,
            "regime": regime, "bc_set": bc_set, "seed": seed}


def compute_norm(name: str, dim: int, root=None):
    d_in = system_dir(name, dim, root)
    f = np.load(d_in / "train_fields.npy", mmap_mode="r")
    ch = np.load(d_in / "train_chan.npy")
    ss, cnt = np.zeros(C.N_CH), np.zeros(C.N_CH)
    for i in range(f.shape[0]):
        x = np.asarray(f[i], np.float64)
        for s in range(f.shape[2]):
            ss[ch[i, s]] += np.sum(x[:, s] ** 2)
            cnt[ch[i, s]] += x[:, s].size
    scale = [float(np.sqrt(ss[c] / cnt[c])) if cnt[c] > 0 else 1.0 for c in range(C.N_CH)]
    scale = [s if s > 0 else 1.0 for s in scale]
    (d_in / "norm.json").write_text(json.dumps({"scale": scale}))
    return scale


def generate_system(name: str, dim: int, root=None, splits=None, n_internal=None, log=True):
    """All standard splits of one system and its normalisation; the function to call for
    the full datasets. splits: optional {split: n} override (tests)."""
    sizes = dict(SPLITS[dim]) if splits is None else dict(splits)
    out = []
    for split, regime, bc_set in JOBS:
        if split not in sizes:
            continue
        out.append(generate(name, split, sizes[split], seed_for(name, dim, split), regime,
                            bc_set, dim, root=root, n_internal=n_internal))
        print(out[-1], flush=True)
    out.append({"norm_scale": compute_norm(name, dim, root)})
    if log and root is None:
        C.save_json(out, C.ROOT / "results" / "datagen_grid" / f"datagen_{name}_d{dim}.json")
    return out


def subset_hash(fields, chan):
    """SHA-256 of stored float32 fields (k, F, S, *grid) and channel slots (k, S), as bytes."""
    import hashlib
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(fields, np.float32).tobytes())
    h.update(np.ascontiguousarray(chan, np.int8).tobytes())
    return h.hexdigest()


def split_job(name, dim, split):
    """(n, seed, regime, bc_set) of a standard split."""
    for sp, regime, bc_set in JOBS:
        if sp == split:
            return SPLITS[dim][split], seed_for(name, dim, split), regime, bc_set
    raise KeyError(split)


def reference_subset(name, dim, split, n_batches=1):
    """The first n_batches generation batches of a standard split, recomputed through the
    production path (_batches): returns (fields, chan) exactly as generate() stores them in
    rows 0 .. k-1, k the number of trajectories in those batches."""
    n, seed, regime, bc_set = split_job(name, dim, split)
    S = STORED_CHANNELS[name]
    fs, cs = [], []
    for j, (bc, trajs) in enumerate(_batches(name, n, seed, regime, bc_set, dim, None, None)):
        if j >= n_batches:
            break
        for tr in trajs:
            ch = list(tr.channels)
            fs.append(tr.fields[:, ch])
            cs.append(ch)
    return np.stack(fs).astype(np.float32), np.array(cs, np.int8).reshape(len(cs), S)


class Store:
    """Memory-mapped split of one system, giving fields in the common format
    (frames, 3, *grid), normalised by the system's training scale unless normalise=False
    (physical units)."""

    def __init__(self, name, split, dim, root=None, normalise=True):
        d_in = system_dir(name, dim, root)
        self.dir = d_in
        self.name, self.split, self.dim = name, split, dim
        self.f = np.load(d_in / f"{split}_fields.npy", mmap_mode="r")
        self.chan = np.load(d_in / f"{split}_chan.npy")
        self.bc = np.load(d_in / f"{split}_bc.npy")
        self.scale = np.array(json.loads((d_in / "norm.json").read_text())["scale"], np.float32)
        self.normalise = normalise
        self.n, self.n_frames = self.f.shape[0], self.f.shape[1]
        self.grid = self.f.shape[3:]

    def __len__(self):
        return self.n

    def window(self, i, t, k):
        """Frames t .. t+k-1 of trajectory i, (k, 3, *grid) float32."""
        raw = np.asarray(self.f[i, t:t + k])
        out = np.zeros((k, C.N_CH) + self.grid, np.float32)
        for s in range(raw.shape[1]):
            c = int(self.chan[i, s])
            out[:, c] = raw[:, s] / self.scale[c] if self.normalise else raw[:, s]
        return out

    def full(self, idx=None):
        idx = range(self.n) if idx is None else idx
        return np.stack([self.window(i, 0, self.n_frames) for i in idx])

    def meta(self):
        return json.loads((self.dir / f"{self.split}_meta.json").read_text())


def estimate_disk(dim: int, systems=None):
    """Bytes of the full standard splits (fields only; the other files are small)."""
    g = C.GEOM[dim]
    per_slot = g["n_frames"] * g["n_out"] ** dim * 4
    n = sum(SPLITS[dim].values())
    systems = systems or (TRAIN_SYSTEMS + HELD_OUT)
    rows = {s: n * STORED_CHANNELS[s] * per_slot for s in systems}
    full = n * C.N_CH * per_slot * len(systems)
    return {"dim": dim, "trajectories_per_system": n, "bytes_per_system": rows,
            "total_bytes": sum(rows.values()), "total_GB": sum(rows.values()) / 1e9,
            "if_all_three_channels_stored_GB": full / 1e9}
