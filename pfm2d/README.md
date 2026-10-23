# pfm2d: the Cartesian-grid rounds (d = 2 and d = 3)

This package holds the two- and three-dimensional version of the study: solvers on a true Cartesian grid, the dataset layout, the patch-transformer surrogate, its training loop and the evaluation metrics. The pre-registration, every design decision and the results are in PLAN.md, section 23 and its sub-sections 23a to 23d. This kit does not include PLAN.md, the GPU queue (`tools/gpu_queue.py`, `gpu/jobs.txt`) or the audit records, so references to them below point to the original research record. Every command below is run from the repository root. All paths are relative to it.

## Geometry and format

| | d = 2 | d = 3 |
|---|---|---|
| Output grid | 64 x 64 cells of 0.2 mm | 32^3 cells of 0.4 mm |
| Frames per trajectory | 32 | 24 |
| Default internal grid | 128^2 | 64^3 |
| Internal grid of the two sound systems | 256^2 | 96^3 |

The tissue window is 12.8 mm in every direction, the same as in 1D. A trajectory is stored as an array of shape (frames, 3, *grid) in float32. Channel 0 is fluence rate, channel 1 is temperature rise and channel 2 is acoustic pressure, as in 1D.

## Modules

| File | What it does |
|---|---|
| `common.py` | Geometry table `GEOM`, boundary-condition sets, cell centres and meshes, cell-average downsampling, the `Solver` interface, and the results directories (`results/verification2d`, `results/verification3d`). |
| `solvers/grid.py` | Shared finite-volume numerics for any d. It provides the two-point diffusion operator with periodic, Dirichlet and Neumann walls, and a Gershgorin bound for the RK4 step. It provides the staggered acoustic operator with its discrete energy. It also provides sparse assembly of the steady diffusion operator, solved by SuperLU in 2D and by Jacobi-preconditioned conjugate gradients in 3D. |
| `solvers/tissue.py` | Optical tissue draws in d dimensions: the 1D chromophore model, plus disc, ball or vessel-segment absorbers, under the mu_a / mu_s' <= 0.3 cap. |
| `solvers/light.py` | Time-resolved diffusion of light (channel 0). |
| `solvers/heat.py` | Heat conduction with hot spots and heaters (channel 1). |
| `solvers/sound.py` | Linear acoustics in a uniform medium, or layered as bands or inclusions (channel 2). Pulse widths are drawn as 3 to 10 output cells. |
| `solvers/chain.py` | The photoacoustic chain (held out) and its two single-link systems, photothermal and thermoacoustic. It uses quasi-static fluence, a sin^2 heating pulse and coupled temperature-pressure stepping. |
| `solvers/advdiff.py` | Generic dimensionless advection-diffusion, written to a random channel. |
| `solvers/references.py` | Exact radial references for the acoustic checks: closed forms in 3D, and a Hankel transform with a checked spline in 2D. |
| `solvers/reduction.py` | Consistency check: a problem that is uniform transversally must reproduce the unchanged 1D solvers. |
| `data.py` | Dataset generation, storage of only the occupied channels, normalisation, `Store` (memory-mapped reading into the common format) and reference subsets for the determinism check. Splits: 2D train/val/test/test_bc/test_ood = 600/60/100/100/100; 3D = 200/16/50/50/50. Seeds are crc32("<system>/d<d>/<split>"). |
| `model.py` | Patch transformer: 8^d patches (64 tokens), four input frames, derivative channels, factorised position table, width 192, depth 6, 6 heads. It predicts a rate; the next frame is one Euler step. |
| `train.py` | Training loop. It uses CUDA when present unless `PFM_FORCE_CPU` is set. It can prefetch the next batch in a thread with identical batches, and it resumes from `state.pt`. The CPU path is bit-identical to the pre-GPU version (`results/verification2d/train_cpu_regression.json`). |
| `evaluate.py` | Error metric and persistence baseline. Also one-step and rollout errors, seed-frame and physical-unit input factors, and bootstrap and paired-bootstrap intervals. |

## Reproducing the verification (CPU, one core per process)

```
export OMP_NUM_THREADS=1
python tests2d/verify_light.py 2 3
python tests2d/verify_heat.py 2 3
python tests2d/verify_sound.py 2 3
python tests2d/verify_advdiff.py 2 3
python tests2d/verify_chain.py 2 3          # chain, photothermal, thermoacoustic
python tests2d/verify_consistency.py 2 3    # reduction to 1D, link consistency, operator
python tests2d/resolution_sound.py 3 2      # stored-data accuracy of the sound systems
python tools/grid_verification_table.py     # the markdown table of all results
```

Each script writes `results/verification<d>d/<system>_d<d>.json`. The 3D checks take about 2 hours in total on a 2-core machine; most of that is the light check at 128^3 and the sound resolution check at 192^3. The training regression check needs the pre-change training loop, for example:

```
git show e1e7159:pfm2d/train.py > train_old.py
python tests2d/check_train_device.py train_old.py
```

## Data generation and the cross-machine check

1. On the reference machine, fingerprint the first generation batch of every system and split:
   `python experiments/gen_data_grid.py --reference --dim 2 3` writes `results/data_manifest_grid.json`. The fingerprint is a SHA-256 plus a tolerance fingerprint: relative tolerance 1e-6, 1,000 fixed-position values, and per-frame norms and maxima.
2. On the training machine, generate the full splits:
   `python experiments/gen_data_grid.py --dim 2 --workers 4` (and `--dim 3`). Each split is compared with the fingerprints as it finishes. The job stops with "DATA MISMATCH" unless the match is strict or within tolerance. It writes `results/datagen_grid/manifest_full_d<d>.json` (SHA-256 of every file), `norm_d<d>_<system>.json` and `ref_seed_rms_d<d>.json`. The full data are about 6 GB (2D) and 14 GB (3D).
3. `python tools/check_grid_data.py <d>` re-verifies the data before every training job.
4. Locked test sets are made once, on the CPU machine only: `python experiments/make_locked_grid.py` (d = 2: 200 trajectories, d = 3: 100, read-only, hashes in `results/final_locked_manifest_grid.json`). `--check` verifies them. The development chain splits can be regenerated locally with `python experiments/gen_data_grid.py --local-dev --dim 2 3`.

## Training on a GPU machine

`experiments/train_jobs_grid.py` defines the 14 pre-registered runs. In each dimension these are three seeds of the main network on the seven training systems, three seeds without the two coupled systems, and a chain oracle. They use 30,000 steps (oracles 12,000) at batch 64.

```
python tools/check_grid_data.py 2 && python experiments/train_jobs_grid.py g2_s0
```

Each run writes `runs_grid/<name>/model.pt` and `log.json`. On an NVIDIA L4 a 2D run takes about 10 minutes and a 3D run about 44 minutes (`results/timing_grid/timing_gpu.json`, from `python experiments/time_grid.py`). The repository's GPU queue (`tools/gpu_queue.py`, `gpu/jobs.txt`) ran these jobs. The lines of `gpu/jobs.txt` that produced the results, in order, are listed below. The earlier lines g2_s0 to g2oracle_s0 and g3_s0 failed on a checker bookkeeping error and are kept only as a record (PLAN.md section 23b).
- `gtime1`: timing on synthetic arrays.
- `gdata3bg`: 3D data, started in the background.
- `gdata2`: 2D data.
- `g2_s0_r2`, `g2nc_s0_r2`, `g2_s1_r2`, `g2_s2_r2`, `g2nc_s1_r2`, `g2nc_s2_r2`, `g2oracle_s0_r2`: the 2D runs.
- `g3nc_s0`, `g3_s1`, `g3_s2`, `g3nc_s1`, `g3nc_s2`, `g3oracle_s0`, `g3_s0_r2`: the 3D runs.

## Evaluation

```
python tools/fetch_gpu_grid.py g2_s0 g2_s1 g2_s2 g2nc_s0 g2nc_s1 g2nc_s2 g2oracle_s0   # models and norm files from branch gpu
python experiments/run_eval_grid.py --dim 2 --split test --models g2_s0 g2_s1 g2_s2 g2nc_s0 g2nc_s1 g2nc_s2 g2oracle_s0 \
    --group g2=g2_s0,g2_s1,g2_s2 --group g2nc=g2nc_s0,g2nc_s1,g2nc_s2 --out results/eval_grid/dev_d2.json
python tools/summarise_grid_eval.py results/eval_grid/dev_d2.json
```

Each run reports persistence and, for every network, three input conditions. The primary condition is physical-unit scaling from the training systems, the secondary is seed-frame scaling, and physmaskL is a diagnostic only. Results include one-step and rollout errors (10, 25, 28 steps in 2D; 10, 20 in 3D), per-channel and per-boundary-condition errors, and paired bootstraps against persistence and between networks. The locked commands, run once, are in PLAN.md sections 23c and 23d. `--cache` is for development splits only and is refused for `final_locked`.

## Limits stated with the results

- Sound pulses cover 3 to 10 output cells (2D 0.6 to 2.0 mm, 3D 1.2 to 4.0 mm), not the 1D millimetre range.
- The worst-case stored pressure error is 0.90 per cent in 2D and 1.44 per cent in 3D.
- 3D has 4,000 distinct training windows per system, against 36,000 in 1D.
- The 3D development set has 50 trajectories and the 3D locked set 100.
