# Photoacoustic Physics Foundation Model

Abhay Chakra Sadineni · Advisor: Prof. Gerard Marriott · Version 1.0, October 2026

One transformer learned light, heat and sound in tissue separately, together with each single link between them. It was then asked to predict the full photoacoustic chain (light heats tissue, heat launches sound) that it had never been shown. On locked test sets, opened once, its one-step forecast error was between 0.056 and 0.177 of the error of the obvious baseline, repeating the last frame. That held in one dimension, in cylinder and sphere geometry, and on true 2D and 3D grids.

This folder holds everything needed to run that model: the solvers that generate the physics, the network code, the four trained networks behind the headline results, worked examples, and the scripts that reproduce a locked result from scratch. The networks also run, with no installation, in a web browser at https://www.abhaychakra.com/p/photoacoustic-pfm/.

## Results

Locked test sets of 200 chain trajectories each (100 in 3D). One-step values are ratios to the error of repeating the last frame; lower is better. The pre-registered target was 0.2 or less.

| Geometry | One step, this network | One step, same design without the single-link systems | One step, a network trained on the chain itself | Ten steps, this network (error) | Ten steps, repeating the last frame (error) |
|---|---|---|---|---|---|
| 1D slab | 0.177 | 0.529 | 0.017 | 0.245 | 0.688 |
| Cylinder | 0.058 | 0.096 (a) | not trained | 0.177 | 0.684 |
| Sphere | 0.056 | 0.095 (a) | not trained | 0.178 | 0.642 |
| 2D grid, 64 × 64 | 0.089 | 0.358 | 0.020 | 0.425 | 0.934 |
| 3D grid, 32 × 32 × 32 | 0.100 | 0.458 | 0.030 | 0.464 | 1.171 |

(a) For the cylinder and sphere, the comparison is the 1D network trained with the same recipe but no cylinder or sphere data. That comparison does not isolate the radial data, because the radial network also trained three times longer on a different device.

Network values are over three training seeds. The single networks in this folder are seed 0, so their numbers differ slightly from the table.

## What the results do not show

These limits are part of the result. Anyone building on the model needs them.

1. The advantage fades with the forecast horizon. By 50 steps the 1D, cylinder and sphere networks are no better than repeating the last frame, or worse. On the grids the late advantage comes from light and heat, not sound.
2. Sound pressure is the weakest quantity. One step ahead it sits at 0.08 to 0.19 of the baseline in the cylinder, sphere and grids, and at 0.385 in 1D, against 0.02 to 0.06 for light. Hiding the light input from the pressure prediction cuts the pressure error four- to ninefold, so the network does not yet keep the two fields apart.
3. The chain contains no new field. Its fields are exactly those of the two single-link training systems. What is new is light and sound appearing together. The test is transfer of known links, not discovery of new physics.
4. The results hold when inputs are scaled to the amplitudes of the training systems. Those scales lie within 1.5 to 4.7 per cent of the chain's own scales. Under the input rule first written for the 1D test, the target is missed in 1D (0.65), in the cylinder and sphere (0.31 to 0.44) and in 2D (0.29). Only 3D meets it under that rule (0.15).
5. A network trained directly on the chain is still 3 to 10 times better one step ahead.
6. Everything is simulation at millimetre scale. There are no measured cells and no cell-scale light model. In 2D and 3D, sound pulses are 3 to 10 grid cells wide, and the 3D training data are nine times less varied than the 1D data.

## Install

Tested with Python 3.13.16 on Linux, CPU only.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins numpy 2.5.3, scipy 1.18.1, torch 2.14.1 and matplotlib 3.11.2. The default torch wheel on Linux includes the CUDA libraries, which run to several gigabytes. A machine without an NVIDIA GPU can install the smaller CPU build first, then the rest:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.5.3 scipy==1.18.1 matplotlib==3.11.2
```

Every command below runs from this folder. The scripts use one CPU thread when `OMP_NUM_THREADS=1` is set, which makes results reproducible to the last digit:

```bash
export OMP_NUM_THREADS=1           # Windows PowerShell: $env:OMP_NUM_THREADS=1
```

## First forecast: the 1D slab

```bash
python experiments/gen_data.py chain
python examples/predict_example.py --run v2_s0 --geom chain --traj 0 --steps 10
```

The first line simulates the photoacoustic chain for the slab (36 s on the test machine). The second loads the network, forecasts ten steps from four true frames, scores the forecast and draws `examples/predict_example.png` (5 s). Its output:

```text
chain/test trajectory 0: boundary periodic, frame interval 6.406e-08 s, length 12.8 mm
input constants per channel (W m^-2, K, Pa): [1.9011299e+06 2.8676564e-05 1.3349426e+01]
pressure after 10 steps (frame 13), first 4 cells (Pa)
   predicted: [0.49508178 1.2410281  1.8439242  1.2644002 ]
   true:      [3.8416975 3.8308733 3.8038404 3.762753 ]
one-step error (mean over the 60 starts): network 0.010788, persistence 0.038953
10-step error: network 0.159524, persistence 0.396632
stored in results/eval_r2/dev.json: one-step 0.010788, 10-step 0.159525
wrote examples/predict_example.png
```

The last printed line compares the result with the value stored by the original evaluation. A match confirms that the installation reproduces the published model. Ten steps ahead the pressure in the first cells is far from the true value, even though the pooled error is 0.40 of the baseline's. Light and heat dominate the pooled error, which is limit 2 above.

## Cylinder, sphere, 2D and 3D

```bash
python experiments/gen_data.py chain_r2                       # cylinder; chain_r3 for the sphere
python examples/predict_example.py --run v3Lplain_s0 --geom chain_r2 --traj 0 --plot examples/predict_cylinder.png
python examples/predict_grid_example.py --dim 2 --run g2_s0 --traj 0
python examples/predict_grid_example.py --dim 3 --run g3_s0 --traj 0
```

The grid examples need no data on disk. Each one recomputes the first batch of chain trajectories (16 in 2D, 4 in 3D) and checks its SHA-256 against `results/data_manifest_grid.json` before using it. On the test machine the 2D line took 12 s and the 3D line 45 s. Both printed errors equal to the stored values: one step 0.006633 against 0.087428 for the baseline in 2D, and 0.011684 against 0.126171 in 3D. Each draws the true and predicted fields and their difference, for every quantity.

## Forecasting your own fields

The network forecasts the next frame from four past frames of three quantities:

| Channel | Quantity | Unit |
|---|---|---|
| 0 | fluence rate | W m^-2 |
| 1 | temperature rise | K |
| 2 | acoustic pressure | Pa |

In 1D, cylinder and sphere geometry a trajectory has the shape (1, frames, 3, 128): 128 cells over 12.8 mm. On the grids a frame is (3, 64, 64) in 2D, with cells of 0.2 mm, or (3, 32, 32, 32) in 3D, with cells of 0.4 mm. The frame interval of each geometry is printed by the examples.

The input must be scaled before it reaches the network. Each channel is divided by a constant taken from the training systems, and the output is multiplied back. The examples read these constants from `results/datagen/` and `results/datagen_grid/` and print them. To forecast fields that are not stored, replace the loading lines of `examples/predict_example.py` or `examples/predict_grid_example.py` with an array of the right shape in the units above. Only the four seed frames are needed for a forecast; a score also needs the true later frames.

Fields far from the training amplitudes, or from the tissue parameter ranges in `params/tissue_params.json`, are outside what was tested (limit 4).

## Reproducing a locked result

This rebuilds the slab's locked test set from its seed, checks it against the stored fingerprint and scores the network on it:

```bash
python experiments/gen_data.py light heat sound_uniform sound_layered sound_lossy transport burgers fisher chain telegraph photothermal thermoacoustic
python tools/check_manifest.py
python experiments/make_locked_set.py
python experiments/run_eval_r23.py --split final_locked --geoms chain --models v2_s0 --out results/my_check.json
```

On a fresh copy of this folder, the data step took 772 s on one core, the manifest check 1 s (69 files, no mismatch), the locked set 8 s and the evaluation 35 s. In `results/my_check.json`, the value at geoms → chain → v2_s0_phys → one_step → median is 0.00913 (0.009127 printed in full), against 0.0515 for repeating the last frame: 0.177 of it.

## Verifying the solvers

The training data come from finite-volume solvers, each checked against exact solutions (second-order convergence on smooth data) or, where none exists, by reduction to a verified solver. To rerun the checks:

```bash
for t in verify_transport verify_burgers verify_fisher verify_telegraph verify_light verify_heat \
         verify_sound verify_chain verify_coupled verify_radial; do python tests/$t.py; done
for t in e2e_transport e2e_burgers e2e_fisher e2e_telegraph; do python tests/$t.py; done
python tests2d/verify_light.py 2 3        # likewise verify_heat, verify_sound, verify_advdiff,
                                          # verify_chain, verify_consistency, resolution_sound
```

Each script writes its numbers to `results/verification*/` and exits with status 1 if a check fails. `e2e_transport` and `e2e_telegraph` exit with status 1 by design: they record a known discretisation error rather than a pass. The stored files hold the original values for comparison; rerunning rewrites them. On a fresh copy of this folder, the fourteen 1D checks took 441 s on one core. Every result matched the stored files except the timing fields, the recorded process priority, and seven spectral radii in `sound.json`, which changed in their last two or three digits through eigenvalue rounding.

## Training

`experiments/train_jobs.py` (1D, cylinder and sphere) and `experiments/train_jobs_grid.py` (grids) list every registered training run by name. For example, after the data step of the section above, `python experiments/train_jobs.py v2_s0` retrains the slab network on the CPU. A run writes into `runs/<name>/` or `runs_grid/<name>/` and replaces the network of the same name, so copy the published one aside first. Grid training needs the grid data (`python experiments/gen_data_grid.py`) and, in practice, a GPU. On an NVIDIA L4 a 2D run took about 10 minutes and a 3D run about 44 minutes. A network retrained on a different type of device is not bit-identical to the published one.

## What is in this folder

| Path | Contents |
|---|---|
| `pfm/` | 1D, cylinder and sphere: solvers, dataset, transformer, training, evaluation |
| `pfm2d/` | 2D and 3D grids: the same parts; `pfm2d/README.md` describes them in detail |
| `runs/v2_s0/`, `runs/v3Lplain_s0/` | the slab network and the cylinder and sphere network (`model.pt` with its `log.json`) |
| `runs_grid/g2_s0/`, `runs_grid/g3_s0/` | the 2D and 3D grid networks |
| `examples/` | worked forecasts, and `regenerate_locked.py` for every locked set |
| `experiments/` | data generation, locked sets, training jobs, evaluation |
| `tests/`, `tests2d/` | solver verification |
| `tools/` | fingerprint and manifest checks, summary tables |
| `params/` | tissue parameters with their published sources |
| `results/` | data fingerprints, input constants, verification records and stored evaluation results |

Each `model.pt` must stay beside its `log.json`, because the loader reads the network's architecture from it. The networks have 2.7 to 8.9 million parameters.

The study trained 73 networks in all, including seeds, ablations and reference networks. All 73 are attached to the release [v1.0](https://github.com/abhaysadineni/photoacoustic-physics-foundation-model/releases/tag/v1.0) as `pfm-models-v1.0.zip` (1,039,810,932 bytes, SHA-256 `ac66693d320980eb50d6fa365ee7716c7e704d1be25e833a244f52e124c31770`). Download it and its checksum file into this folder, check it and unzip it here:

```bash
gh release download v1.0 -R abhaysadineni/photoacoustic-physics-foundation-model -p 'pfm-models-v1.0.zip*'
sha256sum -c pfm-models-v1.0.zip.sha256          # macOS: shasum -a 256 -c pfm-models-v1.0.zip.sha256
unzip -o pfm-models-v1.0.zip
```

Without the GitHub command line, download the two files from the release page in a browser. On Windows, `certutil -hashfile pfm-models-v1.0.zip SHA256` prints the checksum to compare. The archive fills `runs/` and `runs_grid/` and adds its own two notes, `README.txt` and `LICENSE.txt`; the four networks already here are byte-identical to their copies in it. `experiments/train_jobs.py` and `experiments/train_jobs_grid.py` describe what each network was trained on. Comments in the code refer to PLAN.md and the audit records of the original study, which are not part of this repository.

## Licence and citation

The code is licensed under the Apache License, Version 2.0 (`LICENSE-APACHE-2.0`). The trained networks, the parameter tables with their notes, and the result files are licensed under Creative Commons Attribution 4.0 International. `LICENSE` states the split exactly. The sources cited in `params/` remain under their publishers' terms.

```text
Sadineni, Abhay Chakra. Photoacoustic Physics Foundation Model. Version 1.0 (2026).
Code Apache-2.0; trained networks and results CC BY 4.0.
```
