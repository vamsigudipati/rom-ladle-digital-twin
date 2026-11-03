# Reduced-order models and a hybrid digital twin of a steel ladle

These three notebooks follow proper orthogonal decomposition (POD) from a textbook reduced-order model into a working digital twin. In Stage 1 I compress the solutions of a parametric heat-conduction problem into 15 POD modes and build two reduced models on them: an intrusive POD-Galerkin model that steps the solver's own operator inside the subspace, and a data-driven one that regresses the POD coefficients on the input parameters. Both are fast and accurate inside the parameter range they were built on and fail outside it. Stage 2 uses the same backbone on an industrial question: the full temperature field of a gas-stirred steel ladle from 16 thermocouples and the gas-flow reading. It compares a physics core, the core corrected by a residual learned from the sensors (the hybrid twin), a pure-data soft sensor and a small neural network, checks the plant KPIs, and adds a guard that flags fields the basis cannot represent. Stage 3 stresses that twin: how many sensors it needs and where to put them, and how its guard behaves as the plant drifts towards an abnormal regime and as the sensors get noisier.

Portfolio page: https://vamsigudipati.github.io/rom-ladle-digital-twin.html

## Notebooks

| Notebook | Stage | Results | Models | Animations |
| --- | --- | --- | --- | --- |
| `notebooks/01_pod_rom_intrusive_and_data_driven.ipynb` | Stage 1 — POD reduced-order models, intrusive and data-driven | `results/01_pod_rom_intrusive_and_data_driven/` | — | `animations/01_*` |
| `notebooks/02_ladle_hybrid_digital_twin.ipynb` (with `ladle_v2.py`) | Stage 2 — A hybrid digital twin of a gas-stirred ladle | `results/02_ladle_hybrid_digital_twin/` | `models/02_ladle_hybrid_digital_twin/` | `animations/02_*` |
| `notebooks/03_sensors_and_guard.ipynb` | Stage 3 — Sensors and the guard under stress | `results/03_sensors_and_guard/` | — | — |

Each notebook is also exported as HTML next to it, with all outputs. In `results/`, the `exercises/` folder of Stages 1 and 2 holds the tables (CSV) and figures (PNG) of the experiments and of the three-seed repeats; Stage 3 writes its tables and figures directly into its results folder. `models/` holds the soft-sensor network's checkpoints; `animations/` holds the two videos and their final frames.

## Reproduce

I ran everything on a laptop CPU (Apple M3 Max) with Python 3.12, NumPy 2.5, SciPy 1.18, scikit-learn 1.8 and PyTorch 2.14. The frozen environment is in `requirements.txt` (a macOS freeze; it includes packages the notebooks never import). `requirements-test.txt` is the subset the notebooks and the tests import, at the same pins, and is what the Linux CI installs. The videos use the ffmpeg binary that `imageio-ffmpeg` installs.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-test.txt ipykernel==7.4.0 nbconvert==7.17.1 nbclient==0.11.0
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu   # CPU wheel; on macOS plain `pip install torch==2.14.1`
pip install -e .                                                             # the ladle_twin package (src/)
python -m pytest -q                                                          # 40 unit tests, about 10 s
cd notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 01_pod_rom_intrusive_and_data_driven.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 02_ladle_hybrid_digital_twin.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 03_sensors_and_guard.ipynb
```

All paths inside the notebooks are relative to `notebooks/`, so run them from there. The `--ExecutePreprocessor.kernel_name=python3` flag matters: the notebooks' metadata names the author's local kernels (`py312-torch-mps`, `py312-pinn`), which do not exist elsewhere. The notebooks overwrite `results/`, `models/` and `animations/` with the same numbers.

Expected wall time on the CPU, whole notebook including the experiments, the three-seed repeats and the animation:

| Notebook | Apple M3 Max (author) | 4-core Linux x86-64 (CI-class machine, measured for this repository's PR) |
| --- | --- | --- |
| Stage 1 | about 75 s | 180 s |
| Stage 2 | about 20 s | 50 s |
| Stage 3 | a few seconds | 8 s |

On the same machine every number is reproduced exactly on a re-run; only the timings change. On a different CPU and BLAS (the Linux re-run above, compared cell by cell with the committed `results/`), every NumPy/SciPy/scikit-learn ridge number in `results/` agrees to at least 12 significant digits, the two trained networks move in the third digit (the Stage 2 neural soft sensor 0.2605 → 0.2614, the Stage 1 Experiment 3 MLP 4.61 % → 4.19 % in the box), and the gappy-POD errors of the near-singular random sensor layouts of Stage 3 (m = 8 to 12, condition numbers of 10¹⁰ to 10¹⁵) differ at the per cent level, as such inverses must. Nothing stated on this page depends on those.

**Why the CPU.** One training step of the Stage 2 neural soft sensor (a 17–64–8 network on a full batch of 300 heats) took 0.24 ms on the CPU and 0.88 ms on the Apple GPU (MPS) when I measured it interactively (that timing is not recorded in any notebook output); everything else is NumPy and SciPy.

**Why three seeds.** Wherever a result depends on a random draw (the training and test parameters of Stage 1, the sensor noise and the network's initial weights of Stage 2, the random sensor layouts and the noise of Stage 3), I repeat it with the seeds 0, 1 and 2 and report the spread; seed 0 is always the notebook's own draw. The repeats show that the hybrid twin and the linear soft sensor are close (the hybrid is lower in each of the three seeds, by less than two standard deviations), and that the network does not beat the linear map.

## Code, tests and CI

The functions the notebooks define inline (the Stage 1 solver and ROM builders, the ladle generator, the volume-weighted POD, the error measures and KPIs, the twin's sensors, physics core and guards, the plotting helpers) are collected unchanged in the package `src/ladle_twin/`; `notebooks/ladle_v2.py` is now a thin shim onto it. `tests/` holds 40 pytest unit tests on tiny synthetic inputs, including a check that the notebooks' inline definitions and the package return identical results, and `.github/workflows/tests.yml` runs them on every push and pull request on a CPU-only Ubuntu runner. `docs/REPO_MAP.md` lists every file, the pipeline order and the results file or output cell behind each number on this page.

## Data

There is no external data. Stage 1's reference is its own finite-difference solver, and the ladle fields of Stages 2 and 3 come from the analytic generator `src/ladle_twin/ladle.py` (imported by the notebooks as `ladle_v2`): six independent, smooth physical patterns (gas-flow plume, wall and floor boundary layers, slag layer, stratification, recirculation tilt, fill level), each driven by a slowly drifting plant variable, plus a late abnormal double-plume regime.

## What the three stages show

- POD-Galerkin is accurate inside its parameter box (1.4 % error over three draws) and degrades smoothly outside it; the regression route is accurate inside and explodes outside (degree-4 polynomial, about 2,700 % at the far corner), and it cannot see a variable it was not given (68 % with a source twice as wide).
- In the ladle, the physics core alone misses a third of the field's variation; corrected by a learned residual it misses 19 %, about the same as a pure-data soft sensor, and a neural network does worse than the linear map.
- Placement matters more than count: at the 8 POD pivots the field error is down to 13–14 %, while random positions need about 24 sensors for the same result. A guard working on sensor readings detects a drift towards the abnormal regime earlier than the field-based guard, but only if its alarm threshold is calibrated at the plant's real noise level; calibrated on clean readings it flags every heat.

Every number above comes from a results file or a notebook output cell; `docs/REPO_MAP.md` gives the exact source of each one. The 1.4 % and the "about 2,700 %" are the means over the three seeds (`results/01_pod_rom_intrusive_and_data_driven/exercises/seeds_main_summary.csv`: 1.42 ± 0.08 % and 2,696 ± 173 %; seed 0 alone gives 1.51 % and 2,764 %); the 33 %, 19 % and 19 % of the ladle are seed 0's (Stage 2 cells 20, 24 and 28; over three seeds 33.4 %, 18.8 ± 0.3 % and 19.2 ± 0.2 %, the network 23.7 ± 2.2 %); the 13–14 % and the 24 sensors are medians over the three seeds (`results/03_sensors_and_guard/a_sensors_summary.csv`).

## Limitations

- **No real data.** The ladle fields are an analytic surrogate with six smooth, independent patterns (`src/ladle_twin/ladle.py`), so the field is almost exactly 8-dimensional and a linear map is the right model class by construction. The 19 % error, the sensor counts and the guard thresholds say how the method behaves on that surrogate, not on a plant. Stage 1's reference is its own finite-difference solver on a 48 × 48 interior grid.
- **Small, chronological data.** Stage 2 trains on 300 heats and tests on 125; the three-seed spreads are over the sensor noise and the network's initialisation only, with one draw of the operating points (seed 0 of `generate`). Stage 1's three seeds redraw the parameters, but on one fixed solver.
- **Error measures.** The ladle errors are relative to the *variation* of the field around the training mean (about 13 K), so 19 % is about 4.5 K RMS on a 1,873 K bath; Stage 1's errors are plain relative L2 norms of the whole field.
- **The guard is a reconstruction residual.** It detects fields outside the 8-mode span. A drift that stays inside the span (an unusual but representable combination of the six patterns) is not flagged, and the sensor-space guard must be calibrated at the plant's real noise level, as Stage 3 shows.
- **Timings.** The wall times in the notebooks' own text ("a few minutes" for the Stage 1 experiments, "~2–3 minutes on a CPU" in Stage 2) are older estimates; the measured times are in the table above. The 0.24 ms / 0.88 ms step timing was measured interactively and is not reproduced by any cell.
- **The notebooks are not re-executed by CI.** CI runs the unit tests on the extracted functions (seconds); reproducing the results means running the three notebooks as above.
