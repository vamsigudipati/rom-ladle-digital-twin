# Repository map

What every file does, the order the pipeline runs in, and where each number in
the README comes from. The notebooks were read by parsing their JSON (sources
and stored outputs); nothing here was produced by re-executing them.

## Layout

| Path | What it is |
| --- | --- |
| `README.md` | Overview of the three stages, how to run them, what they show. |
| `requirements.txt` | The author's full frozen environment (macOS, Python 3.12). Includes packages the notebooks never import (TensorFlow, DeepXDE, PyVista, pyobjc, ...). |
| `requirements-test.txt` | The subset of `requirements.txt` that the notebooks and tests import, same pins. CI installs this plus CPU-only PyTorch. |
| `pyproject.toml` | Packaging for `src/ladle_twin` and the pytest configuration (`pythonpath = src`). |
| `notebooks/01_pod_rom_intrusive_and_data_driven.ipynb` | Stage 1 (68 cells). Parametric heat-conduction solver, POD compression, intrusive POD-Galerkin ROM, non-intrusive polynomial-ridge ROM, two failure modes, five experiments, three-seed repeats, stepping animation. |
| `notebooks/02_ladle_hybrid_digital_twin.ipynb` | Stage 2 (51 cells). Ladle fields from the generator, 8-mode volume-weighted POD, physics core, residual-corrected hybrid twin, linear and neural soft sensors, KPIs, guard, training animation, three-seed repeats. |
| `notebooks/03_sensors_and_guard.ipynb` | Stage 3 (10 cells). Field error against the number and placement of sensors (A); the guard under drift towards the abnormal regime and under sensor noise (B). |
| `notebooks/*.html` | HTML export of each notebook with all outputs. |
| `notebooks/ladle_v2.py` | Thin shim: re-exports the generator from `ladle_twin.ladle` under the name the notebooks import. Running it as a script runs the generator self-test CLI. |
| `src/ladle_twin/plate.py` | Stage 1 solver (`build_problem`, `as_grid`), `snapshot_basis`, `build_rom`, `poly_features`, `poly_features_3d`, `fit_ridge`. Copied from notebook 01 cells 10–12, 24, 32–33, 48, 57. |
| `src/ladle_twin/ladle.py` | The analytic gas-stirred-ladle generator, moved unchanged from `notebooks/ladle_v2.py`: `geom`, `temperature`, `draw_operating_point`, `volume_weights`, `generate`, `place_sensors` (pivoted QR), `gappy_reconstruct`; plus `design_numbers` (the old `__main__` self-test as a function). |
| `src/ladle_twin/pod.py` | Volume-weighted POD of Stages 2–3: `build_basis`, `coeffs`, `lift`, `leftover_energy` (the field-space guard score). Notebook 02 cells 16 and 39, notebook 03 cells 2 and 7. |
| `src/ladle_twin/metrics.py` | `rel_error`, `knee` (notebook 01 cell 48), `rel_l2`, `rms_kelvin` (notebook 02 cells 16, 28), `kpi_masks`, `kpis`, `r2`, `kpi_r2` (notebook 02 cells 36, 50). |
| `src/ladle_twin/twin.py` | `cell_index`, `two_lance_probes`, `gas_flow_input`, `read_sensors`, `physics_core`, `residual_basis`, `residual_coeffs` (notebook 02 cells 20, 24), `sensor_score` (notebook 03 cell 7), `make_soft_sensor`, `train_soft_sensor` (notebook 02 cell 33). |
| `src/ladle_twin/plotting.py` | `heat` (notebook 01 cell 7), `heat_ladle` (notebook 02 cell 5), `panels3` (notebook 01 cell 8), the two palettes. |
| `src/ladle_twin/cli.py` | Argument parser and entry point of the generator self-test: `python -m ladle_twin.cli [--n-single --n-double --seed --modes --sensors]`. |
| `tests/` | Unit tests (see below). |
| `.github/workflows/tests.yml` | Runs the tests on push and pull request, ubuntu-latest, CPU. |
| `results/01_pod_rom_intrusive_and_data_driven/exercises/` | Stage 1 experiment tables (`exp1_rank.csv`, `exp2_box.csv`, `exp3_regressors.csv`, `exp4_width_parameter.csv`, `exp5_grid_steps.csv`), figures (`exp2_box_sweep.png`, `exp5_spectrum.png`), three-seed runs and summaries (`seeds_*_runs.csv`, `seeds_*_summary.csv`, `seeds_summary.png`). |
| `results/02_ladle_hybrid_digital_twin/exercises/` | Stage 2 three-seed runs (`seeds_runs.csv`) and their mean/std (`seeds_summary.csv`). |
| `results/03_sensors_and_guard/` | Stage 3 tables and figures: `a_sensors_runs.csv`, `a_sensors_summary.csv`, `a_sensors.png`, `b_drift_runs.csv`, `b_drift_summary.csv`, `b_noise_runs.csv`, `b_noise_summary.csv`, `b_guard.png`. |
| `models/02_ladle_hybrid_digital_twin/checkpoints/` | `soft_sensor_step0150.pth` … `step1500.pth`: state dicts of the neural soft sensor every 150 optimiser steps (written by the animation hook, notebook 02 cell 32). |
| `animations/` | `01_stepping_dashboard.mp4` + `01_final_frame.png` (Stage 1, FOM and ROM stepping side by side); `02_training_dashboard.mp4` + `02_final_frame.png` (Stage 2, the neural soft sensor training). |

The notebooks still carry their own inline copies of these functions; the
package is the tested copy. `tests/test_notebook_equivalence.py` pulls the
inline definitions out of the `.ipynb` JSON with `ast` and checks that they
return exactly what the package returns, so the two cannot drift silently.

The only script is the generator self-test (`notebooks/ladle_v2.py` →
`ladle_twin.cli`). The notebooks have no command-line interface; they are run
with `jupyter nbconvert --execute` from `notebooks/`.

## Pipeline order

Each notebook is self-contained and runs top to bottom from `notebooks/`
(all paths are relative to that folder). The three stages share no files
except the generator; Stage 3 rebuilds Stage 2's data, basis and split itself.

1. **Stage 1** (`01_…ipynb`). `build_problem(50)` → one FOM solve → coarse POD
   warm-up → 40 solves at `default_rng(0)` parameters give the snapshot matrix
   and basis → rank-15 POD-Galerkin ROM, error and speed-up on 40 test
   parameters → polynomial-ridge ROM on 100 fields → corner sweep and wide
   source (failure) → Experiments 1–5 → three-seed repeats → stepping
   animation. Writes `results/01_…/exercises/*` and `animations/01_*`.
2. **Stage 2** (`02_…ipynb`, imports `ladle_v2`). `generate()` (500 normal +
   120 abnormal heats, seed 0) → chronological split, training heats 0–299,
   test heats 375–499 → 8-mode volume-weighted POD → physics core → residual
   basis and ridge map (hybrid twin) → linear soft sensor → neural soft sensor
   (1,500 Adam steps, hook stores checkpoints) → KPIs → guard → training
   animation → three-seed repeats and KPI R². Writes
   `results/02_…/exercises/*`, `models/02_…/checkpoints/*`, `animations/02_*`.
3. **Stage 3** (`03_…ipynb`, imports `ladle_v2`). Rebuilds Stage 2's data and
   basis, adds a calibration block (heats 300–374) → A: sensors 2–32, QR/DEIM
   versus random layouts, gappy POD and ridge map, seeds 0–2 → B: drift
   towards the double plume and sensor noise, field-space and sensor-space
   guards. Writes `results/03_…/*`.

The tests (`python -m pytest`) need none of the above to have run.

## Where the README's numbers come from

"Output" means the stored text output of that notebook cell (cell indices
are 0-based over all cells, as stored in the JSON). "File" means a results
file in the repository. Every number in the README is listed.

| README statement | Source | Value there |
| --- | --- | --- |
| 15 POD modes (Stage 1) | notebook 01 cell 25 (`RANK = 15`), output of cell 25 | `using r = 15` |
| 16 thermocouples, the gas-flow reading | notebook 02 cell 24 (`probes`: 2 lances × 8), cell 0 text | 16 probes + Q |
| three-seed repeats, seeds 0, 1 and 2 | notebook 01 cell 62, notebook 02 cell 50, notebook 03 cell 2 (`SEEDS = (0, 1, 2)`) | — |
| Python 3.12, NumPy 2.5, SciPy 1.18, scikit-learn 1.8, PyTorch 2.14 | `requirements.txt`; outputs of notebook 01 cells 3, 8 and notebook 02 cell 3 | `Python 3.12.14`, `numpy 2.5.3`, `PyTorch 2.14.1` |
| Stage 1 took about 75 s, Stage 2 about 20 s, Stage 3 a few seconds (author's laptop) | not recorded in any output cell. For this PR, copies of the notebooks were re-executed on a 4-core Linux x86-64 CPU with the pinned versions: 180 s, 50 s and 8 s; those are in the README's "Reproduce" table | 180 s / 50 s / 8 s |
| one training step 0.24 ms on the CPU and 0.88 ms on MPS | not recorded in any notebook or results file (an interactive measurement by the author); the README now says so | — |
| "Every number is reproduced exactly on a re-run" | re-run for this PR (above) compared cell by cell with `results/*.csv`: all NumPy/SciPy/ridge numbers agree to ≥ 12 digits; the Stage 2 neural soft sensor (0.2605 → 0.2614, `seeds_runs.csv`), the Stage 1 Experiment 3 MLP (0.0461 → 0.0419, `exp3_regressors.csv`) and the random layouts with condition numbers ≥ 10¹⁰ in `a_sensors_runs.csv` differ; the README now qualifies the claim | — |
| 40 unit tests, about 10 s | `python -m pytest -q` on this branch | 40 passed in 4.6 s (Linux); CI adds the install time |
| 48 × 48 interior grid, 300 heats / 125 test heats, about 13 K variation, 1,873 K bath | outputs of notebook 01 cell 13, notebook 03 cell 2, notebook 02 cell 7; `T_bath=1873.0` in `ladle.temperature` | `interior 48 x 48`; `300 75 125 120`; `~13 K` |
| 33.4 %, 18.8 ± 0.3 %, 19.2 ± 0.2 %, 23.7 ± 2.2 % (three seeds, ladle) | file `results/02_…/exercises/seeds_summary.csv` | 0.3338; 0.1878 ± 0.0027; 0.1921 ± 0.0021; 0.2375 ± 0.0215 |
| 1.51 % and 2,764 % (seed 0, Stage 1) | outputs of notebook 01 cells 26 and 39 | `1.51%`; `2764%` |
| 4.5 K RMS | output of notebook 02 cell 28; `seeds_summary.csv` row `linear_rms_K` | 4.5 K; 4.52 ± 0.05 |
| 17–64–8 network on a full batch of 300 heats | notebook 02 cell 33 (`nn.Linear(Xtr.shape[1], 64)`, `nn.Linear(64, 8)`, `Xtr.shape[1]` = 16 + 1), cell 16 (`tr = s[:300]`) | — |
| hybrid lower than the linear soft sensor in each of the three seeds, by less than two standard deviations | file `results/02_…/exercises/seeds_runs.csv` and `seeds_summary.csv`; output of notebook 02 cell 50 | hybrid 0.1859 / 0.1909 / 0.1868 vs linear 0.1900 / 0.1943 / 0.1922; means 0.1878 ± 0.0027 vs 0.1921 ± 0.0021 |
| the network does not beat the linear map | same file; output of notebook 02 cell 33 figure labels, cell 46 | neural 0.2375 ± 0.0215 vs linear 0.1921 ± 0.0021; seed 0: 0.261 vs 0.190 |
| six independent patterns, late abnormal double-plume regime | `src/ladle_twin/ladle.py` docstring and `temperature`; output of notebook 02 cell 7 | 500 single + 120 double heats |
| POD-Galerkin 1.4 % error over three draws | file `results/01_…/exercises/seeds_main_summary.csv`, row `intrusive_in_box`; output of notebook 01 cell 64 | 0.0142 ± 0.0008 → "1.42 ± 0.08 %" |
| degree-4 polynomial about 2,700 % at the far corner | same file, row `nonintrusive_corner`; output of notebook 01 cell 39 (seed 0) | 26.96 ± 1.73 (2,696 %); seed 0: 2764 % |
| 68 % with a source twice as wide | same file, row `nonintrusive_wide_source`; output of notebook 01 cell 43 | 0.6812 ± 0.0003; seed 0: 68.1 % |
| physics core misses a third of the field's variation | output of notebook 02 cells 20, 24; file `results/02_…/exercises/seeds_summary.csv` row `physics_core` | 33 %; 0.3338 |
| corrected by a learned residual it misses 19 % | output of notebook 02 cell 24; same file row `hybrid` | 19 %; 0.1878 ± 0.0027 |
| about the same as a pure-data soft sensor | output of notebook 02 cell 28; same file row `linear_soft_sensor` | 19 % (4.5 K RMS); 0.1921 ± 0.0021 |
| a neural network does worse than the linear map | output of notebook 02 cell 46; same file row `neural_soft_sensor` | 0.261 vs 0.190; 0.2375 ± 0.0215 |
| at the 8 POD pivots the field error is down to 13–14 % | file `results/03_sensors_and_guard/a_sensors_summary.csv`, rows `QR/DEIM pivots, m = 8`; output of notebook 03 cell 5 | ridge 0.134, gappy 0.143 (medians over seeds) |
| random positions need about 24 sensors for the same result | same file, rows `random, m = 24` and `m = 16`; output of notebook 03 cells 4–5 | ridge median 0.1215 at m = 24 (0.223 at m = 16) |
| the sensor-space guard detects the drift earlier than the field-based guard | file `results/03_sensors_and_guard/b_drift_summary.csv`; output of notebook 03 cells 7, 9 | sensor guard 0.957 at α = 0.3 (90 % reached at α = 0.3); field guard 0.592 at α = 1, never 90 % |
| calibrated on clean readings it flags every heat | file `results/03_sensors_and_guard/b_noise_summary.csv`, column `false_alarm_clean_threshold_mean`; output of notebook 03 cells 8–9 | 1.00 at 1, 5 and 10 % noise; 0.00 recalibrated |

Numbers that appear in notebook text but not in the README were left alone;
two of them disagree with the README's timings and are noted in the README's
"Limitations" (notebook 01 says the experiments take "a few minutes", notebook
02 says "~2–3 minutes on a CPU").

## Tests

`tests/` holds 40 tests (38 run without PyTorch; the two network tests skip
if it is missing):

| File | Covers |
| --- | --- |
| `test_plate.py` | Solver sizes, Laplacian symmetry, source peak, snapshot bookkeeping, positivity and x↔y symmetry of the field, `as_grid`, full-basis Galerkin equals the FOM, ROM error falls with rank to zero at full rank, `poly_features` shapes and values, `fit_ridge` recovers a linear map. |
| `test_ladle.py` | Geometry, temperature field, operating-point clips, `generate` shapes/labels/determinism, volume weights sum to the bath volume, `place_sensors` uniqueness and border, `gappy_reconstruct` exact on in-span fields, `design_numbers`. |
| `test_pod_metrics.py` | Weighted orthonormality of the modes, `coeffs`/`lift` round trip, leftover energy 0 and 1, `rel_l2`/`rms_kelvin` on known cases, `rel_error`, `knee`, KPIs on a constant field, R². |
| `test_twin.py` | Probe layout, sensor readings (shape, reproducibility, noise level), physics core, residual basis, sensor-space guard, 2-sample CPU forward pass of the 17–64–8 network, a short training run with the hook. |
| `test_cli_and_shim.py` | The argument parser builds with its defaults, the CLI runs on a small case, the `ladle_v2` shim re-exports the package functions and runs as a script without installation. |
| `test_notebook_equivalence.py` | The notebooks' inline `build_problem`, `as_grid`, `build_rom`, `poly_features`, `fit_ridge`, `ex_rel`, `ex_knee`, `ex_poly3`, `coeffs`, `relL2`, `leftover_energy`, `kpis`, `cell`, `field_score`, `sensor_score`, `rmsK`, `probe` give the same results as the package. |
