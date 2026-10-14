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

## Setting up and running

I ran everything on a laptop CPU (Apple M3 Max) with Python 3.12, NumPy 2.5, SciPy 1.18, scikit-learn 1.8 and PyTorch 2.14. The frozen environment is in `requirements.txt`; the videos use the ffmpeg binary that `imageio-ffmpeg` installs.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name py312-pinn
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_pod_rom_intrusive_and_data_driven.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_ladle_hybrid_digital_twin.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_sensors_and_guard.ipynb
```

All paths inside the notebooks are relative to `notebooks/`, so run them from there. Stage 1 took about 75 seconds, Stage 2 about 20 and Stage 3 a few seconds. Every number is reproduced exactly on a re-run; only the timings change.

**Why the CPU.** One training step of the Stage 2 neural soft sensor (a 17–64–8 network on a full batch of 300 heats) takes 0.24 ms on the CPU and 0.88 ms on the Apple GPU (MPS); everything else is NumPy and SciPy.

**Why three seeds.** Wherever a result depends on a random draw (the training and test parameters of Stage 1, the sensor noise and the network's initial weights of Stage 2, the random sensor layouts and the noise of Stage 3), I repeat it with the seeds 0, 1 and 2 and report the spread; seed 0 is always the notebook's own draw. The repeats show that the hybrid twin and the linear soft sensor are close (the hybrid is lower in each of the three seeds, by less than two standard deviations), and that the network does not beat the linear map.

## Data

There is no external data. Stage 1's reference is its own finite-difference solver, and the ladle fields of Stages 2 and 3 come from the analytic generator `ladle_v2.py`: six independent, smooth physical patterns (gas-flow plume, wall and floor boundary layers, slag layer, stratification, recirculation tilt, fill level), each driven by a slowly drifting plant variable, plus a late abnormal double-plume regime.

## What the three stages show

- POD-Galerkin is accurate inside its parameter box (1.4 % error over three draws) and degrades smoothly outside it; the regression route is accurate inside and explodes outside (degree-4 polynomial, about 2,700 % at the far corner), and it cannot see a variable it was not given (68 % with a source twice as wide).
- In the ladle, the physics core alone misses a third of the field's variation; corrected by a learned residual it misses 19 %, about the same as a pure-data soft sensor, and a neural network does worse than the linear map.
- Placement matters more than count: at the 8 POD pivots the field error is down to 13–14 %, while random positions need about 24 sensors for the same result. A guard working on sensor readings detects a drift towards the abnormal regime earlier than the field-based guard, but only if its alarm threshold is calibrated at the plant's real noise level; calibrated on clean readings it flags every heat.
