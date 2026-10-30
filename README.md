# AEQGA — Amplitude-Encoded Quantum Genetic Algorithm

Research implementation of Sarracino et al., *A Quantum Genetic Algorithm with
application to Cosmological Parameters Estimation*, arXiv:2602.15459 (2026),
adapted from [Quasar-UniNA/HQGA](https://github.com/Quasar-UniNA/HQGA).

AEQGA searches over **[H0, Omega_m]**, bounded by [60, 80] km/s/Mpc and [0, 0.5].
The CPU evaluates cosmological likelihoods, selects elites and decodes counts.
Quantum circuits prepare amplitudes, apply bidirectional CRy(pi/2) crossover
and Rx(pi/2) mutation, then measure. The best quarter survives unchanged;
an elite-copy quarter and a fresh random half pass through separate circuits.

## Current implementation

- Calibrated Pantheon+ MU_SH0ES likelihood with the matching full STAT+SYS
  covariance. Missing real inputs raise an error; the scientific runner has
  no synthetic fallback.
- Aer, finite-shot statevector sampling, exact-probability simulation and an
  injected IBM Runtime SamplerV2 adapter.
- Default scientific settings: 32 individuals, 50 updates, 4096 shots,
  p_cross=p_mut=0.5. Generation zero is included in histories.
- Separate BAO/CMB likelihoods using CAMB/PICO. Exact reproduction of the
  paper's CMB configuration remains unresolved.
- Independent-run ensembles, provenance checks, scientific figures and
  generation-level hardware recovery.

**Use the `aeqga/` package.** Root-level `aeqga_algorithm.py`,
`pantheon_problem.py` are historical copies. In particular,
the root likelihood contains a synthetic fallback and is unsuitable for
scientific hardware runs. The maintained notebook is `notebooks/AEQGA.ipynb`;
the obsolete root-level notebook has been removed.

## Layout

| Location | Responsibility |
|---|---|
| `aeqga/steps/` | Selection, encoding, genetic gates, decoding and evolution |
| `aeqga/likelihoods/` | Pantheon+, cosmological distances, BAO and CMB |
| `aeqga/execution/ibm_sampler.py` | Ideal/noisy Aer and IBM execution, counts, archives, diagnostics |
| `aeqga/execution/workflow.py` | Scientific identity and full optimizer restart |
| `aeqga/experiments/runner.py` | Resumable independent simulation ensembles |
| `aeqga/analysis/` | Circuit distribution comparisons and offline mitigation study |
| `aeqga/visualization/` | Convergence, objective contours and optimizer distributions |
| `scripts/` | Simulation, hardware pilots, recovery and analysis commands |
| `tests/` | Offline numerical, recovery and integration tests |

## Setup and simulation

```bash
# Python 3.12; simulation and IBM execution share the pinned environment.
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
# requirements-ibm.txt is a compatibility alias for requirements.txt.

# Obtain the real SNe inputs; no credentials are involved.
git clone https://github.com/CobayaSampler/sn_data.git sn_data
git -C sn_data checkout 61d96434cafc2770928322c38e5a750e686368ae

python -m scripts.run_pantheon_aeqga --fast
python -m scripts.run_sne_ensemble --runs 20 --name sne_stage20
# Full simulation ensemble:
python -m scripts.run_sne_ensemble --runs 300 --name sne_production
```

Source/data/configuration changes require a new experiment name. Historical
ensemble statistics are documented in [tasks 8–12](docs/TASKS_8_TO_12_REPORT.md);
they are optimizer outcomes, not automatically posterior uncertainties.

## IBM adaptation and offline pilots

A 32-individual generation submits four circuits with logical widths
**[4, 3, 4, 3]** in a single Sampler call. Generations remain sequential.
A 50-update run uses 200 circuit executions and 819,200 shots at 4096 shots.

```bash
# Does not submit hardware jobs; --backend additionally reads IBM calibration.
python -m scripts.benchmark_hardware --shots 4096

# Real-data offline pilot; repeat the same command to resume.
python -m scripts.run_ibm_generation --generations 3 --shots 1024 \
  --directory outputs/json/sne_offline_pilot

# Credential/backend inspection only; no circuit submission.
python -m scripts.ibm_preflight --backend ibm_kingston

# Retrieve existing results; never resubmits.
python -m scripts.recover_ibm_job \
  --checkpoint outputs/json/sne_offline_pilot/generation_0001.json

# Requires archived circuits and matching real-data provenance.
python -m scripts.compare_generation_populations \
  --directory outputs/json/sne_offline_pilot --samples 1000
python -m scripts.analyze_hardware_runs --directory outputs/json/sne_offline_pilot
```

Physical submission requires **both** `--submit-hardware` and
`--confirm-run-qpu`. Use a separate experiment directory and an agreed budget.
`--session` opts into a supported account's session mode;
`--sampler-options options.json` passes supported Runtime suppression settings.
`--preparation state_preparation` is a separately identified reset-free variant.
`--runs N` uses independent seeds and separate run directories;
`--max-total-shots N` rejects plans above the specified circuit-shot ceiling.
That ceiling excludes calibration jobs and is not a monetary or QPU-time cap.

The benchmark compares preparation methods, optimization levels, ideal Aer,
backend-derived noisy Aer and an explicit readout-correction analysis. Its
default backend is **synthetic**, not a measured IBM device. Corrected
probabilities never silently replace raw optimizer counts.

See [IBM hardware guide](docs/IBM_HARDWARE.md) for restart semantics, pilot
criteria and remaining live-hardware validation. Historical hardware records
exist, but the original circuits are missing; exact same-circuit provenance
cannot be certified. No full hardware ensemble is claimed.

## Accuracy against the classical answer

The [latest 300-run plot results](docs/SNE_PLOT_RESULTS.md) include graph previews,
PNG/SVG/PDF downloads, measured accuracy statistics and the saved numeric reports.

Notebook section **9a** now reports how closely AEQGA matches the classical
fit: signed/absolute parameter errors, percentage errors, bias, MAE, RMSE,
and delta chi-squared. A four-panel accuracy figure shows every run, including
outliers. The notebook compares against both the configured objective and a
separately re-evaluated direct-distance minimum. Neither fitted reference is
assumed to be the true value of the universe's parameters.

The success fraction counts runs with delta chi-squared < 2.30. Its 95% Wilson
interval concerns the success rate of independent comparable runs, not
cosmological confidence. Parameter percentage errors are ordinary relative
errors; they are never presented as a percentage probability of correctness.
Covariance/KDE contours still describe optimizer scatter, while likelihood
contours describe the configured model/data under a chi-squared approximation.

`scripts.run_sne_ensemble` writes `NAME_accuracy.json` and accuracy figures in
PNG/SVG/PDF. Hardware analysis writes the same accuracy metrics into
`hardware_summary.json` and exports `hardware_accuracy.*`. The single-run
simulation runner also writes `pantheon_accuracy.json` and figures.

See [graph accuracy and interpretation](docs/GRAPH_ACCURACY.md) for exactly
what is checked, the distance-grid limitations, and what the graphs cannot prove.

## Supplied noisy/QPU examples

The demo uploaded to GitHub `main` is copied here as
`qml_demo_noisy_and_qpu.zip` and unpacked in `qml_demo_noisy_and_qpu/`.
The ZIP preserves the original files. The extracted
[`test_ionq_sim.ipynb`](qml_demo_noisy_and_qpu/test_ionq_sim.ipynb) now defaults
to 100 shots and includes executed counts, a plot, and an interpretation.
With no IonQ provider package or API credential in this environment, its
recorded result uses local Aer with IBM FakeBrisbane noise; it is **not** an
IonQ `aria-1` result. The extracted `test_noisy_sim.py` also runs at 100 shots.
The other two examples require an IBM account or target hardware and were not
run. The measured local run executed one circuit in 0.919 seconds with zero
cloud service charges; its [plot](outputs/png/ionq_reference_demo_100shot.png)
shows the observed counts against ideal probabilities. For a hosted IonQ
`aria-1` run, install `qiskit-ionq`, configure the IonQ credential locally,
and set `RUN_IONQ_CLOUD = True` in the extracted notebook.

## Verification

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
python -m scripts.test_hardware_compilation
python -m scripts.test_full_generation  # requires real Pantheon+ data
```

Dataset-dependent tests skip when their inputs are absent. Recovery tests use
explicit synthetic test objectives and mocked IBM jobs; they consume no QPU
allocation. Never place IBM credentials in notebooks, configuration, logs,
checkpoints or version control.
