# AEQGA cosmological parameter estimation

Implementation of [Sarracino et al.](https://arxiv.org/html/2602.15459v1), with
classical simulation and IBM Quantum execution using the same calibrated
Pantheon+ objective and package algorithm.

## Install and run

Validated with Python 3.12. From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m ipykernel install --sys-prefix --name aeqga --display-name "Python (AEQGA venv)"
git clone https://github.com/CobayaSampler/sn_data.git sn_data
git -C sn_data checkout 61d96434cafc2770928322c38e5a750e686368ae
jupyter notebook notebooks/AEQGA.ipynb
```

On Windows activate with `venv\Scripts\activate` instead. Select **Python
(AEQGA venv)** in Jupyter. Restart the kernel after updating code or packages.
`requirements.txt` pins the tested direct dependencies for both simulation and
IBM Runtime. `requirements-ibm.txt` includes that same file for compatibility;
there is no separate IBM environment to maintain.

## One notebook and one scientific workflow

[notebooks/AEQGA.ipynb](notebooks/AEQGA.ipynb) is the sole maintained notebook.
The former root notebook was removed; its source remains recoverable in Git.
The notebook uses package imports throughout and requires real Pantheon+ data.
Missing data raise an error. Parameters are ordered `[H0, Omega_m]`.

Parts 1–12 retain the calibrated SNe, separate CMB+BAO, and HQGA workflows.
The default SNe experiment uses 32 individuals, 50 updates, 4096 shots and a
resumable 300-run ensemble; `MODE="demo"` selects a smaller simulation.
`EXECUTION_MODE="statevector_shots"` samples the actual post-gate state;
`"aer"` selects native Aer. CMB, exact-probability and HQGA branches are opt-in.

Part 13 adds the IBM experiment with its own budget. Set `RUN_IBM=True` to
rehearse locally using `DryRunExecutor`. For physical execution, additionally
set `IBM_EXECUTION="qpu"`, select an accessible backend, and set
`CONFIRM_QPU_SUBMISSION=True`. Credentials come from the local IBM account or
environment; never store tokens in notebooks or Git. Four circuits per
32-individual generation have widths `[4, 3, 4, 3]`; generations are sequential.

The notebook and ensemble CLI use provenance-checked atomic checkpoints.
The notebook uses `checkpoint_policy="namespace"`: compatible checkpoints
resume; changed data, settings, source or library versions use isolated
fingerprint-named checkpoints. The library/CLI is strict by default. Every
run retains populations, best-pair/chi-squared histories and seeds.
Hardware checkpoints retain submitted job IDs and counts for replay.

## Package and commands

`aeqga/steps/` contains selection, encoding, genetic gates, decoding and the
canonical generation loop. `aeqga/likelihoods/` contains calibrated SNe,
BAO and CMB likelihoods. `aeqga/execution/` provides IBM and offline adapters;
`aeqga/experiments/` provides independent ensembles; `aeqga/analysis/` and
`aeqga/visualization/` provide scientific comparisons and figures.
Root-level legacy Python modules remain historical compatibility code; the
notebook and hardware generation entry points use the package modules.

```bash
python -m scripts.run_sne_ensemble --runs 300 --name sne_production
python -m scripts.run_ibm_generation  # offline rehearsal by default
python -m scripts.test_hardware_compilation
python -m scripts.fetch_cmb_data      # optional CMB inputs
python -m scripts.run_bao_cmb_aeqga --backend camb --classical-only
MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
python -m pip check
```

## Scientific scope

Raw SNe chi-squared near 1753 is expected for the 1701-row objective;
convergence is measured against the verified minimum near 1752.904.
Optimizer scatter describes repeated algorithm outcomes, rather than posterior
uncertainty. PNG/SVG/PDF exports derive from the validated records.
Paper-identical CMB+BAO remains limited by unidentified author emulator/TT
settings. Live QPU submission requires IBM account access and allocation.

See [replication specification](docs/REPLICATION_SPEC.md),
[tasks 8–12 results](docs/TASKS_8_TO_12_REPORT.md),
[IBM hardware details](docs/IBM_HARDWARE.md), and
[HQGA integration](docs/HQGA_INTEGRATION.md).
