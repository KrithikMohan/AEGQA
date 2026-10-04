# AEQGA cosmological parameter estimation

Implementation of Sarracino et al.'s [AEQGA paper](https://arxiv.org/html/2602.15459v1), adapted from [HQGA](https://github.com/Quasar-UniNA/HQGA).

## Project layout

```
aeqga/
  steps/
    selection/          # Classical elites, duplicates, fresh random individuals
    encoding/           # L2 normalization and amplitude state preparation
    genetic_operators/  # Bidirectional CRY crossover and RX mutation
    decoding/           # Measured counts to classical parameters
    evolution/          # Fitness evaluation, generation loop, history and best fit
  likelihoods/          # Calibrated Pantheon+, BAO, Planck TT, distances
  emulators/            # Checksum-pinned PICO runtime
  experiments/          # Seeded independent runs and strict atomic checkpoints
  integrations/         # Bidirectional HQGA objective and Gray-code adapters
  visualization/        # Publication-style circuit drawing
  paths.py              # Working-directory-independent data/output locations
scripts/                # Experiment runners, data fetch, benchmark, environment setup
tests/                  # Offline, layout and real-data regression tests
notebooks/              # Current calibrated workflow; costly branches are opt-in
docs/                   # Replication specification, methodology report, original diagnosis
outputs/
  png/                  # Circuit, convergence and contour raster figures
  svg/                  # Vector circuit diagrams
  pdf/                  # Printable circuit diagrams
  json/                 # Numerical results and benchmark reports
sn_data/                # External SNe data repository (ignored)
data/                   # External CMB/model and BAO reference downloads (ignored)
```

The generation cycle is fitness evaluation → selection/duplication → amplitude encoding → genetic gates → measurement/decoding → recombination. Each parameter and subset has its own circuit; 25% of individuals bypass the circuit as unchanged elites.

## Setup and execution

From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -e .
git clone https://github.com/CobayaSampler/sn_data.git sn_data
python -m scripts.fetch_cmb_data
python -m scripts.run_pantheon_aeqga --pop 8 --gen 2 --shots 512 --no-contour
python -m scripts.run_bao_cmb_aeqga --backend camb --classical-only
python -m scripts.benchmark_replication
python -m scripts.run_sne_ensemble --runs 300 --name sne_production
python -m aeqga.visualization.circuit_diagram
```

Use module commands (`python -m …`), not the former root-level script paths. Alternatively, `python -m scripts.setup_env` prepares the root virtual environment. Editable installation lets the package and notebook imports work outside the repository root. Default data/output paths always resolve relative to the repository.

## Tests

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
python -m tests.test_quantum_primitives
```

Real-data regression tests are explicitly skipped if their downloads are absent. Fetch both datasets/model before a full integration run. Offline distance, gate, encoding and decoding checks are included in unittest discovery.

## Scientific scope and limitations

Default SNe estimation uses calibrated Pantheon+ distance moduli and the total covariance; H0 is identifiable. CMB+BAO runs separately and never includes SNe. The notebook checks real inputs without mock fallback and now defaults to 32 individuals, 50 updates and a resumable 300-run SNe ensemble. CMB, exact-probability and native HQGA runs remain opt-in. MODE="demo" explicitly selects the shorter exploratory workflow.

Strict `--backend pico` rejects points outside the public model's training domain. `camb` and `hybrid` are explicit reference alternatives, not claims of an exact match to the paper's unidentified emulator. Likelihood contours, descriptive covariance ellipses and mass-integrated KDE regions are now separately labeled. Optimizer scatter is never claimed as posterior uncertainty.

Start with `jupyter notebook notebooks/AEQGA.ipynb`, then Run All for the configured paper-sized SNe workflow. The notebook saves `notebook_`-prefixed figures and current JSON records and never loads the old legacy output files. Deprecated APIs and the old `--dataset pantheon` path were retired; see the [feature audit and future implementation decisions](docs/DEPRECATED_CODE.md).

The notebook and ensemble CLI use provenance-checked atomic checkpoints. The explicit fast CPU quantum simulation evaluates post-gate states and samples finite-shot counts; `--engine aer` selects native Aer instead. Choose a new experiment name if changing data, settings, engine or scientific source. Every run's populations, best-pair/chi-squared histories and seed remain available. Shared figures export PNG/SVG/PDF.

Raw SNe chi-squared near 1753 is expected for the configured 1701-row objective. Convergence is assessed by the excess above the verified minimum near 1752.904, not by forcing the raw value toward zero. See the [tasks 8–12 report](docs/TASKS_8_TO_12_REPORT.md) for actual 300-run results, exact plot-data checks, residual optimizer bias and remaining CMB limitations.

See the [replication specification](docs/REPLICATION_SPEC.md) and [tasks 1–7 methodology/report](docs/TASKS_1_TO_7_REPORT.md). The [original diagnosis](docs/code-summary.md) is historical and may describe deficiencies that have since been fixed.

HQGA interoperability is retained in a separate tested bridge. See [connection examples and comparison methodology](docs/HQGA_INTEGRATION.md) for using existing HQGA objectives in AEQGA or the same calibrated cosmology objective in HQGA's native runner. The notebook includes a Gray-code contract check and an opt-in HQGA run; legacy HQGA runtime compatibility requires a separate compatible environment or an explicit execution-layer port.

Generated scientific PNG/SVG/PDF/JSON results and downloaded datasets/models stay local; previously tracked SVG/PDF circuit examples remain preserved.
