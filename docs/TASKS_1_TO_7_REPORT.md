# Tasks 1–7: implementation and validation

Historical implementation report. The subsequent notebook/repository audit
retired the legacy APIs that this report originally described as retained.
For current replacements and future feature decisions, see
[DEPRECATED_CODE.md](DEPRECATED_CODE.md); use the calibrated notebook/README.

Tasks were handled in the requested order. Tasks 1–6 are implemented and
validated. Task 7 has functioning BAO, PICO and CAMB paths, with a documented
exact-reproduction limitation: the authors' PICO training file and exact TT
configuration remain unidentified. At the time of this original report no
production ensemble had been run; the subsequent [tasks 8–12 report](TASKS_8_TO_12_REPORT.md)
documents the now-completed 300-run SNe ensemble and scientific plotting workflow.

## 1. Freeze the replication specification

Created `REPLICATION_SPEC.md` with paper references, parameter order, domain,
population split, rotations, readout, decoding, main/study hyperparameters,
data requirements and validation gates. Distinguished requirements from
unspecified choices rather than treating old local defaults as paper facts.
Shot count, seeds, elite-box expansion, TT selection and model identity are
explicit uncertainties. This prevents claiming bit-for-bit reproduction.

## 2. Correct circuit measurement ordering

Construct state preparation without measurement; apply crossover/mutation;
add final measurement only in shot mode. Statevector mode uses the same gate
path. Validate with a deliberately nonuniform four-state population, both
genetic operations forced on, and 32768 shots. Post-gate probabilities differ
from initial probabilities and measured frequencies match them within 0.012.
This tests the computation, not just the visual circuit ordering.

## 3. Correct generation accounting and decoding bounds

Evaluate and log the initial population as generation zero, then perform
exactly `max_gen` updates. Track `pop_size*(max_gen+1)` objective evaluations.
Use one shared implementation for shot and statevector modes. Validate domain,
probabilities, sizes and finite populations. Preserve the elite quarter;
expose relative elite margin (default zero), clip to global bounds, retain
collapsed intervals, and apply Algorithm 2 lower-bound resampling to both
random and elite-copy decoding. Tests cover both backends, monotonic best
fitness, preserved elites, maximize/zero-update behavior and invalid settings.

## 4. Implement a dedicated Pantheon+ loader

Read named columns, not legacy numeric positions. Keep calibrated MU_SH0ES,
redshifts, errors, CID, calibrator flags, Cepheid metadata and source row
indices. Parse the 1701x1701 total covariance, check element count/finite
values/symmetry, apply the same row mask to both axes, and factor it once.
Support all rows or 1580 Hubble-flow non-calibrator rows. Retain multiple
light curves; do not deduplicate by CID. Default to zHD integral/zHEL prefactor.
Tests reconstruct the covariance from its Cholesky factor and verify masks.

## 5. Make H0 identifiable and use one objective

Add `PantheonPlusProblem`: calibrated distance-modulus residuals and full
covariance chi-squared, without arbitrary M or H0-erasing M marginalization.
Both optimizer and objective-grid/contour code call this objective. Keep the
old Pantheon likelihood explicitly available as legacy, with an H0 warning.
Compare the whitened quadratic to a dense linear solve and profile H0
analytically for an independent direct-distance baseline.

Default all-row, zHD/zHEL direct fit:

| Quantity | Measured |
|---|---:|
| H0 (km/s/Mpc) | 72.8373494 |
| Omega_m | 0.36236864 |
| chi-squared | 1752.9033102 |

The paper reports approximately (72.82, 0.363). Alternative data selections
are exposed for sensitivity tests, not tuned to force agreement.

## 6. Cache covariance and reproduce the paper's distance approximation

Add vectorized 48-node Gauss–Legendre distances and exact H0 scaling. Test
against adaptive quadrature over representative redshifts/densities. Remove
rounded-H0 cache keys. Precompute 100/300 uniformly spaced Omega_m nodes,
nearest-neighbor lookup without interpolation, and whitened quadratic
coefficients. Grid-node objectives agree with direct objectives to seven
decimal places. `benchmark_replication.py` repeats the accuracy/speed audit
over 500 fixed-seed points (H0=70..75, Omega_m=0.30..0.42).

| Grid | Fitted H0 | Fitted Omega_m | Minimum chi-squared excess vs direct |
|---|---:|---:|---:|
| 100 | 72.824096 | 0.363636 | 0.004590 |
| 300 | 72.832041 | 0.362876 | 0.000737 |

One measured run: direct 500 evaluations 1.117 s; grid 300 0.00162 s
(about 690x, excluding loading/precomputation; timing varies). Across the
tested rectangle, max/RMS chi-squared differences are 11.764/3.173 (100)
and 4.022/1.048 (300). Nearest-neighbor lookup is discontinuous; these numbers
are not a claim of uniformly negligible contour error. Use direct distances
for numerical-reference comparisons.

## 7. Implement separate CMB+BAO estimation

Replace zero-returning stubs with explicit configured likelihoods. Implement
all 16 BAO observables, Eq. 9 sound horizon, original correlated covariance
blocks, and cached whitening. Record the 11-vs-10 correlated-row discrepancy.
Add Planck TT multipole loading, units/error selection, fixed-parameter
mapping and diagonal residual objective. Checksum-pin the official PICO
model and TT data; add a reusable fetch command. Validate the supported PICO
example against matching CAMB spectra (2% relative plus 2 microkelvin-squared
absolute tolerance), test domain rejection, and test explicit hybrid fallback.

`BAOCMBProblem` sums CMB and BAO only; it does not include SNe.
`run_bao_cmb_aeqga.py` exposes backend, multipole/error choices, independent
local classical baseline, quantum run and JSON result output. Physically
impossible fixed-density combinations have infinite fitness.

At the paper target (66.76, 0.323), BAO chi-squared is 38.439337 and the
configured CAMB TT chi-squared is 2581.461296. The CAMB combined local
Nelder–Mead baseline converges to (67.414988, 0.314528), chi-squared
2613.352290. This does **not** reproduce the paper's CMB+BAO center; the
public PICO model rejects that target with the selected 2018 fixed inputs.
The exact author model/TT conventions are therefore still needed to certify
an exact reproduction. No forced PICO extrapolation or fitted-center tuning
is used to hide this limitation.

## Reproduction commands

Final verification: all 15 regression tests passed; whitespace
validation (`git diff --check`) passed. Both real-data smoke runs used
8 individuals, 2 updates and 512 shots, producing 3 history entries and
24 objective evaluations. These are execution checks, not convergence or
uncertainty claims. The final decoding post-process correction is covered
by the additional regression test and real-data smoke checks.

```bash
./venv/bin/python -m scripts.fetch_cmb_data
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv/bin/python -m unittest tests.test_replication -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv/bin/python -m scripts.benchmark_replication
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv/bin/python -m scripts.run_pantheon_aeqga --pop 8 --gen 2 --shots 512 --no-contour
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv/bin/python -m scripts.run_bao_cmb_aeqga --backend camb --classical-only
```

Downloaded data/models and generated outputs are ignored by Git. The source
tree and regression tests contain the reproducible implementation; fetch the
large model/data before the CMB tests. CAMB 2.0.4 is pinned in requirements.
The SNe command can fetch Cobaya's sn_data repository if absent; for the exact
data revision validated here, check out
`61d96434cafc2770928322c38e5a750e686368ae` in that data repository before tests.
