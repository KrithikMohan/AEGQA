# Tasks 8–12: reproducible experiments and verified scientific figures

Source: [Sarracino et al., arXiv:2602.15459v1](https://arxiv.org/html/2602.15459v1),
§§2–4 and Figures 3–4. This change implements tasks 8–12 in order and executes
the full **SNe** ensemble. CMB+BAO remains a separately labeled reference path;
its unidentified author emulator/TT settings still prevent paper-identical
production replication. No classical optimum was injected into AEQGA.

## 8. Reproducible, resumable experiment runner

`aeqga.experiments.runner.run_ensemble` returns every outcome, full population
history, best-pair history, chi-squared history, seed, discovery update,
optimizer evaluation count and elapsed time. Atomic per-run JSON files are
committed before advancing the manifest. An interrupted run resumes only
unfinished work. A changed configuration, input hash, scientific-source hash
or numerical dependency version is rejected; corrupt histories are also
rejected. Do not overwrite an experiment: choose a new name. This is a
single-writer API, not a concurrent manifest writer.

Population/decoding draws, gate choices and readout are seeded. A separate RNG
stream supplies new circuit sampling seeds instead of reusing the same Aer
seed at every generation. Aer transpilation is also seeded. The explicit
`statevector_shots` CPU engine evaluates the actual post-gate Qiskit state and
draws finite-shot multinomial counts; it is not shot-free decoding. Default
production uses this fast classical simulation, with Aer separately validated.
Both engines follow the same physical gates/selection/decoding. Different
engines need not produce identical trajectories from the same seed.
An additional real-data native Aer run with seed 23, 32 individuals and 50
updates gave chi-squared 1754.606527 (gap 1.702480), with zero history
re-evaluation error. Backend validation is not a demand for identical fits.

Data SHA256s, selections, redshift mode, numerical grid, bounds, engine,
dependency versions and source digest are retained. `scripts.run_sne_ensemble`
and the notebook use identical experiment identities and checkpoint protocols.
The CMB+BAO CLI/opt-in notebook path now uses this runner independently, with
TT/model hashes, explicit backend and CAMB version. Optimizer evaluation budgets
exclude extra reference fits, history rechecks and plotting evaluations.

## 9. Staged validation and actual 300-run production ensemble

First run the complete regression suite and a 20-run real-data stage. Then run
300 independent SeedSequence-derived seeds, root seed 23, with n_p=32,
n_g=50, p_c=p_m=0.5, shots=4096, elite margin=0, Omega_m grid=300,
all 1701 Pantheon+ light-curve rows, MU_SH0ES/full STAT+SYS, zHD/zHEL.
Keep all outcomes, including poor fits. There is no favorable-seed selection,
local polishing, posterior sampler or synthetic fallback.

| Quantity | Measured production result |
|---|---:|
| Completed independent runs | 300 |
| Objective evaluations in AEQGA | 489,600 |
| Mean H0 / standard deviation (ddof=0) | 72.873758 / 0.177593 |
| Mean Omega_m / standard deviation (ddof=0) | 0.356091 / 0.014490 |
| Configured classical [H0, Omega_m] | [72.832041, 0.362876] |
| Configured classical chi-squared | 1752.904047 |
| Median / 90th percentile / largest optimization gap | 0.307812 / 2.535458 / 16.882487 |
| Fraction with delta chi-squared < 2.30 | 88.333% |
| Largest logged-chi-squared re-evaluation error | 0 |

The paper reports optimizer means/scatter near H0=72.81±0.22 and
Omega_m=0.362±0.016. Our distribution has comparable scale but is **not
identical or demonstrably unbiased**: relative to the configured minimum,
mean offsets are +0.041718 and −0.006785. These offsets are smaller than
single-run scatter, but are not negligible compared with the standard error
of a 300-run mean. Do not claim exact statistical replication. The restricted
elite interval and shot count are unspecified in the paper; no tuning was
performed to force agreement. About 11.7% of runs exceed the diagnostic
2.30 optimization-gap tolerance. This fraction is not a posterior coverage test.

The direct-distance classical minimum is [72.837349, 0.362369], chi-squared
1752.903310. Re-evaluating all 300 fits with direct distances gives a median
gap of 0.335368 and the same 88.333% below 2.30. Grid-vs-direct objective
differences at these outcomes have maximum 0.339051 and RMS 0.071296.

Experiment fingerprint:
`2eae33bd9b41c1f452081e692d0f8d4b88cf7afbc973e990369d38fd99849ee5`.
Complete data/version/source identifiers reside in the local manifest
`outputs/json/sne_production_checkpoint.json`; every run is separately stored
in `sne_production_checkpoint.runs/`. Generated data/figures remain local,
not a large Git attachment; the committed runner reproduces them.

## 10. Correct objective and optimizer-distribution contours

Resolved objective grids use the optimizer's exact configured objective and
subtract its independently profiled minimum, not the coarsest plotting sample.
Joint two-parameter thresholds use chi-square(df=2) quantiles at the Gaussian
1–5 sigma enclosed masses; the first three are 2.295749, 6.180074 and
11.829158. Nearest-neighbor objective grids use actual Omega_m nodes.

Optimizer covariance uses ddof=1. Covariance ellipses are descriptive Gaussian
approximations to outcomes, not posterior credible regions. KDE uses Scott
bandwidth after unit standardization and a 300×300 grid padded beyond the
sample range. Highest-density thresholds are found from integrated mass,
with a captured-mass check and ascending unique rendering levels. Singular
or insufficient samples raise; no arbitrary jitter or peak-fraction sigma
labels are used. KDE regions are explicitly labeled optimizer mass regions.

## 11. Shared plotting and exact history accuracy

`aeqga.visualization.scientific_plots` supplies objective contours, independent
scatter/mean/covariance ellipse, KDE regions, likelihood/optimizer overlays,
and four-panel raw chi-squared / convergence-gap / H0 / Omega_m histories.
All plots share units, restrained fills, black boundaries and compact legends.
PNG, SVG and PDF exports derive from the same records. Density labels were
moved to the legend after visual inspection found overlap on correlated clouds.
Figure sidecars link the experiment fingerprint/configuration, plotting-source
hash, Matplotlib version and actual PNG/SVG/PDF file hashes. Restyling can reuse
validated optimizer records without rerunning the ensemble.

Every plotted parameter pair is the **same global-best individual at that
update** as its plotted objective value. It is not a mean of random fresh
population members or a pair assembled from independent coordinate extrema.
Plot-data tests inspect the actual Matplotlib line arrays. All 15,300 production
best-pair values were re-evaluated; all histories include generation zero and
exactly 50 updates, are finite/in-bounds, and preserve monotonic best fitness.
Parameter trajectories need not be monotonic even when chi-squared decreases.

### Why the earlier chi-squared looked high

The former default was only 16 individuals/10 updates/1024 shots. Reproducing
its seed-23 run gives [72.762333,0.369770], raw chi-squared 1753.050107,
**only 0.146060 above** the configured minimum. A raw quadratic summed over
1701 correlated measurements is not expected to approach zero. A high absolute
number alone is not evidence of an incorrect fit; subtracting a mock constant
would conceal the science. The old fixed simulator-seed reuse was corrected,
but it did not invalidate the already-verified calibrated likelihood.

The new default single trajectory (seed 23, finite-shot CPU engine,
32 individuals/50 updates) gives [72.909151,0.351116], raw chi-squared
1753.408165 and gap 0.504118, with zero logged-pair re-evaluation error.
The default is no longer the undersized demo. Its actual raw values and
classical reference appear in one panel; the convergence gap has its own
clearly labeled panel. A seed's plateau is reported, never replaced by a
chosen favorable trajectory.

## 12. Notebook and replication report

The notebook defaults to paper-sized SNe settings and enables the full
resumable ensemble. Costly CMB, exact-probability and native HQGA branches
remain explicit opt-ins. Duplicated scientific plotting was replaced by shared
functions; figures are regenerated from validated experiment records.
Source execution counts/outputs are cleared to avoid presenting stale results
as current evidence. Fresh executed verification is performed separately.

The actual notebook executed with no errors and rendered all SNe figures.
35 discoverable tests passed, including true checkpoint interruption/resume,
configuration mismatch rejection, same-engine reproducibility, post-gate
finite-shot frequencies, integrated KDE masses, and plotted line identities.
Native HQGA runner compatibility and the paper-identical CMB ensemble remain
separate limitations, not silently simulated as completed work.
The CMB+BAO resumable CLI also passed a real-data CAMB smoke run (8 individuals,
one update, 16 evaluations). Its local reference chi-squared was 2613.352290;
its deliberately tiny optimizer run stopped at 3187.749613. This is an
execution/history check, not a converged CMB fit or production result.
The optional CMB+BAO notebook branch also executed successfully on real inputs
with its separate minimal budget and validated best-pair diagnostic histories.

## Reproduce

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.run_sne_ensemble --runs 20 --name sne_stage20
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.run_sne_ensemble --runs 300 --name sne_production
jupyter notebook notebooks/AEQGA.ipynb
python -m scripts.run_bao_cmb_aeqga --backend camb --pop 8 --gen 1 --shots 256 --output outputs/json/cmb_reference_smoke.json
```

Re-running the same ensemble uses/rechecks compatible run files. Change the
experiment name when altering source, data, settings or engine. The CLI rejects
mismatches instead of mixing results. Keep CMB reference and SNe experiment
identities separate.
