# Deprecated-code audit and future implementation decisions

Scope: all repository-owned Python modules, scripts, tests and the AEQGA
notebook, cross-checked against imports, direct callers and documented module
entry points. Third-party virtual environments, downloaded repositories,
data/models and generated historical PNG/JSON files were excluded. No dataset
or user-generated result file was deleted. Git retains the old source and
notebook outputs for recovery. The cleanup and retained integration bridge are
included together in the notebook/HQGA integration change set.

An uncalled function is not automatically obsolete: dynamic emulator symbols,
script entry points and numerical verification helpers were audited separately.
The following summaries replace executable deprecated alternatives rather
than leaving inactive code paths that can be mistaken for the paper workflow.

## Retired features

| Removed code | Features it provided | Future implementation decision | Current replacement |
|---|---|---|---|
| `load_pantheon`, `PantheonProblem` | Read 1048-row standardized apparent magnitudes and a legacy covariance; exposed an AEQGA problem | Not required for Pantheon+ replication. A future comparison with an uncalibrated dataset must explicitly model its calibration/nuisance degeneracy and cannot claim a two-parameter H0 constraint | `load_pantheon_plus`, `PantheonPlusProblem` with MU_SH0ES and matching total covariance |
| `chi2_pantheon`, `_chi2_diag` | Analytic marginalization of an additive absolute-magnitude offset, with full or diagonal covariance | Marginalization remains a valid technique for another model, but is not required here: it removes the H0 information the calibrated fit must retain. Reintroduce only in a separately specified nuisance-parameter experiment | Calibrated residuals, cached Cholesky whitening and a non-marginalized quadratic |
| `chi2_pantheon_fixed_M` | Fixed an arbitrary M=-19.36 to obtain elliptical apparent-magnitude contours | Not required. Never restore it as a cosmetic contour fix: it makes optimization and visualization disagree unless a separately justified calibration is included in both | `PantheonPlusProblem.objective_grid` evaluates the optimizer's exact configured objective |
| `luminosity_distance_flat_lcdm`, `distance_modulus`, `_D_L_CACHE` | Midpoint integration, scalar modulus evaluation and an unbounded per-distance cache | Not required; duplicated the vectorized implementation. Future expansion models belong in the common cosmology module with quadrature-reference tests and explicit cache semantics | Gauss–Legendre `distance_integral` and `distance_moduli`; bounded quadrature-node cache and paper-style Omega_m precomputation |
| `chi2_bao`, `chi2_cmb` compatibility wrappers | Forwarded calls to supplied likelihood instances and rejected missing placeholder data | Not required; no actual pipeline called them. A future common likelihood protocol should be typed and shared, rather than attached to the SNe module | `BAOLikelihood.chi2`, `PlanckTTLikelihood.chi2`, `BAOCMBProblem` |
| `compute_kde_contours` | Gaussian KDE over best-fit pairs; returned density grid, mean/std and peak | Density visualization is useful for future paper-like optimizer contours, but this uncalled helper did not implement probability-mass contour thresholds or handle singular samples. Do not label fractions of peak density as confidence probabilities | Notebook plots raw independent-run outcomes. Future density tools need enough full-rank samples, normalization/mass-integrated thresholds, ascending unique levels, convergence diagnostics and tests |
| `build_amplitude_circuit_with_measure` | Convenience state initialization immediately followed by readout; unused shots argument | Not required and unsafe to compose with later crossover/mutation. State preparation must not consume the final readout step | `build_amplitude_circuit` plus `build_dimension_circuit`, which measures after genetic gates |
| `run_aeqga` | Deprecated single-run alias forwarding to the dual-subset loop | Not required. External consumers should migrate to the actual independent-parameter dual-subset API | `run_aeqga_dual` |
| Old evolution-module `run_qga_aeqga` shim | Converted HQGA real objectives and a subset of parameters in one direction | Retain interoperability for future HQGA comparisons, but replace this incomplete shim with explicit, tested interfaces in a separate integration module | `aeqga.integrations.hqga`: real-objective adapter, budget adapter, native AEQGA runner bridge and reverse Gray-code problem wrapper; see [connection guide](HQGA_INTEGRATION.md) |
| `run_aeqga_iterations` and single-run `n_iterations` field | Repeated runs, computed mean/std, but retained only the final history and no sample/seed provenance | Repeated runs are required for future production statistics; this lossy duplicate implementation is not. Future library orchestration should return every fit/history, independent seeds, configurations and failure status | Notebook's opt-in ensemble records all best points, chi-squared values and run seeds; ensemble count is a workflow setting, not a single-run parameter |
| `plot_convergence` | Generic global pyplot figure, fixed filename, interactive show | Not required; only the retired legacy script called it, and the CLI's import was unused. Future shared plotting should accept axes/output targets and use identical scientific baselines | Current notebook and SNe runner plot their actual histories |
| `test_pantheon_aeqga.py` | Ad-hoc offline assertions plus legacy loading, absolute-magnitude fits, stochastic H0 assertions and optional interactive figures | Legacy calibration-dependent fit assertions are not useful future validation. Distance units, amplitude preparation, Eq.14/15 gate matrices and decoding edges remain required | Useful checks migrated to `test_quantum_primitives.py`; real-data and generation checks remain in `test_replication.py`. Summary at `tests/LEGACY_TESTS.md` |
| Old notebook mock problem, saved fitted centers and failed KDE logs | Demonstrated optimization without real data; carried obsolete H0 results and a descending-level contour failure | Not required as scientific evidence. A future synthetic experiment must be clearly named, independent, and validated against known generative truth | Real-data preflight with an actionable exception; current provenance-aware results, sorted likelihood levels and explicitly labeled demo/ensemble modes |
| CLI's coarse plot-grid minimum subtraction | Used the lowest sample on a sparse 30x30 plot as the likelihood zero | Not required and can shift every contour when the sparse grid misses the true minimum. Future plotting must use an independently established reference and show sampling limitations | Current CLI and notebook use the configured classical minimum; the plotting grid resolves the narrow fit region and preserves the actual objective |
| `_sum_m_nu` in the SNe module and unused imports | A disconnected neutrino-mass constant and stale import paths | No future requirement for disconnected declarations. Physical inputs must live where they affect the prediction | Active CMB parameter mapping retains its explicit neutrino configuration; environment verification uses current package imports |

The legacy CLI `--dataset pantheon` path was also retired. The SNe runner
always uses calibrated Pantheon+; `--data` selects a Pantheon+ directory.
No error is hidden by restoring legacy data or a mock objective.

## Kept after review

- `run_aeqga_sv`: optional exact-probability diagnostic used by tests/notebook;
  not equivalent to a deterministic shot-based experiment.
- `verify_u_cross`, `verify_rx_pi2`: scientifically meaningful gate-matrix
  regression helpers; now exercised by discoverable tests.
- `chi2_pantheon_plus`: independent direct-distance likelihood used to verify
  the accelerated problem against a dense covariance solve.
- `sound_horizon`: Eq.9 is actively called by BAO; not an unused SNe feature.
- PICO `PICO`, `CantUsePICO`, `create_pico` protocol symbols: imported
  dynamically by the checksum-pinned model source. The training callback raises
  explicitly because this is a runtime adapter, not a training implementation.
- Circuit illustration helpers: a documented executable visualization,
  not optimizer logic; CNOT/RY decomposition remains mathematically useful.
- Fetch, benchmark and setup entry points: documented workflows even when
  no library imports their `main` functions. Stale setup import paths were fixed.
- Full covariance, source row indices, Cepheid/calibrator metadata and alternate
  row/redshift selections: necessary for calibration/provenance sensitivity tests.
- The original diagnosis and tasks 1–7 report: historical documentation,
  retained with current-status notices rather than misrepresented as live APIs.

## Validation approach

Check notebook structure/code, absence of retired scientific calls, opt-in
defaults and output provenance; execute the actual default notebook with no
mock data; visually inspect its generated likelihood diagnostic; rerun the
whole discoverable regression suite and the cleaned SNe CLI smoke test.
Production 300-run uncertainties and exact paper-identical CMB configuration
are not certified by a short notebook execution.

Validation results: 29 discoverable tests passed, including the retained HQGA
bridge tests; the default 16-individual,
10-update notebook completed without cell errors and displayed its inline
likelihood figure. A separate real-data smoke notebook enabled three independent
ensemble runs and the exact-statevector diagnostic with 8 individuals and one
update per run. The cleaned SNe CLI completed with both convergence and contour
outputs. Source notebook execution counts/outputs remain cleared deliberately:
opening it cannot confuse old cached results with the newly configured run.

The optional CMB+BAO branch was also executed end-to-end on the actual Planck
TT table and correlated BAO data, with CAMB explicitly selected and a minimal
8-individual, one-update run. Its independently fitted local reference was
finite, the AEQGA run used 16 evaluations, and the result was saved under a
smoke-specific filename with `exact_paper_reproduction: false`. No paper-sized
ensemble or forced PICO extrapolation was run.
