# Accuracy against a reference and graph validation

The maintained notebook has a new section **9a: Accuracy against the classical
answer**. Its report is implemented in `aeqga.analysis.accuracy`; its plot is
`aeqga.visualization.scientific_plots.accuracy_comparison`. Simulation and
hardware analysis use the same functions, so their metrics have the same meaning.

## What the new report measures

For each parameter and each outcome, signed error is estimate minus reference;
absolute error is its magnitude; absolute percentage error is 100 times that
magnitude divided by the absolute reference value. Percentage error is undefined
at a zero reference and is saved as JSON null. It is not a confidence score.

Across outcomes, bias is mean signed error, MAE is mean absolute error, and
RMSE is the square root of mean squared error. Small bias alone can conceal
large errors that cancel, so MAE/RMSE and every run are retained.

Delta chi-squared is the objective evaluated at the AEQGA result minus the
minimum of the identical objective. Smaller means better optimization. Negative
gaps below numerical tolerance raise an error: the supplied reference or
objective identity needs checking. Tiny negative rounding residuals are
retained rather than silently changed to zero.

The configured grid and direct-distance objectives are evaluated separately
in the notebook. Subtracting a direct-distance minimum from a grid objective
would confound approximation error with optimizer accuracy. The direct check
uses the same AEQGA points, without polishing or replacing any outcome.

Success means delta chi-squared < 2.30, an explicitly stated diagnostic
optimization tolerance. The Wilson 95% interval describes uncertainty in an
independent-run success fraction. With only one run it is wide. Hardware drift,
correlated runs or changing settings violate the independent comparable-trial
interpretation; the interval must then be treated as descriptive. Neither the
success fraction nor that interval is a probability that cosmology is true.

The reference is a classical fit to this dataset and model. It is not known
physical truth. An optional labeled external point can be supplied to the
report for parameter agreement only. A paper point, a different experiment and
a synthetic ground truth must not be silently treated as equivalent references.

## How reliable are the graphs?

| Graph | Verified behavior | Interpretation/limitation |
|---|---|---|
| Parameter/chi-squared histories | Tests inspect actual Matplotlib line arrays; experiment validation recomputes logged best-pair objectives | One trajectory does not establish an ensemble success rate |
| New reference-accuracy plot | Tests check actual offsets, gaps and empirical cumulative fractions; outliers are kept | Coordinate distance and objective distance answer different questions |
| Joint likelihood contours | Tests verify 2-parameter thresholds and numerically evaluate actual rendered contour vertices on a known correlated quadratic | Gaussian-equivalent/asymptotic likelihood regions conditional on the model, data and covariance; not calibrated coverage for this cosmology |
| Covariance ellipse | Tests check center, sample covariance eigenvalues and joint Gaussian radius | Descriptive Gaussian approximation to optimizer outcomes |
| KDE regions | Tests integrate probability mass above thresholds and reject singular/insufficient outcomes | Smoothed optimizer distribution; bandwidth/grid and finite sample size matter |
| Classical reference lines | Single-run convergence now uses the classical minimum rather than chi-squared at a quoted paper point | A baseline is only comparable when objective/data/settings match |

The contour-vertex test uses a known analytic correlated Gaussian objective;
rendering is checked to absolute delta-chi-squared error <0.01 at its first
three contour boundaries. This is a test of that grid/rendering example, not
a universal error bound on real-data contours. Hand-calculated examples check
MAE, RMSE, percentage errors, gap accounting and interval behavior.

## Numerical and scientific limits

The Pantheon+ fast objective uses nearest-neighbor Omega_m lookup. With a
300-node grid spanning [0,0.5], node separation is about 0.00167. Contour
rendering interpolates between evaluated grid nodes and can look smoother than
the actual stepwise objective. Multiple Omega_m coordinates within a lookup
cell may have identical objective values; a nonzero coordinate difference
there need not mean a worse fit. Direct distances provide a separate check.
The classical direct minimum uses numerical quadrature and a scanned/profiled
search, rather than a proof of a global minimum for every possible cosmology.

For the historical 300-run study, the repository report records grid-versus-
direct chi-squared differences up to 0.339051 at the returned fits. Those are
historical results, not a new measurement. The historical source report is
[TASKS_8_TO_12_REPORT.md](TASKS_8_TO_12_REPORT.md). Without its original datasets
and validated records present locally, new production accuracy numbers are
not inferred from old images or filled in from the paper.

Current tests establish numerical/plotting consistency on their checked cases.
They do not validate the data calibration, prove the assumed cosmological
model, establish exact frequentist coverage, or certify live QPU fidelity.
Historical root-level PNGs/notebooks may predate the maintained plotting code;
use newly exported figures with matching experiment/figure provenance.

## Outputs

- Notebook: `EXPERIMENT_NAME_notebook_accuracy.json`, configured and direct
  reports, grid-versus-direct objective differences, and PNG/SVG/PDF figures.
- Simulation ensemble: `NAME_accuracy.json`, `NAME_accuracy.*`, and an accuracy
  field in the ensemble summary.
- Hardware analysis: accuracy field in `hardware_summary.json` and
  `hardware_accuracy.png/.svg/.pdf` alongside raw execution records.

No fresh 300-run production result or new hardware confidence claim is created
by adding this reporting section. Real-data generation requires the corresponding
inputs; source notebook outputs remain cleared until an actual execution.
