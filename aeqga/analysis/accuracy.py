"""Agreement with a specified reference, separate from cosmological uncertainty."""
import numpy as np
from scipy.stats import norm

PARAMETERS = ("H0", "Omega_m")


def parameter_errors(points, reference):
    """Signed errors and aggregate accuracy in each parameter's own units.

    Relative errors at a zero reference are undefined and serialized as null.
    A literature comparison measures agreement; it does not establish truth.
    """
    points = np.asarray(points, dtype=float)
    target = np.array([reference[name] for name in PARAMETERS], dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) == 0
            or not np.all(np.isfinite(points)) or not np.all(np.isfinite(target))):
        raise ValueError("Need finite nonempty [H0, Omega_m] points and reference")
    errors = points-target
    result = {}
    for index, name in enumerate(PARAMETERS):
        signed = errors[:, index]
        absolute = np.abs(signed)
        denominator = abs(target[index])
        result[name] = dict(reference=float(target[index]), estimates=points[:, index].tolist(),
                            signed_errors=signed.tolist(), absolute_errors=absolute.tolist(),
                            absolute_percent_errors=(100*absolute/denominator).tolist() if denominator else None,
                            mean=float(points[:, index].mean()), bias=float(signed.mean()),
                            mae=float(absolute.mean()), rmse=float(np.sqrt(np.mean(signed**2))),
                            max_absolute_error=float(absolute.max()),
                            mean_absolute_percent_error=float(100*absolute.mean()/denominator) if denominator else None)
    return result


def accuracy_report(points, objective_values, reference, *, gap_tolerance=2.30,
                    reference_label="Classical minimum of the configured objective",
                    comparison_reference=None):
    """Quantify optimizer accuracy, never a probability that cosmology is true.

    The Wilson interval describes a binomial success rate only for independent
    runs with comparable settings. Correlated hardware drift violates that model.
    Do not mix direct-distance chi2 values with a grid-objective minimum.
    """
    parameters = parameter_errors(points, reference)
    values = np.asarray(objective_values, dtype=float)
    if (values.shape != (len(points),) or not np.all(np.isfinite(values))
            or not np.isfinite(reference["chi2"]) or not np.isfinite(gap_tolerance)
            or gap_tolerance <= 0):
        raise ValueError("Need matching finite objectives/reference and positive gap tolerance")
    gaps = values-float(reference["chi2"])
    if np.any(gaps < -1e-7):
        raise ValueError("Candidate beats the supplied minimum; verify reference/objective consistency")
    # Preserve tiny numerical negative residuals; do not conceal reference errors.
    success = gaps < gap_tolerance
    n, k = len(gaps), int(success.sum())
    fraction = k/n
    z = float(norm.ppf(.975))
    center = (fraction+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(fraction*(1-fraction)/n+z*z/(4*n*n))/(1+z*z/n)
    report = dict(reference_label=reference_label,
                  interpretation="Agreement with a fitted reference, not probability of physical correctness",
                  runs=n, parameters=parameters, objective_values=values.tolist(),
                  reference_chi2=float(reference["chi2"]), delta_chi2=gaps.tolist(),
                  gap_quantiles=dict(zip(("min", "median", "p90", "max"),
                                         np.quantile(gaps, [0, .5, .9, 1]).tolist())),
                  gap_tolerance=float(gap_tolerance), success_rule="delta_chi2 < gap_tolerance",
                  successful_runs=k, success_fraction=fraction,
                  success_rate_wilson_95=[max(0., float(center-radius)), min(1., float(center+radius))],
                  interval_assumption="Independent comparable runs; descriptive only for dependent/drifting hardware",
                  percent_error_note="100 * absolute parameter difference / absolute reference; not percent confidence")
    if comparison_reference is not None:
        if not comparison_reference.get("label"):
            raise ValueError("External reference needs an explicit source/interpretation label")
        report["external_comparison"] = dict(label=comparison_reference["label"],
            interpretation="Parameter agreement only; this point is not assumed to be ground truth",
            parameters=parameter_errors(points, comparison_reference))
    return report


def accuracy_markdown(report):
    """Compact notebook table, computed from the same JSON report as the plot."""
    rows = [f"**Reference:** {report['reference_label']}", "",
            "| Parameter | Reference | AEQGA mean | Signed bias | MAE | RMSE | Mean absolute error (%) |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for name, item in report["parameters"].items():
        percent = item["mean_absolute_percent_error"]
        text = "undefined (zero reference)" if percent is None else f"{percent:.4f}"
        rows.append(f"| {name} | {item['reference']:.6f} | {item['mean']:.6f} | "
                    f"{item['bias']:+.6f} | {item['mae']:.6f} | {item['rmse']:.6f} | {text} |")
    low, high = report["success_rate_wilson_95"]
    quantiles = report["gap_quantiles"]
    rows.extend(["", f"Δχ² above reference: median **{quantiles['median']:.6g}**, "
                 f"90th percentile {quantiles['p90']:.6g}, maximum {quantiles['max']:.6g}.",
                 "", f"Runs with Δχ² < {report['gap_tolerance']:g}: "
                 f"**{report['successful_runs']}/{report['runs']} ({100*report['success_fraction']:.1f}%)**. "
                 f"95% Wilson interval for independent-run success rate: {100*low:.1f}–{100*high:.1f}%.",
                 "", "This measures agreement with the fitted reference. It is not confidence that the cosmological parameters are true."])
    return "\n".join(rows)
