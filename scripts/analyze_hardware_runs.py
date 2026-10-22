"""Analyze completed hardware/offline experiments without submitting work."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from aeqga.execution.ibm_sampler import CheckpointManager
from aeqga.execution.workflow import pantheon_provenance
from aeqga.experiments.runner import validate_history
from aeqga.steps.evolution.aeqga_algorithm import AEQGAParameters
from aeqga.likelihoods.pantheon_problem import PantheonPlusProblem
from aeqga.analysis.accuracy import accuracy_report
from aeqga.visualization.scientific_plots import run_diagnostics, optimizer_scatter, accuracy_comparison


def analyze(directory, *, data=None, max_gap=2.30, max_tvd=None):
    if not np.isfinite(max_gap) or max_gap <= 0 or (max_tvd is not None and
            (not np.isfinite(max_tvd) or not 0 <= max_tvd <= 1)):
        raise ValueError("Gap tolerance must be positive and TVD threshold in [0, 1]")
    directory = Path(directory)
    paths = ([directory/"result.json"] if (directory/"result.json").exists()
             else sorted(directory.glob("run_*/result.json")))
    if not paths:
        raise ValueError("No completed runs found")
    records = [json.loads(path.read_text()) for path in paths]
    config = records[0]["configuration"]
    provenance = config["provenance"]
    kwargs = dict(selection=provenance["selection"], redshift=provenance["redshift"],
                  grid_size=provenance["grid_size"])
    problem = PantheonPlusProblem(data, **kwargs) if data else PantheonPlusProblem(**kwargs)
    if pantheon_provenance(problem) != provenance:
        raise ValueError("Analysis objective/data mismatch")
    params = AEQGAParameters(**config["parameters"])
    identity = {k:v for k,v in config.items() if k != "seed"}
    job_metrics, comparisons, shots = [], [], 0
    for path, record in zip(paths, records):
        if {k:v for k,v in record["configuration"].items() if k != "seed"} != identity:
            raise ValueError("Do not combine different hardware configurations/calibrations studies")
        validate_history(problem, params, record)
        for ckpt in sorted(path.parent.glob("generation_*.json")):
            saved = json.loads(ckpt.read_text())
            if saved.get("submission_state") != "completed":
                raise ValueError("A generation is incomplete")
            shots += saved["shots"] * saved["num_circuits"]
            job_metrics.append(dict(job_id=saved["job_id"], metrics=saved.get("job_metrics", {})))
        for comparison in path.parent.glob("comparison_*.json"):
            result = json.loads(comparison.read_text())
            if result.get("fingerprint") != record["fingerprint"]:
                raise ValueError("Comparison belongs to a different experiment")
            comparisons.extend(result["hardware_tvds"])
    reference = problem.classical_minimum()
    points = np.asarray([r["best_x"] for r in records])
    gaps = np.array([r["best_chi2"]-reference["chi2"] for r in records])
    accuracy = accuracy_report(points, [r["best_chi2"] for r in records], reference,
                               gap_tolerance=max_gap)
    usage_seconds = [item["metrics"].get("usage", {}).get("quantum_seconds") for item in job_metrics]
    complete_usage = bool(usage_seconds) and all(isinstance(t, (int, float)) for t in usage_seconds)
    report = dict(runs=len(records), classical_reference=reference,
                  accuracy=accuracy,
                  mean=points.mean(axis=0).tolist(), std_ddof0=points.std(axis=0).tolist(),
                  optimization_gaps=gaps.tolist(), max_gap=max_gap,
                  fraction_meeting_gap=accuracy["success_fraction"],
                  optimizer_tolerance_note="Optimization diagnostic, not posterior coverage",
                  recorded_shots=shots, elapsed_seconds=sum(r["elapsed_seconds"] for r in records),
                  reported_qpu_seconds=sum(usage_seconds) if complete_usage else None,
                  job_metrics=job_metrics, circuit_tvds=comparisons,
                  tvd_acceptance=(all(t <= max_tvd for t in comparisons)
                                  if max_tvd is not None and comparisons else None),
                  all_fits_meet_gap=bool(np.all(gaps < max_gap)))
    CheckpointManager.atomic_write(directory/"hardware_summary.json", report)
    figures = [(run_diagnostics(records[0], reference), "hardware_convergence"),
               (accuracy_comparison(accuracy), "hardware_accuracy")]
    if len(points) >= 3:
        figures.append((optimizer_scatter(points, reference), "hardware_scatter"))
    for figure, name in figures:
        for kind in ("png", "svg", "pdf"):
            figure.savefig(directory/f"{name}.{kind}", bbox_inches="tight")
        plt.close(figure)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--max-gap", type=float, default=2.30)
    parser.add_argument("--max-tvd", type=float)
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory, data=args.data, max_gap=args.max_gap, max_tvd=args.max_tvd), indent=2))
