"""Read-only compatibility audit of the historical PR #10 IBM experiments.

Reconstructs the documented logical inputs and replays saved measurements through
the current algorithm. This cannot establish the historical compiled-circuit
identity: the old ledger contains no QPY archive or optimizer provenance.
"""
import argparse
import json
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from aeqga.paths import PROJECT_ROOT
from aeqga.execution.ibm_sampler import BaseExecutor, _counts
from aeqga.experiments.runner import atomic_json, file_digest
from aeqga.likelihoods.pantheon_problem import PantheonPlusProblem
from aeqga.steps.evolution.aeqga_algorithm import (
    AEQGAParameters, build_dimension_circuit, run_aeqga_dual,
)


def completed_legacy_counts(record, widths):
    """Validate completed schema-v1 counts; never alter or upgrade their identity."""
    if (record.get("schema_version") != 1 or record.get("state") != "completed"
            or record.get("num_circuits") != len(widths)
            or len(record.get("counts", [])) != len(widths)):
        raise ValueError("Expected a completed legacy schema-v1 ledger with matching circuit count")
    shots = record.get("shots")
    if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
        raise ValueError("Legacy ledger shots must be a positive integer")
    return [_counts(c, "", shots, width)
            for c, width in zip(record["counts"], widths, strict=True)]


class HistoricalReplayExecutor(BaseExecutor):
    """One-generation offline replay, explicitly distinct from verified QPU resume."""
    def __init__(self, record):
        self.counts = completed_legacy_counts(record, [4, 3, 4, 3])
        super().__init__(shots=record["shots"])
        self.calls = 0

    def run(self, circuits, generation=0, optimizer_state=None):
        self.validate_circuits(circuits)
        if generation != 1 or self.calls or [c.num_qubits for c in circuits] != [4, 3, 4, 3]:
            raise ValueError("PR #10 replay requires generation 1 and circuit widths [4, 3, 4, 3]")
        self.calls += 1
        return [dict(c) for c in self.counts]


def analyze_pilot(directory=None):
    directory = Path(directory or PROJECT_ROOT / "outputs/json/ibm_first_pilot")
    record = json.loads((directory / "generation_0001.json").read_text())
    counts = completed_legacy_counts(record, [4])[0]
    recovered = json.loads((directory / "recovered_counts.json").read_text())
    if recovered != counts:
        raise ValueError("Pilot recovered counts disagree with its ledger")
    # The archived pilot used default_rng(42), [0.1, 1.0], and both gates forced on.
    state = random.getstate()
    try:
        random.seed(42)
        circuit = build_dimension_circuit(
            np.random.default_rng(42).uniform(.1, 1., 16), 1., 1., record["shots"])
    finally:
        random.setstate(state)
    ideal = Statevector(circuit.remove_final_measurements(inplace=False)).probabilities()
    measured = np.array([counts.get(format(i, "04b"), 0) for i in range(16)]) / record["shots"]
    tvd = float(np.abs(measured - ideal).sum() / 2)
    samples = np.random.default_rng(12345).multinomial(record["shots"], ideal, 10000)
    reference = np.abs(samples / record["shots"] - ideal).sum(axis=1) / 2
    return dict(job_id=record["job_id"], backend=record["backend"], shots=record["shots"],
                observed_states=len(counts), tvd=tvd,
                max_probability_error=float(np.max(np.abs(measured - ideal))),
                monte_carlo_trials=10000, finite_shot_mean_tvd=float(reference.mean()),
                finite_shot_p95=float(np.quantile(reference, .95)),
                finite_shot_p99=float(np.quantile(reference, .99)),
                trials_at_least_observed=int(np.sum(reference >= tvd)),
                interpretation="Conditional on reconstructed logical input; TVD is not gate fidelity")


def analyze_generation(directory=None, data=None):
    directory = Path(directory or PROJECT_ROOT / "outputs/json/ibm_generation_01")
    record = json.loads((directory / "generation_0001.json").read_text())
    executor = HistoricalReplayExecutor(record)
    problem = PantheonPlusProblem(data or PROJECT_ROOT / "sn_data/PantheonPlus")
    params = AEQGAParameters(pop_size=32, max_gen=1, num_shots=record["shots"], progress_bar=False)
    best, populations, _ = run_aeqga_dual(
        problem, params, seed=42, hardware_executor=executor, quiet=True)
    sim_best, sim_populations, _ = run_aeqga_dual(
        problem, params, seed=42, execution_mode="statevector_shots", quiet=True)
    actual = dict(hardware_initial=populations[0], hardware_final=populations[1],
                  simulator_initial=sim_populations[0], simulator_final=sim_populations[1])
    with np.load(directory / "population_comparison.npz") as archived:
        errors = {name: float(np.max(np.abs(values - archived[name])))
                  for name, values in actual.items()}
    fit = np.array([problem.compute_fitness(x) for x in populations[0]])
    elites = populations[0][np.argsort(fit)[:8]]
    reference = problem.classical_minimum()
    return dict(job_id=record["job_id"], backend=record["backend"], shots_per_circuit=record["shots"],
                circuit_widths=[4, 3, 4, 3], evaluations=best.evaluations,
                best_x=best.x.tolist(), best_chi2=float(best.fitness),
                simulator_best_x=sim_best.x.tolist(), simulator_best_chi2=float(sim_best.fitness),
                configured_classical_reference=reference,
                delta_chi2=float(best.fitness-reference["chi2"]),
                archived_population_max_abs_errors=errors,
                retained_elites_equal=bool(np.array_equal(elites, populations[1][:8])),
                hardware_simulator_final_max_abs_difference=float(np.max(np.abs(populations[1]-sim_populations[1]))),
                interpretation="Historical count replay; retained classical elite is not evidence of QPU improvement")


def compatibility_report(data=None):
    paths = [PROJECT_ROOT / "outputs/json/ibm_first_pilot" / name
             for name in ("generation_0001.json", "recovered_counts.json")]
    paths += [PROJECT_ROOT / "outputs/json/ibm_generation_01" / name
              for name in ("generation_0001.json", "population_comparison.npz")]
    return dict(pull_request="https://github.com/KrithikMohan/AEQGA/pull/10",
                merged_main_commit="599927f018fc5d299d7180f3d9424714b656ff76",
                live_qpu_submissions=0, pilot=analyze_pilot(), generation=analyze_generation(data=data),
                versions={name: version(name) for name in ("numpy", "scipy", "qiskit", "qiskit-aer", "qiskit-ibm-runtime")},
                artifact_sha256={str(p.relative_to(PROJECT_ROOT)): file_digest(p) for p in paths},
                legacy_identity_verified=False,
                limitations="Legacy compiled fingerprints cannot be verified without original ISA circuits. "
                "Historical records remain separate from new resumable jobs; use a new output directory.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compatibility_report(data=args.data)
    if args.output:
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
