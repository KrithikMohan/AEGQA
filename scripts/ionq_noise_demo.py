"""100-shot local Aer noise pilot on real Pantheon+; never submits cloud jobs."""
import argparse
import random
from time import perf_counter

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector

from aeqga.execution.ionq_noise import BASIS, IonQDemoExecutor, IonQDemoNoise
from aeqga.execution.ibm_sampler import CheckpointManager, get_provenance_metadata
from aeqga.execution.workflow import pantheon_provenance
from aeqga.experiments.runner import scientific_source_digest, file_digest
from aeqga.likelihoods.pantheon_problem import PantheonPlusProblem
from aeqga.paths import PROJECT_ROOT, output_path
from aeqga.steps.evolution.aeqga_algorithm import (
    AEQGAParameters, build_dimension_circuit, run_aeqga_dual,
)


def _probabilities(counts, width, shots):
    return np.array([counts.get(format(i, f"0{width}b"), 0)
                     for i in range(2**width)]) / shots


def circuit_probe(shots=100, seed=23, noise=None):
    """Separate exact gate-noise effects from finite-shot sampling on fixed circuits.

    Exact probabilities are obtained before measurement, then the independent
    symmetric readout channel is applied analytically. This avoids estimating
    a small noise effect from only 100 samples.
    """
    noise = noise if noise is not None else IonQDemoNoise()
    records = []
    rng = np.random.default_rng(seed)
    state = random.getstate()
    try:
        for width in (3, 4):
            random.seed(seed+width)
            circuit = build_dimension_circuit(rng.uniform(60, 80, 2**width),
                                              1., 1., shots, preparation="state_preparation")
            ideal = Statevector(circuit.remove_final_measurements(inplace=False)).probabilities()
            noisy_executor = IonQDemoExecutor(shots=shots, seed=seed, noise=noise)
            ideal_executor = IonQDemoExecutor(shots=shots, seed=seed, noise=IonQDemoNoise(0, 0, 0))
            compiled = transpile(circuit, basis_gates=BASIS, optimization_level=1, seed_transpiler=seed)
            clean = compiled.remove_final_measurements(inplace=False)
            error = float(np.max(np.abs(Statevector(clean).probabilities()-ideal)))
            if error > 1e-10:
                raise ValueError("Compilation changed ideal probabilities")
            clean.save_probabilities()
            exact = np.asarray(noisy_executor.simulator.run(clean).result().data(0)["probabilities"])
            readout = np.array([[1-noise.readout, noise.readout],
                                [noise.readout, 1-noise.readout]])
            assignment = readout
            for _ in range(width-1):
                assignment = np.kron(readout, assignment)
            exact = assignment @ exact
            ideal_counts = ideal_executor.run([circuit])[0]
            noisy_counts = noisy_executor.run([circuit])[0]
            records.append(dict(width=width, ideal_probability_error=error,
                                ideal_probabilities=ideal.tolist(), noisy_probabilities=exact.tolist(),
                                exact_noise_tvd=float(np.abs(exact-ideal).sum()/2),
                                ideal_sampling_tvd=float(np.abs(_probabilities(ideal_counts, width, shots)-ideal).sum()/2),
                                noisy_sampling_tvd=float(np.abs(_probabilities(noisy_counts, width, shots)-exact).sum()/2),
                                ideal_counts=ideal_counts, noisy_counts=noisy_counts,
                                compilation=noisy_executor.diagnostics.summary()[0]))
    finally:
        random.setstate(state)
    return records


def run_demo(*, shots=100, generations=5, seeds=(23, 24, 25, 26, 27), noise=None):
    """Small independent paired trials; no convergence or hardware-cost claim."""
    noise = noise if noise is not None else IonQDemoNoise()
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use distinct independent seeds")
    params = AEQGAParameters(pop_size=32, max_gen=generations, num_shots=shots,
                             progress_bar=False, state_preparation="state_preparation")
    params._validate()
    if generations < 1:
        raise ValueError("At least one generation required")
    started = perf_counter()
    probes = circuit_probe(shots, seeds[0], noise)
    problem = PantheonPlusProblem()
    reference = problem.classical_minimum()
    records = []
    for seed in seeds:
        for label, config in (("ideal", IonQDemoNoise(0, 0, 0)), ("noisy", noise)):
            executor = IonQDemoExecutor(shots=shots, seed=seed, noise=config)
            tick = perf_counter()
            best, populations, history = run_aeqga_dual(
                problem, params, hardware_executor=executor, seed=seed, quiet=True)
            elapsed = perf_counter()-tick
            if not np.all(np.isfinite(populations)) or any(
                    np.any(p < problem.lower_bounds) or np.any(p > problem.upper_bounds)
                    for p in populations):
                raise ValueError("Nonfinite or out-of-bounds decoded population")
            values = np.asarray([h[1] for h in history])
            if np.any(np.diff(values) > 1e-8):
                raise ValueError("Elitism failed: best fitness increased")
            if abs(problem.compute_fitness(best.x)-best.fitness) > 1e-8:
                raise ValueError("Saved best fitness does not match the objective")
            for batch in executor.batches:
                if [c["original_width"] for c in batch["compilation"]] != [4, 3, 4, 3]:
                    raise ValueError("Unexpected generation circuit widths")
            row = dict(mode=label, seed=seed, seconds=elapsed, best_x=best.x.tolist(),
                       chi2=best.fitness, delta_chi2=best.fitness-reference["chi2"],
                       parameter_error=(best.x-np.array([reference["H0"], reference["Omega_m"]])).tolist(),
                       history=history, batches=executor.batches)
            records.append(row)
            print(f"{label:5} seed={seed}: {elapsed:.3f}s, chi2={best.fitness:.4f}, delta={row['delta_chi2']:.4f}", flush=True)
    circuits = sum(len(b["counts"]) for r in records for b in r["batches"])
    report = dict(schema_version=1, noise=noise.metadata(), shots_per_circuit=shots,
                  parameters=vars(params), seeds=list(seeds), reference=reference,
                  data= pantheon_provenance(problem), versions=get_provenance_metadata(),
                  scientific_source=scientific_source_digest(),
                  demo_source={str(p.relative_to(PROJECT_ROOT)): file_digest(p) for p in
                               (PROJECT_ROOT/"scripts/ionq_noise_demo.py", PROJECT_ROOT/"aeqga/execution/ionq_noise.py")},
                  probes=probes, runs=records, elapsed_seconds=perf_counter()-started,
                  workload=dict(optimizer_circuits=circuits, probe_sampled_circuits=4,
                                sampled_circuit_shots=(circuits+4)*shots,
                                exact_density_matrix_probes=2,
                                optimizer_objective_evaluations=len(records)*32*(generations+1),
                                cloud_jobs=0, cloud_service_cost_usd=0,
                                note="Local CPU/electricity cost unpriced; runtime is not a QPU cost estimate"))
    return report


def save_demo(report):
    """Export readable numeric evidence and plots from the same run."""
    import matplotlib.pyplot as plt
    CheckpointManager.atomic_write(output_path("json", "ionq_noise_demo.json"), report)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode, color in (("ideal", "tab:blue"), ("noisy", "tab:orange")):
        rows = [r for r in report["runs"] if r["mode"] == mode]
        gaps = np.array([[h[1]-report["reference"]["chi2"] for h in r["history"]] for r in rows])
        for row in gaps:
            axes[0].plot(np.arange(len(row)), row, color=color, alpha=.25)
        axes[0].plot(np.arange(gaps.shape[1]), np.median(gaps, axis=0), color=color, label=mode)
        axes[1].scatter([r["seconds"] for r in rows], [r["delta_chi2"] for r in rows], color=color, label=mode)
    axes[0].set(xlabel="Generation (0 = initial)", ylabel="Best chi-squared minus classical minimum",
                title="Individual trials and median")
    axes[1].set(xlabel="Local execution seconds (includes compilation)", ylabel="Final delta chi-squared",
                title="100-shot pilot" if report["shots_per_circuit"] == 100 else "Local pilot")
    for axis in axes:
        axis.legend()
    fig.suptitle("Illustrative Aer noise assumptions; no IonQ calibration")
    fig.tight_layout()
    for kind in ("png", "svg", "pdf"):
        fig.savefig(output_path(kind, f"ionq_noise_demo.{kind}"))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    report = run_demo(shots=args.shots, generations=args.generations, seeds=range(23, 23+args.runs))
    save_demo(report)
    print(report["workload"])


if __name__ == "__main__":
    main()
