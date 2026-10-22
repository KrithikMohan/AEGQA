"""Compare archived generation circuits/counts against finite-shot references.

Rejects historical records without circuit identity. Never submits hardware jobs.
"""
import argparse
from pathlib import Path
import numpy as np
from qiskit import qpy
from aeqga.likelihoods.pantheon_problem import PantheonPlusProblem
from aeqga.execution.ibm_sampler import CheckpointManager, compute_circuit_fingerprint, _counts
from aeqga.execution.workflow import pantheon_provenance
from aeqga.analysis import (compute_circuit_ideal_probabilities, run_finite_shot_ensemble,
                           compare_counts_against_ensemble)


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable(item) for item in value]
    return value


def run_comparison(directory, *, data=None, generation=1, samples=1000, seed=12345):
    directory = Path(directory)
    checkpoint = directory/f"generation_{generation:04d}.json"
    saved = CheckpointManager.load(checkpoint)
    optimizer = CheckpointManager.load(directory/"optimizer.json")
    archive = checkpoint.with_suffix(".logical.qpy")
    if saved is None or optimizer is None or not archive.is_file():
        raise ValueError("Verified logical circuits and optimizer provenance are required; historical replay refused")
    if samples < 1:
        raise ValueError("samples must be positive")
    with archive.open("rb") as stream:
        circuits = qpy.load(stream)
    if [compute_circuit_fingerprint(c) for c in circuits] != saved["circuit_fingerprints"]:
        raise ValueError("Archived circuit identity mismatch")
    state = saved.get("optimizer_state")
    if state is None:
        raise ValueError("Missing pre-generation optimizer state")
    config = optimizer["configuration"]
    provenance = config["provenance"]
    problem = PantheonPlusProblem(data or Path(__file__).resolve().parents[1]/"sn_data/PantheonPlus",
                                   selection=provenance["selection"], redshift=provenance["redshift"],
                                   grid_size=provenance["grid_size"])
    if pantheon_provenance(problem) != provenance:
        raise ValueError("Analysis data/objective differs from hardware experiment")
    initial = np.asarray(state["current_population"])
    fits = np.array([problem.compute_fitness(x) for x in initial])
    if not np.allclose(fits, state["current_fitnesses"], atol=1e-8, rtol=0):
        raise ValueError("Stored fitness disagrees with real-data objective")
    elites = initial[np.argsort(fits)[:len(initial)//4]]
    expected = [int(np.log2(len(initial)))-i for i in (1, 2, 1, 2)]
    if [c.num_qubits for c in circuits] != expected or len(saved["raw_counts"]) != 4:
        raise ValueError("Wrong circuit/count ordering or cardinality")
    counts = [_counts(c, "", saved["shots"], width)
              for c, width in zip(saved["raw_counts"], expected)]
    probabilities = compute_circuit_ideal_probabilities(circuits)
    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        ensemble = run_finite_shot_ensemble(problem, circuits, elites, probabilities,
                                            n_samples=samples, shots=saved["shots"],
                                            ensemble_seed=seed, elite_margin=config["parameters"]["elite_margin"])
        result = compare_counts_against_ensemble(
            problem, counts, probabilities, elites, ensemble, initial,
            simulator_final_pop=ensemble["final_populations"][0],
            elite_margin=config["parameters"]["elite_margin"])
    finally:
        np.random.set_state(rng_state)
    result.update(fingerprint=optimizer["fingerprint"], generation=generation,
                  seed=seed, samples=samples,
                  decoding_note="Conditional re-decoding with analysis seed; not the recorded optimizer trajectory",
                  finite_shot_tvd_99=ensemble["tvd_99ths"])
    result = serializable(result)
    CheckpointManager.atomic_write(directory/f"comparison_{generation:04d}.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--generation", default=1, type=int)
    parser.add_argument("--samples", default=1000, type=int)
    args = parser.parse_args()
    result = run_comparison(args.directory, data=args.data, generation=args.generation, samples=args.samples)
    print("TVD:", result["hardware_tvds"])
