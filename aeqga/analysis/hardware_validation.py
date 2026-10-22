"""Offline preparation/noise comparisons and explicit mitigation policy.

Corrected probabilities are analysis variants. Always retain raw integer counts.
The default optimizer consumes raw counts; mitigation is never silently enabled.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from aeqga.execution.ibm_sampler import CompilationDiagnostics, DryRunExecutor, verify_mps_probability_equivalence
from aeqga.steps.encoding.amplitude_encoding import build_amplitude_circuit
from aeqga.steps.decoding.measurement_decoding import decode_random_subset, decode_elite_subset
from aeqga.steps.evolution.aeqga_algorithm import build_dimension_circuit


def corrected_probabilities(counts, assignment_matrix):
    """Linear least squares, then clip negatives and normalize.

    Columns of the assignment matrix are measured probabilities conditional on
    each prepared computational basis state. This projection is biased; report
    clipped mass and keep raw distributions alongside corrected distributions.
    """
    matrix = np.asarray(assignment_matrix, dtype=float)
    size = len(matrix)
    if (matrix.shape != (size, size) or size < 2 or size & (size-1)
            or not np.all(np.isfinite(matrix)) or np.any(matrix < 0)
            or not np.allclose(matrix.sum(axis=0), 1, atol=1e-8)
            or np.linalg.matrix_rank(matrix) != size):
        raise ValueError("Assignment matrix must be finite, full rank and column-stochastic")
    width = int(np.log2(size))
    from aeqga.execution.ibm_sampler import _counts
    total = sum(counts.values())
    raw = _counts(counts, "", total, width)
    if total <= 0:
        raise ValueError("Positive calibration/measurement shots required")
    observed = np.array([raw.get(format(i, f"0{width}b"), 0) for i in range(size)]) / total
    quasi = np.linalg.solve(matrix, observed)
    clipped_mass = float(-quasi[quasi < 0].sum())
    probabilities = np.maximum(quasi, 0)
    probabilities /= probabilities.sum()
    return probabilities, dict(policy="linear inversion, clip negative values, renormalize",
                               clipped_mass=clipped_mass,
                               condition_number=float(np.linalg.cond(matrix)))


def calibration_circuits(width):
    circuits = []
    for state in range(2**width):
        qc = QuantumCircuit(width)
        for q in range(width):
            if state & (1 << q):
                qc.x(q)
        qc.measure_all()
        circuits.append(qc)
    return circuits


def assignment_matrix(calibration_counts, width):
    from aeqga.execution.ibm_sampler import _counts
    if len(calibration_counts) != 2**width:
        raise ValueError("One calibration result per prepared basis state required")
    columns = []
    for counts in calibration_counts:
        shots = sum(counts.values())
        if shots <= 0:
            raise ValueError("Empty calibration")
        raw = _counts(counts, "", shots, width)
        columns.append([raw.get(format(i, f"0{width}b"), 0)/shots for i in range(2**width)])
    return np.asarray(columns).T


def benchmark_circuits(backend, *, shots=4096, seed=42):
    """Compare fixed 3/4-qubit circuits, preparations and optimization levels.

    Backend may be a synthetic GenericBackendV2 or an authenticated IBM backend.
    This function only executes Aer, including backend-derived noise. It never
    submits hardware jobs. Gate RNGs are restored after circuit construction.
    """
    import random
    original_rng = random.getstate()
    rng = np.random.default_rng(seed)
    report = dict(backend=backend.name, shots=shots, seed=seed, records=[])
    try:
        for width in (3, 4):
            values = rng.uniform(60, 80, 2**width)
            reference = None
            for preparation in ("initialize", "state_preparation"):
                random.seed(seed + width)
                circuit = build_dimension_circuit(values, 1., 1., shots,
                                                   preparation=preparation)
                probabilities = Statevector.from_instruction(
                    circuit.remove_final_measurements(inplace=False)).probabilities()
                if reference is None:
                    reference = probabilities
                error = float(np.max(np.abs(probabilities-reference)))
                if error > 1e-10:
                    raise ValueError("Preparation changed ideal probabilities")
                for level in (0, 1, 2, 3):
                    compiled = transpile(circuit, backend=backend,
                                         optimization_level=level, seed_transpiler=seed)
                    diagnostics = CompilationDiagnostics([circuit], [compiled], backend.name, level, seed)
                    report["records"].append(dict(width=width, preparation=preparation,
                                                   ideal_probability_error=error,
                                                   compiled_probability_error=verify_mps_probability_equivalence(circuit, compiled),
                                                   **diagnostics.summary()[0]))
                ideal_executor = DryRunExecutor(shots=shots, seed_simulator=seed)
                noisy_executor = DryRunExecutor(shots=shots, seed_simulator=seed,
                                               simulation_backend=backend)
                ideal = ideal_executor.run([circuit])[0]
                noisy = noisy_executor.run([circuit])[0]
                get_probs = lambda counts: np.array([counts.get(format(i, f"0{width}b"), 0)
                                                     for i in range(2**width)])/shots
                noisy_prob = get_probs(noisy)
                # Match the experiment's selected physical region for all basis preparations.
                layout = noisy_executor.diagnostics.summary()[0]["final_layout"]
                calibration_executor = DryRunExecutor(shots=shots, seed_simulator=seed,
                                                     simulation_backend=backend, initial_layout=layout)
                calibration = calibration_executor.run(calibration_circuits(width))
                corrected, policy = corrected_probabilities(noisy, assignment_matrix(calibration, width))
                # Use identical decoding RNG for a paired sensitivity comparison.
                decoded = {}
                for label, counts in (("ideal", ideal), ("noisy", noisy),
                                      ("mitigated", {format(i, f"0{width}b"): p
                                                     for i, p in enumerate(corrected)})):
                    state = np.random.get_state()
                    np.random.seed(seed)
                    decoded[label] = dict(random=decode_random_subset(counts, width, 60., 80., 2**width).tolist(),
                                          elite=decode_elite_subset(counts, width, 70., 74., 2**width).tolist())
                    np.random.set_state(state)
                report.setdefault("noise", []).append(dict(
                    width=width, preparation=preparation, raw_ideal_counts=ideal,
                    physical_readout_layout=layout,
                    raw_noisy_counts=noisy, raw_calibration_counts=calibration,
                    corrected_probabilities=corrected.tolist(), mitigation=policy,
                    ideal_sampling_tvd=float(np.abs(get_probs(ideal)-probabilities).sum()/2),
                    noisy_tvd=float(np.abs(noisy_prob-probabilities).sum()/2),
                    mitigated_tvd=float(np.abs(corrected-probabilities).sum()/2), decoded=decoded))
    finally:
        random.setstate(original_rng)
    return report
