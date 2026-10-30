"""
circuit_comparison.py
=====================
Controlled same-circuit scientific comparison pipeline for AEQGA.

Decouples circuit generation from execution and decoding to enable:
1. Frozen generation and circuit serialization.
2. Calculation of exact ideal measurement probabilities.
3. Controlled finite-shot Monte Carlo ensembles.
4. Evaluation of real hardware counts against ideal finite-shot distributions.
5. Distribution-based metrics (Wasserstein, KS) avoiding arbitrary index ordering artifacts.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

import numpy as np
from scipy import stats
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

try:
    from aeqga.steps.encoding.amplitude_encoding import n_qubits_for_population
    from aeqga.steps.decoding.measurement_decoding import decode_random_subset, decode_elite_subset
    from aeqga.steps.evolution.aeqga_algorithm import build_dimension_circuit, AEQGAParameters
    from aeqga.steps.selection.population import split_population
except ImportError:
    from amplitude_encoding import (  # type: ignore
        decode_random_subset,
        decode_elite_subset,
        n_qubits_for_population,
    )
    from aeqga_algorithm import (  # type: ignore
        build_dimension_circuit,
        split_population,
        AEQGAParameters,
    )
from aeqga.execution.ibm_sampler import compute_circuit_fingerprint


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Total Variation Distance: TVD(p, q) = 0.5 * sum(|p_i - q_i|)."""
    return float(0.5 * np.sum(np.abs(np.asarray(p, dtype=float) - np.asarray(q, dtype=float))))


def compute_circuit_ideal_probabilities(circuits: Sequence[QuantumCircuit]) -> list[np.ndarray]:
    """Compute exact ideal basis-state probabilities for a list of circuits.

    Strips measurement operations and classical bits to extract the pure statevector.
    """
    ideal_probs = []
    for circuit in circuits:
        pure_qc = QuantumCircuit(circuit.num_qubits)
        for instruction in circuit.data:
            op = instruction.operation
            if op.name not in ("measure", "barrier"):
                q_args = [pure_qc.qubits[circuit.find_bit(q).index] for q in instruction.qubits]
                pure_qc.append(op, q_args)
        sv = Statevector(pure_qc)
        probs = np.asarray(sv.probabilities(), dtype=float)
        ideal_probs.append(probs)
    return ideal_probs


def generate_frozen_generation_circuits(
    problem: Any,
    pop_size: int = 32,
    seed: int = 42,
    p_cross: float = 0.5,
    p_mut: float = 0.5,
    shots: int = 1024,
) -> tuple[np.ndarray, np.ndarray, list[QuantumCircuit], dict[str, Any]]:
    """Deterministically generate and freeze the 4 AEQGA circuits for Generation 1.

    Separates circuit construction from execution and decoding.

    Returns
    -------
    population : initial population (pop_size, n_dim)
    fitnesses : initial fitnesses (pop_size,)
    circuits : list of 4 QuantumCircuit [H0_rand, H0_elite, Om_rand, Om_elite]
    metadata : provenance and diagnostic metadata
    """
    np.random.seed(seed)
    random.seed(seed)

    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)
    n_dim = int(problem.n_dim)
    minimise = not (hasattr(problem, "is_max_problem") and problem.is_max_problem())

    # Step 1: Initial population
    population = np.random.uniform(lower, upper, size=(pop_size, n_dim))
    fitnesses = np.array([problem.compute_fitness(population[i]) for i in range(pop_size)])

    # Step 2: Split population
    P_elite, P_elite_copy, P_rand = split_population(
        population, fitnesses, minimise, lower, upper
    )

    # Step 3: Build 4 circuits in strict order [H0_rand, H0_elite, Om_rand, Om_elite]
    circuits = []
    for d in range(n_dim):
        qc_rand = build_dimension_circuit(
            P_rand[:, d],
            p_cross=p_cross,
            p_mut=p_mut,
            num_shots=shots,
            mode="measure",
        )
        qc_elite = build_dimension_circuit(
            P_elite_copy[:, d],
            p_cross=p_cross,
            p_mut=p_mut,
            num_shots=shots,
            mode="measure",
        )
        circuits.append(qc_rand)
        circuits.append(qc_elite)

    circuit_fingerprints = [compute_circuit_fingerprint(c) for c in circuits]
    circuit_widths = [c.num_qubits for c in circuits]

    metadata = {
        "seed": seed,
        "pop_size": pop_size,
        "p_cross": p_cross,
        "p_mut": p_mut,
        "shots": shots,
        "circuit_widths": circuit_widths,
        "circuit_fingerprints": circuit_fingerprints,
        "num_circuits": len(circuits),
        "P_elite": P_elite.copy(),
        "P_rand": P_rand.copy(),
    }

    return population, fitnesses, circuits, metadata


def run_finite_shot_ensemble(
    problem: Any,
    circuits: Sequence[QuantumCircuit],
    P_elite: np.ndarray,
    ideal_probs: Sequence[np.ndarray],
    n_samples: int = 1000,
    shots: int = 1024,
    ensemble_seed: int = 12345,
    elite_margin: float = 0.0,
) -> dict[str, Any]:
    """Execute controlled ideal finite-shot ensemble on fixed circuits.

    Generates independent multinomial samples from ideal probabilities,
    decodes them with AEQGA's standard decoder, and records population
    and fitness metrics.
    """
    rng = np.random.default_rng(ensemble_seed)
    n_p = len(P_elite) * 4  # P_elite is 25% of pop_size
    n_elite = len(P_elite)
    n_random = n_p // 2
    n_dim = int(problem.n_dim)
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)
    minimise = not (hasattr(problem, "is_max_problem") and problem.is_max_problem())

    n_q_random = n_qubits_for_population(n_p) - 1
    n_q_elite = n_qubits_for_population(n_p) - 2

    tvds_per_circuit = [[] for _ in range(len(circuits))]
    best_fitnesses = []
    best_h0s = []
    best_oms = []
    final_pops = []

    for _ in range(n_samples):
        # Draw finite-shot counts for all 4 circuits
        counts_list = []
        for c_idx, probs in enumerate(ideal_probs):
            sampled_counts_arr = rng.multinomial(shots, probs)
            sample_prob_dist = sampled_counts_arr / float(shots)
            tvd = total_variation_distance(sample_prob_dist, probs)
            tvds_per_circuit[c_idx].append(tvd)

            # Convert to dictionary format
            n_q = circuits[c_idx].num_qubits
            counts_dict = {}
            for i, cnt in enumerate(sampled_counts_arr):
                if cnt > 0:
                    bitstr = format(i, f"0{n_q}b")
                    counts_dict[bitstr] = int(cnt)
            counts_list.append(counts_dict)

        # Decode counts with existing AEQGA decoder
        decoded_random = np.empty((n_random, n_dim))
        decoded_elite = np.empty((n_elite, n_dim))

        for d in range(n_dim):
            c_rand = counts_list[2 * d]
            c_elite = counts_list[2 * d + 1]

            decoded_random[:, d] = decode_random_subset(
                c_rand, n_q_random, float(lower[d]), float(upper[d]), n_random
            )

            n_min_e = float(np.min(P_elite[:, d]))
            n_max_e = float(np.max(P_elite[:, d]))
            margin = elite_margin * (n_max_e - n_min_e)
            n_min_e = max(float(lower[d]), n_min_e-margin)
            n_max_e = min(float(upper[d]), n_max_e+margin)
            decoded_elite[:, d] = decode_elite_subset(
                c_elite, n_q_elite, n_min_e, n_max_e, n_elite
            )

        new_pop = np.vstack([P_elite, decoded_random, decoded_elite])
        fits = np.array([problem.compute_fitness(new_pop[i]) for i in range(n_p)])

        best_idx = int(np.argmin(fits) if minimise else np.argmax(fits))
        best_fitnesses.append(float(fits[best_idx]))
        best_h0s.append(float(new_pop[best_idx, 0]))
        best_oms.append(float(new_pop[best_idx, 1]))
        final_pops.append(new_pop)

    tvds_arr = np.array(tvds_per_circuit)  # shape (4, n_samples)
    final_pops_arr = np.array(final_pops)  # shape (n_samples, n_p, n_dim)

    return {
        "n_samples": n_samples,
        "shots": shots,
        "ensemble_seed": ensemble_seed,
        "tvds_per_circuit": tvds_arr,
        "tvd_means": np.mean(tvds_arr, axis=1),
        "tvd_95ths": np.percentile(tvds_arr, 95, axis=1),
        "tvd_99ths": np.percentile(tvds_arr, 99, axis=1),
        "best_fitnesses": np.array(best_fitnesses),
        "best_h0s": np.array(best_h0s),
        "best_oms": np.array(best_oms),
        "final_populations": final_pops_arr,
        "population_mean_across_ensemble": np.mean(final_pops_arr, axis=0),
    }


def compare_counts_against_ensemble(
    problem: Any,
    hardware_counts: list[dict[str, int]],
    ideal_probs: Sequence[np.ndarray],
    P_elite: np.ndarray,
    ensemble_results: dict[str, Any],
    initial_pop: np.ndarray,
    simulator_final_pop: np.ndarray | None = None,
    elite_margin: float = 0.0,
) -> dict[str, Any]:
    """Compare real hardware counts against the controlled ideal finite-shot ensemble.

    Calculates:
    - TVD against ideal distribution for each circuit
    - Percentile position of hardware TVD in finite-shot ensemble
    - Index-wise population differences (mean, max, RMS)
    - Distributional differences (Wasserstein distance, KS test)
    - Best solution and elite retention
    """
    n_p = len(P_elite) * 4
    n_elite = len(P_elite)
    n_random = n_p // 2
    n_dim = int(problem.n_dim)
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)
    minimise = not (hasattr(problem, "is_max_problem") and problem.is_max_problem())

    n_q_random = n_qubits_for_population(n_p) - 1
    n_q_elite = n_qubits_for_population(n_p) - 2

    # Step 1: Compute TVD for hardware counts
    hw_tvds = []
    for c_idx, counts in enumerate(hardware_counts):
        probs = ideal_probs[c_idx]
        n_states = len(probs)
        hw_dist = np.zeros(n_states)
        shots = sum(counts.values())
        for bitstr, cnt in counts.items():
            idx = int(bitstr, 2)
            if idx < n_states:
                hw_dist[idx] = cnt / float(shots)
        hw_tvd = total_variation_distance(hw_dist, probs)
        hw_tvds.append(hw_tvd)

    # Step 2: Compare TVD against ensemble distribution
    tvd_percentiles = []
    for c_idx, hw_tvd in enumerate(hw_tvds):
        ensemble_tvds = ensemble_results["tvds_per_circuit"][c_idx]
        pct = float(np.mean(ensemble_tvds >= hw_tvd))
        tvd_percentiles.append(pct)

    # Step 3: Decode hardware counts to produce hardware final population
    decoded_random = np.empty((n_random, n_dim))
    decoded_elite = np.empty((n_elite, n_dim))

    for d in range(n_dim):
        c_rand = hardware_counts[2 * d]
        c_elite = hardware_counts[2 * d + 1]

        decoded_random[:, d] = decode_random_subset(
            c_rand, n_q_random, float(lower[d]), float(upper[d]), n_random
        )

        n_min_e = float(np.min(P_elite[:, d]))
        n_max_e = float(np.max(P_elite[:, d]))
        margin = elite_margin * (n_max_e - n_min_e)
        n_min_e = max(float(lower[d]), n_min_e-margin)
        n_max_e = min(float(upper[d]), n_max_e+margin)
        decoded_elite[:, d] = decode_elite_subset(
            c_elite, n_q_elite, n_min_e, n_max_e, n_elite
        )

    hw_final_pop = np.vstack([P_elite, decoded_random, decoded_elite])
    hw_fitnesses = np.array([problem.compute_fitness(hw_final_pop[i]) for i in range(n_p)])

    hw_best_idx = int(np.argmin(hw_fitnesses) if minimise else np.argmax(hw_fitnesses))
    hw_best_fitness = float(hw_fitnesses[hw_best_idx])
    hw_best_x = hw_final_pop[hw_best_idx].copy()
    hw_elite_retained = bool(hw_best_idx < n_elite)

    results: dict[str, Any] = {
        "hardware_tvds": hw_tvds,
        "fraction_ensemble_tvd_ge_hardware": tvd_percentiles,
        "hardware_best": {
            "H0": float(hw_best_x[0]),
            "Omega_m": float(hw_best_x[1]),
            "chi2": float(hw_best_fitness),
            "elite_retained": hw_elite_retained,
        },
        "hardware_final_population": hw_final_pop,
    }

    # Step 4: Distributional & index-wise population comparisons
    if simulator_final_pop is not None:
        diff_h0 = np.abs(hw_final_pop[:, 0] - simulator_final_pop[:, 0])
        diff_om = np.abs(hw_final_pop[:, 1] - simulator_final_pop[:, 1])

        # Wasserstein distance (1D Earth Mover's Distance)
        w_h0 = float(stats.wasserstein_distance(hw_final_pop[:, 0], simulator_final_pop[:, 0]))
        w_om = float(stats.wasserstein_distance(hw_final_pop[:, 1], simulator_final_pop[:, 1]))

        # Two-sample Kolmogorov-Smirnov test
        ks_h0 = stats.ks_2samp(hw_final_pop[:, 0], simulator_final_pop[:, 0])
        ks_om = stats.ks_2samp(hw_final_pop[:, 1], simulator_final_pop[:, 1])

        results["index_wise_metrics"] = {
            "H0": {
                "mean_abs_diff": float(np.mean(diff_h0)),
                "max_abs_diff": float(np.max(diff_h0)),
                "rms_diff": float(np.sqrt(np.mean(diff_h0**2))),
            },
            "Omega_m": {
                "mean_abs_diff": float(np.mean(diff_om)),
                "max_abs_diff": float(np.max(diff_om)),
                "rms_diff": float(np.sqrt(np.mean(diff_om**2))),
            },
        }
        results["distributional_metrics"] = {
            "H0": {
                "wasserstein_distance": w_h0,
                "ks_statistic": float(ks_h0.statistic),
                "ks_pvalue": float(ks_h0.pvalue),
            },
            "Omega_m": {
                "wasserstein_distance": w_om,
                "ks_statistic": float(ks_om.statistic),
                "ks_pvalue": float(ks_om.pvalue),
            },
        }

    return results
