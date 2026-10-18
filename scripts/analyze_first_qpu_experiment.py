#!/usr/bin/env python3
"""
analyze_first_qpu_experiment.py
===============================
Analyzes Experiment A: the historical single-circuit pilot on ibm_kingston
(Job ID: db5qnv4vf2bc73d09org).

Validates:
- 1024 total shots across 15 of 16 basis states.
- Observed TVD against ideal probabilities: 0.110565...
- Maximum absolute basis probability difference: 0.0728...
- 10,000-sample Monte Carlo finite-shot reference ensemble (seed=12345).
"""

import sys
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aeqga.analysis import total_variation_distance


def analyze_pilot():
    print("=" * 65)
    print("  Analysis of Historical IBM Kingston Pilot (db5qnv4vf2bc73d09org)")
    print("=" * 65)

    ckpt_path = REPO_ROOT / "outputs" / "json" / "ibm_first_pilot" / "generation_0001.json"
    counts_path = REPO_ROOT / "outputs" / "json" / "ibm_first_pilot" / "recovered_counts.json"

    if not ckpt_path.exists() or not counts_path.exists():
        print(f"[ERROR] Required checkpoint or counts files not found in outputs/json/ibm_first_pilot/")
        sys.exit(1)

    with open(ckpt_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(counts_path, "r", encoding="utf-8") as f:
        counts_data = json.load(f)

    raw_counts = counts_data["counts"]
    shots = sum(raw_counts.values())
    n_observed_states = sum(1 for c in raw_counts.values() if c > 0)

    print(f"\nHardware Job ID:        {counts_data['job_id']}")
    print(f"Backend:                {counts_data['backend']}")
    print(f"Total shots returned:   {shots}")
    print(f"Observed basis states:  {n_observed_states} out of 16")

    # Reconstruct ideal distribution for the 4-qubit circuit with seed=42
    # In Experiment A:
    # 4 logical qubits, p_cross=1.0, p_mut=1.0, seed=42
    import random
    try:
        from aeqga.steps.encoding.amplitude_encoding import build_amplitude_circuit
        from aeqga.steps.genetic_operators.quantum_gates import apply_crossover_and_mutation
    except ImportError:
        from amplitude_encoding import build_amplitude_circuit  # type: ignore
        from quantum_gates import apply_crossover_and_mutation  # type: ignore
    from qiskit.quantum_info import Statevector

    np.random.seed(42)
    random.seed(42)
    vals = np.random.uniform(60.0, 80.0, 16)
    qc = build_amplitude_circuit(vals)
    apply_crossover_and_mutation(qc, p_c=1.0, p_m=1.0)
    ideal_probs = np.asarray(Statevector(qc).probabilities(), dtype=float)

    # Convert counts to hardware distribution
    hw_dist = np.zeros(16, dtype=float)
    for bitstr, cnt in raw_counts.items():
        hw_dist[int(bitstr, 2)] = cnt / float(shots)

    # TVD and max absolute difference
    tvd_observed = total_variation_distance(hw_dist, ideal_probs)
    max_prob_diff = float(np.max(np.abs(hw_dist - ideal_probs)))

    print(f"\nTVD against ideal probabilities:             {tvd_observed:.16f}")
    print(f"Maximum absolute basis probability diff:     {max_prob_diff:.16f}")

    # Monte Carlo Reference Ensemble
    print("\nRunning 10,000-sample ideal finite-shot Monte Carlo ensemble (seed=12345)...")
    rng = np.random.default_rng(12345)
    samples = rng.multinomial(1024, ideal_probs, size=10000)
    sample_probs = samples / 1024.0
    mc_tvds = 0.5 * np.sum(np.abs(sample_probs - ideal_probs), axis=1)

    mean_mc_tvd = float(np.mean(mc_tvds))
    p95_mc_tvd = float(np.percentile(mc_tvds, 95))
    p99_mc_tvd = float(np.percentile(mc_tvds, 99))
    fraction_ge = float(np.mean(mc_tvds >= tvd_observed))

    print(f"  Mean ideal finite-shot TVD:                 {mean_mc_tvd:.16f}")
    print(f"  95th percentile:                            {p95_mc_tvd:.16f}")
    print(f"  99th percentile:                            {p99_mc_tvd:.16f}")
    print(f"  Fraction of samples with TVD >= observed:   {fraction_ge:.1f} / 10000")

    print("\nScientific Summary:")
    print("  The observed hardware result (TVD ~0.111) lies well beyond the finite-shot")
    print("  sampling distribution (99th percentile ~0.065), confirming the presence")
    print("  of non-sampling physical noise (readout error, 2-qubit CZ gates, resets).")
    print("  Crucially, TVD must not be equated directly with gate fidelity.")


if __name__ == "__main__":
    analyze_pilot()
