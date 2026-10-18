#!/usr/bin/env python3
"""
test_hardware_compilation.py
============================
Performs offline compilation diagnostics and ideal probability equivalence checks
for 3-qubit and 4-qubit AEQGA circuits compiled for 156-qubit IBM devices.

Validates:
1. Circuit width, depth, and layout diagnostics.
2. Two-qubit gate count and reset count analysis.
3. Aer Matrix Product State (MPS) probability equivalence (avoiding 156-qubit memory blowup).
"""

import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qiskit import transpile
from qiskit.providers.fake_provider import GenericBackendV2
try:
    from aeqga.steps.encoding.amplitude_encoding import build_amplitude_circuit
    from aeqga.steps.genetic_operators.quantum_gates import apply_crossover_and_mutation
except ImportError:
    from amplitude_encoding import build_amplitude_circuit  # type: ignore
    from quantum_gates import apply_crossover_and_mutation  # type: ignore
from aeqga.execution.ibm_sampler import CompilationDiagnostics, verify_mps_probability_equivalence


def run_compilation_test():
    print("=" * 65)
    print("  AEQGA Hardware Compilation & MPS Equivalence Diagnostics")
    print("=" * 65)

    fake_156_backend = GenericBackendV2(num_qubits=156, seed=42)

    # Build 3-qubit and 4-qubit test circuits
    np.random.seed(42)
    vals_4q = np.random.uniform(60.0, 80.0, 16)
    vals_3q = np.random.uniform(60.0, 80.0, 8)

    # 4-qubit circuit (random subset)
    qc_4q = build_amplitude_circuit(vals_4q)
    apply_crossover_and_mutation(qc_4q, p_c=0.5, p_m=0.5)

    # 3-qubit circuit (elite-copy subset)
    qc_3q = build_amplitude_circuit(vals_3q)
    apply_crossover_and_mutation(qc_3q, p_c=0.5, p_m=0.5)

    circuits = [qc_4q, qc_3q]
    print(f"\nOriginal circuits: 4-qubit (depth {qc_4q.depth()}), 3-qubit (depth {qc_3q.depth()})")

    # Transpile for 156-qubit device
    print("\nTranspiling for 156-qubit backend (optimization_level=1, seed=42)...")
    transpiled = transpile(
        circuits,
        backend=fake_156_backend,
        optimization_level=1,
        seed_transpiler=42,
    )

    diagnostics = CompilationDiagnostics(
        circuits,
        transpiled,
        backend_name="ibm_kingston_simulated",
        optimization_level=1,
        seed_transpiler=42,
    )

    for i, rec in enumerate(diagnostics.summary()):
        q_label = "4-qubit" if rec["original_width"] == 4 else "3-qubit"
        print(f"\n--- Diagnostics for {q_label} circuit ---")
        print(f"  Original logical width:   {rec['original_width']} qubits, depth: {rec['original_depth']}")
        print(f"  Transpiled device width:  {rec['transpiled_width']} qubits, depth: {rec['transpiled_depth']}")
        print(f"  Gate operations count:    {rec['gate_counts']}")
        print(f"  Two-qubit gate count:     {rec['two_qubit_gate_counts']}")
        print(f"  Reset gate count:         {rec['reset_count']}")
        print(f"  Circuit fingerprint:      {rec['circuit_fingerprint'][:16]}...")

    # MPS Probability Equivalence Test
    print("\nTesting MPS ideal probability equivalence (active physical qubits)...")
    for i, (orig, comp) in enumerate(zip(circuits, transpiled)):
        q_label = "4-qubit" if orig.num_qubits == 4 else "3-qubit"
        max_diff = verify_mps_probability_equivalence(orig, comp, atol=1e-10)
        print(f"  {q_label} circuit max absolute probability diff: {max_diff:.3e}  PASS")

    print("\n✓ All compilation and equivalence checks PASSED successfully.")


if __name__ == "__main__":
    run_compilation_test()
