"""
quantum_gates.py
================
Quantum gate operations for the AEQGA described in
Sarracino et al., arXiv:2602.15459 (2026), §3.2 Eq.12-15.

Crossover:  CRy(π/2) bidirectionally  → U_cross (Eq.14)
Mutation:   Rx(π/2) on one random qubit → U_mut (Eq.15)
"""

import random
import numpy as np
from qiskit import QuantumCircuit


def apply_crossover(qc: QuantumCircuit, p_c: float) -> None:
    """Apply quantum crossover with probability p_c.

    Paper §3.2 Eq.12-14: CRy(π/2) bidirectionally on two randomly picked
    qubits.  The total unitary U_cross = CRy_{0→1}(π/2)·CRy_{1→0}(π/2).
    """
    if random.random() >= p_c:
        return
    n = qc.num_qubits
    if n < 2:
        return
    q0, q1 = random.sample(range(n), 2)
    qc.cry(np.pi / 2, q0, q1)
    qc.cry(np.pi / 2, q1, q0)


def apply_mutation(qc: QuantumCircuit, p_m: float) -> None:
    """Apply quantum mutation with probability p_m.

    Paper §3.2 Eq.15: single-qubit Rx(π/2) on one randomly picked qubit.
    """
    if random.random() >= p_m:
        return
    n = qc.num_qubits
    if n < 1:
        return
    q = random.randrange(n)
    qc.rx(np.pi / 2, q)


def apply_crossover_and_mutation(qc: QuantumCircuit, p_c: float, p_m: float) -> None:
    """Apply crossover then mutation to the circuit."""
    apply_crossover(qc, p_c)
    apply_mutation(qc, p_m)


def verify_u_cross():
    """Verify that the CRy(π/2) bidirectional product matches Eq.14.

    Note: Qiskit uses little-endian bit ordering (qubit 0 = rightmost bit).
    The paper's Eq.14 uses big-endian convention. Use reverse_bits()
    to align the matrix representations.
    """
    from qiskit.quantum_info import Operator
    qc = QuantumCircuit(2)
    qc.cry(np.pi / 2, 0, 1)
    qc.cry(np.pi / 2, 1, 0)
    # Reverse bit ordering to match paper's big-endian convention
    mat = Operator(qc).reverse_qargs().data
    # Eq.14 matrix (big-endian: |00⟩, |01⟩, |10⟩, |11⟩):
    U_cross = np.array([
        [1, 0, 0, 0],
        [0, 1/np.sqrt(2), -1/2, -1/2],
        [0, 0, 1/np.sqrt(2), -1/np.sqrt(2)],
        [0, 1/np.sqrt(2), 1/2, 1/2],
    ], dtype=complex)
    return mat, U_cross


def verify_rx_pi2():
    """Verify Rx(π/2) matches Eq.15."""
    from qiskit.quantum_info import Operator
    qc = QuantumCircuit(1)
    qc.rx(np.pi / 2, 0)
    return Operator(qc).data
