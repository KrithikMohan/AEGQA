"""
amplitude_encoding.py
======================
Amplitude-encoding helpers for the AEQGA described in
Sarracino et al., arXiv:2602.15459 (2026), §3.1 Eq.11.

Given a classical array x = [x₀, …, x_{N-1}], normalise ‖x‖₂=1 and map to
|ψ⟩ = Σ xᵢ |i⟩ on n = log₂ N qubits via qiskit.initialize().
"""

import numpy as np
from qiskit import QuantumCircuit


def _normalise(x: np.ndarray) -> np.ndarray:
    """L2-normalise a real vector; handle zero-norm edge case."""
    nrm = float(np.linalg.norm(x))
    if nrm == 0.0:
        return np.full_like(x, 1.0 / np.sqrt(len(x)))
    return x / nrm


def n_qubits_for_population(n_p: int) -> int:
    """Number of qubits needed for amplitude encoding of n_p individuals.

    Paper §3.1 p.9: n = log₂(n_p) must be integer (power-of-two constraint).
    """
    n = int(np.log2(n_p))
    assert 2 ** n == n_p, (
        f"pop_size {n_p} is not a power of two; "
        f"amplitude encoding requires n_qubits = log2(n_p) integer"
    )
    return n


def build_amplitude_circuit(values: np.ndarray) -> QuantumCircuit:
    """Build a quantum circuit encoding `values` via amplitude encoding.

    Parameters
    ----------
    values : shape (n_p,) — 1-D real array, n_p must be power of two.

    Returns
    -------
    QuantumCircuit with n_qubits = log2(len(values)) and no classical register.
    """
    n_p = len(values)
    n_qubits = n_qubits_for_population(n_p)
    norm = _normalise(values)
    qc = QuantumCircuit(n_qubits)
    qc.initialize(norm, range(n_qubits))
    return qc


def build_amplitude_circuit_with_measure(values: np.ndarray, shots: int) -> QuantumCircuit:
    """Same as build_amplitude_circuit but appends measurement register."""
    qc = build_amplitude_circuit(values)
    qc.measure_all()
    return qc
