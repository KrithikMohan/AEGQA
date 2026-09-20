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


def get_basis_counts(counts: dict, n_qubits: int) -> np.ndarray:
    """Extract basis-state counts from a Qiskit counts dict.

    Qiskit uses little-endian convention (rightmost bit = qubit 0).
    Returns array c of length 2^n_qubits indexed by int(bitstring, 2).
    """
    c = np.zeros(2 ** n_qubits, dtype=float)
    for bitstring, shots in counts.items():
        idx = int(bitstring, 2)
        if idx < len(c):
            c[idx] += shots
    return c


def decode_random_subset(
    counts: dict,
    n_qubits: int,
    a: float,
    b: float,
    n: int,
) -> np.ndarray:
    """Decode counts for the random subset per Eq.16 (§3.3).

    x_i = a + (b-a)·(c_i - c_min)/(c_max - c_min)

    Post-process (Alg.2 L11): if x_i == a exactly, replace with U(a,b) draw.
    """
    c = get_basis_counts(counts, n_qubits)
    c_min, c_max = c.min(), c.max()
    if c_max == c_min:
        return np.random.uniform(a, b, size=n)
    x = a + (b - a) * (c - c_min) / (c_max - c_min)
    # Replace exact lower-bound values with random draw
    mask = (x == a)
    if mask.any():
        x[mask] = np.random.uniform(a, b, size=mask.sum())
    return x[:n]


def decode_elite_subset(
    counts: dict,
    n_qubits: int,
    n_min: float,
    n_max: float,
    n: int,
) -> np.ndarray:
    """Decode counts for the elite-copy subset per Eq.17-18 (§3.3).

    p_i = c_i / Σ c_j
    x_i = n_min + (n_max - n_min)·√p_i
    """
    c = get_basis_counts(counts, n_qubits)
    total = c.sum()
    if total == 0:
        return np.random.uniform(n_min, n_max, size=n)
    p = c / total
    x = n_min + (n_max - n_min) * np.sqrt(p)
    return x[:n]
