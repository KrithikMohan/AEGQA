"""Post-measurement decoding: paper Eqs. 16–18 and Algorithm 2."""
import numpy as np


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
    # Algorithm 2 line 11 follows BOTH branches, including zero-count elites.
    mask = (x == n_min)
    if mask.any():
        x[mask] = np.random.uniform(n_min, n_max, size=mask.sum())
    return x[:n]
