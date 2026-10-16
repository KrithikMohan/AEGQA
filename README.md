# AEQGA — Amplitude-Encoded Quantum Genetic Algorithm
### Applied to Pantheon SNe Ia Cosmological Parameter Estimation

Implementation of the AEQGA described in:
> Sarracino et al., *"A Quantum Genetic Algorithm with application to
> Cosmological Parameters Estimation"*, arXiv:2602.15459 (2026)

Adapted from the Quasar-UniNA HQGA repository:
https://github.com/Quasar-UniNA/HQGA

## Architecture

```
amplitude_encoding.py   — Amplitude encoding via qiskit.initialize()
                          (§3.1, Eq.11) + basis-state count decoding (Eq.16/17-18)
quantum_gates.py        — CRy(π/2) crossover (Eq.12-14) + Rx(π/2) mutation (Eq.15)
aeqga_algorithm.py      — Main AEQGA loop (Alg.1)
pantheon_problem.py     — SNe Ia χ² likelihood + BAO/CMB stubs (§2)
run_pantheon_aeqga.py   — End-to-end runner
test_pantheon_aeqga.py  — Integration + quantum-correctness tests
code-summary.md         — Full diagnosis and replication plan
```

## Key Points

- **Amplitude encoding** uses `qiskit.initialize()` on normalized vectors (not RY angle encoding).
- **Population size must be a power of two** (`n_p = 2^k`), required for `log₂(n_p)` qubit allocation.
- **Dual-circuit structure**: 25% elite (bypass), 25% duplicate (circuit A), 50% random (circuit B).
- **Search ranges** per paper §3: `Ω_M ∈ [0.0, 0.5]`, `H0 ∈ [60, 80]`.
- **Optimal gate probabilities**: `p_cross = p_mut = 0.5` (paper §4.1).

## Requirements

```
pip install qiskit qiskit-aer tqdm numpy matplotlib
```

## Usage

```bash
python run_pantheon_aeqga.py --data sn_data/Pantheon
python run_pantheon_aeqga.py --fast
python test_pantheon_aeqga.py --data sn_data/Pantheon
```

## Note

This implementation covers the **SNe Ia** component only (paper §2.1). BAO (§2.2) and CMB (§2.3) paths are stubbed for future extension. See `code-summary.md` for details.
