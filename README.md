# AEQGA — Amplitude-Encoded Quantum Genetic Algorithm
### Applied to Pantheon SNe Ia Cosmological Parameter Estimation

Implementation of the AEQGA described in:
> Sarracino et al., *"A Quantum Genetic Algorithm with application to
> Cosmological Parameters Estimation"*, arXiv:2602.15459 (2026)

Adapted from the Quasar-UniNA HQGA repository:
https://github.com/Quasar-UniNA/HQGA

## Architecture

```
aeqga_algorithm.py                      — Main AEQGA loop (Alg.1) supporting simulation & QPU injection
amplitude_encoding.py                   — Amplitude encoding via qiskit.initialize() (§3.1, Eq.11)
                                          + basis-state count decoding (Eq.16/17-18)
quantum_gates.py                        — CRy(π/2) crossover (Eq.12-14) + Rx(π/2) mutation (Eq.15)
pantheon_problem.py                     — SNe Ia χ² likelihood + Pantheon+ support + BAO/CMB stubs (§2)
run_pantheon_aeqga.py                   — End-to-end simulation runner
test_pantheon_aeqga.py                  — Paper integration + quantum-correctness tests (T1-T4)
code-summary.md                         — Full diagnosis, replication plan, and hardware documentation

aeqga/
  execution/ibm_sampler.py              — IBMSamplerExecutor, DryRunExecutor, CheckpointManager
  steps/                                — Modular pipeline (encoding, operators, decoding, evolution)
  likelihoods/pantheon_problem.py       — PantheonPlusProblem offline calibrated likelihood
  analysis/circuit_comparison.py        — Controlled same-circuit MC ensemble & QPU comparison

scripts/
  test_hardware_compilation.py          — 156-qubit Heron transpilation & Aer MPS numerical check
  test_full_generation.py               — Full offline 1-generation execution test
  compare_generation_populations.py     — Finite-shot MC ensemble vs hardware comparison
  recover_ibm_job.py                    — Idempotent job recovery & count parsing
  ibm_preflight.py                      — Offline-safe credentials & backend inspector
  run_ibm_generation.py                 — Guarded generation runner (defaults to dry-run)

tests/test_ibm_adapter.py               — 13 automated unit & integration tests
docs/IBM_HARDWARE.md                    — Comprehensive IBM Quantum hardware integration guide
```

## Key Points

- **Amplitude encoding** uses `qiskit.initialize()` on normalized vectors (not RY angle encoding).
- **Population size must be a power of two** (`n_p = 2^k`), required for `log₂(n_p)` qubit allocation.
- **Dual-circuit structure**: 25% elite (bypass), 25% duplicate (circuit A), 50% random (circuit B).
- **Search ranges** per paper §3: `Ω_M ∈ [0.0, 0.5]`, `H0 ∈ [60, 80]`.
- **Optimal gate probabilities**: `p_cross = p_mut = 0.5` (paper §4.1).
- **Hybrid CPU/QPU execution**: Classical CPU performs population management, likelihood evaluations, elitism, and decoding; IBM QPU executes amplitude state preparation, crossover, mutation, and measurement.
- **Four-circuit batching**: Each generation batches 4 circuits in width order `[4, 3, 4, 3]` into a single Sampler call. Generations remain strictly sequential.
- **156-qubit device compatibility**: Classical register validation compares against logical circuit width (3 or 4 bits), ensuring compatibility with 156-qubit Heron processors.
- **Zero-hardware-submission safety**: All tests, preflight scripts, and runners default to offline simulation (`DryRunExecutor`). Physical submission requires explicit confirmation flags.

## Requirements

Core simulation:
```bash
pip install -r requirements.txt
```

IBM Quantum hardware execution / Runtime:
```bash
pip install -r requirements-ibm.txt
```

## Usage

### Simulation & Paper Tests
```bash
python run_pantheon_aeqga.py --data sn_data/Pantheon
python run_pantheon_aeqga.py --fast
python test_pantheon_aeqga.py --data sn_data/Pantheon
```

### IBM Hardware Integration & Offline Tests
```bash
# Run full automated test suite (13 tests)
python -m unittest discover -s tests -p "test_*.py" -v

# Test 156-qubit compilation diagnostics and Aer MPS equivalence
python scripts/test_hardware_compilation.py

# Test full offline generation execution
python scripts/test_full_generation.py

# Run controlled same-circuit statistical comparison (MC ensemble vs QPU)
python scripts/compare_generation_populations.py
```

## Note

This implementation covers the **SNe Ia** component (paper §2.1) with both Pantheon and Pantheon+ likelihoods. BAO (§2.2) and CMB (§2.3) paths are stubbed for future extension. See `docs/IBM_HARDWARE.md` and `code-summary.md` for complete technical details.
