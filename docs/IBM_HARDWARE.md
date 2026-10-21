# IBM Quantum Hardware Integration for AEQGA

This documentation describes the IBM Quantum Hardware integration for the Amplitude-Encoded Quantum Genetic Algorithm (AEQGA) applied to cosmological parameter estimation ($H_0$ and $\Omega_m$), based on:

> Sarracino et al., *A Quantum Genetic Algorithm with application to Cosmological Parameters Estimation*, arXiv:2602.15459 (2026).

---

## 1. Architectural Overview

AEQGA is a hybrid classical-quantum optimizer. The division of responsibilities is:

### Classical CPU
- Population initialization and boundary constraints
- Cosmological likelihood and $\chi^2$ evaluation ($SNe$ Ia Pantheon/Pantheon+)
- Fitness calculation and ranking
- Elite selection (top 25% elites preserved directly)
- Measurement-result decoding (random subset Eq.16, elite-copy subset Eq.17–18)
- Generation history, convergence logging, and atomic checkpoint persistence

### IBM Quantum Hardware
- Amplitude-encoded quantum state preparation via `initialize()`
- Quantum crossover operations via bidirectional $CR_y(\pi/2)$ gates
- Quantum mutation operations via single-qubit $R_x(\pi/2)$ gates
- Full-register measurement into classical bitstrings

```
                    ┌────────────────────────────────────────────────────────┐
                    │                   Classical CPU Host                   │
                    │                                                        │
                    │  1. Evaluate fitness χ²(x) on SNe Ia likelihood        │
                    │  2. Preserve top 25% elites (P_elite)                  │
                    │  3. Duplicate elites (P_elite_copy, 25%)               │
                    │  4. Draw fresh uniform individuals (P_rand, 50%)       │
                    └─────────────────────────┬──────────────────────────────┘
                                              │
                      Construct 4 Circuits    │ Width order: [4, 3, 4, 3]
                                              ▼
                    ┌────────────────────────────────────────────────────────┐
                    │             IBM Quantum Runtime (SamplerV2)            │
                    │                                                        │
                    │  Circuit 0: H0 random subset (4 logical qubits)        │
                    │  Circuit 1: H0 elite-copy (3 logical qubits)           │
                    │  Circuit 2: Ω_m random subset (4 logical qubits)       │
                    │  Circuit 3: Ω_m elite-copy (3 logical qubits)          │
                    │                                                        │
                    │  - Transpiled for 156-qubit processor (e.g. Kingston)  │
                    │  - Executed in a single batch (1024 or 4096 shots)     │
                    └─────────────────────────┬──────────────────────────────┘
                                              │
                      Raw Bitstring Counts    │ [counts_0, counts_1, ...]
                                              ▼
                    ┌────────────────────────────────────────────────────────┐
                    │                   Classical CPU Host                   │
                    │                                                        │
                    │  5. Decode basis counts (Eq. 16, 17-18)                │
                    │  6. Assemble next generation: P ← P_elite ∪ P_decoded  │
                    │  7. Record atomic checkpoint and update global best    │
                    └────────────────────────────────────────────────────────┘
```

---

## 2. Injected Execution Interface

The core evolution algorithm exposes a pluggable `hardware_executor` keyword argument in `run_aeqga_dual`:

```python
from aeqga.steps.evolution.aeqga_algorithm import AEQGAParameters, run_aeqga_dual
from aeqga.execution.ibm_sampler import IBMSamplerExecutor, DryRunExecutor
from aeqga.likelihoods.pantheon_problem import PantheonPlusProblem

problem = PantheonPlusProblem()
params = AEQGAParameters(pop_size=32, max_gen=1, num_shots=1024)

# Production IBM Hardware execution (requires account credentials)
executor = IBMSamplerExecutor(
    backend_name="ibm_kingston",
    shots=1024,
    optimization_level=1,
    seed_transpiler=42,
    checkpoint_dir="outputs/json/ibm_generation_01",
)

g_best, pop_evol, bests_log = run_aeqga_dual(
    problem,
    params,
    hardware_executor=executor,
    seed=42,
)
```

The interface preserves existing callers:
- `execution_mode="aer"` (default): uses local `AerSimulator()`.
- `execution_mode="statevector"`: uses exact quantum state probabilities without shot noise.
- `execution_mode="ibm"` or passing `hardware_executor`: executes via the injected hardware adapter.

---

## 3. Circuit Batching & Sequential Generations

For a standard population size $n_p = 32$ over 2 cosmological parameters ($H_0$ and $\Omega_m$):
- $n_q(\text{random}) = \log_2(n_p) - 1 = 4$ logical qubits
- $n_q(\text{elite}) = \log_2(n_p) - 2 = 3$ logical qubits

The 4 circuits are submitted as **one single IBM Sampler batch** per generation in the validated width order:
```python
[4, 3, 4, 3]
```

**Generations are strictly sequential**: Generation $t+1$ depends on the classical fitness evaluation, sorting, and splitting of the population decoded from Generation $t$. Dependent generations are never batched together.

---

## 4. Logical vs. Physical Measurement Register Width

When compiling 3-qubit or 4-qubit logical circuits for a 156-qubit physical processor (e.g., Heron `ibm_kingston` or `ibm_fez`), the transpiled circuit has:
```python
compiled_circuit.num_qubits == 156
```
However, the measurement register only contains bits for the measured logical qubits (size 3 or 4).

The adapter explicitly validates classical register size against the **original logical circuit width**:
```python
for original, circuit, pub in zip(circuits, compiled, result):
    registers = list(circuit.cregs)
    logical_width = original.num_qubits

    if len(registers) != 1 or registers[0].size != logical_width:
        raise ValueError("Exactly one full-width measurement register required")

    counts.append(_counts(pub, registers[0].name, self.shots, logical_width))
```

This prevents the `ValueError: Exactly one full-width measurement register required` error that previously occurred when comparing register width to physical device width.

---

## 5. Checkpointing & Job Recovery

Per-generation checkpoints are saved to:
```text
outputs/json/ibm_generation_01/generation_0001.json
```

### Safety & Idempotency Rules
1. **Atomic writes**: Checkpoints are written to a temporary file (`.tmp`), flushed, fsynced, and renamed via `os.replace` to prevent corrupted partial files.
2. **Pre-execution checkpoint**: When submitting a hardware job, the job ID and state `"submitted"` are persisted *before* awaiting results.
3. **Recovery without duplication**: If a checkpoint contains a valid `job_id`, the adapter retrieves the existing job rather than submitting a duplicate.
4. **Fingerprint protection**: Before reusing a checkpoint, the adapter verifies backend name, shot count, number of circuits, and circuit SHA-256 fingerprints. Any configuration mismatch immediately raises `ValueError` rather than reusing incorrect results.

---

## 6. Hardware Compilation & Aer MPS Equivalence

On physical processors, amplitude encoding via `initialize()` decomposes into single-qubit rotations, resets, and two-qubit entangling gates ($CZ$ / $CX$).

Compilation diagnostics record:
- Original vs. transpiled width and depth
- Layout mapping from logical to physical qubits
- Gate counts ($SX$, $RZ$, $CZ$/$CX$, $Reset$, $Measure$)
- Two-qubit gate counts

### Numerical Equivalence Check
To verify transpiler correctness without running out of host memory:
- **Do NOT allocate full 156-qubit statevectors** (fails with `ValueError: Maximum allowed dimension exceeded`).
- Use **Aer Matrix Product State (MPS)** simulation targeting only the active physical qubits:
  ```python
  from aeqga.execution.ibm_sampler import verify_mps_probability_equivalence
  max_diff = verify_mps_probability_equivalence(orig_circuit, transpiled_circuit, atol=1e-10)
  ```
  Expected maximum absolute probability difference: $\approx 10^{-11} - 10^{-16}$.

---

## 7. Controlled Same-Circuit Scientific Comparison

To determine the exact impact of hardware execution versus finite-shot sampling, the pipeline decouples circuit construction from execution:

1. **Deterministic freeze**: Generate Generation 1 circuits with fixed seed (42).
2. **Ideal probabilities**: Calculate exact basis-state probability distribution $P_{\text{ideal}}$.
3. **Finite-shot ensemble**: Draw 10,000 independent multinomial samples of 1024 shots each.
4. **Statistical comparison**: Compare hardware counts TVD against the finite-shot empirical TVD distribution.
5. **Distribution metrics**: Evaluate population differences using Wasserstein distance and Kolmogorov-Smirnov tests to avoid artifacts from arbitrary individual indexing.

Run the comparison:
```bash
python scripts/compare_generation_populations.py
```

---

## 8. Historical Experiments

### Experiment A: Single-Circuit Pilot
- **Backend**: `ibm_kingston`
- **Job ID**: `db5qnv4vf2bc73d09org`
- **Configuration**: 4 logical qubits, 1024 shots, $p_{\text{cross}}=1.0$, $p_{\text{mut}}=1.0$, seed=42.
- **Observed TVD**: $0.110565$
- **Finite-shot reference ensemble**: Mean TVD $0.040679$, 99th percentile $0.064987$.
- **Finding**: Hardware noise lies beyond pure finite-shot sampling uncertainty.

### Experiment B: Complete Generation 1
- **Backend**: `ibm_kingston`
- **Job ID**: `db5r25imb58s738at2v0`
- **Configuration**: Pop 32, [4, 3, 4, 3] circuits, 1024 shots per circuit, seed=42.
- **End-of-generation best**: $H_0 = 71.958$, $\Omega_m = 0.461$, $\chi^2 = 1778.76$.
- **Likelihood evaluations**: 64 (32 initial + 32 generation 1).
- **Caveat**: The best individual was retained via classical elitism.

---

## 9. Offline Verification & Tests

Run the full offline test suite without credentials or QPU jobs:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Run compilation diagnostics:
```bash
python scripts/test_hardware_compilation.py
```

Run full generation offline test:
```bash
python scripts/test_full_generation.py
```

## Consolidated notebook and environment

Use `notebooks/AEQGA.ipynb`, Part 13, for offline rehearsal and opt-in physical
execution on the calibrated Pantheon+ objective. The former root notebook has
been removed. Install `requirements.txt` for both simulation and IBM execution;
`requirements-ibm.txt` now delegates to it. Select the `Python (AEQGA venv)`
kernel after registering it with `python -m ipykernel install --sys-prefix
--name aeqga --display-name "Python (AEQGA venv)"`.
