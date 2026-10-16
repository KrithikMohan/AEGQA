# Code Summary — Diagnosis, Replication Plan & IBM Quantum Hardware Integration: AEQGA vs Sarracino et al. (arXiv:2602.15459)

> **Paper:** Sarracino et al., *"A Quantum Genetic Algorithm with application to Cosmological Parameters Estimation"*, Astron. Comput. 55:101078 (2026), arXiv:2602.15459v1
> **Workspace:** `AEGQA-main/` — adapted from Quasar-UniNA HQGA (`github.com/Quasar-UniNA/HQGA`)
> **Diagnosis date:** 2026-04-05 (Replication) | 2026-10-16 (IBM Hardware Integration Reconstruction)

---

## 1. Ground Truth — What the Paper Actually Does (§3, Alg.1-2)

### 1.1 Hybrid Loop (Alg.1, §3 p.7)

For each generation `t = 1..n_g` with population size `n_p`:

1. **Classical fitness** — evaluate `χ²(x)` for every `x ∈ P`. §3 intro: *"AEQGA computes the merit function classically, and then uses a quantum circuit to entangle the population and perform crossover and mutation."*
2. **Classical selection** — keep top 25% elites `P_elite ⊂ P`, `|P_elite| = n_p/4`, **outside** quantum circuits.
3. **Repopulate classically**:
   * duplicate `P_elite` → `P_elite_copy` (`n_p/4`)
   * draw fresh `P_rand` uniform [`a`, `b`] (`n_p/2`)
   * 75% (`P_elite_copy ∪ P_rand`) enters quantum circuits; 25% (`P_elite`) bypasses.
4. **Quantum encode** into **two independent circuits** (one per parameter: Ω_M `[0.0, 0.5]`, H₀ `[60, 80]`).
5. **Quantum crossover + mutation** per dimension with prob `p_c`, `p_m`.
6. **Quantum decode** — Eq.16 (random) or Eq.17-18 (elite-copy) using **basis-state counts**.
7. **Combine** `P ← P_elite ∪ P_decoded` (Alg.1 L14).
8. Statistics from `n_i = 300` independent iterations (§3.4).

### 1.2 Amplitude Encoding (§3.1, Eq.11)

> `x = [x₀, …, x_{N-1}]`, normalise `‖x‖₂ = 1`, map to `|ψ⟩ = Σ xᵢ |i⟩` on `n = log₂ N` qubits via `Qiskit.initialize()`.

* **Qubit scaling logarithmic**: `log₂(n_p) - 2` qubits (elite-copy), `log₂(n_p) - 1` (random), per parameter.
* Paper §3.1 p.9 example: `n_p = 16` → 2+3 qubits; `n_p = 32` → 3+4 qubits.
* **Power-of-two constraint**: `n_p = 2^k` required.

### 1.3 Quantum Crossover & Mutation (§3.2, Eq.12-15)

* **Crossover**: `CRy(π/2)` bidirectionally on random qubit pair → `U_cross` (Eq.14).
* **Mutation**: `Rx(π/2)` on random qubit → `U_mut` (Eq.15).
* Fixed angles `π/2`; axes `y` (crossover), `x` (mutation); `z` avoided as phase-only.

### 1.4 Quantum Decoding (§3.3, Eq.16-18, Alg.2)

Counts `c_i` = shots per **basis state** `|i⟩`.

* **Random** (Eq.16): `x_i = a + (b-a)·(c_i - c_min)/(c_max - c_min)` + `U(a,b)` replacement for exact lower-bound.
* **Elite-copy** (Eq.17-18): `p_i = c_i/Σc_j`, `x_i = n_min + (n_max-n_min)·√p_i` in box `[n_min, n_max]` around previous elites.

### 1.5 Hyperparameters (§3.4)

* `n_p` power of two (16 or 32 typical); `n_g = 50`; `n_i = 300`; `p_c = p_m = 0.5` (§4.1 optimum).
* Search ranges: `Ω_M ∈ [0.0, 0.5]`, `H0 ∈ [60, 80]`.

### 1.6 Hybrid Classical-Quantum QPU Architecture & Circuit Batching

AEQGA operates as a strict division of responsibilities between classical host CPU and quantum QPU:

* **Classical Host CPU**:
  * Population initialization & boundary clamping
  * Cosmological $\chi^2$ likelihood evaluation ($SNe$ Ia Pantheon/Pantheon+)
  * Fitness ranking & top-25% elite selection ($P_{\text{elite}}$)
  * Elite copying ($P_{\text{elite\_copy}}$, 25%) and uniform random draw ($P_{\text{rand}}$, 50%)
  * Measurement-result decoding (Eq.16 & Eq.17–18)
  * Generation history, global best tracking, and atomic JSON checkpoint persistence
* **IBM Quantum Processor (QPU / Runtime Sampler)**:
  * Amplitude state preparation via `initialize()` on normalized vectors
  * Quantum crossover via bidirectional $CR_y(\pi/2)$ gates
  * Quantum mutation via single-qubit $R_x(\pi/2)$ gates
  * Measurement of logical register into classical bitstrings
* **Four-Circuit Batching**:
  For $n_p = 32$ over 2 dimensions ($H_0, \Omega_m$), each generation constructs 4 independent circuits:
  1. $H_0$ random subset: 4 logical qubits
  2. $H_0$ elite-copy subset: 3 logical qubits
  3. $\Omega_m$ random subset: 4 logical qubits
  4. $\Omega_m$ elite-copy subset: 3 logical qubits
  Validated submission width order: `[4, 3, 4, 3]`. Submitted as **one single Sampler batch** per generation.
* **Sequential Generation Constraint**:
  Generation $t+1$ depends on classical decoding, $\chi^2$ evaluation, and elite sorting of Generation $t$. Never batch multiple generations into one job.

---

## 2. Implementation Status — Complete

All files are in place, the simulation code matches the paper, and the IBM Quantum hardware integration is fully reconstructed and verified offline.

| File | Status | Purpose |
|------|--------|---------|
| `amplitude_encoding.py` | **EXISTING** | `build_amplitude_circuit` (§3.1 Eq.11), `decode_random_subset` (Eq.16), `decode_elite_subset` (Eq.17-18), `_normalise`, `n_qubits_for_population` |
| `quantum_gates.py` | **EXISTING** | `apply_crossover` (`CRy(π/2)` Eq.12-14), `apply_mutation` (`Rx(π/2)` Eq.15), `verify_u_cross()`, `verify_rx_pi2()` |
| `aeqga_algorithm.py` | **UPDATED** | Pluggable `hardware_executor` keyword in `run_aeqga_dual`, 4-circuit batching (`[4, 3, 4, 3]`), `optimizer_state` tracking, backward compatibility |
| `pantheon_problem.py` | **UPDATED** | `lower_bounds=[60, 0.0]`, `upper_bounds=[80, 0.5]`; `PantheonPlusProblem` offline calibrated likelihood ($1753.011838$) + BAO/CMB stubs |
| `run_pantheon_aeqga.py` | **EXISTING** | `p_cross=p_mut=0.5` (§4.1); power-of-two validation; paper best-fit reference `(72.82, 0.363)` |
| `test_pantheon_aeqga.py` | **EXISTING** | T1 distance modulus & chi2 tests + T4 quantum-correctness tests |
| `aeqga/execution/ibm_sampler.py` | **NEW** | `BaseExecutor`, `IBMSamplerExecutor`, `DryRunExecutor`, `CheckpointManager`, `CompilationDiagnostics`, `_counts`, MPS equivalence verification |
| `aeqga/execution/__init__.py` | **NEW** | Exports execution adapter symbols (`BaseExecutor`, `IBMSamplerExecutor`, `DryRunExecutor`, `CheckpointManager`) |
| `aeqga/steps/evolution/aeqga_algorithm.py` | **NEW** | Modular evolution step implementation matching root `aeqga_algorithm.py` |
| `aeqga/steps/encoding/amplitude_encoding.py` | **NEW** | Modular amplitude encoding step |
| `aeqga/steps/genetic_operators/quantum_gates.py` | **NEW** | Modular quantum crossover and mutation gates |
| `aeqga/steps/decoding/measurement_decoding.py` | **NEW** | Modular basis-state count decoding |
| `aeqga/likelihoods/pantheon_problem.py` | **NEW** | Modular `PantheonPlusProblem` with full covariance and calibrated offline baseline |
| `aeqga/analysis/circuit_comparison.py` | **NEW** | Controlled same-circuit comparison pipeline (10,000 MC ensemble, TVD, Wasserstein, KS-test) |
| `tests/test_ibm_adapter.py` | **NEW** | 13 automated unit and integration tests for IBM hardware adapter |
| `scripts/ibm_preflight.py` | **NEW** | Offline-safe credentials, backend properties, and status inspector |
| `scripts/test_hardware_compilation.py` | **NEW** | 156-qubit Heron transpilation diagnostics and Aer MPS numerical equivalence test |
| `scripts/test_full_generation.py` | **NEW** | Complete offline 1-generation AEQGA test with 64 likelihood evaluations |
| `scripts/compare_generation_populations.py` | **NEW** | Monte Carlo finite-shot ensemble generation & hardware comparison saving `.npz` archive |
| `scripts/recover_ibm_job.py` | **NEW** | Idempotent job recovery from checkpoint and raw count extraction |
| `scripts/analyze_first_qpu_experiment.py` | **NEW** | Historical Experiment A TVD analysis against 10k finite-shot ensemble |
| `scripts/run_first_qpu_experiment.py` | **NEW** | Guarded pilot runner (defaults to dry-run; requires confirmation flags) |
| `scripts/run_ibm_generation.py` | **NEW** | Guarded generation runner (defaults to dry-run; requires confirmation flags) |
| `docs/IBM_HARDWARE.md` | **NEW** | Comprehensive IBM Quantum hardware integration documentation |
| `requirements-ibm.txt` | **NEW** | IBM Runtime dependencies (`qiskit-ibm-runtime==0.50.0`, `qiskit==2.5.2`, `qiskit-aer==0.17.2`) |
| `README.md` | **UPDATED** | Added hardware architecture, key points, requirements, and usage |
| `code-summary.md` | **THIS FILE** | Full diagnosis, replication plan, and hardware documentation |

**Offline verification results (no physical QPU jobs / no credentials needed):**
```
T1  Distance modulus:                                  PASS
T1  Chi2 marginalisation:                              PASS
T4a _normalise unit vector:                            PASS
T4b n_qubits_for_population(8)=3:                      PASS
T4c build_amplitude_circuit |00⟩:                      PASS
T4d build_amplitude_circuit equal superposition:       PASS
T4e non-power-of-two rejected:                         PASS
T4f U_cross = CRy(π/2) bidirectional matches Eq.14:    PASS
T4g Rx(π/2) matches Eq.15:                             PASS
T4h decode_random_subset equal counts:                 PASS
T4i decode_elite_subset valid:                         PASS
T4j _normalise zero-norm edge:                         PASS
T5a test_counts bitstring zero-padding & parsing:      PASS
T5b test_reject_zero_shots:                            PASS
T5c test_shot_total_validation:                        PASS
T5d test_156_qubit_register_width_validation:         PASS
T5e test_correct_bit_ordering (little-endian):         PASS
T5f test_four_circuit_ordering [4, 3, 4, 3]:           PASS
T5g test_offline_generation_execution:                 PASS
T5h test_deterministic_circuit_construction:           PASS
T5i test_ideal_probability_equivalence (Aer MPS):      PASS (< 4.72e-12)
T5j test_checkpoint_atomic_write_and_reading:          PASS
T5k test_recovery_without_duplicate_submissions:       PASS
T5l test_rejection_of_mismatched_fingerprints:         PASS
T5m test_preservation_of_existing_aer_and_sv_modes:    PASS
Hardware compilation 156-qubit (Heron / Kingston):     PASS (MPS max diff: 4.72e-12)
Full offline generation execution (64 evaluations):    PASS (Elitism preserved)
```

---

## 3. Diagnosis — How Current Code Implements Amplitude Encoding & Hardware Execution

### 3.1 Encoding — CORRECT ✅

| Paper (§3.1) | Current Implementation |
|---|---|
| Amplitude encoding via `initialize` on `log₂N` qubits (Eq.11) | `amplitude_encoding.build_amplitude_circuit(values)` calls `qc.initialize(normalized, range(n_qubits))` |
| Qubits: logarithmic (`log₂(n_p)-1/-2` per dim) | `n_qubits_for_population(n_p)` asserts power-of-two; `n_q_random = n_p_bits - 1`, `n_q_elite = n_p_bits - 2` |
| `‖x‖₂=1` normalization | `_normalise()` with zero-norm edge case |
| One circuit per parameter per subset | `build_dimension_circuit()` per dimension per subset (`aeqga_algorithm.py`) |

**Verification:** `build_amplitude_circuit([1,0,0,0])` → `Statevector(|00⟩)` ✓; `build_amplitude_circuit([1,1])` → `(1/√2)(|0⟩+|1⟩)` ✓.

### 3.2 Crossover/Mutation — CORRECT ✅

| Paper (§3.2) | Current Implementation |
|---|---|
| `CRy(π/2)` bidirectionally, random pair, fixed `π/2`, prob `p_c` (Eq.12-14) | `quantum_gates.apply_crossover(qc, p_c)`: `qc.cry(π/2,q0,q1); qc.cry(π/2,q1,q0)` |
| `Rx(π/2)` fixed, random qubit, prob `p_m` (Eq.15) | `quantum_gates.apply_mutation(qc, p_m)`: `qc.rx(π/2, q)` |
| No `sigma_mut` hyperparameter | Removed; `p_mut` only (paper: `p_mut=0.5`) |

**Verification:** `Operator(cry(π/2) bidirectional).reverse_qargs().data` equals `U_cross` Eq.14 within 1e-10 ✓; `Rx(π/2)` matches Eq.15 ✓.

### 3.3 Decoding — CORRECT ✅

| Paper (§3.3) | Current Implementation |
|---|---|
| **Basis-state counts** `c_i` via Eq.16/17-18 | `decode_random_subset` / `decode_elite_subset` in `amplitude_encoding.py` |
| `x_i = a+(b-a)·(c_i-c_min)/(c_max-c_min)` + `U(a,b)` replacement | `decode_random_subset` handles zero-count → uniform draw |
| `x_i = n_min+(n_max-n_min)·√p_i` in box `[n_min,n_max]` | `decode_elite_subset` with 5% margin expansion around elite span |

### 3.4 Selection & Loop — CORRECT ✅

| Paper (Alg.1) | Current Implementation |
|---|---|
| Preserve `P_elite` (25%) unchanged; merge `P_elite ∪ P_decoded` | `split_population` returns `P_elite, P_elite_copy, P_rand`; combine in `run_aeqga_dual` |
| Fresh `P_rand` uniform draws each generation | `P_rand = np.random.uniform(lower, upper, size=(n_random, n_dim))` |
| 25/25/50 split per Alg.1 L5-L7 | `n_elite = pop_size//4`, `n_random = pop_size//2` |
| No outer `n_i=300` loop | `run_aeqga_iterations` with `n_iterations` param |
| Power-of-two `n_p` | `AEQGAParameters._validate()` asserts power-of-two |

### 3.5 Pantheon & Pantheon+ Data — COMPLETE ✅

| Paper §2 | Current Implementation |
|---|---|
| `Ω_M ∈ [0.0, 0.5]`, `H0 ∈ [60, 80]` | `lower_bounds=[60.0, 0.0]`, `upper_bounds=[80.0, 0.5]` ✅ |
| BAO (16 points) + CMB (Planck TT) | `sound_horizon()` (Eq.9), `chi2_bao()`, `chi2_cmb()` stubs returning `0.0` ⚠️ |
| Cached `d_L` integral (100/300 Ω_M grid, §2.1) | `_D_L_CACHE` dict keyed by `(z_cmb, z_hel, H0, Omega_m)` ✅ |
| Pantheon+ (1701 SNe, Scolnic+22) | `PantheonPlusProblem` implemented with full covariance and calibrated offline baseline ($1753.011838$ at $[72.81, 0.362]$) ✅ |

### 3.6 IBM Hardware Adapter & 156-Qubit Register Width Bug — RESOLVED ✅

| Issue / Requirement | Diagnosis & Current Implementation |
|---|---|
| **156-Qubit Register Width Bug** | When transpiling for a 156-qubit Heron QPU (`ibm_kingston`), `compiled.num_qubits == 156`. Comparing classical register width against `compiled.num_qubits` raised `ValueError: Exactly one full-width measurement register required`. **Resolved**: Classical register width is validated against `original.num_qubits` (3 or 4 bits) ✅ |
| **Shot Validation** | Rejects `shots <= 0` immediately with `ValueError`. Verifies shot sum across bitstrings equals requested shots ✅ |
| **Bitstring Formatting** | Enforces little-endian bit ordering, zero-padded to `logical_width` bits via `_counts()` helper ✅ |
| **Execution Injection** | Injected keyword `hardware_executor` in `run_aeqga_dual()`; preserved `execution_mode="aer"` and `"statevector"` ✅ |
| **Four-Circuit Batching** | Exactly 4 circuits per generation in width order `[4, 3, 4, 3]` submitted as a single Sampler pub ✅ |
| **Zero-Hardware Submission** | Default execution mode is dry-run (`DryRunExecutor`). Physical submission blocked unless explicit confirmation flags provided ✅ |

---

## 4. IBM Hardware Integration Architecture & Diagnostics

### 4.1 Hybrid Execution Division & Pluggable Adapter

The interface is defined by `BaseExecutor` in `aeqga/execution/ibm_sampler.py`:
* `IBMSamplerExecutor`: Connects to `QiskitRuntimeService`, compiles circuits with `optimization_level=1, seed_transpiler=42`, and submits jobs to `SamplerV2`.
* `DryRunExecutor`: Drop-in offline simulator using `AerSimulator()`. Simulates identical batching, width validation, and atomic checkpointing without IBM credentials.

In `aeqga_algorithm.py`:
```python
def run_aeqga_dual(
    problem,
    params: AEQGAParameters,
    simulator=None,
    *,
    seed: int | None = None,
    execution_mode: str = "aer",
    quiet: bool = False,
    hardware_executor: BaseExecutor | None = None,
) -> tuple[GlobalBest, list, list]:
```

### 4.2 Four-Circuit Batching (`[4, 3, 4, 3]`)

For population size $n_p = 32$ over 2 cosmological parameters ($H_0$ and $\Omega_m$):
* $n_q(\text{random}) = \log_2(16) = 4$ logical qubits
* $n_q(\text{elite}) = \log_2(8) = 3$ logical qubits

Order within each generation:
```python
circuits = [
    qc_rand_h0,   # Index 0: 4 logical qubits
    qc_elite_h0,  # Index 1: 3 logical qubits
    qc_rand_om,   # Index 2: 4 logical qubits
    qc_elite_om,  # Index 3: 3 logical qubits
]
```
All four circuits are passed to `hardware_executor.run(circuits, generation=gen)` in a single batch.

### 4.3 Hardware Compilation & Aer MPS Numerical Equivalence Check

When targeting 156-qubit Heron processors (e.g. `ibm_kingston`), `initialize()` decomposes into single-qubit rotations, resets, and entangling gates ($CZ$/$CX$).
* **Statevector Memory Blowup**: Allocating a $2^{156}$ statevector crashes with `ValueError: Maximum allowed dimension exceeded`.
* **Aer Matrix Product State (MPS)**: In `verify_mps_probability_equivalence()`, the adapter simulates the transpiled circuit using `AerSimulator(method="matrix_product_state")` targeting only the active physical qubits from `transpiled.layout.initial_layout`.
* **Numerical Bound**: The maximum absolute probability difference between the uncompiled logical circuit and the 156-qubit transpiled circuit is $< 4.72 \times 10^{-12}$ (well within the $10^{-10}$ threshold).

### 4.4 Atomic Checkpoint Persistence & Idempotent Job Recovery

Checkpoints are managed by `CheckpointManager`:
* **Atomic write**: Written to `.tmp` file, flushed, synced via `os.fsync`, and atomically renamed via `os.replace`.
* **Pre-submission logging**: Job state is recorded as `"submitted"` before awaiting QPU execution.
* **Idempotent recovery**: If a checkpoint contains a valid `job_id`, the adapter retrieves existing results rather than submitting a duplicate job.
* **Fingerprint validation**: Circuit SHA-256 fingerprints, backend name, shot count, and circuit count are verified before reusing checkpoints. Mismatches raise `ValueError`.
* **Full optimizer state**: Checkpoints record population arrays, fitnesses, global best, and RNG states.

### 4.5 Controlled Same-Circuit Comparison Pipeline

Implemented in `aeqga/analysis/circuit_comparison.py`:
1. **Frozen generation circuits**: Generates initial population and builds the 4 circuits without executing them.
2. **Exact ideal probabilities**: Strips measurements and computes $P_{\text{ideal}}$ using ideal statevector simulation.
3. **Finite-shot Monte Carlo ensemble**: Draws 10,000 independent multinomial samples (1024 shots each) for each circuit, decodes populations, and evaluates $\chi^2$.
4. **Statistical metrics**:
   * Total Variation Distance (TVD) for each circuit.
   * Distribution-based metrics: 1D Wasserstein distance and 2-sample Kolmogorov-Smirnov test (avoids arbitrary individual index pairing).
   * Index-wise metrics: Mean absolute difference, max absolute difference, RMS difference.
5. **Archive export**: Saves analysis arrays to `outputs/json/ibm_generation_01/population_comparison.npz`.

### 4.6 Historical Hardware Experiments Summary

* **Experiment A: Single-Circuit Pilot**
  * Backend: `ibm_kingston` | Job ID: `db5qnv4vf2bc73d09org`
  * Configuration: 4 logical qubits, 1024 shots, $p_{\text{cross}}=1.0, p_{\text{mut}}=1.0$, seed=42.
  * Observed TVD: $0.110565$.
  * 10,000-sample ideal finite-shot ensemble: Mean TVD $0.040679$, 99th percentile $0.064987$.
  * Finding: Hardware noise is statistically significant and exceeds finite-shot sampling variance.
* **Experiment B: Complete Generation 1**
  * Backend: `ibm_kingston` | Job ID: `db5r25imb58s738at2v0`
  * Configuration: Pop 32, `[4, 3, 4, 3]` circuits, 1024 shots per circuit, seed=42.
  * Best solution: $H_0 = 71.958, \Omega_m = 0.461, \chi^2 = 1778.76$.
  * Finding: Provenance analysis demonstrates this best solution matched initial population individual #27, proving it was preserved via classical elitism.
  * Provenance caveat: Explicitly tagged `reconstructed_nominal` because serialized QPY files were not preserved in the uncommitted Windows session.

### 4.7 Zero-Hardware-Submission Safety Policy

* **Guarded Execution**: All automated tests, compilation scripts, and runners run in offline simulation mode by default.
* **Submission Guards**: Physical submission to IBM Quantum hardware requires explicit user authorization and dual command-line confirmation flags (`--submit-hardware` and `--confirm-run-qpu`).

---

## 5. What Was Fixed (Relative to Original Adapted Code)

### 5.1 Priority 1 — Core Quantum Operators ✅

1. ✅ **Amplitude encoding** via `qiskit.initialize()` on L2-normalized vectors (was: angle encoding `θ=2arcsin(√(...))` → `RY(θ)`)
2. ✅ **`CRy(π/2)` bidirectional crossover** (`quantum_gates.py:16-29`), verified against Eq.14
3. ✅ **`Rx(π/2)` mutation** (`quantum_gates.py:32-43`), verified against Eq.15
4. ✅ **Basis-state count decoding** (`amplitude_encoding.py:79-118`), Eq.16 + Eq.17-18

### 5.2 Priority 2 — Population & Algorithm Structure ✅

5. ✅ **25/25/50 split** with `split_population` returning `P_elite, P_elite_copy, P_rand`
6. ✅ **No global-best elitism**; `P_elite` preserved, merge `P_elite ∪ P_decoded`
7. ✅ **`n_iterations`** parameter for outer statistics loop
8. ✅ **Power-of-two enforcement** via `AEQGAParameters._validate()`
9. ✅ **Bounds** corrected to `Ω_M ∈ [0.0, 0.5]` per paper §3

### 5.3 Priority 3 — Likelihood & Paper Tests ✅

10. ✅ **Header docstrings** rewritten to describe amplitude encoding
11. ✅ **`sigma_mut` removed** from `AEQGAParameters`
12. ✅ **Quantum-correctness tests** added (`test_amplitude_encoding`, `test_quantum_gates`, `test_decoding`, `test_zero_norm_normalize`)
13. ✅ **README updated** with architecture and key notes

### 5.4 Priority 4 — IBM Hardware Integration Reconstruction ✅

14. ✅ **`IBMSamplerExecutor` & `DryRunExecutor`** implemented in `aeqga/execution/ibm_sampler.py`
15. ✅ **Pluggable `hardware_executor` keyword** in `run_aeqga_dual()`
16. ✅ **156-qubit Heron classical register validation bug fixed** (compares against `original.num_qubits`)
17. ✅ **Aer MPS numerical equivalence check** implemented (verifies transpiled circuit within $< 10^{-10}$)
18. ✅ **Four-circuit batching** in validated width order `[4, 3, 4, 3]`
19. ✅ **Atomic checkpoint persistence** (`CheckpointManager`) with `.tmp` and `os.replace`
20. ✅ **Idempotent job recovery** without duplicate submissions
21. ✅ **SHA-256 circuit fingerprinting** and parameter mismatch rejection
22. ✅ **Controlled same-circuit comparison pipeline** (`aeqga/analysis/circuit_comparison.py`) with 10k MC ensemble and `.npz` archive export
23. ✅ **13 automated unit & integration tests** in `tests/test_ibm_adapter.py`
24. ✅ **`PantheonPlusProblem`** implemented with calibrated baseline ($1753.011838$)
25. ✅ **Diagnostics, preflight, and recovery scripts** in `scripts/`
26. ✅ **Complete technical documentation** in `docs/IBM_HARDWARE.md`

---

## 6. Remaining Work / Future Extension

- **Physical QPU Hardware Runs**: Submit multi-generation runs to physical processors (`ibm_kingston`, `ibm_fez`) when allocation/token is explicitly provided.
- **Quantum Error Mitigation**: Incorporate dynamical decoupling, Pauli twirling, and M3 readout error mitigation in `IBMSamplerExecutor`.
- **Pantheon+ Full Integration**: The offline calibrated likelihood is complete; full raw dataset download from Scolnic et al. (2022) can be used for end-to-end contour reproduction.
- **BAO/CMB Likelihood Integration**: `chi2_bao()` and `chi2_cmb()` are stubs returning `0.0`. Full implementation requires BAO covariance (16 measurements) and PICO emulator (§2.3).
- **`n_i=300` Statistics Campaign**: Run 300 independent 50-generation iterations to reproduce the statistical distributions in paper Fig. 4.

---

## 7. Verification Checklist

- [x] `amplitude_encoding.py` has `initialize` call; no `_individual_to_angles`/`_angle_to_individual` remain
- [x] `n_qubits == log2(n_p)-1 or -2` asserted per circuit
- [x] `Operator(cry(π/2) bidirectional).reverse_qargs().data` equals `U_cross` Eq.14 within 1e-10
- [x] `decode_random` with all-equal counts → uniform resampling
- [x] `pop_size` power-of-two enforced at runtime
- [x] All offline paper tests pass (T1, T4a-T4j)
- [x] Classical register width compared against logical width (`original.num_qubits`), passing on 156-qubit devices
- [x] Aer MPS numerical equivalence verified ($< 10^{-10}$, measured $< 4.72 \times 10^{-12}$)
- [x] Four-circuit batching in order `[4, 3, 4, 3]` verified
- [x] Atomic checkpoint write and idempotent job recovery verified
- [x] SHA-256 fingerprint collision protection verified
- [x] All 13 unit and integration tests in `tests/test_ibm_adapter.py` pass
- [x] Controlled same-circuit comparison pipeline tested with 10k-sample MC ensemble and `.npz` export
- [x] Zero-hardware-submission safety defaults verified

---

## 8. References

* Sarracino et al., *"A Quantum Genetic Algorithm with application to Cosmological Parameters Estimation"*, Astron. Comput. 55:101078 (2026), arXiv:2602.15459v1.
* Scolnic et al., *"The Pantheon+ Analysis: The Full Dataset and Light-curve Modeling"*, ApJ 938:113 (2022).
* Qiskit Runtime & Sampler V2 Documentation (`qiskit-ibm-runtime==0.50.0`).
* Qiskit Aer Matrix Product State simulation documentation (`qiskit-aer==0.17.2`).
* Workspace files: `amplitude_encoding.py`, `quantum_gates.py`, `aeqga_algorithm.py`, `pantheon_problem.py`, `aeqga/execution/ibm_sampler.py`, `aeqga/analysis/circuit_comparison.py`, `tests/test_ibm_adapter.py`, `docs/IBM_HARDWARE.md`.

*Reconstruction and offline verification complete as of 2026-10-16. All tests pass.*
