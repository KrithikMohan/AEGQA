# Code Summary — Diagnosis & Implementation of AEQGA vs Sarracino et al. (arXiv:2602.15459)

> Paper: Sarracino et al., *"A Quantum Genetic Algorithm with application to
> Cosmological Parameters Estimation"*, Astron. Comput. 55:101078 (2026), arXiv:2602.15459v1
> Workspace: `AEGQA-main/` — originally adapted from Quasar-UniNA HQGA

---

## 1. What the Paper Actually Does (Ground Truth)

### 1.1 Overall Hybrid Loop — Alg. 1 §3 (p.7)

For each generation `t = 1..n_g` with population size `n_p`:

1. **Classical fitness** — evaluate `χ²(x)` for every `x ∈ P`. §3 intro: *"AEQGA computes the merit function classically, and then uses a quantum circuit to entangle the population and perform crossover and mutation"*.
2. **Classical selection** — keep top 25% elites `P_elite ⊂ P`, `|P_elite|=n_p/4`, *outside* quantum circuits.
3. **Repopulate classically**:
   * duplicate `P_elite` → `P_elite_copy` (n_p/4)
   * draw fresh `P_rand` uniform (n_p/2)
   * 75% (`P_elite_copy ∪ P_rand`) enters quantum circuits; 25% (`P_elite`) bypasses.
4. **Quantum encode** into **two independent circuits** (one per parameter: Ω_M `[0.0,0.5]`, H₀ `[60,80]`).
5. **Quantum crossover + mutation** per dimension with prob `p_c`, `p_m`.
6. **Quantum decode** — Eq.16 (random) or Eq.17-18 (elite-copy) using basis-state counts.
7. **Combine** `P ← P_elite ∪ P_decoded` (Alg.1 L14).
8. Statistics from `n_i=300` independent iterations (§3.4).

### 1.2 Amplitude Encoding — §3.1, Eq.11

> `x = [x₀,...,x_{N-1}]`, normalize `Σ|x_i|²=1`, map to `|ψ⟩ = Σ x_i |i⟩` in `n=log₂ N` qubits via `Qiskit.initialize()`.

* **Qubit scaling logarithmic**: `log₂(n_p)-2` qubits (elite-copy), `log₂(n_p)-1` (random), per parameter.
* Paper §3.1 p.9 example: `n_p=16` → 2+3 qubits; `n_p=32` → 3+4 qubits.
* **Power-of-two constraint**: `n_p = 2^k` required.

### 1.3 Quantum Crossover & Mutation — §3.2, Eq.12-15

* **Crossover**: `CRy(π/2)` bidirectionally on random qubit pair → `U_cross` (Eq.14).
* **Mutation**: `Rx(π/2)` on random qubit → `U_mut` (Eq.15).
* Fixed angles `π/2`; axes `y` (crossover), `x` (mutation); `z` avoided as phase-only.

### 1.4 Quantum Decoding — §3.3, Eq.16-18, Alg.2

Counts `c_i` = shots per **basis state** `|i⟩`.

* **Random** (Eq.16): `x_i = a + (b-a)·(c_i-c_min)/(c_max-c_min)` + `U(a,b)` replacement for exact lower-bound.
* **Elite-copy** (Eq.17-18): `p_i = c_i/Σc_j`, `x_i = n_min + (n_max-n_min)·√p_i` in box `[n_min,n_max]` around previous elites.

### 1.5 Hyperparameters — §3.4

* `n_p` power of two (16 or 32 typical); `n_g=50`; `n_i=300`; `p_c=p_m=0.5` (§4.1 optimum).
* Search ranges: `Ω_M ∈ [0.0, 0.5]`, `H0 ∈ [60, 80]`.

---

## 2. Implementation Status — All Plan Steps Completed

### Files created/updated

| File | Status | Purpose |
|------|--------|---------|
| `amplitude_encoding.py` | **NEW** | `build_amplitude_circuit` (§3.1 Eq.11), `decode_random_subset` (Eq.16), `decode_elite_subset` (Eq.17-18), `_normalise`, `n_qubits_for_population` |
| `quantum_gates.py` | **NEW** | `apply_crossover` (`CRy(π/2)` Eq.12-14), `apply_mutation` (`Rx(π/2)` Eq.15), `verify_u_cross()`, `verify_rx_pi2()` |
| `aeqga_algorithm.py` | **REWRITTEN** | Per-dimension per-subset circuits (Alg.1), `run_aeqga_dual` with 25/25/50 split, `run_aeqga_sv` statevector variant, `run_aeqga_iterations` (§3.4 n_i=300) |
| `pantheon_problem.py` | **UPDATED** | `lower_bounds=[60,0.0]`, `upper_bounds=[80,0.5]` per paper §3; BAO/CMB stubs; cached `d_L`; `sound_horizon()` (Eq.9) |
| `run_pantheon_aeqga.py` | **UPDATED** | `p_cross=p_mut=0.5` (§4.1); power-of-two validation; paper best-fit reference `(72.82, 0.363)` |
| `test_pantheon_aeqga.py` | **UPDATED** | T4 quantum-correctness tests added: `test_amplitude_encoding`, `test_quantum_gates`, `test_decoding`, `test_zero_norm_normalize` |
| `README.md` | **UPDATED** | Architecture diagram, key notes (power-of-two, SNe-only scope) |
| `code-summary.md` | **THIS FILE** | Diagnosis + implementation status |

### Verification results (offline, no data needed)

```
T1  Distance modulus:           PASS
T1  Chi2 marginalisation:       PASS
T4a _normalise unit vector:     PASS
T4b n_qubits_for_population(8)=3 PASS
T4c build_amplitude_circuit |00⟩: PASS
T4d build_amplitude_circuit equal superposition: PASS
T4e non-power-of-two rejected:  PASS
T4f U_cross = CRy(π/2) bidirectional matches Eq.14: PASS
T4g Rx(π/2) matches Eq.15:      PASS
T4h decode_random_subset equal counts: PASS
T4i decode_elite_subset valid:  PASS
T4j _normalise zero-norm edge:  PASS
```

---

## 3. Diagnosis — Original Code vs Paper (Now Fixed)

### 3.1 Encoding — **FIXED**

| Paper (§3.1) | Old Code | New Implementation |
|---|---|---|
| Amplitude encoding via `initialize` on `log₂N` qubits (Eq.11) | **Angle encoding**: `θ=2arcsin(√(...))` → `RY(θ)` per individual per dim | `amplitude_encoding.build_amplitude_circuit(values)` with `qc.initialize(normalized, range(n_qubits))` |
| Qubits: logarithmic (`log₂(n_p)-1/-2` per dim) | Qubits: **linear** (`pop_size * n_dim`) | `n_qubits_for_population(n_p)` asserts power-of-two |
| `Qiskit.initialize` (paper p.9) | No `initialize`; only `qc.ry` loops | `amplitude_encoding.py:36-43` |
| One circuit per parameter per subset | Single monolithic circuit mixing all | `build_dimension_circuit` per dimension per subset |
| `‖x‖₂=1` normalization | No normalization | `_normalise()` with zero-norm edge case |

### 3.2 Crossover/Mutation — **FIXED**

| Paper (§3.2) | Old Code | New Implementation |
|---|---|---|
| `CRy(π/2)` bidirectionally, random pair, fixed `π/2`, prob `p_c` (Eq.12-14) | `RY(θ)-CNOT-RY(-θ)` adjacent pair, θ-dependent, prob `p_cross` | `quantum_gates.apply_crossover(qc, p_c)`: `qc.cry(π/2,q0,q1); qc.cry(π/2,q1,q0)` |
| `Rx(π/2)` fixed, random qubit, prob `p_m` (Eq.15) | `RY(δ), δ~N(0,σ)` per qubit, prob `p_mut` | `quantum_gates.apply_mutation(qc, p_m)`: `qc.rx(π/2, q)` |
| No `sigma_mut` hyperparameter | `sigma_mut` in `AEQGAParameters` | Removed; `p_mut` only (paper: `p_mut=0.5`) |

### 3.3 Decoding — **FIXED**

| Paper (§3.3) | Old Code | New Implementation |
|---|---|---|
| **Basis-state counts** `c_i` via Eq.16/17-18 | **Per-qubit marginal** `P(|1⟩) = ones/num_shots` | `decode_random_subset` / `decode_elite_subset` in `amplitude_encoding.py` |
| `x_i = a+(b-a)·(c_i-c_min)/(c_max-c_min)` + `U(a,b)` replacement | `θ=2arcsin(√p)` → `_angle_to_individual` | `decode_random_subset` handles zero-count → uniform draw |
| `x_i = n_min+(n_max-n_min)·√p_i` in box `[n_min,n_max]` | Always spans full `[lower,upper]` | `decode_elite_subset` with 5% margin expansion around elite span |

### 3.4 Selection & Loop — **FIXED**

| Paper (Alg.1) | Old Code | New Implementation |
|---|---|---|
| Preserve `P_elite` (25%) unchanged; merge `P_elite ∪ P_decoded` | Sort by fitness, inject single `g_best` | `split_population` returns `P_elite, P_elite_copy, P_rand`; combine in `run_aeqga_dual` |
| Fresh `P_rand` uniform draws each generation | `np.random.choice` from previous order | `P_rand = np.random.uniform(lower, upper, size=(n_random, n_dim))` |
| 25/25/50 split per Alg.1 L5-L7 | `n_elite = pop_size//2` | `n_elite = pop_size//4`, `n_random = pop_size//2` |
| No outer `n_i=300` loop | Single trajectory | `run_aeqga_iterations` with `n_iterations` param |
| Power-of-two `n_p` | Any `pop_size` allowed | `AEQGAParameters._validate()` asserts power-of-two |

### 3.5 Pantheon Data — **PARTIALLY FIXED**

| Paper §2 | Old Code | New Implementation |
|---|---|---|
| `Ω_M ∈ [0.0, 0.5]`, `H0 ∈ [60, 80]` | `Ω_M ∈ [0.10, 0.60]`, `H0 ∈ [60, 80]` | `lower_bounds=[60.0, 0.0]`, `upper_bounds=[80.0, 0.5]` |
| BAO (16 points) + CMB (Planck TT) | Ignored | `sound_horizon()` (Eq.9), `chi2_bao()`, `chi2_cmb()` stubs |
| Cached `d_L` integral (100/300 Ω_M grid, §2.1) | Per-SN numerical midpoint per χ² eval | `_D_L_CACHE` dict keyed by `(z_cmb, z_hel, H0, Omega_m)` |
| Pantheon+ (1701 SNe) | Pantheon (1048 SNe) | Unchanged; documented in `PantheonProblem` docstring |

---

## 4. What Was Fixed (Summary)

### 4.1 Priority 1 — Complete ✅

1. ✅ **Amplitude encoding** via `qiskit.initialize()` on L2-normalized vectors
2. ✅ **`CRy(π/2)` bidirectional crossover** (`quantum_gates.py:16-29`), verified against Eq.14
3. ✅ **`Rx(π/2)` mutation** (`quantum_gates.py:32-43`), verified against Eq.15
4. ✅ **Basis-state count decoding** (`amplitude_encoding.py:79-118`), Eq.16 + Eq.17-18

### 4.2 Priority 2 — Complete ✅

5. ✅ **25/25/50 split** with `split_population` returning `P_elite, P_elite_copy, P_rand`
6. ✅ **No global-best elitism**; `P_elite` preserved, merge `P_elite ∪ P_decoded`
7. ✅ **`n_iterations`** parameter for outer statistics loop
8. ✅ **Power-of-two enforcement** via `AEQGAParameters._validate()`
9. ✅ **Bounds** corrected to `Ω_M ∈ [0.0, 0.5]` per paper §3

### 4.3 Priority 3 — Complete ✅

10. ✅ **Header table & docstring** rewritten to describe amplitude encoding
11. ✅ **`sigma_mut` removed** from `AEQGAParameters`
12. ✅ **Quantum-correctness tests** added (`test_amplitude_encoding`, `test_quantum_gates`, `test_decoding`, `test_zero_norm_normalize`)
13. ✅ **README updated** with architecture and key notes

---

## 5. Remaining Work / Future Extension

- **Pantheon+ data**: The paper uses Pantheon+ (1701 SNe, Scolnic+22). Current implementation uses Pantheon (1048 SNe). To fully reproduce paper's `0.363±0.016, 72.81±0.22`, replace `lcparam_full_long_zhel.txt` with Pantheon+ file.
- **BAO/CMB integration**: `chi2_bao()` and `chi2_cmb()` are stubs returning `0.0`. Full implementation requires BAO covariance (16 measurements) and PICO emulator (§2.3).
- **End-to-end validation**: Requires Pantheon data clone to run `test_pantheon_aeqga.py --data sn_data/Pantheon` and verify convergence to paper's Fig.4 values.
- **`n_i=300` statistics**: `run_aeqga_iterations` implemented but untested without data (requires 300 × 50-gen runs).

---

## 6. Verification Checklist

- [x] `amplitude_encoding.py` has `initialize` call; no `_individual_to_angles`/`_angle_to_individual` remain
- [x] `n_qubits == log2(n_p)-1 or -2` asserted per circuit
- [x] `Operator(cry(π/2) bidirectional).reverse_qargs().data` equals `U_cross` Eq.14 within 1e-10
- [x] `decode_random` with all-equal counts → uniform resampling
- [x] `pop_size` power-of-two enforced at runtime
- [x] All offline tests pass (T1, T4a-T4j)
- [ ] End-to-end with Pantheon data (requires data clone) — see §5

---

## 7. References

* Sarracino et al. arXiv:2602.15459v1, §§3.1-3.4, Alg.1-2, Eq.11-18, Figs.1-2.
* Qiskit `initialize` docs — amplitude encoding via state preparation.
* Current workspace files: `amplitude_encoding.py`, `quantum_gates.py`, `aeqga_algorithm.py`, `pantheon_problem.py`, `run_pantheon_aeqga.py`, `test_pantheon_aeqga.py`.
* Paper HTML fetched from https://arxiv.org/html/2602.15459v1.

*Diagnosis and implementation generated 2026-04-05. All blocking fixes (§4.1-4.2) and cleanup (§4.3) are complete. Offline verification passed.*
