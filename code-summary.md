# Code Summary — Diagnosis of AEQGA Workspace vs Sarracino et al. (arXiv:2602.15459)

> Paper: Sarracino et al., *"A Quantum Genetic Algorithm with application to Cosmological Parameters Estimation"*, Astron. Comput. 55:101078 (2026), arXiv:2602.15459v1 — no attached code repository.
> Workspace: `AEGQA-main/` adaptation from Quasar-UniNA HQGA (`aeqga_algorithm.py`, `pantheon_problem.py`, `run_pantheon_aeqga.py`, `test_pantheon_aeqga.py`)

---

## 1. What the Paper Actually Does (Ground Truth)

### 1.1 Overall Hybrid Loop — Alg. 1 §3 (p.7)

For each generation `t = 1..n_g` with population size `n_p`:

1. **Classical fitness** — evaluate `χ²(x)` for every `x ∈ P` (SNe Ia, BAO, CMB+BAO separately, §2). §3 intro: *"AEQGA computes the merit function classically, and then uses a quantum circuit to entangle the population and perform crossover and mutation"*.
2. **Classical selection** — keep top 25% elites `P_elite ⊂ P`, `|P_elite|=n_p/4`, *outside* quantum circuits for elitism (§3 p.7, Alg.1 L5).
3. **Repopulate classically before quantum step**:
   * duplicate `P_elite` → `P_elite_copy`, size `n_p/4` (Alg.1 L6)
   * draw fresh random `P_rand`, size `n_p/2`, uniform in `[a,b]` per parameter (Alg.1 L7: `P_rand + P_elite + P_elite_copy = P`)
   * 75% (`P_elite_copy ∪ P_rand`) enters quantum circuits; 25% (`P_elite`) bypasses them (p.7 bullet list).
4. **Quantum encode** — `P_elite_copy` and `P_rand` into **two independent circuits** (Alg.1 L8). Furthermore Ω_M and H₀ go into **separate but identical circuits** per parameter because bounds differ (`[0.0,0.5]` vs `[60,80]`, §3.1 p.9).
5. **Quantum crossover + mutation** per dimension with prob `p_c`, `p_m` (Alg.1 L9-L11, §3.2).
6. **Quantum decode** — two different decoding schemes for the two circuits (§3.3, Eq.16 vs Eq.17-18, Alg.2).
7. **Combine** `P ← P_elite ∪ P_decoded` (Alg.1 L14); iterate. Final best = argmin over last population; statistics from `n_i=300` independent iterations (§3.4).

### 1.2 Amplitude Encoding — §3.1, Eq.11

> Given `x = [x₀,...,x_{N-1}]`, normalize `Σ|x_i|²=1`, map to `|ψ⟩ = Σ x_i |i⟩` in `n=log₂ N` qubits. Basis is computational `|i⟩`, amplitudes are the data.

* Implemented via `Qiskit.initialize()` on normalized vector — Fig.1 shows decomposition into `U`, `CNOT`, `R` gates (p.8).
* **Qubit scaling is logarithmic**: for `n_p` individuals, `log₂(n_p)-2` qubits for duplicate circuit, `log₂(n_p)-1` for random circuit (p.9: example `n_p=16` → 2 and 3 qubits; `n_p=32` → 3 and 4 qubits). One such circuit **per parameter**.
* No per-individual `RY(θ)` angle embedding; no per-qubit = per-individual mapping. Entanglement in the initializer already links all qubits.

### 1.3 Quantum Crossover & Mutation — §3.2, Eq.12-15, Fig.2

* Crossover: **Coupled controlled-Y rotation `CRy(π/2)` bidirectionally** on two randomly picked qubits. Fixed angle `π/2` chosen to maximize probability variation. Matrix `U_cross` is Eq.14 (4×4). Applied with probability `p_c`.
  ```
  CRy(π/2) = |0⟩⟨0|⊗I + |1⟩⟨1|⊗Ry(π/2),  Ry(π/2)=1/√2[[1,-1],[1,1]]   (Eq.12-13)
  U_cross = CRy_{0→1}(π/2)·CRy_{1→0}(π/2)  (Eq.14)
  ```
* Mutation: **Single-qubit `Rx(π/2)`** on one randomly picked qubit. Fixed `π/2`. Matrix `U_mut = Rx(π/2)=1/√2[[1,-i],[-i,1]]` (Eq.15). Applied with probability `p_m`.
* No `RY(theta) – CNOT – RY(-theta)` pattern, no Gaussian `RY(δ)` mutation, no `sigma_mut` hyperparameter. Rotation axes are `y` (crossover) and `x` (mutation); `z` is intentionally avoided as phase-only.

### 1.4 Quantum Decoding — §3.3, Eq.16-18, Alg.2

Counts `c_i` = shots per basis state `|i⟩` (not per-qubit `P(|1⟩)` marginal).

* **Random subset** (`flag=random`, 8-individual example) — full-range min-max scaling (Eq.16):
  ```
  x_i = a + (b-a)·(c_i - c_min)/(c_max - c_min),  a,b = [0,0.5] or [60,80]
  c_min = min_i c_i, c_max = max_i c_i
  ```
  Post-process: if `x_i == a` exactly (i.e. `c_i == c_min`), replace with `U(a,b)` random draw (footnote 2, Alg.2 L11) to avoid collapse to lower bound when multiple states have zero counts.

* **Elite-copy subset** (`flag=best`, 4-individual example) — restricted-box probability decoding (Eq.17-18):
  ```
  p_i = c_i / Σ c_j
  x_i = n_min + (n_max - n_min)·√p_i
  ```
  where `[n_min,n_max]` is a **box around previous generation's elites** ("to remember results", p.10). `√p_i` phenomenologically focuses samples toward box center.

* Both use **basis-state counts**, not single-qubit marginals, not angle inversion via `θ=2 arcsin(√p)`.

### 1.5 Hyperparameters — §3.4

* `n_p` typically 16 (crossover/mutation studies) or 32 (example runs Fig.4); `n_g=50`; `n_i=300` iterations for mean±std. `n_p` must be power of two (log₂ constraint — paper only tests powers of two). Current broad intervals: Ω_M `[0.0,0.5]`, H₀ `[60,80]` — paper's intervals, not `[0.10,0.60]`.

---

## 2. Plan to Replicate Amplitude Encoding Faithfully

### Step A — Correction Layer (no data needed)

1. **Normalize + initialize**: implement `encode_amplitude(x, n_qubits)` that L2-normalizes `x` (`‖x‖₂=1`, handle `‖x‖=0` edge) and calls `qc.initialize(normalized, range(n_qubits))`. Verify against Qiskit statevector that amplitudes equal input.
2. **Crossover gate**: implement `apply_crossover(qc, q_a, q_b)` as `qc.cry(π/2, q_a, q_b); qc.cry(π/2, q_b, q_a)` (or `qc.append(CRyGate(π/2))` both directions) matching Eq.14. Unit-test matrix vs Eq.14 via `Operator`.
3. **Mutation gate**: implement `apply_mutation(qc, q)` as `qc.rx(π/2, q)` matching Eq.15.
4. **Random-replacement post-process**: after decoding, replace any `x_i == lower` with uniform draw.

### Step B — Circuit Builders

5. **Per-parameter, per-subset circuits**: for population `P` of shape `(n_p, n_dim)`:
   * split classically into `P_elite`, `P_elite_copy`, `P_rand` (25/25/50)
   * for each dimension `d`: build `qc_elite_copy_d` with `n_q = log2(n_p)-2` qubits initialized from column `P_elite_copy[:,d]`, and `qc_rand_d` with `n_q = log2(n_p)-1` qubits from `P_rand[:,d]`.
   * assert `n_p` is power of two; raise otherwise.
6. **Add quantum ops per dimension** with prob `p_c`, `p_m` on random qubit pairs/singles (loop `for d in range(n_dim)` as paper's Alg.1 does).

### Step C — Decoding

7. **From counts**: obtain `counts` dict from `AerSimulator` shots; compute `c_i` per basis state ordered by `int(bitstring,2)` (little-endian reversal as Qiskit uses). Map via Eq.16 or Eq.17-18 depending on `flag`.
8. **Box calculation for elite-copy**: compute `[n_min,n_max]` as e.g. `±σ` box around `P_elite[:,d]` min/max or mean±range (paper says "box around region delimited by best individuals" — define explicitly and document; tune width as hyperparameter, initially `±10%` of global range or previous elite span expanded by factor).

### Step D — Loop Integration

9. **Selection preservation**: after decode, `P_next = concat(P_elite, decoded_elite_copy, decoded_rand)` (no sorting, no global-best injection). Elites already not degraded.
10. **Outer iterations**: add `n_iterations` outer loop to compute mean±std across 300 runs (paper Fig.4 values: SNe `0.362±0.016, 72.81±0.22`, CMB+BAO `0.324±0.008, 66.68±0.56`).

### Step E — Validation

11. **Unit tests**: amplitude normalization, qubit count formula, `U_cross` matrix, `Rx` matrix, decoding edge case (`c_i==c_min` multiple zeros), end-to-end reproducibility with fixed seed vs paper's Fig.4 within 1σ.
12. **Integration with SNe/BAO/CMB χ²**: verify precomputed distance integrals match analytic (paper §C.1).

---

## 3. Diagnosis — How Current Workspace Implements It

### 3.1 `aeqga_algorithm.py` — Core Algorithm

#### ENCODING — Fundamentally Wrong Scheme (`aeqga_algorithm.py:25-27,132-146,153-229,471-520`)

| Paper | Current Code |
|-------|--------------|
| Amplitude encoding: `normalize(x) → |ψ⟩=Σ x_i|i⟩` via `initialize` on `log₂ N` qubits, `N=n_p` per subset per dimension (§3.1, Eq.11, `aeqga_algorithm.py:269` quote) | **Angle encoding**: `θ_i = 2·arcsin(√((x_i - lower)/(upper-lower)))` per individual per dimension → `RY(θ_i)` on qubit `q=i·n_dim+d` (`aeqga_algorithm.py:25-26,132-142,185-188,493-494`). Header table `aeqga_algorithm.py:14` claims "Amplitude / angle encoding" conflating two distinct schemes. |
| Qubits: logarithmic — `log₂(n_p)-2` and `log₂(n_p)-1` per dimension (§3.1 p.9) | Qubits: **linear** — `n_qubits = pop_size * n_dim` (`aeqga_algorithm.py:167,475`). For `pop_size=32, n_dim=2` paper uses 3+4=7 qubits/dim split; code uses 64 qubits → exponentially larger Hilbert space (2⁶⁴ vs 2⁴), infeasible for statevector. |
| `Qiskit.initialize` (paper p.9) | No `initialize` call anywhere; only `qc.ry` loops. |
| One circuit per parameter per subset | Single monolithic circuit mixing all individuals & dimensions (`aeqga_algorithm.py:179-233`, `aeqga_algorithm.py:485-518`) |
| Population array normalized to `‖x‖=1` | No normalization; raw cosmological values (60–80, 0.0–0.5) directly mapped to angles |

**Consequence**: Encoding does not reproduce paper's probability distribution; shot statistics measure per-qubit marginals of product-like `RY` state, not amplitude-encoded basis probabilities.

#### CROSSOVER — Wrong Gate, Wrong Ansatz (`aeqga_algorithm.py:33-34,190-223,500-508`)

* Paper: `CRy(π/2)` bidirectionally, fixed `π/2`, random qubit pair, prob `p_c` (§3.2 Eq.12-14, Fig.2).
* Code: `RY(θ_i)` – `CNOT` – `RY(-θ_i)` on adjacent pair `(i,i+1)` per dimension with prob `p_cross` (`aeqga_algorithm.py:33-34,216-221`). This is a partially entangling `θ`-dependent SWAP-inspired gate from HQGA heritage, not `CRy(π/2)`. Angle `θ` reuses encoding angle, introducing parameter dependence absent in paper; target qubit selection is deterministic adjacent, not random.

#### MUTATION — Wrong Gate, Wrong Distribution (`aeqga_algorithm.py:36-38,225-229,510-515`)

* Paper: `Rx(π/2)` fixed, one random qubit, prob `p_m` (§3.2 Eq.15).
* Code: `RY(δ)`, `δ ∼ N(0, sigma_mut)` per qubit with prob `p_mut` (`aeqga_algorithm.py:36-38,227-228`). Attributes: wrong axis (`y` vs `x`), random continuous angle vs fixed `π/2`, Gaussian spread vs deterministic maximal rotation, per-qubit independent vs single-qubit random. Introduces spurious `sigma_mut` hyperparameter (`aeqga_algorithm.py:89,100`) not in paper.

#### DECODING — Wrong Observable, Wrong Formula (`aeqga_algorithm.py:42-44,241-282,521-560`)

* Paper: decode from **basis-state counts** `c_i` via Eq.16 (`random`) or `√p_i` in restricted box (Eq.17-18, `aeqga_algorithm.py:321` note), with zero-count random replacement.
* Code: decode from **per-qubit marginal** `P(qubit=|1⟩)= ones/num_shots` or `Σ|amp|²` where bit `q=1` (`aeqga_algorithm.py:252-272,532-544`), then `θ=2·arcsin(√p)` → `_angle_to_individual` (`aeqga_algorithm.py:278-280,558-559`). Comment `aeqga_algorithm.py:277` falsely says "Inverse of amplitude encoding" but implements inverse of angle encoding.

Missing features:
* No `c_min/c_max` min-max scaling (Eq.16).
* No `√p_i` box (Eq.17) — statevector path `aeqga_algorithm.py:521-560` also uses marginal, not basis counts.
* No `U(a,b)` replacement for `x_i == a` (`aeqga_algorithm.py:338` footnote 2).
* No restricted box `[n_min,n_max]` around elites; decoding always spans full `[lower,upper]`.

#### SELECTION & POPULATION STRUCTURE (`aeqga_algorithm.py:318-343,643-715,686-732`)

* Paper: preserve `P_elite` (25%) unchanged, decode only copies + randoms, merge `P_elite ∪ P_decoded` (Alg.1 L14). Random subset always uniform resampled before each generation.
* Code single-circuit `run_aeqga` (`aeqga_algorithm.py:318-343`): decodes *entire* population (100%), sorts by fitness, injects single global best `next_population[0]=g_best` (`aeqga_algorithm.py:340-341`) — classic elitism of HQGA, not paper's 25% preservation.
* Code dual `run_aeqga_dual` (`aeqga_algorithm.py:656-732`): attempts two circuits but still splits by `n_elite=pop_size//2` (`aeqga_algorithm.py:675`), not 25/25/50; `random_idx` is `np.random.choice` from previous order (`aeqga_algorithm.py:698-704`) rather than fresh uniform draws in `[lower,upper]`; decoding still via angle inverse; post-elitism again single `new_population[0]=g_best` (`aeqga_algorithm.py:723`) overwriting sorted merge.
* No outer `n_i=300` iteration loop; only single trajectory `g_best` tracked.

#### STATEVECTOR VARIANT (`aeqga_algorithm.py:471-560,564-636`)

* Shares same angle-encoding/marginal-decoding flaws; `build_aeqga_circuit_sv` (`aeqga_algorithm.py:471-520`) is copy of `build_aeqga_circuit` without measurement, not an `initialize`-based amplitude circuit. Decoding `decode_population_from_statevector` (`aeqga_algorithm.py:521-560`) computes marginal per qubit, not basis probabilities.

#### HYPERPARAMETERS & DOCSTRING DRIFT

* Docstring Table `aeqga_algorithm.py:14-18` mislabels current scheme as amplitude; §Algorithm loop `aeqga_algorithm.py:20-44` describes code's (incorrect) steps as if they were paper's.
* `AEQGAParameters.sigma_mut` (`aeqga_algorithm.py:89,100`) artifact of Gaussian mutation; no `p_c=0.5, p_m=0.5` optimum point (§4.1) reproduced.
* `num_shots` default 4096 but paper's emulator shots not specified; not validated.

### 3.2 `pantheon_problem.py` — Data & Likelihood

*Largely functional for SNe Ia stat+sys χ², but deviates from paper's data setup:*

* **Dataset**: paper uses **Pantheon+** (1701 light curves, Covariance from Scolnic+22, Brout+22, §2.1 p.5, Eq.5). Code loads **Pantheon** (1048 SNe, `lcparam_full_long_zhel.txt`, `aeqga_algorithm.py:18` vs `pantheon_problem.py:123-150`) with `n_sn==1048` assertion (`test_pantheon_aeqga.py:48`). Not Pantheon+; difference changes global minimum location (paper: `0.363,72.82`; code's smaller sample shifts optimum).
* **Cosmological model handling**: paper fixes `Ω_r, Ω_k, Ω_b h²=0.02237, Ω_ν h²=0.00064, Σm_ν≈0.06` (§2.2 Eq.9-10) for BAO `r_d` scaling. Code implements flat ΛCDM `E(z)` with only `Ω_M, Ω_L` (`pantheon_problem.py:44-54`) and ignores BAO/CMB entirely — BAO `d_V, A(z), d_H, r_d` (Eq.6-9) and CMB `PICO/CAMB TT` (§2.3) not implemented. Hence `run_pantheon_aeqga.py` only reproduces SNe Ia half of paper.
* **Likelihood approximation**: paper precomputes luminosity-distance integral on 100/300 Ω_M grid and nearest-neighbor lookup (§2.1 p.6, §C.1). Code does per-SN numerical midpoint integral with `n_steps=1000` (`pantheon_problem.py:32-40`) per χ² evaluation — correct but 1000× slower, dominating runtime (paper notes merit evaluation is bottleneck, §3 p.7). No caching.
* **M marginalization**: both use analytic `χ²_marg = χ² - B²/A` (`pantheon_problem.py:72-98` vs Conley C1 footnote), so consistent.
* **Bounds**: paper initializes Ω_M ∈ `[0.0,0.5]` (§3 p.7), code uses `[0.10,0.60]` (`pantheon_problem.py:136`). Shifts search box; paper's low-Ω tail not explored.

### 3.3 `run_pantheon_aeqga.py` — Runner

* Correctly wires `AEQGAParameters` → `run_aeqga` but inherits all encoding flaws.
* `args.shots=8192` default differs from `AEQGAParameters.num_shots=4096`; inconsistent.
* Contour grid `30×30` on `[62,74]×[0.20,0.42]` (`run_pantheon_aeqga.py:85-90`) not aligned with paper's full `[0,0.5]×[60,80]` (Fig.3).
* Claims "adapted from HQGA" in README but no indication that adaptation is incomplete.

### 3.4 `test_pantheon_aeqga.py` — Tests

* T1/T2/T3 (distance modulus, diagonal χ², data loading) valid for physics, **do not test quantum encoding correctness** — no test for `initialize` amplitude, `U_cross` matrix, or decoding Eq.16/17.
* T4 checks AEQGA converges to `63<H0<75, 0.20<Ω<0.45` (`test_pantheon_aeqga.py:128`) — too loose to catch paper's tighter `±0.016, ±0.22` precision; passes even with wrong encoding.

### 3.5 Cross-Cutting

* **Typo in task title**: workspace file header `aeqga_algorithm.py:12` says "Amplitude/Aplitude" — indicates hasty adaptation.
* **Reproducibility**: no seed handling for quantum simulator vs paper's emulator seed `s` (Alg.2 L2).
* **Power-of-two constraint**: paper requires `n_p` power of two for amplitude encoding; code allows any `pop_size` (e.g., 8 is okay by accident, but 10 would silently produce non-logarithmic behavior).

---

## 4. Proposed Fixes

### 4.1 Priority 1 — Fix Encoding (Blocking)

**Replace `aeqga_algorithm.py:132-146,167-229,471-520` entirely:**

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, RXGate  # or CRYGate

def _normalize_amplitude(x: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(x)
    if nrm == 0:
        return np.full_like(x, 1/np.sqrt(len(x)))
    return x / nrm

def build_amplitude_circuit(values: np.ndarray) -> QuantumCircuit:
    n = len(values)
    n_qubits = int(np.log2(n))
    assert 2**n_qubits == n, "n_p subset must be power of two for amplitude encoding"
    norm = _normalize_amplitude(values)
    qc = QuantumCircuit(n_qubits)
    qc.initialize(norm, range(n_qubits))
    return qc

# Per dimension, per subset:
# qc_rand_H0 = build_amplitude_circuit(P_rand[:,0])
# qc_rand_Om = build_amplitude_circuit(P_rand[:,1])
# qc_dup_H0  = build_amplitude_circuit(P_dup[:,0])
# similarly for Om
```

* Delete `_individual_to_angles`, `_angle_to_individual`, linear `n_qubits = pop_size * n_dim`.
* Add per-dimension circuits: `build_aeqga_circuit` should take `values: (N,)` 1-D array, return `n_qubits=log2(N)` circuit, not `(pop_size,n_dim)`.

### 4.2 Priority 1 — Fix Crossover/Mutation

**Replace `aeqga_algorithm.py:190-229` with paper's fixed-angle gates:**

```python
def apply_crossover(qc: QuantumCircuit, p_c: float):
    if random.random() >= p_c:
        return
    q0, q1 = random.sample(range(qc.num_qubits), 2)
    qc.cry(np.pi/2, q0, q1)  # CRy(π/2) Eq.12-13
    qc.cry(np.pi/2, q1, q0)  # bidirectional → U_cross Eq.14

def apply_mutation(qc: QuantumCircuit, p_m: float):
    if random.random() >= p_m:
        return
    q = random.randrange(qc.num_qubits)
    qc.rx(np.pi/2, q)  # Eq.15, axis x, fixed π/2
```

* Remove `sigma_mut`, `qc.ry(theta)`, `qc.cx`, `qc.ry(-theta)`, Gaussian draws.
* Apply **once per circuit** per generation with probability `p_c/p_m` (paper's Alg.1 L10-11 inside `for each dimension d` — so per-dimension Bernoulli, not per-qubit/per-pair).

### 4.3 Priority 1 — Fix Decoding

**Replace `decode_population_from_counts` (`aeqga_algorithm.py:241-282`) and `decode_population_from_statevector` (`aeqga_algorithm.py:521-560`):**

```python
def decode_random(counts: dict, a: float, b: float, n: int) -> np.ndarray:
    # counts: basis_state -> shots, a,b global bounds
    c = np.array([counts.get(format(i, f'0{n_qubits}b'), 0) for i in range(n)])
    c_min, c_max = c.min(), c.max()
    if c_max == c_min:
        return np.random.uniform(a, b, size=n)
    x = a + (b-a)*(c - c_min)/(c_max - c_min)
    # post-process: exact lower bound → random draw
    mask = (x == a)
    x[mask] = np.random.uniform(a, b, size=mask.sum())
    return x

def decode_elite(counts: dict, n_min: float, n_max: float, n: int) -> np.ndarray:
    c = np.array([...])
    p = c / c.sum()
    x = n_min + (n_max - n_min)*np.sqrt(p)
    return x
```

* Use basis-state counts, not `ones/num_shots` marginal (`aeqga_algorithm.py:253-272`).
* Implement separate `flag=best/random` paths as Alg.2; compute `[n_min,n_max]` box around `P_elite[:,d]` (e.g., `np.min(P_elite[:,d]), np.max(P_elite[:,d])` expanded by 5% margin per paper description).
* Fix little-endian bitstring reversal — keep consistent but document.

### 4.4 Priority 2 — Fix Selection & Loop

**Rewrite `run_aeqga` generational step (`aeqga_algorithm.py:295-345`):**

```python
# after classical fitness evaluation
order = np.argsort(fitnesses)
P_elite = population[order[:n_p//4]]          # 25% best, not mutated
P_elite_copy = P_elite.copy()                  # duplicate
P_rand = np.random.uniform(lower, upper, size=(n_p//2, n_dim))  # fresh random 50%

# quantum evolve each subset per dimension
decoded_dup = np.column_stack([
    decode_elite(run_circuit(build_amplitude_circuit(P_elite_copy[:,d]), p_c, p_m), n_min_d, n_max_d)
    for d in range(n_dim)
])
decoded_rand = np.column_stack([
    decode_random(run_circuit(build_amplitude_circuit(P_rand[:,d]), p_c, p_m), lower[d], upper[d])
    for d in range(n_dim)
])
population = np.vstack([P_elite, decoded_dup, decoded_rand])
```

* Remove global-best elitism `next_pop[0]=g_best`.
* Enforce `n_p % 4 == 0` and power-of-two.
* Add outer `for iteration in range(n_iterations):` loop for statistics.

Update `run_aeqga_dual` similarly or deprecate (paper's dual is exactly this 25/25/50 per-dimension split — no need for separate `run_aeqga_dual`).

### 4.5 Priority 2 — Align Data With Paper

* `pantheon_problem.py:136`: change `lower_bounds = [0.0, 0.10]` → `[0.0, 60.0]`? Actually `lower=[0.0,60]` for `[Ω_M,H0]` ordering — paper uses `[0.0,60]` not `[60,0.10]`. Currently `lower=[60,0.10]` assumes `[H0,Ω]` ordering; ensure consistent ordering with Alg.1 per-dimension circuits. Document ordering.
* Add BAO (`pantheon_problem.py: add BAOProblem`) implementing Eq.6-9 with fixed `Ω_b h², Ω_ν h²` and covariance, plus CMB `PICO` stub or CAMB emulator note; or explicitly scope workspace to "SNe Ia only" in README and `code-summary`.
* Optionally add cached integral lookup for `d_L` to match paper's optimization and speed up `χ²`.

### 4.6 Priority 3 — Cleanup & Docs

* Fix header table `aeqga_algorithm.py:14` and docstring loop `aeqga_algorithm.py:20-44` to describe amplitude encoding accurately (remove angle references).
* Remove `AEQGAParameters.sigma_mut` or deprecate with warning; add `n_iterations` param.
* Add tests for `U_cross` matrix equality to Eq.14, `initialize` normalization, decoding edge cases.
* Update `README.md:4-10` to note Pantheon vs Pantheon+ and SNe-only scope.
* Fix typo: title request says "Aplitude" → "Amplitude" (intentionally preserved in output filename `code-summary.md`).

---

## 5. What Does NOT Need Fixing

* `pantheon_problem.py:72-98` χ² marginalization and `chi2_pantheon` logic — correct.
* `pantheon_problem.py:18-54` distance modulus integral — correct physics, just uncached.
* `aeqga_algorithm.py:80-115` parameter container structure (aside from `sigma_mut`).
* `aeqga_algorithm.py:774-798` convergence plotting — generic, works after decode fix.
* `test_pantheon_aeqga.py: T1` (distance modulus unit tests) — valid.

---

## 6. Verification Checklist After Fixes

- [ ] `grep -r "initialize" aeqga_algorithm.py` non-empty; `grep -r "_individual_to_angles\|_angle_to_individual"` empty
- [ ] `n_qubits == log2(n_p)-1 or -2` asserted per circuit
- [ ] `Operator(cry(π/2) bidirection).data` equals `U_cross` Eq.14 within 1e-10
- [ ] `decode_random` with all-zero counts triggers uniform resampling, not all-`a`
- [ ] `run_aeqga` with `n_p=16, n_g=50, p_c=p_m=0.5` on Pantheon data yields mean within 1σ of paper's SNe result (run 300 iterations, check `0.363±0.016, 72.82±0.22` envelope)
- [ ] BAO+CMB path either implemented or explicitly documented as future work

---

## 7. References

* Sarracino et al. arXiv:2602.15459v1, §§3.1-3.4, Alg.1-2, Eq.11-18, Figs.1-2 — all section/gate citations above.
* Qiskit `initialize` docs — amplitude encoding via state preparation.
* Current workspace files: `aeqga_algorithm.py:14,25,112-146,153-282,471-560,675,723`, `pantheon_problem.py:32,123,136`, `run_pantheon_aeqga.py:85`, `test_pantheon_aeqga.py:48,128`.

*Diagnosis generated 2026-04-05 by workspace inspection + paper HTML fetch (arxiv.org/html/2602.15459v1). No paper PDF was attached locally; HTML is authoritative for gate definitions.*
