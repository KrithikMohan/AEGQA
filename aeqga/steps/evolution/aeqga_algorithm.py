"""
aeqga_algorithm.py
==================
Amplitude-Encoded Quantum Genetic Algorithm (AEQGA) described in:

  Sarracino et al., "A Quantum Genetic Algorithm with application to
  Cosmological Parameters Estimation", arXiv:2602.15459 (2026).

Adapted from the Quasar-UniNA HQGA repository:
https://github.com/Quasar-UniNA/HQGA

Key differences from the original HQGA (per §3, Alg.1-2):
-----------------------------------------------------------
| Aspect               | HQGA (original)          | AEQGA (this file)        |
|----------------------|---------------------------|---------------------------|
| Encoding             | Binary (Hadamard + theta) | Amplitude via initialize  |
| Circuit readout      | Bitstring counts          | Bitstring counts          |
| Individuals          | Discrete bit-strings      | Continuous real vectors   |
| Crossover            | CNOT entanglement on best | CRy(π/2) bidirectional    |
| Mutation             | Rotation within range     | Rx(π/2) single qubit      |
| Theta update         | Reinforcement rule        | Re-encode from population |
| Elitism              | Q / D / R variants        | 25% elite preserved,      |
|                      |                           | 25% duplicate,            |
|                      |                           | 50% random fresh          |

Algorithm loop (per generation, Alg.1 §3 p.7)
-----------------------------------------------
1.  Classical fitness — evaluate χ²(x) for every x ∈ P
2.  Classical selection — keep top 25% elites P_elite outside circuits
3.  Repopulate classically:
      - duplicate P_elite → P_elite_copy  (n_p/4)
      - draw fresh P_rand uniform [a,b]   (n_p/2)
      - 75% (P_elite_copy ∪ P_rand) enters quantum circuits
4.  Encode P_elite_copy and P_rand into TWO independent circuits
    (one per dimension, log₂(n_p)-2 and log₂(n_p)-1 qubits)
5.  Quantum crossover (CRy(π/2)) + mutation (Rx(π/2)) per dimension
6.  Quantum decode — Eq.16 for random, Eq.17-18 for elite-copy
7.  Combine P ← P_elite ∪ P_decoded
8.  Repeat until max_gen reached; return global best
"""

import copy
import numpy as np
from tqdm import tqdm

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Local helpers
from aeqga.steps.encoding.amplitude_encoding import (
    build_amplitude_circuit,
    n_qubits_for_population,
)
from aeqga.steps.decoding.measurement_decoding import decode_random_subset, decode_elite_subset
from aeqga.steps.genetic_operators.quantum_gates import apply_crossover_and_mutation
from aeqga.steps.selection.population import split_population


# ---------------------------------------------------------------------------
# Data-classes / parameter containers
# ---------------------------------------------------------------------------

class AEQGAParameters:
    """Hyper-parameters for the AEQGA run (per §3.4).

    Parameters
    ----------
    pop_size   : int   – number of individuals (must be power of two)
    max_gen    : int   – number of generations
    p_cross    : float – probability of CRy(π/2) crossover per dimension
    p_mut      : float – probability of Rx(π/2) mutation per dimension
    num_shots  : int   – shots for sampling (0 → statevector mode)
    verbose    : bool  – print per-generation info
    progress_bar: bool – show tqdm bar
    """

    def __init__(
        self,
        pop_size: int = 32,
        max_gen: int = 50,
        p_cross: float = 0.5,
        p_mut: float = 0.5,
        num_shots: int = 4096,
        verbose: bool = False,
        progress_bar: bool = True,
        elite_margin: float = 0.0,
    ):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.p_cross = p_cross
        self.p_mut = p_mut
        self.num_shots = num_shots
        self.verbose = verbose
        self.progress_bar = progress_bar
        self.elite_margin = elite_margin

    def _validate(self):
        for name, minimum in [("pop_size", 8), ("max_gen", 0), ("num_shots", 0)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.pop_size & (self.pop_size - 1):
            raise ValueError("pop_size must be a power of two")
        for name in ["p_cross", "p_mut"]:
            if not np.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not np.isfinite(self.elite_margin) or self.elite_margin < 0:
            raise ValueError("elite_margin must be finite and nonnegative")


class GlobalBest:
    """Stores the best individual found so far."""

    def __init__(self):
        self.x: np.ndarray | None = None
        self.fitness: float = float("inf")
        self.gen: int = 0
        self.evaluations: int = 0

    def display(self):
        print(f"\n[GlobalBest] gen={self.gen}  fitness={self.fitness:.6g}  x={self.x}")


# ---------------------------------------------------------------------------
# Circuit construction (per dimension, per subset)
# ---------------------------------------------------------------------------

def build_dimension_circuit(
    values: np.ndarray,
    p_cross: float,
    p_mut: float,
    num_shots: int,
    mode: str = "measure",
) -> QuantumCircuit:
    """Build a per-dimension quantum circuit for a subset of individuals.

    Encodes `values` (length n_p, power-of-two) via amplitude encoding
    (§3.1), applies crossover + mutation with probability p_cross/p_mut
    (§3.2), and measures or returns statevector.

    Parameters
    ----------
    values : shape (n_p,) — 1-D array of individual values for one dimension
    p_cross, p_mut : gate probabilities (§3.4)
    num_shots : shots for measurement (0 → statevector)
    mode : "measure" or "statevector"

    Returns
    -------
    QuantumCircuit
    """
    n_p = len(values)
    n_qubits = n_qubits_for_population(n_p)

    if mode not in {"measure", "statevector"}:
        raise ValueError("mode must be 'measure' or 'statevector'")
    qc = build_amplitude_circuit(values)

    # Block 2 – Crossover + Mutation (§3.2, Alg.1 L9-L11)
    qc.barrier(label="crossover_mutation")
    apply_crossover_and_mutation(qc, p_cross, p_mut)

    if mode == "measure":
        qc.barrier(label="measure")
        qc.measure_all()

    return qc


# Deprecated single-run/HQGA adapters are summarized in docs/DEPRECATED_CODE.md.
# Main algorithm: current independent-parameter, dual-subset circuits.
def run_aeqga_dual(
    problem,
    params: AEQGAParameters,
    simulator=None,
) -> tuple[GlobalBest, list, list]:
    """Two-circuit AEQGA following the paper's dual-circuit structure (Alg.1).

    Per generation:
      1. Evaluate fitness classically
      2. Split: P_elite (25%), P_elite_copy (25%), P_rand (50%)
      3. Encode each dimension separately into circuits
      4. Apply quantum crossover + mutation
      5. Decode: Eq.16 for random, Eq.17-18 for elite-copy
      6. Combine: P ← P_elite ∪ P_decoded

    Parameters
    ----------
    problem : object with attributes / methods
        - lower_bounds : array-like, shape (n_dim,)
        - upper_bounds : array-like, shape (n_dim,)
        - n_dim        : int
        - compute_fitness(x: np.ndarray) -> float  (minimises)
        - is_max_problem() -> bool  (optional, defaults to False)
    params : AEQGAParameters
    simulator : Qiskit backend (optional). Defaults to AerSimulator().

    Returns
    -------
    g_best           : GlobalBest – best individual found
    population_evol  : list of np.ndarray – population at each generation
    bests_log        : list of [x, fitness] per generation
    """
    params._validate()

    if simulator is None:
        simulator = AerSimulator()

    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)
    n_dim = int(problem.n_dim)
    n_p = params.pop_size
    minimise = not (hasattr(problem, "is_max_problem") and problem.is_max_problem())

    def is_better(a: float, b: float) -> bool:
        return (a < b) if minimise else (a > b)

    if (lower.shape != (n_dim,) or upper.shape != (n_dim,)
            or not np.all(np.isfinite([lower, upper])) or np.any(lower >= upper)):
        raise ValueError("Problem bounds must be finite ordered vectors of length n_dim")

    # Initialise population in [lower, upper]
    population = np.random.uniform(lower, upper, size=(n_p, n_dim))

    # Evaluate initial fitness
    fitnesses = np.array([problem.compute_fitness(population[i]) for i in range(n_p)])

    g_best = GlobalBest()
    initial_best = int(np.argmin(fitnesses) if minimise else np.argmax(fitnesses))
    g_best.x = population[initial_best].copy()
    g_best.fitness = float(fitnesses[initial_best])
    g_best.evaluations = n_p
    # History index 0 is the initial population, followed by exactly max_gen updates.
    population_evol = [population.copy()]
    bests_log = [[g_best.x.tolist(), g_best.fitness]]

    gen_range = range(1, params.max_gen + 1)
    if params.progress_bar:
        gen_range = tqdm(gen_range, desc="AEQGA generations")

    for gen in gen_range:
        P_elite, P_elite_copy, P_rand = split_population(
            population, fitnesses, minimise, lower, upper
        )
        n_elite = len(P_elite)  # = n_p // 4
        n_random = n_p // 2

        # ── Step 2: Encode + quantum ops + decode per dimension ──
        decoded_elite = np.empty((n_elite, n_dim))
        decoded_random = np.empty((n_random, n_dim))

        use_measure = params.num_shots > 0

        for d in range(n_dim):
            # Qubit counts per paper §3.1 p.9:
            #   random: log2(n_p) - 1
            #   elite_copy: log2(n_p) - 2
            n_q_random = n_qubits_for_population(n_p) - 1
            n_q_elite = n_qubits_for_population(n_p) - 2

            # ── Circuit for random subset ──────────────────────
            qc_rand = build_dimension_circuit(
                P_rand[:, d],
                p_cross=params.p_cross,
                p_mut=params.p_mut,
                num_shots=params.num_shots,
                mode="measure" if use_measure else "statevector",
            )
            if use_measure:
                transpiled = transpile(qc_rand, simulator)
                result = simulator.run(transpiled, shots=params.num_shots).result()
                counts_rand = result.get_counts()
                decoded_random[:, d] = decode_random_subset(
                    counts_rand, n_q_random,
                    float(lower[d]), float(upper[d]), n_random,
                )
            else:
                qc_rand.save_statevector()
                t = transpile(qc_rand, simulator)
                sv = np.asarray(
                    simulator.run(t).result().get_statevector(t), dtype=complex
                )
                probs = np.abs(sv) ** 2
                n_states = 2 ** n_q_random
                counts_rand = {}
                for s in range(n_states):
                    bitstr = format(s, f'0{n_q_random}b')
                    counts_rand[bitstr] = probs[s]
                decoded_random[:, d] = decode_random_subset(
                    counts_rand, n_q_random,
                    float(lower[d]), float(upper[d]), n_random,
                )

            # ── Circuit for elite-copy subset ──────────────────
            qc_elite = build_dimension_circuit(
                P_elite_copy[:, d],
                p_cross=params.p_cross,
                p_mut=params.p_mut,
                num_shots=params.num_shots,
                mode="measure" if use_measure else "statevector",
            )
            if use_measure:
                transpiled = transpile(qc_elite, simulator)
                result = simulator.run(transpiled, shots=params.num_shots).result()
                counts_elite = result.get_counts()
                n_min_e = float(np.min(P_elite[:, d]))
                n_max_e = float(np.max(P_elite[:, d]))
                margin = params.elite_margin * (n_max_e - n_min_e)
                n_min_e = max(float(lower[d]), n_min_e - margin)
                n_max_e = min(float(upper[d]), n_max_e + margin)
                decoded_elite[:, d] = decode_elite_subset(
                    counts_elite, n_q_elite,
                    n_min_e, n_max_e, n_elite,
                )
            else:
                qc_elite.save_statevector()
                t = transpile(qc_elite, simulator)
                sv = np.asarray(
                    simulator.run(t).result().get_statevector(t), dtype=complex
                )
                probs = np.abs(sv) ** 2
                n_states = 2 ** n_q_elite
                counts_elite = {}
                for s in range(n_states):
                    bitstr = format(s, f'0{n_q_elite}b')
                    counts_elite[bitstr] = probs[s]
                n_min_e = float(np.min(P_elite[:, d]))
                n_max_e = float(np.max(P_elite[:, d]))
                margin = params.elite_margin * (n_max_e - n_min_e)
                n_min_e = max(float(lower[d]), n_min_e - margin)
                n_max_e = min(float(upper[d]), n_max_e + margin)
                decoded_elite[:, d] = decode_elite_subset(
                    counts_elite, n_q_elite,
                    n_min_e, n_max_e, n_elite,
                )

        # ── Step 3: Combine P_elite ∪ P_decoded (Alg.1 L14) ──
        new_population = np.vstack([P_elite, decoded_random, decoded_elite])
        if (not np.all(np.isfinite(new_population))
                or np.any(new_population < lower) or np.any(new_population > upper)):
            raise RuntimeError("Decoded population violates problem bounds")

        # ── Step 4: Evaluate fitness on new population ───────
        fitnesses = np.array([
            problem.compute_fitness(new_population[i])
            for i in range(n_p)
        ])

        g_best.evaluations += n_p

        # ── Step 5: Update global best ───────────────────────
        best_idx = int(np.argmin(fitnesses) if minimise else np.argmax(fitnesses))
        best_fitness_gen = fitnesses[best_idx]

        if g_best.x is None or is_better(best_fitness_gen, g_best.fitness):
            g_best.x = new_population[best_idx].copy()
            g_best.fitness = float(best_fitness_gen)
            g_best.gen = gen

        bests_log.append([g_best.x.tolist(), g_best.fitness])
        population_evol.append(new_population.copy())

        # ── Step 6: Prepare next generation ──────────────────
        population = new_population

    g_best.display()
    print(f"Total merit-function evaluations: {g_best.evaluations}")
    return g_best, population_evol, bests_log


# ---------------------------------------------------------------------------
# Statevector-based variant (paper's "probabilities" path)
# ---------------------------------------------------------------------------

def run_aeqga_sv(
    problem,
    params: AEQGAParameters,
    simulator=None,
) -> tuple[GlobalBest, list, list]:
    """AEQGA using the AerSimulator in statevector mode.

    Uses exact quantum state probabilities for decoding, removing
    shot noise from the inner loop.  Otherwise identical to run_aeqga_dual.

    Parameters / return values are identical to run_aeqga_dual().
    """
    sv_params = copy.copy(params)
    sv_params.num_shots = 0
    if simulator is None:
        simulator = AerSimulator(method="statevector")
    return run_aeqga_dual(problem, sv_params, simulator)
