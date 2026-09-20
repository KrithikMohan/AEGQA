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

import math
import random
import numpy as np
from tqdm import tqdm

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Local helpers
from amplitude_encoding import (
    build_amplitude_circuit,
    build_amplitude_circuit_with_measure,
    decode_random_subset,
    decode_elite_subset,
    n_qubits_for_population,
)
from quantum_gates import apply_crossover_and_mutation


# ---------------------------------------------------------------------------
# Data-classes / parameter containers
# ---------------------------------------------------------------------------

class AEQGAParameters:
    """Hyper-parameters for the AEQGA run (per §3.4).

    Parameters
    ----------
    pop_size   : int   – number of individuals (must be power of two)
    max_gen    : int   – number of generations
    n_iterations : int – outer runs for statistics (paper: 300)
    p_cross    : float – probability of CRy(π/2) crossover per dimension
    p_mut      : float – probability of Rx(π/2) mutation per dimension
    num_shots  : int   – shots for sampling (0 → statevector mode)
    verbose    : bool  – print per-generation info
    progress_bar: bool – show tqdm bar
    """

    def __init__(
        self,
        pop_size: int = 8,
        max_gen: int = 100,
        n_iterations: int = 1,
        p_cross: float = 0.5,
        p_mut: float = 0.5,
        num_shots: int = 4096,
        verbose: bool = False,
        progress_bar: bool = True,
    ):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.n_iterations = n_iterations
        self.p_cross = p_cross
        self.p_mut = p_mut
        self.num_shots = num_shots
        self.verbose = verbose
        self.progress_bar = progress_bar

    def _validate(self):
        assert (self.pop_size & (self.pop_size - 1)) == 0, \
            f"pop_size {self.pop_size} must be a power of two"
        assert self.pop_size >= 4, \
            f"pop_size must be ≥ 4 (need n_p/4 ≥ 1 elites)"


class GlobalBest:
    """Stores the best individual found so far."""

    def __init__(self):
        self.x: np.ndarray | None = None
        self.fitness: float = float("inf")
        self.gen: int = 0

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

    if mode == "measure":
        qc = build_amplitude_circuit_with_measure(values, num_shots)
    else:
        qc = build_amplitude_circuit(values)

    # Block 2 – Crossover + Mutation (§3.2, Alg.1 L9-L11)
    qc.barrier(label="crossover_mutation")
    apply_crossover_and_mutation(qc, p_cross, p_mut)

    if mode == "measure":
        qc.barrier(label="measure")

    return qc


# ---------------------------------------------------------------------------
# Population splitting (Alg.1 L5-L7)
# ---------------------------------------------------------------------------

def split_population(
    population: np.ndarray,
    fitnesses: np.ndarray,
    minimise: bool,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split population into elites (25%), elite-copies (25%), random (50%).

    Paper §3 p.7: keep top 25% as P_elite (bypass circuits), duplicate
    to P_elite_copy (enters circuit), draw fresh P_rand uniform (enters
    circuit).  75% total enters quantum circuits.

    Returns
    -------
    P_elite, P_elite_copy, P_rand
    """
    order = np.argsort(fitnesses) if minimise else np.argsort(-fitnesses)
    n_p = len(population)
    n_elite = n_p // 4

    P_elite = population[order[:n_elite]].copy()
    P_elite_copy = P_elite.copy()  # duplicate (Alg.1 L6)
    n_random = n_p // 2
    P_rand = np.random.uniform(lower, upper, size=(n_random, population.shape[1]))

    return P_elite, P_elite_copy, P_rand


# ---------------------------------------------------------------------------
# Main algorithm: single-circuit (legacy shim, deprecated)
# ---------------------------------------------------------------------------

def run_aeqga(
    problem,
    params: AEQGAParameters,
    simulator=None,
) -> tuple[GlobalBest, list, list]:
    """Run the AEQGA (single-circuit legacy wrapper around dual mode).

    For backwards compatibility; delegates to run_aeqga_dual.
    """
    return run_aeqga_dual(problem, params, simulator=simulator)


# ---------------------------------------------------------------------------
# Core algorithm: dual-circuit per paper Alg.1
# ---------------------------------------------------------------------------

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

    # Initialise population in [lower, upper]
    population = np.random.uniform(lower, upper, size=(n_p, n_dim))

    # Evaluate initial fitness
    fitnesses = np.array([problem.compute_fitness(population[i]) for i in range(n_p)])

    g_best = GlobalBest()
    population_evol: list = []
    bests_log: list = []

    gen_range = range(params.max_gen + 1)
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
                margin = 0.05 * (n_max_e - n_min_e) if n_max_e > n_min_e else 0.05
                n_min_e -= margin
                n_max_e += margin
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
                margin = 0.05 * (n_max_e - n_min_e) if n_max_e > n_min_e else 0.05
                n_min_e -= margin
                n_max_e += margin
                decoded_elite[:, d] = decode_elite_subset(
                    counts_elite, n_q_elite,
                    n_min_e, n_max_e, n_elite,
                )

        # ── Step 3: Combine P_elite ∪ P_decoded (Alg.1 L14) ──
        new_population = np.vstack([P_elite, decoded_random, decoded_elite])

        # ── Step 4: Evaluate fitness on new population ───────
        fitnesses = np.array([
            problem.compute_fitness(new_population[i])
            for i in range(n_p)
        ])

        # ── Step 5: Update global best ───────────────────────
        best_idx = int(np.argmin(fitnesses) if minimise else np.argmax(fitnesses))
        best_fitness_gen = fitnesses[best_idx]

        if g_best.x is None or is_better(best_fitness_gen, g_best.fitness):
            g_best.x = new_population[best_idx].copy()
            g_best.fitness = float(best_fitness_gen)
            g_best.gen = gen

        bests_log.append([g_best.x.copy(), g_best.fitness])
        population_evol.append(new_population.copy())

        # ── Step 6: Prepare next generation ──────────────────
        population = new_population

    g_best.display()
    print(f"Total merit-function evaluations: {params.pop_size * (params.max_gen + 1)}")
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
    params._validate()

    if simulator is None:
        simulator = AerSimulator(method="statevector")

    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)
    n_dim = int(problem.n_dim)
    n_p = params.pop_size
    minimise = not (hasattr(problem, "is_max_problem") and problem.is_max_problem())

    def is_better(a: float, b: float) -> bool:
        return (a < b) if minimise else (a > b)

    population = np.random.uniform(lower, upper, size=(n_p, n_dim))
    fitnesses = np.array([problem.compute_fitness(population[i]) for i in range(n_p)])

    g_best = GlobalBest()
    population_evol: list = []
    bests_log: list = []

    gen_range = range(params.max_gen + 1)
    if params.progress_bar:
        gen_range = tqdm(gen_range, desc="AEQGA-SV generations")

    for gen in gen_range:
        P_elite, P_elite_copy, P_rand = split_population(
            population, fitnesses, minimise, lower, upper
        )
        n_elite = len(P_elite)
        n_random = n_p // 2

        decoded_elite = np.empty((n_elite, n_dim))
        decoded_random = np.empty((n_random, n_dim))

        for d in range(n_dim):
            n_q_random = n_qubits_for_population(n_p) - 1
            n_q_elite = n_qubits_for_population(n_p) - 2

            # Random subset (statevector)
            qc_rand = build_dimension_circuit(
                P_rand[:, d], params.p_cross, params.p_mut, 0, mode="statevector"
            )
            qc_rand.save_statevector()
            t = transpile(qc_rand, simulator)
            sv = np.asarray(simulator.run(t).result().get_statevector(t), dtype=complex)
            probs = np.abs(sv) ** 2
            n_states = 2 ** n_q_random
            counts_rand = {}
            for s in range(n_states):
                bitstr = format(s, f'0{n_q_random}b')
                counts_rand[bitstr] = probs[s]
            decoded_random[:, d] = decode_random_subset(
                counts_rand, n_q_random, float(lower[d]), float(upper[d]), n_random
            )

            # Elite-copy subset (statevector)
            qc_elite = build_dimension_circuit(
                P_elite_copy[:, d], params.p_cross, params.p_mut, 0, mode="statevector"
            )
            qc_elite.save_statevector()
            t = transpile(qc_elite, simulator)
            sv = np.asarray(simulator.run(t).result().get_statevector(t), dtype=complex)
            probs = np.abs(sv) ** 2
            n_states = 2 ** n_q_elite
            counts_elite = {}
            for s in range(n_states):
                bitstr = format(s, f'0{n_q_elite}b')
                counts_elite[bitstr] = probs[s]
            n_min_e = float(np.min(P_elite[:, d]))
            n_max_e = float(np.max(P_elite[:, d]))
            margin = 0.05 * (n_max_e - n_min_e) if n_max_e > n_min_e else 0.05
            n_min_e -= margin
            n_max_e += margin
            decoded_elite[:, d] = decode_elite_subset(
                counts_elite, n_q_elite, n_min_e, n_max_e, n_elite
            )

        new_population = np.vstack([P_elite, decoded_random, decoded_elite])
        fitnesses = np.array([problem.compute_fitness(new_population[i]) for i in range(n_p)])

        best_idx = int(np.argmin(fitnesses) if minimise else np.argmax(fitnesses))
        best_fitness_gen = fitnesses[best_idx]

        if g_best.x is None or is_better(best_fitness_gen, g_best.fitness):
            g_best.x = new_population[best_idx].copy()
            g_best.fitness = float(best_fitness_gen)
            g_best.gen = gen

        bests_log.append([g_best.x.copy(), g_best.fitness])
        population_evol.append(new_population.copy())
        population = new_population

    g_best.display()
    print(f"Total merit-function evaluations: {params.pop_size * (params.max_gen + 1)}")
    return g_best, population_evol, bests_log


# ---------------------------------------------------------------------------
# Outer-iteration loop (paper §3.4 n_i=300)
# ---------------------------------------------------------------------------

def run_aeqga_iterations(
    problem,
    params: AEQGAParameters,
    simulator=None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Run AEQGA for n_iterations independent runs (paper §3.4).

    Returns mean ± std of best-fit (H0, Ω_M) across iterations,
    plus the list of bests_log entries from the last run.

    Paper Fig.4 values for SNe Ia: Ω_M = 0.362±0.016, H0 = 72.81±0.22
    """
    params._validate()
    means = []
    last_bests_log = []

    for it in range(params.n_iterations):
        if params.verbose:
            print(f"\n{'='*40}")
            print(f"  Iteration {it+1}/{params.n_iterations}")
            print(f"{'='*40}")
        g_best, _, bests_log = run_aeqga_dual(problem, params, simulator=simulator)
        means.append(g_best.x.copy())
        last_bests_log = bests_log

    means = np.array(means)  # shape (n_iterations, n_dim)
    stds = means.std(axis=0)
    means = means.mean(axis=0)

    print(f"\n[AEQGA iterations] mean ± std:")
    for i, name in enumerate(["H0", "Omega_m"]):
        print(f"  {name}: {means[i]:.4f} ± {stds[i]:.4f}")
    return means, stds, last_bests_log


# ---------------------------------------------------------------------------
# Convenience: drop-in replacement shim for the original HQGA interface
# ---------------------------------------------------------------------------

def run_qga_aeqga(problem_hqga, params_hqga, simulator=None):
    """Thin wrapper so existing code calling hqga_algorithm.runQGA() can
    switch to AEQGA with minimal changes.

    Wraps a legacy HQGA-style problem object (which exposes
    `.lower_bounds`, `.upper_bounds`, `.dim`, `.num_bit_code`,
    `.computeFitness`, `.isMaxProblem`) into the interface expected
    by run_aeqga_dual().
    """

    class _ProblemAdapter:
        def __init__(self, p):
            self._p = p
            self.lower_bounds = np.asarray(p.lower_bounds, dtype=float)
            self.upper_bounds = np.asarray(p.upper_bounds, dtype=float)
            self.n_dim = int(p.dim)

        def compute_fitness(self, x):
            return self._p.computeFitness(x)

        def is_max_problem(self):
            return self._p.isMaxProblem()

    adapted_problem = _ProblemAdapter(problem_hqga)

    aeqga_params = AEQGAParameters(
        pop_size = params_hqga.pop_size,
        max_gen = params_hqga.max_gen,
        n_iterations = getattr(params_hqga, "n_iterations", 1),
        p_cross = getattr(params_hqga, "p_cross", 0.5),
        p_mut = getattr(params_hqga, "p_mut", 0.5),
        num_shots = getattr(params_hqga, "num_shots", 4096),
        verbose = getattr(params_hqga, "verbose", False),
        progress_bar = getattr(params_hqga, "progressBar", True),
    )

    return run_aeqga_dual(adapted_problem, aeqga_params, simulator=simulator)


# ---------------------------------------------------------------------------
# Utility: convergence plotting
# ---------------------------------------------------------------------------

def plot_convergence(bests_log: list, title: str = "AEQGA Convergence") -> None:
    """Plot best fitness vs generation from the bests_log returned by run_aeqga*."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping convergence plot.")
        return

    gens = list(range(len(bests_log)))
    fitvals = [entry[1] for entry in bests_log]

    plt.figure(figsize=(8, 4))
    plt.plot(gens, fitvals, linewidth=1.8, color="steelblue")
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("aeqga_convergence.png", dpi=150)
    plt.show()
    print("Convergence plot saved to aeqga_convergence.png")
