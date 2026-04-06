"""
aeqga_algorithm.py
==================
Adaptation of the Quasar-UniNA HQGA (hqga_algorithm.py) to the
Amplitude-Encoded Quantum Genetic Algorithm (AEQGA) described in:

  Sarracino et al., "A Quantum Genetic Algorithm with application to
  Cosmological Parameters Estimation", arXiv:2602.15459, 2026.

Key differences from the original HQGA
---------------------------------------
| Aspect               | HQGA (original)            | AEQGA (this file)          |
|----------------------|----------------------------|----------------------------|
| Encoding             | Binary (Hadamard + theta)  | Amplitude / angle encoding |
| Circuit readout      | Bitstring counts           | Statevector probabilities  |
| Individuals          | Discrete bit-strings       | Continuous real vectors    |
| Crossover            | CNOT entanglement on best  | CNOT pairs, prob. p_cross  |
| Mutation             | Rotation within range      | RY(delta), prob. p_mut     |
| Theta update         | Reinforcement rule         | Re-encode from population  |
| Elitism              | Q / D / R variants         | Keep global best (classic) |

Algorithm loop (per generation)
---------------------------------
1.  Encode population  → build a fresh circuit; map each individual
    x_i ∈ [lower, upper] to RY(theta_i) where
       theta_i = 2 * arcsin(sqrt((x_i - lower) / (upper - lower)))
    One qubit per individual (n_pop qubits) per parameter dimension.
    The paper uses two separate circuits for the n_p individuals:
    one for the 'elites' selected by classical fitness, one for the
    remaining randomly drawn ones.

2.  Quantum crossover  → for each adjacent pair (q_i, q_{i+1}):
    with probability p_cross apply
       RY(theta_i) ─ CNOT ─ RY(-theta_i)
    which entangles the pair before measurement.

3.  Quantum mutation   → for each qubit, with probability p_mut
    apply RY(delta) with delta drawn from N(0, sigma_mut).

4.  Simulate / execute circuit; extract statevector or shot-based
    probabilities.

5.  Decode            → map measurement probabilities back to real
    values in [lower, upper]:
       x_decoded_j = lower + (upper - lower) * P(|j>) / max_P

6.  Evaluate merit function classically on decoded individuals.

7.  Select new population:
    - Keep global best individual (elitism).
    - Replace remaining slots with decoded individuals ordered by
      fitness.

8.  Repeat until max_gen reached; return global best.
"""

import math
import copy
import random
import numpy as np
from tqdm import tqdm

# Qiskit imports (compatible with qiskit >= 1.0)
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler

# Optional statevector support
try:
    from qiskit_aer import StatevectorSimulator  # noqa: F401 – presence check only
    _HAS_SV = True
except ImportError:
    _HAS_SV = False


# ---------------------------------------------------------------------------
# Data-classes / parameter containers
# ---------------------------------------------------------------------------

class AEQGAParameters:
    """Hyper-parameters for the AEQGA run.

    Parameters
    ----------
    pop_size   : int   – number of individuals in the population
    max_gen    : int   – number of generations
    p_cross    : float – probability of applying quantum crossover to a pair
    p_mut      : float – probability of mutating a qubit
    sigma_mut  : float – std-dev of the Gaussian RY rotation used for mutation
    num_shots  : int   – shots for sampling (set 0 to use statevector mode)
    verbose    : bool  – print per-generation info
    progress_bar: bool – show tqdm bar
    """

    def __init__(
        self,
        pop_size: int = 8,
        max_gen: int = 100,
        p_cross: float = 0.7,
        p_mut: float = 0.1,
        sigma_mut: float = 0.1,
        num_shots: int = 4096,
        verbose: bool = False,
        progress_bar: bool = True,
    ):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.p_cross = p_cross
        self.p_mut = p_mut
        self.sigma_mut = sigma_mut
        self.num_shots = num_shots
        self.verbose = verbose
        self.progress_bar = progress_bar


class GlobalBest:
    """Stores the best individual found so far."""

    def __init__(self):
        self.x: np.ndarray | None = None   # real-valued parameter vector
        self.fitness: float = float("inf")
        self.gen: int = 0

    def display(self):
        print(f"\n[GlobalBest] gen={self.gen}  fitness={self.fitness:.6g}  x={self.x}")


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _individual_to_angles(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Map a real-valued individual to RY rotation angles in [0, pi].

    Each parameter p_i is mapped via:
        theta_i = 2 * arcsin( sqrt( (p_i - lower_i) / (upper_i - lower_i) ) )

    so that P(|1>) = sin^2(theta_i / 2) encodes the normalised position.
    """
    normed = np.clip((x - lower) / (upper - lower), 0.0, 1.0)
    return 2.0 * np.arcsin(np.sqrt(normed))


def _angle_to_individual(theta: float, lower: float, upper: float) -> float:
    """Inverse mapping: RY angle → real parameter value."""
    return lower + (upper - lower) * math.sin(theta / 2.0) ** 2


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------

def build_aeqga_circuit(
    population: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    p_cross: float,
    p_mut: float,
    sigma_mut: float,
) -> QuantumCircuit:
    """Build the AEQGA quantum circuit for one generation.

    The circuit layout (per arXiv:2602.15459, Fig. 2):

      Block 1 – Encoding
        For each individual i and dimension d:
          apply RY(theta_{i,d}) to qubit q = i * n_dim + d

      Block 2 – Crossover (entanglement)
        For each adjacent pair (i, i+1) and each dimension d:
          with probability p_cross:
            RY(theta_{i,d})
            CNOT(control=q_i_d, target=q_{i+1}_d)
            RY(-theta_{i,d})

      Block 3 – Mutation
        For each qubit, with probability p_mut:
          apply RY(delta),  delta ~ N(0, sigma_mut)

      Block 4 – Measurement
        Measure all qubits.

    Parameters
    ----------
    population : shape (pop_size, n_dim)
    lower, upper : shape (n_dim,) – search-space bounds per dimension
    p_cross, p_mut, sigma_mut – circuit hyper-parameters

    Returns
    -------
    QuantumCircuit with classical register already appended.
    """
    pop_size, n_dim = population.shape
    n_qubits = pop_size * n_dim

    # Pre-compute all encoding angles
    angles = np.array(
        [_individual_to_angles(population[i], lower, upper) for i in range(pop_size)]
    )  # shape: (pop_size, n_dim)

    qc = QuantumCircuit(n_qubits, n_qubits)

    # ── Block 1: Encoding ──────────────────────────────────────────────────
    qc.barrier(label="encode")
    for i in range(pop_size):
        for d in range(n_dim):
            q = i * n_dim + d
            qc.ry(float(angles[i, d]), q)

    # ── Block 2: Quantum crossover ─────────────────────────────────────────
    qc.barrier(label="crossover")
    for i in range(pop_size - 1):
        for d in range(n_dim):
            if random.random() < p_cross:
                q_ctrl = i * n_dim + d
                q_tgt  = (i + 1) * n_dim + d
                theta  = float(angles[i, d])
                # Entangle: partial-SWAP style from the paper
                qc.ry(theta, q_ctrl)
                qc.cx(q_ctrl, q_tgt)
                qc.ry(-theta, q_ctrl)

    # ── Block 3: Mutation ──────────────────────────────────────────────────
    qc.barrier(label="mutation")
    for q in range(n_qubits):
        if random.random() < p_mut:
            delta = float(np.random.normal(0.0, sigma_mut))
            qc.ry(delta, q)

    # ── Block 4: Measurement ───────────────────────────────────────────────
    qc.barrier(label="measure")
    qc.measure(range(n_qubits), range(n_qubits))

    return qc


# ---------------------------------------------------------------------------
# Decoding from shot-based counts
# ---------------------------------------------------------------------------

def decode_population_from_counts(
    counts: dict,
    pop_size: int,
    n_dim: int,
    lower: np.ndarray,
    upper: np.ndarray,
    num_shots: int,
) -> np.ndarray:
    """Decode measurement counts into real-valued individuals.

    Following arXiv:2602.15459: the probability P(|1>) on each qubit is
    estimated from shot counts, then mapped back via the inverse encoding.

    Returns
    -------
    decoded : shape (pop_size, n_dim)
    """
    n_qubits = pop_size * n_dim

    # Count how many times each qubit was measured as '1'.
    # Qiskit bit-string convention: the RIGHTMOST character is qubit 0
    # (little-endian), so we reverse each bitstring before indexing.
    ones = np.zeros(n_qubits, dtype=float)
    for bitstring, count in counts.items():
        bits_lsb_first = bitstring[::-1]   # index 0 → qubit 0
        for q_idx, bit in enumerate(bits_lsb_first):
            if bit == "1":
                ones[q_idx] += count

    prob_one = ones / num_shots  # P(|1>) for each qubit

    decoded = np.empty((pop_size, n_dim))
    for i in range(pop_size):
        for d in range(n_dim):
            q = i * n_dim + d
            p = np.clip(prob_one[q], 0.0, 1.0)
            # Inverse of the amplitude encoding:
            #   P(|1>) = sin^2(theta/2)  =>  theta = 2*arcsin(sqrt(P))
            theta = 2.0 * math.asin(math.sqrt(p))
            decoded[i, d] = _angle_to_individual(theta, lower[d], upper[d])

    return decoded


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def run_aeqga(
    problem,
    params: AEQGAParameters,
    simulator=None,
) -> tuple[GlobalBest, list, list]:
    """Run the Amplitude-Encoded Quantum Genetic Algorithm.

    Parameters
    ----------
    problem : object with attributes / methods
        - lower_bounds : array-like, shape (n_dim,)
        - upper_bounds : array-like, shape (n_dim,)
        - n_dim        : int
        - compute_fitness(x: np.ndarray) -> float
            Returns a scalar; the algorithm *minimises* this value.
            (Pass a wrapper that negates for maximisation problems.)
        - is_max_problem() -> bool  (optional, defaults to False)

    params : AEQGAParameters

    simulator : Qiskit backend (optional).
        Defaults to AerSimulator().

    Returns
    -------
    g_best           : GlobalBest – best individual found
    population_evol  : list of np.ndarray – population at each generation
    bests_log        : list of [x, fitness] per generation
    """
    if simulator is None:
        simulator = AerSimulator()

    lower  = np.asarray(problem.lower_bounds, dtype=float)
    upper  = np.asarray(problem.upper_bounds, dtype=float)
    n_dim  = int(problem.n_dim)
    minimise = not (hasattr(problem, "is_max_problem") and problem.is_max_problem())

    def is_better(a: float, b: float) -> bool:
        """True if fitness a is strictly better than b."""
        return (a < b) if minimise else (a > b)

    # ── Initialisation: random population in [lower, upper] ───────────────
    population = np.random.uniform(
        lower, upper, size=(params.pop_size, n_dim)
    )

    g_best = GlobalBest()
    population_evol: list = []
    bests_log: list = []

    gen_range = range(params.max_gen + 1)
    if params.progress_bar:
        gen_range = tqdm(gen_range, desc="AEQGA generations")

    for gen in gen_range:

        # ── Step 1: Build and run circuit ──────────────────────────────────
        qc = build_aeqga_circuit(
            population,
            lower,
            upper,
            p_cross=params.p_cross,
            p_mut=params.p_mut,
            sigma_mut=params.sigma_mut,
        )

        transpiled = transpile(qc, simulator)
        job    = simulator.run(transpiled, shots=params.num_shots)
        result = job.result()
        counts = result.get_counts()

        # ── Step 2: Decode new candidate population ────────────────────────
        new_population = decode_population_from_counts(
            counts,
            params.pop_size,
            n_dim,
            lower,
            upper,
            params.num_shots,
        )

        if params.verbose:
            print(f"\n[gen {gen}] decoded population:\n{new_population}")

        # ── Step 3: Evaluate fitness ───────────────────────────────────────
        fitnesses = np.array(
            [problem.compute_fitness(new_population[i]) for i in range(params.pop_size)]
        )

        if params.verbose:
            print(f"[gen {gen}] fitnesses: {fitnesses}")

        # ── Step 4: Identify best individual in this generation ────────────
        best_idx = int(np.argmin(fitnesses) if minimise else np.argmax(fitnesses))
        best_fitness_gen = fitnesses[best_idx]

        # ── Step 5: Update global best ─────────────────────────────────────
        if g_best.x is None or is_better(best_fitness_gen, g_best.fitness):
            g_best.x       = new_population[best_idx].copy()
            g_best.fitness = float(best_fitness_gen)
            g_best.gen     = gen

        bests_log.append([g_best.x.copy(), g_best.fitness])
        population_evol.append(new_population.copy())

        # ── Step 6: Elitism + selection → next population ─────────────────
        # Sort by fitness (best first).
        order = (
            np.argsort(fitnesses)
            if minimise
            else np.argsort(-fitnesses)
        )
        sorted_pop = new_population[order]

        # Always keep the global best in slot 0 (elitism).
        next_population = sorted_pop.copy()
        next_population[0] = g_best.x.copy()

        population = next_population

    g_best.display()
    print(f"Total merit-function evaluations: {params.pop_size * (params.max_gen + 1)}")
    return g_best, population_evol, bests_log


# ---------------------------------------------------------------------------
# Convenience: drop-in replacement shim for the original HQGA interface
# ---------------------------------------------------------------------------

def run_qga_aeqga(problem_hqga, params_hqga, simulator=None):
    """Thin wrapper so existing code calling hqga_algorithm.runQGA() can
    switch to AEQGA with minimal changes.

    Wraps a legacy HQGA-style problem object (which exposes
    `.lower_bounds`, `.upper_bounds`, `.dim`, `.num_bit_code`,
    `.computeFitness`, `.isMaxProblem`) into the interface expected
    by run_aeqga().

    Usage
    -----
    Replace:
        gBest, evol, bests = hqga_algorithm.runQGA(device_features, circuit, params, problem)
    With:
        gBest, evol, bests = aeqga_algorithm.run_qga_aeqga(problem, params)
    """

    class _ProblemAdapter:
        def __init__(self, p):
            self._p = p
            self.lower_bounds = np.asarray(p.lower_bounds, dtype=float)
            self.upper_bounds = np.asarray(p.upper_bounds, dtype=float)
            self.n_dim = int(p.dim)

        def compute_fitness(self, x):
            # HQGA's computeFitness takes a binary string; here we pass
            # the real vector directly – override if needed.
            return self._p.computeFitness(x)

        def is_max_problem(self):
            return self._p.isMaxProblem()

    adapted_problem = _ProblemAdapter(problem_hqga)

    aeqga_params = AEQGAParameters(
        pop_size   = params_hqga.pop_size,
        max_gen    = params_hqga.max_gen,
        p_cross    = getattr(params_hqga, "p_cross",   0.7),
        p_mut      = getattr(params_hqga, "prob_mut",  0.1),
        sigma_mut  = getattr(params_hqga, "sigma_mut", 0.1),
        num_shots  = getattr(params_hqga, "num_shots", 4096),
        verbose    = getattr(params_hqga, "verbose",   False),
        progress_bar = getattr(params_hqga, "progressBar", True),
    )

    return run_aeqga(adapted_problem, aeqga_params, simulator=simulator)


# ---------------------------------------------------------------------------
# Statevector-based decoder  (more accurate; mirrors paper's "probabilities"
# emphasis rather than individual-shot counts)
# ---------------------------------------------------------------------------

def build_aeqga_circuit_sv(
    population: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    p_cross: float,
    p_mut: float,
    sigma_mut: float,
) -> QuantumCircuit:
    """Same as build_aeqga_circuit but WITHOUT measurement gates.

    Used with the statevector simulator so we can read exact |amplitude|^2
    rather than relying on shot-based estimates.
    """
    pop_size, n_dim = population.shape
    n_qubits = pop_size * n_dim

    angles = np.array(
        [_individual_to_angles(population[i], lower, upper) for i in range(pop_size)]
    )

    qc = QuantumCircuit(n_qubits)

    # Block 1 – Encoding
    for i in range(pop_size):
        for d in range(n_dim):
            qc.ry(float(angles[i, d]), i * n_dim + d)

    # Block 2 – Quantum crossover
    qc.barrier()
    for i in range(pop_size - 1):
        for d in range(n_dim):
            if random.random() < p_cross:
                q_ctrl = i * n_dim + d
                q_tgt  = (i + 1) * n_dim + d
                theta  = float(angles[i, d])
                qc.ry(theta, q_ctrl)
                qc.cx(q_ctrl, q_tgt)
                qc.ry(-theta, q_ctrl)

    # Block 3 – Mutation
    qc.barrier()
    for q in range(n_qubits):
        if random.random() < p_mut:
            delta = float(np.random.normal(0.0, sigma_mut))
            qc.ry(delta, q)

    # No measurement — statevector read-out instead
    return qc


def decode_population_from_statevector(
    statevector: np.ndarray,
    pop_size: int,
    n_dim: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Decode exact statevector probabilities into real-valued individuals.

    For an n_qubit system the statevector has 2^n entries.  We compute the
    marginal probability P(qubit q = |1>) by summing |amplitude|^2 over all
    basis states where bit q is 1.

    This is the "probabilities" path described in the paper.
    """
    n_qubits = pop_size * n_dim
    assert len(statevector) == 2 ** n_qubits, (
        f"Statevector length {len(statevector)} does not match 2^{n_qubits}"
    )

    probs = np.abs(statevector) ** 2  # shape: (2^n_qubits,)

    # Marginal P(qubit q = 1) — Qiskit statevector is stored in little-endian
    # order: basis state index j has qubit q = (j >> q) & 1
    prob_one = np.zeros(n_qubits)
    for j, p in enumerate(probs):
        if p == 0.0:
            continue
        for q in range(n_qubits):
            if (j >> q) & 1:
                prob_one[q] += p

    decoded = np.empty((pop_size, n_dim))
    for i in range(pop_size):
        for d in range(n_dim):
            q = i * n_dim + d
            p = float(np.clip(prob_one[q], 0.0, 1.0))
            theta = 2.0 * math.asin(math.sqrt(p))
            decoded[i, d] = _angle_to_individual(theta, lower[d], upper[d])

    return decoded


def run_aeqga_sv(
    problem,
    params: AEQGAParameters,
) -> tuple[GlobalBest, list, list]:
    """AEQGA using the AerSimulator in statevector mode.

    This variant closely mirrors the paper's description: it uses exact
    quantum state probabilities (rather than shot-sampled counts) for decoding,
    which removes shot noise from the inner loop.

    Parameters / return values are identical to run_aeqga().
    """
    simulator = AerSimulator(method="statevector")

    lower   = np.asarray(problem.lower_bounds, dtype=float)
    upper   = np.asarray(problem.upper_bounds, dtype=float)
    n_dim   = int(problem.n_dim)
    minimise = not (hasattr(problem, "is_max_problem") and problem.is_max_problem())

    def is_better(a: float, b: float) -> bool:
        return (a < b) if minimise else (a > b)

    population = np.random.uniform(lower, upper, size=(params.pop_size, n_dim))

    g_best = GlobalBest()
    population_evol: list = []
    bests_log: list = []

    gen_range = range(params.max_gen + 1)
    if params.progress_bar:
        gen_range = tqdm(gen_range, desc="AEQGA-SV generations")

    for gen in gen_range:

        # Build circuit without measurement gates; save statevector
        qc = build_aeqga_circuit_sv(
            population, lower, upper,
            p_cross=params.p_cross,
            p_mut=params.p_mut,
            sigma_mut=params.sigma_mut,
        )
        qc.save_statevector()

        transpiled = transpile(qc, simulator)
        result     = simulator.run(transpiled).result()
        sv         = np.asarray(result.get_statevector(transpiled), dtype=complex)

        new_population = decode_population_from_statevector(
            sv, params.pop_size, n_dim, lower, upper
        )

        if params.verbose:
            print(f"\n[gen {gen}] decoded population:\n{new_population}")

        fitnesses = np.array(
            [problem.compute_fitness(new_population[i]) for i in range(params.pop_size)]
        )

        if params.verbose:
            print(f"[gen {gen}] fitnesses: {fitnesses}")

        best_idx        = int(np.argmin(fitnesses) if minimise else np.argmax(fitnesses))
        best_fitness_gen = fitnesses[best_idx]

        if g_best.x is None or is_better(best_fitness_gen, g_best.fitness):
            g_best.x       = new_population[best_idx].copy()
            g_best.fitness = float(best_fitness_gen)
            g_best.gen     = gen

        bests_log.append([g_best.x.copy(), g_best.fitness])
        population_evol.append(new_population.copy())

        order        = np.argsort(fitnesses) if minimise else np.argsort(-fitnesses)
        sorted_pop   = new_population[order]
        next_pop     = sorted_pop.copy()
        next_pop[0]  = g_best.x.copy()
        population   = next_pop

    g_best.display()
    print(f"Total merit-function evaluations: {params.pop_size * (params.max_gen + 1)}")
    return g_best, population_evol, bests_log


# ---------------------------------------------------------------------------
# Dual-circuit variant  (paper's two-circuit structure)
# ---------------------------------------------------------------------------
# The paper splits the population into two groups processed by two circuits:
#   Circuit A  – the n_elite individuals ranked best by classical fitness
#   Circuit B  – the remaining individuals (randomly drawn or worst)
# After quantum ops + decode on each circuit, the two sub-populations are
# merged, re-evaluated, and the global best is propagated.

def run_aeqga_dual(
    problem,
    params: AEQGAParameters,
    n_elite: int | None = None,
    simulator=None,
) -> tuple[GlobalBest, list, list]:
    """Two-circuit AEQGA following the paper's dual-circuit structure.

    Parameters
    ----------
    n_elite : int, optional
        Size of the 'elite' sub-population processed by Circuit A.
        Defaults to pop_size // 2 (half-half split).
    All other parameters identical to run_aeqga().
    """
    if simulator is None:
        simulator = AerSimulator()

    lower   = np.asarray(problem.lower_bounds, dtype=float)
    upper   = np.asarray(problem.upper_bounds, dtype=float)
    n_dim   = int(problem.n_dim)
    minimise = not (hasattr(problem, "is_max_problem") and problem.is_max_problem())

    if n_elite is None:
        n_elite = max(1, params.pop_size // 2)
    n_random = params.pop_size - n_elite

    def is_better(a: float, b: float) -> bool:
        return (a < b) if minimise else (a > b)

    def _run_sub_circuit(sub_pop: np.ndarray) -> np.ndarray:
        """Build, run, and decode a circuit for a sub-population."""
        qc = build_aeqga_circuit(
            sub_pop, lower, upper,
            p_cross=params.p_cross,
            p_mut=params.p_mut,
            sigma_mut=params.sigma_mut,
        )
        t  = transpile(qc, simulator)
        r  = simulator.run(t, shots=params.num_shots).result()
        return decode_population_from_counts(
            r.get_counts(), len(sub_pop), n_dim, lower, upper, params.num_shots
        )

    # Initial random population + fitness
    population = np.random.uniform(lower, upper, size=(params.pop_size, n_dim))
    fitnesses  = np.array(
        [problem.compute_fitness(population[i]) for i in range(params.pop_size)]
    )

    g_best = GlobalBest()
    population_evol: list = []
    bests_log: list = []

    gen_range = range(params.max_gen + 1)
    if params.progress_bar:
        gen_range = tqdm(gen_range, desc="AEQGA-dual generations")

    for gen in gen_range:

        # ── Split: elite vs. random ────────────────────────────────────────
        order  = np.argsort(fitnesses) if minimise else np.argsort(-fitnesses)
        elite_idx  = order[:n_elite]
        random_idx = order[n_elite:]
        if n_random > 0:
            # Paper draws the non-elite individuals randomly from the range
            random_idx = np.random.choice(
                order[n_elite:] if len(order) > n_elite else order,
                size=n_random,
                replace=False,
            )

        sub_elite  = population[elite_idx]
        sub_random = population[random_idx] if n_random > 0 else np.empty((0, n_dim))

        # ── Circuit A: elite sub-population ───────────────────────────────
        new_elite = _run_sub_circuit(sub_elite)

        # ── Circuit B: random sub-population (if any) ──────────────────────
        if n_random > 0:
            new_random = _run_sub_circuit(sub_random)
            new_population = np.vstack([new_elite, new_random])
        else:
            new_population = new_elite

        if params.verbose:
            print(f"\n[gen {gen}] new population:\n{new_population}")

        # ── Evaluate merged population ─────────────────────────────────────
        fitnesses = np.array(
            [problem.compute_fitness(new_population[i]) for i in range(params.pop_size)]
        )

        best_idx        = int(np.argmin(fitnesses) if minimise else np.argmax(fitnesses))
        best_fitness_gen = fitnesses[best_idx]

        if g_best.x is None or is_better(best_fitness_gen, g_best.fitness):
            g_best.x       = new_population[best_idx].copy()
            g_best.fitness = float(best_fitness_gen)
            g_best.gen     = gen

        bests_log.append([g_best.x.copy(), g_best.fitness])
        population_evol.append(new_population.copy())

        # Elitism: inject global best
        new_population[0] = g_best.x.copy()
        fitnesses[0]      = g_best.fitness
        population        = new_population

    g_best.display()
    print(f"Total merit-function evaluations: {params.pop_size * (params.max_gen + 1)}")
    return g_best, population_evol, bests_log


# ---------------------------------------------------------------------------
# Utility: convergence plotting
# ---------------------------------------------------------------------------

def plot_convergence(bests_log: list, title: str = "AEQGA Convergence") -> None:
    """Plot best fitness vs generation from the bests_log returned by run_aeqga*.

    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping convergence plot.")
        return

    gens     = list(range(len(bests_log)))
    fitvals  = [entry[1] for entry in bests_log]

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
