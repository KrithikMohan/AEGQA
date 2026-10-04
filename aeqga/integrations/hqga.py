"""Objective-level interoperability with Quasar-UniNA/HQGA real problems.

Keep each algorithm's circuits, encoding and result types independent.
See docs/HQGA_INTEGRATION.md for the upstream calling convention.
"""
import numpy as np

from aeqga.steps.evolution.aeqga_algorithm import AEQGAParameters, run_aeqga_dual


def _bounds(problem, dimension):
    lower = np.asarray(problem.lower_bounds, dtype=float).copy()
    upper = np.asarray(problem.upper_bounds, dtype=float).copy()
    if (lower.shape != (dimension,) or upper.shape != (dimension,)
            or not np.all(np.isfinite([lower, upper])) or np.any(lower >= upper)):
        raise ValueError("Expected finite per-dimension bounds with lower < upper")
    return lower, upper


class HQGAProblemAdapter:
    """Expose an HQGA RealProblem's continuous phenotype objective to AEQGA.

    BinaryProblem (e.g. OneMax) is intentionally unsupported. Do not call
    evaluate here: upstream evaluate expects a Gray-coded chromosome.
    """
    def __init__(self, problem):
        self.problem = problem
        self.n_dim = problem.dim
        if not isinstance(self.n_dim, (int, np.integer)) or self.n_dim < 1:
            raise ValueError("HQGA dim must be a positive integer")
        self.lower_bounds, self.upper_bounds = _bounds(problem, self.n_dim)

    def compute_fitness(self, x):
        return float(self.problem.computeFitness(np.asarray(x, dtype=float).tolist()))

    def is_max_problem(self):
        return bool(self.problem.isMaxProblem())


def parameters_from_hqga(params, **overrides):
    """Transfer common budgets only, not HQGA's reinforcement/mutation rules.

    AEQGA probabilities retain their own defaults unless explicitly overridden.
    Invalid HQGA population sizes are rejected, never silently rounded.
    """
    values = dict(pop_size=params.pop_size, max_gen=params.max_gen,
                  num_shots=getattr(params, "num_shots", 4096),
                  verbose=getattr(params, "verbose", False),
                  progress_bar=getattr(params, "progressBar", True))
    values.update(overrides)
    result = AEQGAParameters(**values)
    result._validate()
    return result


def run_qga_aeqga(problem_hqga, params_hqga, simulator=None, **overrides):
    """Run AEQGA on an HQGA real objective; return native AEQGA results.

    Not a replacement for runQGA(device_features, circuit, params, problem).
    """
    return run_aeqga_dual(HQGAProblemAdapter(problem_hqga),
                          parameters_from_hqga(params_hqga, **overrides), simulator)


class AEQGAProblemForHQGA:
    """Expose a continuous AEQGA objective through HQGA's Gray-code protocol.

    Decoding matches upstream convertFromBinToFloat: dimension-major chunks,
    inclusive endpoints, resolution (upper-lower)/(2**num_bit_code-1).
    Chromosomes are AFTER HQGA.fromQtoC, not raw Qiskit count keys.
    """
    def __init__(self, problem, num_bit_code=8, name="AEQGA objective"):
        if (not isinstance(num_bit_code, (int, np.integer))
                or isinstance(num_bit_code, bool) or not 1 <= num_bit_code <= 52):
            raise ValueError("num_bit_code must be an integer in [1, 52]")
        self.problem = problem
        self.dim = problem.n_dim
        if not isinstance(self.dim, (int, np.integer)) or self.dim < 1:
            raise ValueError("n_dim must be a positive integer")
        self.num_bit_code = int(num_bit_code)
        self.name = name
        self.lower_bounds, self.upper_bounds = _bounds(problem, self.dim)
        self.evaluations = 0

    @property
    def resolution(self):
        return (self.upper_bounds - self.lower_bounds) / (2**self.num_bit_code - 1)

    def convert(self, chromosome):
        if (not isinstance(chromosome, str)
                or len(chromosome) != self.dim * self.num_bit_code
                or set(chromosome) - {"0", "1"}):
            raise ValueError("Expected a dimension-major Gray-coded chromosome")
        indices = []
        for start in range(0, len(chromosome), self.num_bit_code):
            gray = int(chromosome[start:start + self.num_bit_code], 2)
            binary = gray
            while gray:
                gray >>= 1
                binary ^= gray
            indices.append(binary)
        return (self.lower_bounds + np.asarray(indices) * self.resolution).tolist()

    convertToReal = convert

    def computeFitness(self, phenotype):
        x = np.asarray(phenotype, dtype=float)
        if (x.shape != (self.dim,) or not np.all(np.isfinite(x))
                or np.any(x < self.lower_bounds) or np.any(x > self.upper_bounds)):
            raise ValueError("Phenotype must be finite and within the shared bounds")
        self.evaluations += 1
        return float(self.problem.compute_fitness(x))

    def evaluate(self, chromosome):
        return self.computeFitness(self.convert(chromosome))

    def isMaxProblem(self):
        return bool(getattr(self.problem, "is_max_problem", lambda: False)())
