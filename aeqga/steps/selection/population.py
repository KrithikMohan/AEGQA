"""Classical selection, duplication and fresh draws (Algorithm 1 lines 5–7)."""
import numpy as np


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
