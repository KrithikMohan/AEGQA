"""
aeqga.analysis
==============
Scientific analysis and verification tools for AEQGA:
- Controlled same-circuit comparison pipeline
- Finite-shot Monte Carlo ensembles
- Hardware vs ideal finite-shot distribution metrics
"""

from .circuit_comparison import (
    generate_frozen_generation_circuits,
    compute_circuit_ideal_probabilities,
    run_finite_shot_ensemble,
    compare_counts_against_ensemble,
    total_variation_distance,
)

__all__ = [
    "generate_frozen_generation_circuits",
    "compute_circuit_ideal_probabilities",
    "run_finite_shot_ensemble",
    "compare_counts_against_ensemble",
    "total_variation_distance",
]
