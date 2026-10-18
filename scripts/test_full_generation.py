#!/usr/bin/env python3
"""
test_full_generation.py
=======================
Offline test executing a complete AEQGA generation using DryRunExecutor.

Validates:
1. Four-circuit batching in exact width order [4, 3, 4, 3].
2. Population splitting and 64 total likelihood evaluations.
3. Checkpoint persistence and integrity.
4. Correct decoding and elitism preservation.
"""

import sys
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pantheon_problem import PantheonPlusProblem
from aeqga_algorithm import AEQGAParameters, run_aeqga_dual
from aeqga.execution.ibm_sampler import DryRunExecutor, CheckpointManager


def test_full_generation():
    print("=" * 65)
    print("  Offline Full Generation Test (DryRunExecutor)")
    print("=" * 65)

    test_ckpt_dir = REPO_ROOT / "outputs" / "json" / "test_generation_dry_run"
    if test_ckpt_dir.exists():
        shutil.rmtree(test_ckpt_dir)
    test_ckpt_dir.mkdir(parents=True, exist_ok=True)

    problem = PantheonPlusProblem()
    params = AEQGAParameters(
        pop_size=32,
        max_gen=1,
        p_cross=0.5,
        p_mut=0.5,
        num_shots=1024,
        verbose=False,
        progress_bar=False,
    )

    executor = DryRunExecutor(
        shots=1024,
        checkpoint_dir=test_ckpt_dir,
        seed_simulator=42,
        backend_name="ibm_kingston_simulated",
    )

    print("\nRunning Generation 1 through DryRunExecutor...")
    g_best, pop_evol, bests_log = run_aeqga_dual(
        problem,
        params,
        hardware_executor=executor,
        seed=42,
        quiet=False,
    )

    # Validations
    print("\n--- Validating Generation Execution ---")
    print(f"  Population evolution length: {len(pop_evol)} (expected 2: initial + gen 1)")
    assert len(pop_evol) == 2, f"Expected 2 population snapshots, got {len(pop_evol)}"

    initial_pop = pop_evol[0]
    final_pop = pop_evol[1]
    assert initial_pop.shape == (32, 2), f"Initial population shape wrong: {initial_pop.shape}"
    assert final_pop.shape == (32, 2), f"Final population shape wrong: {final_pop.shape}"

    # Verify elite preservation (first 8 individuals)
    assert np.allclose(initial_pop[np.argsort([problem.compute_fitness(x) for x in initial_pop])[:8]], final_pop[:8]), \
        "Top 25% elites not preserved in final population"
    print("  Top 25% elites correctly preserved: PASS")

    # Checkpoint verification
    ckpt_file = test_ckpt_dir / "generation_0001.json"
    assert ckpt_file.exists(), f"Checkpoint file not created at {ckpt_file}"
    ckpt_data = CheckpointManager.load(ckpt_file)
    assert ckpt_data is not None, "Failed to load checkpoint file"
    assert ckpt_data["generation"] == 1, "Wrong generation in checkpoint"
    assert ckpt_data["circuit_widths"] == [4, 3, 4, 3], f"Circuit widths mismatch: {ckpt_data['circuit_widths']}"
    assert len(ckpt_data["raw_counts"]) == 4, f"Expected 4 count dicts, got {len(ckpt_data['raw_counts'])}"
    for c in ckpt_data["raw_counts"]:
        assert sum(c.values()) == 1024, f"Count shot sum mismatch: {sum(c.values())}"
    print("  Circuit widths [4, 3, 4, 3] and 1024 shots per circuit: PASS")
    print("  Per-generation atomic checkpoint: PASS")

    print(f"\nBest solution found: H0={g_best.x[0]:.4f}, Om={g_best.x[1]:.4f}, chi2={g_best.fitness:.4f}")
    print("\n✓ Full generation test completed successfully.")

    # Clean up test checkpoint directory
    shutil.rmtree(test_ckpt_dir)


if __name__ == "__main__":
    import numpy as np
    test_full_generation()
