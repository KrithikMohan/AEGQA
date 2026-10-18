#!/usr/bin/env python3
"""
run_ibm_generation.py
=====================
Executes or simulates a full 1-generation AEQGA experiment with 4 circuits.

SAFETY:
- Guarded: Runs in offline/dry-run mode by default.
- Will NOT submit QPU jobs unless both --submit-hardware and --confirm-run-qpu are passed.
"""

import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pantheon_problem import PantheonPlusProblem
from aeqga_algorithm import AEQGAParameters, run_aeqga_dual
from aeqga.execution.ibm_sampler import IBMSamplerExecutor, DryRunExecutor


def run_generation(backend_name: str = "ibm_kingston", shots: int = 1024, submit_hardware: bool = False, confirm: bool = False):
    print("=" * 65)
    print("  AEQGA Generation 1 Experiment (4 Circuits Batch)")
    print("=" * 65)

    ckpt_dir = REPO_ROOT / "outputs" / "json" / "ibm_generation_01"

    if submit_hardware:
        if not confirm:
            print("[ABORT] Hardware submission requested without --confirm-run-qpu confirmation flag.")
            print("To protect QPU allocation, submission was blocked.")
            sys.exit(1)
        print(f"[ACTION] Initializing Qiskit Runtime Sampler on {backend_name}...")
        executor = IBMSamplerExecutor(
            backend_name=backend_name,
            shots=shots,
            optimization_level=1,
            seed_transpiler=42,
            checkpoint_dir=ckpt_dir,
            mode="runtime",
        )
    else:
        print("[SAFE] Running in OFFLINE DryRun simulation mode (no hardware jobs submitted).")
        executor = DryRunExecutor(
            shots=shots,
            backend_name=f"{backend_name}_dry_run",
            checkpoint_dir=ckpt_dir,
            seed_simulator=42,
        )

    problem = PantheonPlusProblem()
    params = AEQGAParameters(
        pop_size=32,
        max_gen=1,
        p_cross=0.5,
        p_mut=0.5,
        num_shots=shots,
        verbose=False,
        progress_bar=False,
    )

    print("Running AEQGA Generation 1 (4 circuits: [4, 3, 4, 3])...")
    g_best, pop_evol, bests_log = run_aeqga_dual(
        problem,
        params,
        hardware_executor=executor,
        seed=42,
        quiet=False,
    )

    print(f"\nCompleted Generation 1.")
    print(f"Best solution: H0={g_best.x[0]:.4f}, Om={g_best.x[1]:.4f}, chi2={g_best.fitness:.4f}")
    print(f"Checkpoint directory: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AEQGA generation experiment")
    parser.add_argument("--backend", default="ibm_kingston", help="Backend name")
    parser.add_argument("--shots", type=int, default=1024, help="Measurement shots")
    parser.add_argument("--submit-hardware", action="store_true", help="Submit to physical QPU")
    parser.add_argument("--confirm-run-qpu", action="store_true", help="Explicit confirmation for QPU run")
    args = parser.parse_args()
    run_generation(args.backend, args.shots, args.submit_hardware, args.confirm_run_qpu)
