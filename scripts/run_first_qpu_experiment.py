#!/usr/bin/env python3
"""
run_first_qpu_experiment.py
===========================
Executes or simulates the single-circuit pilot experiment on IBM Quantum Hardware.

SAFETY:
- Guarded: Runs in offline/dry-run mode by default.
- Will NOT submit QPU jobs unless both --submit-hardware and --confirm-run-qpu are passed.
"""

import sys
import argparse
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from aeqga.steps.encoding.amplitude_encoding import build_amplitude_circuit
    from aeqga.steps.genetic_operators.quantum_gates import apply_crossover_and_mutation
except ImportError:
    from amplitude_encoding import build_amplitude_circuit  # type: ignore
    from quantum_gates import apply_crossover_and_mutation  # type: ignore
from aeqga.execution.ibm_sampler import IBMSamplerExecutor, DryRunExecutor, CheckpointManager


def run_experiment(backend_name: str = "ibm_kingston", shots: int = 1024, submit_hardware: bool = False, confirm: bool = False):
    print("=" * 65)
    print("  AEQGA Pilot Single-Circuit Experiment")
    print("=" * 65)

    ckpt_dir = REPO_ROOT / "outputs" / "json" / "ibm_first_pilot"

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

    # Build 4-qubit circuit from seed 42
    np.random.seed(42)
    import random
    random.seed(42)
    vals = np.random.uniform(60.0, 80.0, 16)
    qc = build_amplitude_circuit(vals)
    apply_crossover_and_mutation(qc, p_c=1.0, p_m=1.0)
    qc.measure_all()

    print(f"Submitting 1 circuit with {shots} shots...")
    counts_list = executor.run([qc], generation=1)
    counts = counts_list[0]

    print(f"\nExecution complete. Non-zero basis states: {len(counts)}/16")
    print(f"Checkpoint saved to {ckpt_dir}/generation_0001.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run first QPU pilot experiment")
    parser.add_argument("--backend", default="ibm_kingston", help="Backend name")
    parser.add_argument("--shots", type=int, default=1024, help="Measurement shots")
    parser.add_argument("--submit-hardware", action="store_true", help="Submit to physical QPU")
    parser.add_argument("--confirm-run-qpu", action="store_true", help="Explicit confirmation for QPU run")
    args = parser.parse_args()
    run_experiment(args.backend, args.shots, args.submit_hardware, args.confirm_run_qpu)
