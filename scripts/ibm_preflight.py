#!/usr/bin/env python3
"""
ibm_preflight.py
================
Preflight check for IBM Quantum Hardware credentials and backend availability.

Safety:
- Does NOT submit any quantum circuits or Runtime jobs.
- Queries available backends, qubit counts, and status.
"""

import sys
import argparse
import platform
from aeqga.execution.ibm_sampler import get_provenance_metadata


def run_preflight(backend_name: str | None = None):
    print("=" * 60)
    print("Python:", platform.python_version())
    print("Packages/source:", get_provenance_metadata())
    print("  IBM Quantum Hardware Preflight")
    print("=" * 60)

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print("[ERROR] qiskit-ibm-runtime is not installed.")
        print("Install with: pip install -r requirements-ibm.txt")
        sys.exit(1)

    print("\nAttempting authentication via saved Qiskit credentials...")
    try:
        service = QiskitRuntimeService()
        print("✓ Authenticated successfully with IBM Quantum.")
    except Exception as e:
        print(f"[INFO] Saved credentials unavailable or authentication failed ({type(e).__name__}).")
        print("Configure credentials locally using your installed IBM Runtime SDK; never paste keys into this repository.")
        print("\nExiting preflight without error (offline mode).")
        return

    print("\nQuerying available quantum backends:")
    try:
        backends = service.backends()
        for b in backends:
            status = b.status()
            q_count = b.num_qubits
            name = b.name
            op_status = "operational" if status.operational else "not operational"
            pending = status.pending_jobs
            print(f"  - {name:<18} ({q_count:>3} qubits, {op_status}, {pending} jobs in queue)")

        if backend_name:
            target = service.backend(backend_name)
            print(f"\nTarget backend '{backend_name}' status: {target.status().status_msg}")
    except Exception as e:
        print(f"Could not retrieve backend list: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IBM Quantum preflight check")
    parser.add_argument("--backend", default=None, help="Target backend name to inspect")
    args = parser.parse_args()
    run_preflight(args.backend)
