"""Offline preparation and backend-noise benchmark. No hardware submissions."""
import argparse
import json
from pathlib import Path
from qiskit.providers.fake_provider import GenericBackendV2
from aeqga.analysis.hardware_validation import benchmark_circuits
from aeqga.execution.ibm_sampler import CheckpointManager, get_provenance_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", help="Read actual backend properties using saved IBM credentials")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/json/hardware_offline_benchmark.json"))
    args = parser.parse_args()
    if args.backend:
        from qiskit_ibm_runtime import QiskitRuntimeService
        backend = QiskitRuntimeService().backend(args.backend)
    else:
        backend = GenericBackendV2(4, basis_gates=["id", "rz", "sx", "x", "cx"], seed=args.seed)
    report = benchmark_circuits(backend, shots=args.shots, seed=args.seed)
    report["synthetic_backend"] = args.backend is None
    report["versions"] = get_provenance_metadata()
    if args.backend:
        properties = backend.properties()
        report["calibration"] = json.loads(json.dumps(properties.to_dict(), default=str))
    CheckpointManager.atomic_write(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
