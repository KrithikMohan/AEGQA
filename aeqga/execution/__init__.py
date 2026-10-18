"""
aeqga.execution
===============
Execution adapters for AEQGA, providing support for:
- Qiskit Aer simulation
- IBM Quantum Hardware via Qiskit IBM Runtime SamplerV2
- Offline dry-run and mock simulation
"""

from .ibm_sampler import (
    BaseExecutor,
    IBMSamplerExecutor,
    DryRunExecutor,
    CheckpointManager,
    CompilationDiagnostics,
    compute_circuit_fingerprint,
    verify_mps_probability_equivalence,
)

__all__ = [
    "BaseExecutor",
    "IBMSamplerExecutor",
    "DryRunExecutor",
    "CheckpointManager",
    "CompilationDiagnostics",
    "compute_circuit_fingerprint",
    "verify_mps_probability_equivalence",
]
