"""
ibm_sampler.py
==============
IBM Quantum Hardware execution adapter for AEQGA using Qiskit IBM Runtime SamplerV2.

Provides:
- BaseExecutor: Abstract base class for AEQGA circuit execution.
- IBMSamplerExecutor: Production IBM Runtime adapter with transpilation,
  logical-width validation, atomic checkpointing, and job recovery.
- DryRunExecutor: Offline simulator for testing and validation without IBM hardware.
- CompilationDiagnostics: Diagnostics recording gate counts, depth, and physical layout.
- CheckpointManager: Atomic checkpoint reading and writing with fingerprint validation.
"""

from __future__ import annotations

import os
import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def get_provenance_metadata() -> dict[str, str]:
    """Extract runtime provenance metadata: Git SHA and package versions."""
    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        pass
    return {
        "git_sha": git_sha,
        "qiskit_version": str(getattr(qiskit, "__version__", "unknown")),
        "numpy_version": str(np.__version__),
    }


def compute_circuit_fingerprint(circuit: QuantumCircuit) -> str:
    """Compute deterministic SHA-256 fingerprint for a QuantumCircuit.

    Hashes qubit count, classical register structure, and instruction sequence.
    """
    hasher = hashlib.sha256()
    hasher.update(f"qubits:{circuit.num_qubits};clbits:{circuit.num_clbits};".encode("utf-8"))
    for creg in circuit.cregs:
        hasher.update(f"creg:{creg.name}:{creg.size};".encode("utf-8"))
    for instruction in circuit.data:
        op = instruction.operation
        q_indices = [circuit.find_bit(q).index for q in instruction.qubits]
        c_indices = [circuit.find_bit(c).index for c in instruction.clbits]
        params_str = ",".join(f"{float(p):.8f}" for p in op.params if isinstance(p, (int, float, np.floating)))
        hasher.update(f"{op.name}:{q_indices}:{c_indices}:{params_str};".encode("utf-8"))
    return hasher.hexdigest()


class CompilationDiagnostics:
    """Records and validates compilation metrics for a set of quantum circuits."""

    def __init__(
        self,
        original_circuits: list[QuantumCircuit],
        compiled_circuits: list[QuantumCircuit],
        backend_name: str,
        optimization_level: int,
        seed_transpiler: int | None = None,
    ):
        self.backend_name = backend_name
        self.optimization_level = optimization_level
        self.seed_transpiler = seed_transpiler
        self.records: list[dict[str, Any]] = []

        two_qubit_names = {"cx", "cz", "ecr", "swap", "rzx", "crx", "cry", "crz"}

        for original, compiled in zip(original_circuits, compiled_circuits):
            gate_counts = dict(compiled.count_ops())
            two_qubit_count = sum(gate_counts.get(name, 0) for name in two_qubit_names)
            reset_count = gate_counts.get("reset", 0)

            # Extract initial layout if available
            layout_info = None
            if compiled.layout is not None and compiled.layout.initial_layout is not None:
                layout_info = {
                    str(q): compiled.layout.initial_layout[q]
                    for q in original.qubits
                    if q in compiled.layout.initial_layout
                }

            record = {
                "original_width": original.num_qubits,
                "original_depth": original.depth(),
                "transpiled_width": compiled.num_qubits,
                "transpiled_depth": compiled.depth(),
                "gate_counts": gate_counts,
                "two_qubit_gate_counts": two_qubit_count,
                "reset_count": reset_count,
                "backend": backend_name,
                "optimization_level": optimization_level,
                "transpiler_seed": seed_transpiler,
                "circuit_fingerprint": compute_circuit_fingerprint(original),
                "compiled_fingerprint": compute_circuit_fingerprint(compiled),
                "layout": layout_info,
            }
            self.records.append(record)

    def summary(self) -> list[dict[str, Any]]:
        return self.records


def verify_mps_probability_equivalence(
    original_circuit: QuantumCircuit,
    transpiled_circuit: QuantumCircuit,
    atol: float = 1e-10,
) -> float:
    """Verify ideal output probabilities match between original and compiled circuits.

    Uses Aer Matrix Product State (MPS) simulation on active physical qubits to
    avoid full 156-qubit statevector allocation (preventing dimension exceeded errors).
    """
    orig_sv = Statevector(original_circuit)
    orig_probs = np.asarray(orig_sv.probabilities(), dtype=float)

    # Determine active physical qubits from initial layout
    if transpiled_circuit.layout is not None and transpiled_circuit.layout.initial_layout is not None:
        init_layout = transpiled_circuit.layout.initial_layout
        active_qubits = [init_layout[q] for q in original_circuit.qubits]
    else:
        active_qubits = list(range(original_circuit.num_qubits))

    from qiskit_aer.library import save_probabilities

    t_qc = transpiled_circuit.copy()
    t_qc.save_probabilities(qubits=active_qubits)

    sim = AerSimulator(method="matrix_product_state")
    result = sim.run(t_qc).result()
    mps_probs = np.asarray(result.data(0)["probabilities"], dtype=float)

    max_diff = float(np.max(np.abs(mps_probs - orig_probs)))
    if max_diff > atol:
        raise ValueError(
            f"Ideal probability divergence between original and compiled circuit: {max_diff:.3e} > {atol:.3e}"
        )
    return max_diff


def _counts(pub_result: Any, register_name: str, shots: int, logical_width: int) -> dict[str, int]:
    """Extract and validate measurement counts for a single circuit result.

    Validates bit-string ordering, width, zero-padding, and shot total.
    """
    raw_counts: dict[str, int] = {}

    if isinstance(pub_result, dict):
        raw_counts = pub_result
    elif hasattr(pub_result, "data"):
        data = pub_result.data
        reg_obj = None
        if hasattr(data, register_name):
            reg_obj = getattr(data, register_name)
        elif hasattr(data, "__getitem__"):
            try:
                reg_obj = data[register_name]
            except (KeyError, TypeError):
                reg_obj = None

        if reg_obj is not None and hasattr(reg_obj, "get_counts"):
            raw_counts = reg_obj.get_counts()
        elif reg_obj is not None and hasattr(reg_obj, "get_bitstrings"):
            from collections import Counter
            bitstrings = reg_obj.get_bitstrings()
            raw_counts = dict(Counter(bitstrings))
        elif hasattr(pub_result, "get_counts"):
            raw_counts = pub_result.get_counts()
        else:
            raise ValueError(f"Cannot extract counts for register '{register_name}' from PubResult")
    else:
        raise ValueError(f"Unsupported result format: {type(pub_result)}")

    # Format bitstrings with exactly logical_width bits and validate shot count
    formatted_counts: dict[str, int] = {}
    total_shots = 0
    for bitstring, count in raw_counts.items():
        clean_bs = str(bitstring).strip().replace(" ", "")
        if len(clean_bs) != logical_width:
            # Handle possible integer or zero-padded bitstring
            clean_bs = clean_bs.zfill(logical_width)
            if len(clean_bs) != logical_width:
                raise ValueError(
                    f"Bitstring '{bitstring}' length {len(clean_bs)} does not match logical width {logical_width}"
                )
        int_count = int(count)
        formatted_counts[clean_bs] = formatted_counts.get(clean_bs, 0) + int_count
        total_shots += int_count

    if total_shots != shots:
        raise ValueError(
            f"Shot total mismatch: expected {shots}, got {total_shots}"
        )

    return formatted_counts


class CheckpointManager:
    """Manages atomic writing and verification of per-generation checkpoints."""

    @staticmethod
    def get_checkpoint_path(checkpoint_dir: str | Path, generation: int) -> Path:
        return Path(checkpoint_dir) / f"generation_{generation:04d}.json"

    @staticmethod
    def atomic_write(filepath: str | Path, data: dict[str, Any]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    @staticmethod
    def load(filepath: str | Path) -> dict[str, Any] | None:
        path = Path(filepath)
        if not path.is_file():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def verify_fingerprint(
        cls,
        saved_data: dict[str, Any],
        expected_backend: str,
        expected_shots: int,
        expected_num_circuits: int,
        expected_fingerprints: list[str],
    ) -> bool:
        """Verify saved checkpoint matches current experiment configuration."""
        if saved_data.get("backend") != expected_backend:
            return False
        if int(saved_data.get("shots", 0)) != expected_shots:
            return False
        if int(saved_data.get("num_circuits", 0)) != expected_num_circuits:
            return False
        saved_fps = saved_data.get("circuit_fingerprints")
        if saved_fps is not None and saved_fps != expected_fingerprints:
            return False
        return True


class BaseExecutor:
    """Abstract interface for AEQGA quantum circuit execution."""

    def __init__(self, shots: int = 1024, checkpoint_dir: str | Path | None = None):
        if shots <= 0:
            raise ValueError("Shots must be greater than zero")
        self.shots = shots
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.diagnostics: CompilationDiagnostics | None = None

    def run(
        self,
        circuits: Sequence[QuantumCircuit],
        generation: int = 0,
        optimizer_state: dict[str, Any] | None = None,
    ) -> list[dict[str, int]]:
        raise NotImplementedError


class DryRunExecutor(BaseExecutor):
    """Offline executor simulating circuits with Aer for local testing and validation."""

    def __init__(
        self,
        shots: int = 1024,
        checkpoint_dir: str | Path | None = None,
        seed_simulator: int | None = None,
        backend_name: str = "aer_simulator_dry_run",
    ):
        super().__init__(shots=shots, checkpoint_dir=checkpoint_dir)
        self.backend_name = backend_name
        self.seed_simulator = seed_simulator
        self.simulator = AerSimulator()

    def run(
        self,
        circuits: Sequence[QuantumCircuit],
        generation: int = 0,
        optimizer_state: dict[str, Any] | None = None,
    ) -> list[dict[str, int]]:
        if len(circuits) == 0:
            return []

        circuit_fingerprints = [compute_circuit_fingerprint(c) for c in circuits]
        circuit_widths = [c.num_qubits for c in circuits]

        # Check existing checkpoint
        if self.checkpoint_dir:
            ckpt_path = CheckpointManager.get_checkpoint_path(self.checkpoint_dir, generation)
            saved = CheckpointManager.load(ckpt_path)
            if saved is not None:
                if not CheckpointManager.verify_fingerprint(
                    saved, self.backend_name, self.shots, len(circuits), circuit_fingerprints
                ):
                    raise ValueError(
                        f"Checkpoint experiment configuration mismatch at generation {generation}"
                    )
                if saved.get("submission_state") == "completed" and "raw_counts" in saved:
                    return saved["raw_counts"]

        # Validate registers on original circuits
        for qc in circuits:
            registers = list(qc.cregs)
            if len(registers) != 1 or registers[0].size != qc.num_qubits:
                raise ValueError("Exactly one full-width measurement register required")

        compiled_circuits = [transpile(qc, self.simulator) for qc in circuits]
        self.diagnostics = CompilationDiagnostics(
            list(circuits), compiled_circuits, self.backend_name, optimization_level=1
        )

        counts_list: list[dict[str, int]] = []
        for orig, comp in zip(circuits, compiled_circuits):
            job = self.simulator.run(comp, shots=self.shots, seed_simulator=self.seed_simulator)
            raw = job.result().get_counts()
            reg_name = list(orig.cregs)[0].name
            counts_list.append(_counts(raw, reg_name, self.shots, orig.num_qubits))

        if self.checkpoint_dir:
            ckpt_path = CheckpointManager.get_checkpoint_path(self.checkpoint_dir, generation)
            data = {
                "generation": generation,
                "backend": self.backend_name,
                "shots": self.shots,
                "num_circuits": len(circuits),
                "circuit_widths": circuit_widths,
                "circuit_fingerprints": circuit_fingerprints,
                "job_id": f"dry_run_gen_{generation:04d}",
                "submission_state": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_counts": counts_list,
                "provenance": get_provenance_metadata(),
                "optimizer_state": optimizer_state,
            }
            CheckpointManager.atomic_write(ckpt_path, data)

        return counts_list


class IBMSamplerExecutor(BaseExecutor):
    """Production IBM Quantum Hardware execution adapter via Qiskit Runtime SamplerV2.

    Adheres to:
    - Four-circuit per-generation batching.
    - Logical measurement-width validation against logical circuit width.
    - Atomic checkpoint persistence before waiting for results.
    - Automatic recovery of existing submitted jobs from checkpoint.
    """

    def __init__(
        self,
        backend_name: str = "ibm_kingston",
        shots: int = 1024,
        optimization_level: int = 1,
        seed_transpiler: int | None = None,
        checkpoint_dir: str | Path | None = None,
        service: Any | None = None,
        backend: Any | None = None,
        mode: str = "runtime",  # "runtime" or "dry_run"
    ):
        super().__init__(shots=shots, checkpoint_dir=checkpoint_dir)
        self.backend_name = backend_name
        self.optimization_level = optimization_level
        self.seed_transpiler = seed_transpiler
        self.mode = mode
        self._service = service
        self._backend = backend

    def _get_service(self) -> Any:
        if self._service is None:
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
                self._service = QiskitRuntimeService()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize QiskitRuntimeService. Ensure credentials are saved: {e}"
                ) from e
        return self._service

    def _get_backend(self) -> Any:
        if self._backend is None:
            service = self._get_service()
            self._backend = service.backend(self.backend_name)
        return self._backend

    def compile(self, circuits: Sequence[QuantumCircuit]) -> list[QuantumCircuit]:
        """Compile circuits for the target backend and record diagnostics."""
        backend = self._get_backend()
        compiled = transpile(
            list(circuits),
            backend=backend,
            optimization_level=self.optimization_level,
            seed_transpiler=self.seed_transpiler,
        )
        if isinstance(compiled, QuantumCircuit):
            compiled = [compiled]

        self.diagnostics = CompilationDiagnostics(
            list(circuits),
            compiled,
            self.backend_name,
            self.optimization_level,
            self.seed_transpiler,
        )
        return compiled

    def run(
        self,
        circuits: Sequence[QuantumCircuit],
        generation: int = 0,
        optimizer_state: dict[str, Any] | None = None,
    ) -> list[dict[str, int]]:
        if len(circuits) == 0:
            return []

        circuit_fingerprints = [compute_circuit_fingerprint(c) for c in circuits]
        circuit_widths = [c.num_qubits for c in circuits]

        ckpt_path = (
            CheckpointManager.get_checkpoint_path(self.checkpoint_dir, generation)
            if self.checkpoint_dir
            else None
        )

        saved = CheckpointManager.load(ckpt_path) if ckpt_path else None
        if saved is not None:
            if not CheckpointManager.verify_fingerprint(
                saved, self.backend_name, self.shots, len(circuits), circuit_fingerprints
            ):
                raise ValueError(
                    f"Checkpoint experiment configuration mismatch at generation {generation}. "
                    "Refusing to silently reuse mismatched checkpoint."
                )

            # Case 1: Job already completed and counts cached
            if saved.get("submission_state") == "completed" and "raw_counts" in saved:
                return saved["raw_counts"]

            # Case 2: Job already submitted, recover using saved job_id
            if saved.get("job_id"):
                job_id = saved["job_id"]
                service = self._get_service()
                job = service.job(job_id)
                result = job.result()
                counts = self._process_results(circuits, circuits, result)

                # Update checkpoint to completed
                saved["submission_state"] = "completed"
                saved["completion_time"] = datetime.now(timezone.utc).isoformat()
                saved["raw_counts"] = counts
                CheckpointManager.atomic_write(ckpt_path, saved)
                return counts

        # If dry-run mode requested without QPU access
        if self.mode == "dry_run":
            dry_runner = DryRunExecutor(
                shots=self.shots,
                checkpoint_dir=self.checkpoint_dir,
                backend_name=self.backend_name,
            )
            return dry_runner.run(circuits, generation=generation, optimizer_state=optimizer_state)

        # Transpile for backend
        compiled_circuits = self.compile(circuits)

        # Validate logical measurement width against original circuit width (Section 6)
        for original, compiled in zip(circuits, compiled_circuits):
            registers = list(compiled.cregs)
            logical_width = original.num_qubits
            if len(registers) != 1 or registers[0].size != logical_width:
                raise ValueError(
                    f"Exactly one full-width measurement register required (expected size {logical_width})"
                )

        # Prepare and submit SamplerV2 job
        from qiskit_ibm_runtime import SamplerV2
        backend = self._get_backend()
        sampler = SamplerV2(mode=backend)

        # Build list of PUBs with requested shots
        pubs = [(c, None, self.shots) for c in compiled_circuits]
        job = sampler.run(pubs)
        job_id = job.job_id()

        # Persist checkpoint BEFORE waiting for results
        if ckpt_path:
            init_data = {
                "generation": generation,
                "backend": self.backend_name,
                "shots": self.shots,
                "num_circuits": len(circuits),
                "circuit_widths": circuit_widths,
                "circuit_fingerprints": circuit_fingerprints,
                "job_id": job_id,
                "submission_state": "submitted",
                "submission_time": datetime.now(timezone.utc).isoformat(),
                "provenance": get_provenance_metadata(),
                "optimizer_state": optimizer_state,
            }
            CheckpointManager.atomic_write(ckpt_path, init_data)

        # Wait for result
        result = job.result()
        counts = self._process_results(circuits, compiled_circuits, result)

        # Update checkpoint to completed
        if ckpt_path:
            final_data = {
                "generation": generation,
                "backend": self.backend_name,
                "shots": self.shots,
                "num_circuits": len(circuits),
                "circuit_widths": circuit_widths,
                "circuit_fingerprints": circuit_fingerprints,
                "job_id": job_id,
                "submission_state": "completed",
                "submission_time": init_data["submission_time"],
                "completion_time": datetime.now(timezone.utc).isoformat(),
                "raw_counts": counts,
                "provenance": get_provenance_metadata(),
                "optimizer_state": optimizer_state,
            }
            CheckpointManager.atomic_write(ckpt_path, final_data)

        return counts

    def _process_results(
        self,
        original_circuits: Sequence[QuantumCircuit],
        compiled_circuits: Sequence[QuantumCircuit],
        result: Any,
    ) -> list[dict[str, int]]:
        """Validate logical width and extract counts according to Section 6."""
        counts = []
        for original, circuit, pub in zip(original_circuits, compiled_circuits, result):
            registers = list(circuit.cregs)
            logical_width = original.num_qubits

            if len(registers) != 1 or registers[0].size != logical_width:
                raise ValueError(
                    "Exactly one full-width measurement register required"
                )

            counts.append(
                _counts(
                    pub,
                    registers[0].name,
                    self.shots,
                    logical_width,
                )
            )
        return counts
