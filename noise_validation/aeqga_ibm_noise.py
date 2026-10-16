"""AEQGA with IBM public calibration snapshots and scheduled native circuits.

Only the selected connected physical-qubit subgraph is simulated. Snapshot JSON
comes unchanged from qiskit-ibm-runtime 0.50.0; importing its cloud client is not
required. Gate channels use Qiskit Aer's official NoiseModel.from_backend.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.circuit import Delay, Measure, Parameter, Reset
from qiskit.circuit.library import CZGate, IGate, RZGate, SXGate, XGate
from qiskit.providers import BackendV2, Options, QubitProperties
from qiskit.transpiler import InstructionProperties, Target
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError

from aeqga import AEQGA
from aeqga_parallel import circuit_groups, connected_placements, pack_circuits

DEVICES = ("kingston", "fez", "marrakesh")
SNAPSHOT_VERSION = "0.50.0"
DEFAULT_SNAPSHOTS = Path(__file__).resolve().parent / "ibm_backend_snapshots"
NATIVE_GATES = {"id": IGate(), "rz": RZGate(Parameter("theta")),
                "sx": SXGate(), "x": XGate(), "cz": CZGate()}


def named_values(entries):
    """Convert the explicitly unit-tagged calibration values to SI units."""
    factors = {"": 1., "s": 1., "ms": 1e-3, "us": 1e-6, "ns": 1e-9,
               "Hz": 1., "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
    return {v["name"]: v["value"] * factors[v.get("unit", "")] for v in entries}


class SnapshotBackend(BackendV2):
    """Exact selected Target entries; physical IDs are preserved in metadata."""

    def __init__(self, device, width=5, snapshot_root=DEFAULT_SNAPSHOTS, *, physical_qubits=None,
                 include_unusable_wires=False):
        if device not in DEVICES:
            raise ValueError("Use kingston/fez/marrakesh.")
        root = Path(snapshot_root) / device
        self.configuration_json = json.loads((root / f"conf_{device}.json").read_text())
        self.properties_json = json.loads((root / f"props_{device}.json").read_text())
        if physical_qubits is None:
            AEQGA._positive_integer(width, "width")
            if width > self.configuration_json["n_qubits"]:
                raise ValueError("Requested width exceeds the snapshot device capacity.")
            physical_qubits = list(range(width))
        self.physical_ids = list(physical_qubits)
        if (not self.physical_ids or len(set(self.physical_ids)) != len(self.physical_ids)
                or any(isinstance(q, bool) or not isinstance(q, (int, np.integer))
                       or not 0 <= q < self.configuration_json["n_qubits"] for q in self.physical_ids)):
            raise ValueError("physical_qubits must contain distinct valid device qubit IDs.")
        width = len(self.physical_ids)
        virtual = {physical: local for local, physical in enumerate(self.physical_ids)}
        self.device = device
        super().__init__(name=f"snapshot_{device}_{width}q", backend_version=SNAPSHOT_VERSION)
        self.calibrations = [named_values(self.properties_json["qubits"][i])
                             for i in self.physical_ids]
        required = {"T1", "T2", "readout_length", "readout_error",
                    "prob_meas1_prep0", "prob_meas0_prep1"}
        self.usable_indices = {q for q, p in enumerate(self.calibrations)
            if required.issubset(p) and p.get("operational", True) != 0
            and all(np.isfinite(p[k]) for k in required) and p["T1"] > 0 and p["T2"] > 0}
        self.unusable_physical_ids = [physical for q, physical in enumerate(self.physical_ids)
                                      if q not in self.usable_indices]
        if self.unusable_physical_ids and not include_unusable_wires:
            raise ValueError(f"Missing/invalid calibration for qubits {self.unusable_physical_ids}.")
        # IBM configuration JSON stores dt in ns; Target uses seconds.
        target = Target(num_qubits=width, dt=self.configuration_json["dt"] * 1e-9,
                        qubit_properties=[QubitProperties(t1=p.get("T1"), t2=p.get("T2"),
                                                          frequency=p.get("frequency"))
                                          for p in self.calibrations],
                        **self.configuration_json.get("timing_constraints", {}))
        supported = set(self.configuration_json["basis_gates"])
        if supported != set(NATIVE_GATES):
            raise ValueError(f"Unexpected snapshot basis: {supported}")
        entries = {name: {} for name in NATIVE_GATES}
        edges = {tuple(e) for e in self.configuration_json["coupling_map"]}
        for g in self.properties_json["gates"]:
            qargs = tuple(g["qubits"])
            if not set(qargs).issubset(self.physical_ids) or g["gate"] not in entries:
                continue
            if not all(virtual[q] in self.usable_indices for q in qargs):
                continue
            values = named_values(g["parameters"])
            if not values.get("operational", True):
                continue
            if len(qargs) == 2 and qargs not in edges:
                raise AssertionError("Calibration edge is absent from the coupling map.")
            entries[g["gate"]][tuple(virtual[q] for q in qargs)] = InstructionProperties(
                duration=values["gate_length"], error=values["gate_error"])
        # Require a connected calibrated subgraph, including non-prefix placements.
        seen, pending = {0}, [0]
        while pending:
            q = pending.pop()
            for edge in entries["cz"]:
                if q in edge:
                    for neighbor in edge:
                        if neighbor not in seen:
                            seen.add(neighbor)
                            pending.append(neighbor)
        if len(seen) != width and not include_unusable_wires:
            raise ValueError("The selected physical qubits are not connected by calibrated CZ gates.")
        for name, operation in NATIVE_GATES.items():
            target.add_instruction(operation, entries[name])
        target.add_instruction(Measure(), {
            (q,): InstructionProperties(duration=p["readout_length"], error=p["readout_error"])
            for q, p in enumerate(self.calibrations) if q in self.usable_indices})
        # Reset has a reported duration, but no assignment/infidelity calibration.
        # Aer uses ideal reset plus duration-based relaxation at zero temperature.
        reset_times = {g["qubits"][0]: named_values(g["parameters"])["gate_length"]
                       for g in self.properties_json["gates"] if g["gate"] == "reset"}
        target.add_instruction(Reset(), {(q,): InstructionProperties(duration=reset_times[physical])
                                         for q, physical in enumerate(self.physical_ids)
                                         if q in self.usable_indices})
        target.add_instruction(Delay(Parameter("duration")), {(q,): None for q in self.usable_indices})
        self._target = target

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)

    def run(self, run_input, **options):
        raise NotImplementedError("Use the explicitly configured Aer simulator.")

    def readout_matrix(self, physical_qubit):
        p = self.calibrations[physical_qubit]
        p10, p01 = p["prob_meas1_prep0"], p["prob_meas0_prep1"]
        return np.array([[1-p10, p10], [p01, 1-p01]])


def device_noise(backend, *, readout=True):
    # Target's scalar readout_error is symmetric. Preserve the more detailed
    # asymmetric assignment probabilities supplied in the same IBM snapshot.
    noise = NoiseModel.from_backend(backend, gate_error=True, readout_error=False,
                                    thermal_relaxation=True, temperature=0.)
    if readout:
        for q in sorted(backend.usable_indices):
            noise.add_readout_error(ReadoutError(backend.readout_matrix(q)), [q])
    if not noise._custom_noise_passes:
        raise AssertionError("Scheduled delay relaxation is missing.")
    return noise


def simulator(backend, mode="noisy", *, readout=True, method="density_matrix"):
    if mode not in ("ideal", "noisy"):
        raise ValueError("mode must be ideal or noisy")
    noise = device_noise(backend, readout=readout) if mode == "noisy" else None
    return AerSimulator(method=method, noise_model=noise,
                        max_parallel_threads=1, max_parallel_experiments=1)


def compile_native(circuit, backend, seed=12345, *, routing_method="basic",
                   optimization_level=1, layout_method=None):
    """Fixed initial layout, deterministic routing, native decomposition + ALAP."""
    if circuit.num_qubits != backend.num_qubits:
        raise ValueError("Compile on exactly the circuit-width physical prefix.")
    compiled = transpile(circuit, target=backend.target,
                         initial_layout=None if layout_method else list(range(circuit.num_qubits)),
                         layout_method=layout_method,
                         routing_method=routing_method, optimization_level=optimization_level,
                         scheduling_method="alap", seed_transpiler=seed, num_processes=1)
    for inst in compiled.data:
        if inst.operation.name == "barrier":
            continue
        qargs = tuple(compiled.find_bit(q).index for q in inst.qubits)
        if not backend.target.instruction_supported(operation_name=inst.operation.name, qargs=qargs):
            raise AssertionError(f"Non-native instruction: {inst.operation.name}{qargs}")
    if compiled.num_qubits != circuit.num_qubits:
        raise AssertionError("Unexpected ancillary qubits.")
    compiled.metadata = {**(compiled.metadata or {}), "transpiler_seed": seed}
    return compiled


def measurement_order(circuit):
    """Physical qubits in classical-bit order, including final routing swaps."""
    order = [None] * circuit.num_clbits
    for inst in circuit.data:
        if inst.operation.name == "measure":
            order[circuit.find_bit(inst.clbits[0]).index] = circuit.find_bit(inst.qubits[0]).index
    if len(order) != circuit.num_qubits or set(order) != set(range(circuit.num_qubits)):
        raise AssertionError("Expected one final measurement per active physical qubit.")
    return order


def in_classical_order(probabilities, physical_order):
    out = np.zeros_like(probabilities)
    for state, value in enumerate(probabilities):
        measured = sum(((state >> q) & 1) << c for c, q in enumerate(physical_order))
        out[measured] = value
    return out


def asymmetric_readout(probabilities, backend, physical_order):
    out = np.asarray(probabilities, dtype=float).copy()
    for bit, physical_qubit in enumerate(physical_order):
        old = out.copy()
        matrix = backend.readout_matrix(physical_qubit)
        for zero in range(len(out)):
            if zero & (1 << bit):
                continue
            one = zero | (1 << bit)
            out[zero] = old[zero]*matrix[0, 0] + old[one]*matrix[1, 0]
            out[one] = old[zero]*matrix[0, 1] + old[one]*matrix[1, 1]
    return out


class RecordingSimulator:
    """Records the actual seeds, resources and every measured histogram."""
    def __init__(self, backend, records):
        self.backend, self.records = backend, records

    def run(self, circuits, **options):
        result = self.backend.run(circuits, **options).result()
        if not result.success:
            raise RuntimeError(result.status)
        generation = 1 + len(self.records) // 6
        for index, circuit in enumerate(circuits):
            self.records.append({"generation": generation, "circuit_index": index,
                "parameter_index": index // 2, "branch": "local" if index % 2 == 0 else "global",
                "qubits": circuit.num_qubits, "shots": options["shots"],
                "sampling_seed": options["seed_simulator"], "depth": circuit.depth(),
                "transpiler_seed": circuit.metadata["transpiler_seed"],
                "gate_counts": dict(circuit.count_ops()),
                "duration_dt": int(circuit.duration),
                "measurement_physical_order": measurement_order(circuit),
                "counts": dict(result.get_counts(index))})

        class FinishedJob:
            def result(self):
                return result
        return FinishedJob()


class IBMNoiseAEQGA(AEQGA):
    def __init__(self, *args, device, mode="noisy", snapshot_root=DEFAULT_SNAPSHOTS, **kwargs):
        super().__init__(*args, **kwargs)
        self.device, self.mode = device, mode
        self.snapshot_root = snapshot_root
        self.device_backends = {n: SnapshotBackend(device, n, snapshot_root) for n in range(2, 6)}
        self.circuit_records = []
        self._backend = RecordingSimulator(simulator(self.device_backends[5], mode), self.circuit_records)
        self._placement_cache = {}
        self._block_backend_cache = {}
        largest_width = max(2, (max(self.local_count, self.random_count)-1).bit_length())
        self._physical_packing = self.parallel_circuits > 1 or largest_width > 5
        if self._physical_packing:
            config = self.device_backends[5].configuration_json
            self.full_device = SnapshotBackend(device, config["n_qubits"], snapshot_root,
                                               include_unusable_wires=True)
            self._backend = simulator(self.full_device, mode, method="matrix_product_state")
            available = len(self.full_device.usable_indices)
            self.max_parallel_qubits = min(self.max_parallel_qubits or available, available)
            self._adjacency = {q: set() for q in sorted(self.full_device.usable_indices)}
            for a, b in self.full_device.target["cz"]:
                self._adjacency[a].add(b)
                self._adjacency[b].add(a)

    def _compile_circuits(self, circuits, seed):
        return [compile_native(c, self.device_backends[c.num_qubits], seed) for c in circuits]

    def _prepare_execution(self, circuits, seed):
        if not self._physical_packing:
            return super()._prepare_execution(circuits, seed)
        pending = circuit_groups(circuits, self.parallel_circuits, self.max_parallel_qubits)
        bundles = []
        while pending:
            group = pending.pop(0)
            widths = tuple(circuits[i].num_qubits for i in group)
            try:
                if widths not in self._placement_cache:
                    self._placement_cache[widths] = connected_placements(widths, self._adjacency)
                placements = self._placement_cache[widths]
            except ValueError:
                if len(group) == 1:
                    raise
                # Fragmentation can prevent a packing even below the qubit limit.
                pending[:0] = [group[:-1], group[-1:]]
                continue
            compiled = []
            for index, physical in zip(group, placements):
                key = tuple(physical)
                if key not in self._block_backend_cache:
                    self._block_backend_cache[key] = SnapshotBackend(
                        self.device, snapshot_root=self.snapshot_root, physical_qubits=physical)
                backend = self._block_backend_cache[key]
                c = compile_native(circuits[index], backend, seed,
                                   routing_method="sabre", optimization_level=3, layout_method="sabre")
                c.metadata.update({"measurement_physical_order":
                                   [physical[q] for q in measurement_order(c)],
                                   "duration_dt": int(c.duration)})
                compiled.append(c)
            bundle = pack_circuits(compiled, group, qubit_maps=placements,
                                   register_width=self.full_device.num_qubits)
            # Verify against the actual full target, not only the small subgraphs.
            for item in bundle.circuit.data:
                if item.operation.name == "barrier":
                    continue
                qargs = tuple(bundle.circuit.find_bit(q).index for q in item.qubits)
                if not self.full_device.target.instruction_supported(
                        operation_name=item.operation.name, qargs=qargs):
                    raise AssertionError(f"Invalid packed physical instruction: {item.operation.name}{qargs}")
            bundles.append(bundle)
        return bundles

    def _measure_circuits(self, circuits, seed, rng):
        histograms = super()._measure_circuits(circuits, seed, rng)
        if self._physical_packing:
            record = self.execution_history_[-1]
            for execution in record["executions"]:
                for block in execution["blocks"]:
                    index = block["circuit_index"]
                    self.circuit_records.append({**block,
                        "generation": record["generation"], "shots": self.shots,
                        "qubits": block["logical_qubits"], "physical_qubits": block["qubits"],
                        "execution_index": execution["execution_index"],
                        "sampling_seed": record["sampling_seed"], "counts": histograms[index]})
        return histograms

    def train(self, *args, **kwargs):
        self.circuit_records.clear()
        return super().train(*args, **kwargs)
