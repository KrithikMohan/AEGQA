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

    def __init__(self, device, width=5, snapshot_root=DEFAULT_SNAPSHOTS):
        if device not in DEVICES or width not in range(2, 6):
            raise ValueError("Use kingston/fez/marrakesh and 2-5 active qubits.")
        root = Path(snapshot_root) / device
        self.configuration_json = json.loads((root / f"conf_{device}.json").read_text())
        self.properties_json = json.loads((root / f"props_{device}.json").read_text())
        self.physical_ids = list(range(width))
        self.device = device
        super().__init__(name=f"snapshot_{device}_{width}q", backend_version=SNAPSHOT_VERSION)
        self.calibrations = [named_values(self.properties_json["qubits"][i])
                             for i in self.physical_ids]
        for p in self.calibrations:
            if p.get("operational", True) == 0 or p["T1"] <= 0 or p["T2"] <= 0:
                raise ValueError("The preselected physical chain contains an invalid qubit.")
        # IBM configuration JSON stores dt in ns; Target uses seconds.
        target = Target(num_qubits=width, dt=self.configuration_json["dt"] * 1e-9,
                        qubit_properties=[QubitProperties(t1=p["T1"], t2=p["T2"],
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
            values = named_values(g["parameters"])
            if not values.get("operational", True):
                continue
            if len(qargs) == 2 and qargs not in edges:
                raise AssertionError("Calibration edge is absent from the coupling map.")
            entries[g["gate"]][qargs] = InstructionProperties(
                duration=values["gate_length"], error=values["gate_error"])
        for q in range(width - 1):
            if (q, q+1) not in entries["cz"] or (q+1, q) not in entries["cz"]:
                raise ValueError("The predeclared physical chain is not fully calibrated.")
        for name, operation in NATIVE_GATES.items():
            target.add_instruction(operation, entries[name])
        target.add_instruction(Measure(), {
            (q,): InstructionProperties(duration=p["readout_length"], error=p["readout_error"])
            for q, p in enumerate(self.calibrations)})
        # Reset has a reported duration, but no assignment/infidelity calibration.
        # Aer uses ideal reset plus duration-based relaxation at zero temperature.
        reset_times = {g["qubits"][0]: named_values(g["parameters"])["gate_length"]
                       for g in self.properties_json["gates"] if g["gate"] == "reset"}
        target.add_instruction(Reset(), {(q,): InstructionProperties(duration=reset_times[q]) for q in range(width)})
        target.add_instruction(Delay(Parameter("duration")), {(q,): None for q in range(width)})
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
        for q in range(backend.num_qubits):
            noise.add_readout_error(ReadoutError(backend.readout_matrix(q)), [q])
    if not noise._custom_noise_passes:
        raise AssertionError("Scheduled delay relaxation is missing.")
    return noise


def simulator(backend, mode="noisy", *, readout=True):
    if mode not in ("ideal", "noisy"):
        raise ValueError("mode must be ideal or noisy")
    noise = device_noise(backend, readout=readout) if mode == "noisy" else None
    return AerSimulator(method="density_matrix", noise_model=noise,
                        max_parallel_threads=1, max_parallel_experiments=1)


def compile_native(circuit, backend, seed=12345):
    """Fixed initial layout, deterministic routing, native decomposition + ALAP."""
    if circuit.num_qubits != backend.num_qubits:
        raise ValueError("Compile on exactly the circuit-width physical prefix.")
    compiled = transpile(circuit, target=backend.target,
                         initial_layout=list(range(circuit.num_qubits)),
                         routing_method="basic", optimization_level=1,
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
        self.device_backends = {n: SnapshotBackend(device, n, snapshot_root) for n in range(2, 6)}
        self.circuit_records = []
        self._backend = RecordingSimulator(simulator(self.device_backends[5], mode), self.circuit_records)

    def _compile_circuits(self, circuits, seed):
        return [compile_native(c, self.device_backends[c.num_qubits], seed) for c in circuits]

    def train(self, *args, **kwargs):
        self.circuit_records.clear()
        return super().train(*args, **kwargs)
