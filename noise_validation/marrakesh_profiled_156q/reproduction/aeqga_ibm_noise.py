"""Complete parallel subpopulations on archived IBM physical-qubit targets."""
from aeqga import AEQGA
from aeqga_parallel import connected_placements, pack_circuits
from aeqga_ibm_noise_legacy import (
    DEFAULT_SNAPSHOTS, DEVICES, SNAPSHOT_VERSION, NATIVE_GATES, SnapshotBackend,
    named_values, device_noise, simulator, compile_native, measurement_order,
    in_classical_order, asymmetric_readout,
)


def stage_measurement_wires(circuit):
    """Physical wire for each fresh classical bit, including mid-circuit stages."""
    wires = [None]*circuit.num_clbits
    for item in circuit.data:
        if item.operation.name == "measure":
            index = circuit.find_bit(item.clbits[0]).index
            if wires[index] is not None:
                raise ValueError("Every stage must measure into fresh classical bits.")
            wires[index] = circuit.find_bit(item.qubits[0]).index
    if any(q is None for q in wires):
        raise ValueError("A recorded classical bit has no measurement.")
    return wires


class IBMNoiseAEQGA(AEQGA):
    def __init__(self, *args, device, mode="noisy", snapshot_root=DEFAULT_SNAPSHOTS, **kwargs):
        super().__init__(*args, **kwargs)
        self.device, self.mode, self.snapshot_root = device, mode, snapshot_root
        self.device_backends = {5: SnapshotBackend(device, 5, snapshot_root)}
        config = self.device_backends[5].configuration_json
        self.full_device = SnapshotBackend(device, config["n_qubits"], snapshot_root,
                                          include_unusable_wires=True)
        self._physical_packing = True
        available = len(self.full_device.usable_indices)
        if self.used_active_qubits > available:
            raise ValueError(f"The {self.parallel_circuits} complete circuits need "
                             f"{self.used_active_qubits} active qubits, but {device} "
                             f"has {available} calibrated usable qubits in this snapshot.")
        adjacency = {q: set() for q in sorted(self.full_device.usable_indices)}
        for a, b in self.full_device.target["cz"]:
            adjacency[a].add(b)
            adjacency[b].add(a)
        self._adjacency = adjacency
        # Fail explicitly if the requested simultaneous layout cannot be found;
        # do not silently turn K parallel groups into sequential smaller batches.
        self.placements = connected_placements([self.qubits_per_circuit]*self.parallel_circuits,
                                               adjacency)
        self._selected_backends = [SnapshotBackend(device, snapshot_root=snapshot_root,
                                                  physical_qubits=physical)
                                   for physical in self.placements]
        self._backend = simulator(self.full_device, mode, method="matrix_product_state")
        self.circuit_records = []

    def _prepare_execution(self, circuits, seed):
        if len(circuits) != self.parallel_circuits or any(
                c.num_qubits != self.qubits_per_circuit for c in circuits):
            raise ValueError("Expected one equally sized circuit per complete subpopulation.")
        compiled = []
        for circuit, physical, backend in zip(circuits, self.placements, self._selected_backends):
            c = compile_native(circuit, backend, seed, routing_method="sabre",
                               optimization_level=3, layout_method="sabre")
            c.metadata.update({"measurement_physical_order":
                               [physical[q] for q in stage_measurement_wires(c)],
                               "duration_dt": int(c.duration)})
            compiled.append(c)
        bundle = pack_circuits(compiled, qubit_maps=self.placements,
                               register_width=self.full_device.num_qubits)
        for item in bundle.circuit.data:
            if item.operation.name != "barrier":
                qargs = tuple(bundle.circuit.find_bit(q).index for q in item.qubits)
                if not self.full_device.target.instruction_supported(
                        operation_name=item.operation.name, qargs=qargs):
                    raise AssertionError(f"Invalid physical instruction: {item.operation.name}{qargs}")
        return [bundle]

    def _measure_circuits(self, circuits, seed, rng):
        histograms = super()._measure_circuits(circuits, seed, rng)
        record = self.execution_history_[-1]
        for execution in record["executions"]:
            for block in execution["blocks"]:
                index = block["circuit_index"]
                self.circuit_records.append({**block, "generation": record["generation"],
                    "shots": self.shots, "qubits": block["logical_qubits"],
                    "physical_qubits": block["qubits"], "execution_index": execution["execution_index"],
                    "sampling_seed": record["sampling_seed"], "counts": histograms[index]})
        return histograms

    def train(self, *args, **kwargs):
        self.circuit_records.clear()
        return super().train(*args, **kwargs)
