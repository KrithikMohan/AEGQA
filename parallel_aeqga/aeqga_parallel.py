"""Pack independent circuits onto disjoint qubits and recover their counts.

Packing exposes circuit-level parallelism to a device. It does not combine
amplitudes or create candidates; the optimizer determines population sizes.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from qiskit import QuantumCircuit


def marginal_counts(counts, classical_bits):
    """Recover one sequential stage without a dense joint probability vector."""
    output = {}
    for token, frequency in counts.items():
        if isinstance(token, Integral):
            value = int(token)
        else:
            token = token.replace(" ", "")
            value = int(token, 16 if token.startswith("0x") else 2)
        local = sum(((value >> source) & 1) << bit for bit, source in enumerate(classical_bits))
        key = format(local, f"0{len(classical_bits)}b")
        output[key] = output.get(key, 0)+int(frequency)
    return output


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def circuit_groups(circuits, parallel_circuits=1, max_qubits=None):
    """Partition in input order, respecting both block count and qubit budget."""
    limit = positive_integer(parallel_circuits, "parallel_circuits")
    if max_qubits is not None:
        max_qubits = positive_integer(max_qubits, "max_parallel_qubits")
    groups, current, width = [], [], 0
    for index, circuit in enumerate(circuits):
        if circuit.num_qubits < 1 or circuit.num_clbits < 1:
            raise ValueError("Each circuit must have qubits and measured classical bits.")
        if max_qubits is not None and circuit.num_qubits > max_qubits:
            raise ValueError(f"Circuit {index} needs {circuit.num_qubits} qubits, "
                             f"exceeding max_parallel_qubits={max_qubits}.")
        if current and (len(current) == limit or
                        (max_qubits is not None and width+circuit.num_qubits > max_qubits)):
            groups.append(current)
            current, width = [], 0
        current.append(index)
        width += circuit.num_qubits
    if current:
        groups.append(current)
    return groups


@dataclass
class PackedCircuit:
    circuit: QuantumCircuit
    blocks: list[dict]

    @property
    def active_qubits(self):
        return len({q for block in self.blocks for q in block["qubits"]})

    def split_counts(self, counts):
        """Marginalize sparse counts; never allocate a 2**total_qubits vector."""
        output = {b["circuit_index"]: {} for b in self.blocks}
        for bitstring, frequency in counts.items():
            if isinstance(bitstring, Integral):
                value = int(bitstring)
            else:
                token = bitstring.replace(" ", "")
                value = int(token, 16 if token.startswith("0x") else 2)
            for block in self.blocks:
                bits = block["clbits"]
                local = sum(((value >> physical_bit) & 1) << bit
                            for bit, physical_bit in enumerate(bits))
                key = format(local, f"0{len(bits)}b")
                histogram = output[block["circuit_index"]]
                histogram[key] = histogram.get(key, 0) + int(frequency)
        return output


def pack_circuits(circuits, indices=None, *, qubit_maps=None, register_width=None):
    """Compose measured circuits on disjoint wires without a global barrier.

    qubit_maps may use physical backend IDs. Classical bits retain each source
    circuit's order, even when its quantum routing changed measurement order.
    """
    if not circuits:
        raise ValueError("Cannot pack an empty circuit list.")
    indices = list(range(len(circuits))) if indices is None else list(indices)
    if len(indices) != len(circuits) or len(set(indices)) != len(indices):
        raise ValueError("indices must contain one unique ID per circuit.")
    if qubit_maps is None:
        offset, qubit_maps = 0, []
        for circuit in circuits:
            qubit_maps.append(list(range(offset, offset+circuit.num_qubits)))
            offset += circuit.num_qubits
    if len(qubit_maps) != len(circuits):
        raise ValueError("qubit_maps must contain one placement per circuit.")
    used = []
    for circuit, placement in zip(circuits, qubit_maps):
        if len(placement) != circuit.num_qubits or any(
                isinstance(q, bool) or not isinstance(q, Integral) or q < 0 for q in placement):
            raise ValueError("Each placement must contain one valid physical ID per qubit.")
        used.extend(placement)
    if len(set(used)) != len(used):
        raise ValueError("Parallel circuit placements must be disjoint.")
    width = max(used)+1 if register_width is None else positive_integer(register_width, "register_width")
    if max(used) >= width:
        raise ValueError("A placement exceeds the quantum register width.")
    total_bits = sum(c.num_clbits for c in circuits)
    # Preserve the existing single-circuit path, including register names and
    # seeded simulator behaviour, when packing is disabled.
    unchanged = (len(circuits) == 1 and width == circuits[0].num_qubits
                 and list(qubit_maps[0]) == list(range(width)))
    combined = circuits[0] if unchanged else QuantumCircuit(width, total_bits)
    blocks, offset = [], 0
    for index, circuit, placement in zip(indices, circuits, qubit_maps):
        classical = list(range(offset, offset+circuit.num_clbits))
        if not unchanged:
            combined.compose(circuit, qubits=placement, clbits=classical, inplace=True)
        blocks.append({**(circuit.metadata or {}),
                       "circuit_index": index, "qubits": list(map(int, placement)),
                       "clbits": classical, "logical_qubits": circuit.num_qubits,
                       "depth": circuit.depth(), "gate_counts": dict(circuit.count_ops())})
        offset += circuit.num_clbits
    return PackedCircuit(combined, blocks)


def connected_placements(widths, adjacency):
    """Deterministic connected blocks; reuse only between separate executions.

    This bounded greedy allocator is not a global layout optimum. A caller can
    split a group if fragmentation prevents the requested packing.
    """
    available = set(adjacency)
    result = [None] * len(widths)
    for index in sorted(range(len(widths)), key=lambda i: (-widths[i], i)):
        width = widths[index]
        choices = []
        for start in sorted(available, key=lambda q: (len(adjacency[q] & available), q)):
            selected, queue = [start], [start]
            while queue and len(selected) < width:
                node = queue.pop(0)
                neighbors = sorted((adjacency[node] & available)-set(selected),
                                   key=lambda q: (len(adjacency[q] & available), q))
                for neighbor in neighbors:
                    selected.append(neighbor)
                    queue.append(neighbor)
                    if len(selected) == width:
                        break
            if len(selected) == width:
                boundary = sum(len(adjacency[q] & (available-set(selected))) for q in selected)
                choices.append((boundary, tuple(sorted(selected))))
        if not choices:
            raise ValueError(f"Cannot allocate a connected {width}-qubit block from remaining qubits.")
        chosen = list(min(choices)[1])
        result[index] = chosen
        available.difference_update(chosen)
    return result
