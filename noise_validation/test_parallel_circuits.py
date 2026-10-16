"""Validate tensor-product packing, bit ordering, topology and training logs."""
import json
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from aeqga import AEQGA
from aeqga_parallel import circuit_groups, connected_placements, pack_circuits
from aeqga_ibm_noise import (DEVICES, IBMNoiseAEQGA, SnapshotBackend, compile_native,
                             simulator, measurement_order, in_classical_order, asymmetric_readout)
from aeqga_noise import NoiseValidationAEQGA


def quadratic(x, p):
    return p[0]*x[0]**2+p[1]*x[0]+p[2]


class ParallelCircuitTests(unittest.TestCase):
    def test_counts_preserve_classical_order_and_disjoint_physical_wires(self):
        first = QuantumCircuit(2, 2)
        first.x(0)
        first.measure(0, 1)
        first.measure(1, 0)
        second = QuantumCircuit(3, 3)
        second.x(2)
        second.measure(2, 0)
        second.measure(0, 2)
        second.measure(1, 1)
        packed = pack_circuits([first, second], [7, 3], qubit_maps=[[4, 0], [1, 3, 2]])
        result = AerSimulator().run(packed.circuit, shots=100, seed_simulator=8).result()
        self.assertEqual(packed.split_counts(result.get_counts()), {7: {"10": 100}, 3: {"001": 100}})
        # Counts can contain register separators and hexadecimal keys.
        self.assertEqual(packed.split_counts({"001 10": 80, "0x6": 20}),
                         {7: {"10": 100}, 3: {"001": 100}})
        self.assertEqual(packed.active_qubits, 5)

    def test_packed_distribution_is_product_of_original_amplitude_circuits(self):
        model = AEQGA(lambda x,p:p[0], [0], bounds=[(-1, 1)],
                      crossover_probability=1, mutation_probability=1)
        circuits = [model._circuit(np.array(v), np.random.default_rng(seed))
                    for v, seed in [([1., -2., 3., 4.], 7), ([2., 3., -1., 5., 2., 7., 1., 4.], 9)]]
        probabilities = [Statevector.from_instruction(c.remove_final_measurements(inplace=False)).probabilities()
                         for c in circuits]
        packed = pack_circuits(circuits)
        observed = Statevector.from_instruction(
            packed.circuit.remove_final_measurements(inplace=False)).probabilities()
        np.testing.assert_allclose(observed, np.kron(probabilities[1], probabilities[0]), atol=1e-12)

    def test_capacity_splits_groups_without_padding_or_duplicate_circuits(self):
        circuits = [QuantumCircuit(n, n) for n in [4, 5, 4, 5, 4, 5]]
        self.assertEqual(circuit_groups(circuits, 6), [[0, 1, 2, 3, 4, 5]])
        self.assertEqual(circuit_groups(circuits, 4), [[0, 1, 2, 3], [4, 5]])
        self.assertEqual(circuit_groups(circuits, 6, 9), [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(circuit_groups(circuits, 100, 27), [[0, 1, 2, 3, 4, 5]])
        with self.assertRaisesRegex(ValueError, "exceeding"):
            circuit_groups(circuits, 6, 4)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            pack_circuits(circuits[:2], qubit_maps=[[0,1,2,3], [3,4,5,6,7]])
        for value in (0, -1, True, 1.5):
            for name in ("parallel_circuits", "max_parallel_qubits"):
                with self.subTest(value=value, name=name), self.assertRaises(ValueError):
                    AEQGA(lambda x,p:p[0], [0], bounds=[(-1,1)], **{name:value})

    def test_same_generation_training_has_complete_candidate_and_execution_records(self):
        x = np.array([0., 1., 2., 3.]); y = x*x+1
        for limit, expected_widths in [(1, [2,3,2,3,2,3]), (2, [5,5,5]),
                                      (4, [10,5]), (6, [15]), (30, [15])]:
            with self.subTest(limit=limit):
                f = AEQGA(quadratic, [1,0,0], bounds=[(-2,2)]*3, population_size=16,
                          parallel_circuits=limit, shots=64, seed=13)
                with patch.object(f._backend, "run", wraps=f._backend.run) as run:
                    f.train(x, y, sigma=.5, generations=3)
                self.assertEqual(run.call_count, 3)
                self.assertEqual(f.quantum_circuits_, 18)
                self.assertEqual(f.executed_circuits_, 3*len(expected_widths))
                self.assertEqual(f.n_evaluations_, 16+3*12)
                self.assertEqual(len(f.history_), 4)
                for generation in f.history_:
                    for candidate in generation["population"]:
                        p = candidate["p"]
                        residual = (p[0]*x*x+p[1]*x+p[2]-y)/.5
                        self.assertAlmostEqual(candidate["chi2"], residual@residual)
                for record in f.execution_history_:
                    self.assertEqual([e["active_qubits"] for e in record["executions"]], expected_widths)
                    self.assertEqual(sorted(b["circuit_index"] for e in record["executions"] for b in e["blocks"]),
                                     list(range(6)))
                    for execution in record["executions"]:
                        self.assertEqual(sum(execution["counts"].values()), 64)
                    json.dumps(record)

    def test_reproducibility_history_reset_and_no_quantum_branches(self):
        def make():
            return AEQGA(quadratic, [1,0,0], bounds=[(-2,2)]*3, population_size=16,
                          parallel_circuits=4, max_parallel_qubits=8, shots=64, seed=4)
        first, second = make(), make()
        for f in (first, second):
            f.train([0,1,2],[1,2,5],generations=3)
        self.assertEqual(first.history_, second.history_)
        self.assertEqual(first.execution_history_, second.execution_history_)
        first.train([0,1,2],[1,2,5],generations=1)
        self.assertEqual(len(first.execution_history_), 1)
        self.assertEqual(first.execution_history_[0]["generation"], 1)
        self.assertLessEqual(max(e["active_qubits"] for e in first.execution_history_[0]["executions"]), 8)
        f = AEQGA(quadratic, [1,0,0], bounds=[(-2,2)]*3, population_size=8,
                  elite_count=8, local_count=0, random_count=0, parallel_circuits=6)
        f.train([0,1],[0,1],generations=2)
        self.assertEqual((f.quantum_circuits_, f.executed_circuits_), (0,0))
        self.assertEqual(f.execution_history_, [])

    def test_synthetic_noise_supports_packed_execution(self):
        f = NoiseValidationAEQGA(quadratic, [1,0,0], bounds=[(-2,2)]*3, population_size=8,
                                parallel_circuits=6, shots=32, seed=3, noise_profile="medium")
        f.train([0,1,2],[1,2,5],sigma=.5,generations=1)
        self.assertEqual(f.executed_circuits_, 1)
        self.assertEqual(f.execution_history_[0]["executions"][0]["active_qubits"], 12)
        self.assertTrue(np.isfinite(f.loss_))

    def test_connected_allocator_reports_fragmentation(self):
        adjacency = {0:{1}, 1:{0,2}, 2:{1}, 3:{4}, 4:{3}}
        placements = connected_placements([3,2], adjacency)
        self.assertEqual(set(placements[0]), {0,1,2})
        self.assertEqual(set(placements[1]), {3,4})
        with self.assertRaisesRegex(ValueError, "connected"):
            connected_placements([4], adjacency)

    def test_packed_ibm_noisy_marginals_match_independent_density_matrices(self):
        # In particular, retain Kingston's large q1 readout asymmetry on its
        # physical ID after packing and after a classical-bit permutation.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            f = IBMNoiseAEQGA(quadratic, [1,0,0], bounds=[(-2,2)]*3,
                             device="kingston", parallel_circuits=2, mode="noisy")
            originals = []
            for flipped in (0,1):
                c = QuantumCircuit(2,2)
                c.x(flipped)
                c.measure(0,1)
                c.measure(1,0)
                originals.append(c)
            bundle = f._prepare_execution(originals, 17)[0]
            shots = 4096
            raw = f._backend.run(bundle.circuit, shots=shots, seed_simulator=31).result().get_counts()
            marginals = bundle.split_counts(raw)
            for index, block in enumerate(bundle.blocks):
                backend = SnapshotBackend("kingston", physical_qubits=block["qubits"])
                c = compile_native(originals[index], backend, 17, routing_method="sabre",
                                   optimization_level=3, layout_method="sabre")
                measured = measurement_order(c)
                exact = c.remove_final_measurements(inplace=False)
                exact.save_probabilities()
                p = simulator(backend, readout=False).run(exact, shots=1).result().data(0)["probabilities"]
                expected = asymmetric_readout(in_classical_order(p, measured), backend, measured)
                observed = AEQGA._counts(marginals[index], 4)/shots
                tolerance = 6*np.sqrt(expected*(1-expected)/shots)+6/shots
                self.assertTrue(np.all(np.abs(observed-expected) <= tolerance), (observed,expected))

    def test_ibm_packed_native_gates_calibrations_and_marginal_records(self):
        for device in DEVICES:
            with self.subTest(device=device), warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                f = IBMNoiseAEQGA(quadratic, [1,0,0], bounds=[(-2,2)]*3,
                    population_size=16, parallel_circuits=6, max_parallel_qubits=10,
                    shots=32, seed=5, device=device, mode="noisy")
                f.train([0,1,2],[1,2,5],sigma=.5,generations=1)
                if device == "kingston":
                    self.assertIn(146, f.full_device.unusable_physical_ids)
                    self.assertNotIn(146, f._adjacency)
                    with self.assertRaisesRegex(ValueError, "Missing/invalid calibration"):
                        SnapshotBackend(device, physical_qubits=[146])
                self.assertEqual(f.executed_circuits_, 2)
                self.assertEqual(len(f.circuit_records), 6)
                for circuit, execution in zip(f.execution_circuits_, f.execution_history_[0]["executions"]):
                    self.assertLessEqual(execution["active_qubits"], 10)
                    sets = [set(b["qubits"]) for b in execution["blocks"]]
                    self.assertEqual(sum(map(len,sets)), len(set().union(*sets)))
                    for item in circuit.data:
                        qargs = tuple(circuit.find_bit(q).index for q in item.qubits)
                        self.assertTrue(any(set(qargs).issubset(s) for s in sets))
                        if item.operation.name != "barrier":
                            self.assertTrue(f.full_device.target.instruction_supported(
                                operation_name=item.operation.name,qargs=qargs))
                    for block in execution["blocks"]:
                        selected = SnapshotBackend(device, physical_qubits=block["qubits"])
                        for local, physical in enumerate(block["qubits"]):
                            np.testing.assert_array_equal(selected.readout_matrix(local),
                                                          f.full_device.readout_matrix(physical))
                for record in f.circuit_records:
                    self.assertEqual(sum(record["counts"].values()), 32)
                    self.assertEqual(record["generation"], 1)
                json.dumps(f.execution_history_)
                json.dumps(f.circuit_records)


if __name__ == "__main__":
    unittest.main()
