"""Check noise attachment, exact zero-noise limit and AEQGA count conventions."""
import unittest

import numpy as np
from qiskit import QuantumCircuit, transpile

from aeqga.execution.ionq_noise import BASIS, IonQDemoExecutor, IonQDemoNoise
from scripts.ionq_noise_demo import circuit_probe


class IonQNoiseTests(unittest.TestCase):
    def test_zero_noise_matches_compiled_ideal(self):
        for row in circuit_probe(shots=100, noise=IonQDemoNoise(0, 0, 0)):
            self.assertLess(row["ideal_probability_error"], 1e-10)
            self.assertLess(row["exact_noise_tvd"], 1e-10)
            self.assertEqual(row["ideal_counts"], row["noisy_counts"])
            self.assertEqual(sum(row["noisy_counts"].values()), 100)

    def test_readout_and_bit_order(self):
        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.measure_all()
        ideal = IonQDemoExecutor(shots=100, noise=IonQDemoNoise(0, 0, 0)).run([circuit])[0]
        flipped = IonQDemoExecutor(shots=100, noise=IonQDemoNoise(0, 0, 1)).run([circuit])[0]
        self.assertEqual(ideal, {"01": 100})
        self.assertEqual(flipped, {"10": 100})

    def test_gate_channels_have_expected_exact_effect(self):
        for width, gate, config in ((1, "rx", IonQDemoNoise(1, 0, 0)),
                                     (2, "rxx", IonQDemoNoise(0, 1, 0))):
            circuit = QuantumCircuit(width)
            if gate == "rx":
                circuit.rx(.7, 0)
            else:
                circuit.rxx(.7, 0, 1)
            circuit = transpile(circuit, basis_gates=BASIS, optimization_level=0)
            circuit.save_probabilities()
            executor = IonQDemoExecutor(noise=config)
            probs = executor.simulator.run(circuit).result().data(0)["probabilities"]
            np.testing.assert_allclose(probs, np.ones(2**width)/2**width, atol=1e-12)

    def test_default_noise_changes_exact_distribution(self):
        for row in circuit_probe(shots=100):
            self.assertGreater(row["exact_noise_tvd"], 1e-5)
            self.assertEqual(sum(row["noisy_counts"].values()), 100)

    def test_invalid_rates_and_shots(self):
        for value in (-.1, 1.1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                IonQDemoNoise(one_qubit=value)
        with self.assertRaises(ValueError):
            IonQDemoExecutor(shots=0)
