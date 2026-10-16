"""Check decomposition, noise attachment, and probability/readout ordering."""
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from aeqga import AEQGA
from aeqga_noise import (NoiseValidationAEQGA, apply_readout,
                         build_noise_model, compile_basis, make_backend)


class NoiseValidationTests(unittest.TestCase):
    def test_compilation_preserves_probabilities_and_decomposes_initialization(self):
        model = AEQGA(lambda x, p: p[0], [0.], bounds=[(-1,1)])
        model.crossover_probability = model.mutation_probability = 1.
        for size in (2,4,8,16,32):
            with self.subTest(size=size):
                values = np.linspace(-1.,2.,size)
                original = model._circuit(values,np.random.default_rng(7))
                compiled = compile_basis(original)
                self.assertNotIn("initialize",compiled.count_ops())
                self.assertNotIn("cry",compiled.count_ops())
                self.assertGreater(compiled.count_ops().get("cx",0),0)
                p0 = Statevector.from_instruction(original.remove_final_measurements(inplace=False)).probabilities()
                p1 = Statevector.from_instruction(compiled.remove_final_measurements(inplace=False)).probabilities()
                np.testing.assert_allclose(p0,p1,rtol=0,atol=1e-11)

    def test_readout_bit_order_and_exact_convolution(self):
        p = np.array([1.,0.,0.,0.])
        np.testing.assert_allclose(apply_readout(p,.2),[.64,.16,.16,.04])
        np.testing.assert_array_equal(apply_readout(p,1.),[0.,0.,0.,1.])
        np.testing.assert_array_equal(apply_readout(p,0.),p)

    def test_gate_noise_matches_analytic_depolarizing_channel(self):
        from qiskit_aer import AerSimulator
        config = {"one_qubit_depolarizing":.2,"two_qubit_depolarizing":0.,"readout_flip":0.}
        noise = build_noise_model(config)
        backend = AerSimulator(method="density_matrix",noise_model=noise)
        circuit = QuantumCircuit(1)
        circuit.u(0.,0.,0.,0)
        circuit.save_probabilities()
        p = backend.run(circuit,shots=1).result().data(0)["probabilities"]
        np.testing.assert_allclose(p,[.9,.1],atol=1e-12)

    def test_actual_aer_readout_matches_convolution(self):
        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.measure_all()
        compiled = compile_basis(circuit)
        result = make_backend("medium_readout_only").run(compiled,shots=40000,seed_simulator=7).result()
        counts = AEQGA._counts(result.get_counts(),4)/40000
        expected = apply_readout([0.,1.,0.,0.],.02)
        np.testing.assert_allclose(counts,expected,rtol=0,atol=.006)

    def test_noise_training_history_and_objective_remain_aligned(self):
        model = NoiseValidationAEQGA(lambda x,p:p[0]*x[0]+p[1], [0.,0.],
                                     bounds=[(-4,4),(-2,2)],population_size=8,shots=128,
                                     seed=7,noise_profile="high")
        x = np.array([0.,1.,2.]); y = np.array([1.,2.,3.])
        model.train(x,y,sigma=.5,generations=2)
        self.assertEqual(len(model.history_),3)
        for record in model.history_:
            self.assertEqual(len(record["population"]),8)
            for candidate in record["population"]:
                residual = (candidate["p"][0]*x+candidate["p"][1]-y)/.5
                self.assertAlmostEqual(candidate["chi2"],float(residual@residual))
        self.assertEqual(model.quantum_circuits_,8)


if __name__ == "__main__":
    unittest.main()
