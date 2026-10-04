"""Useful offline regressions retained from the retired legacy test script."""
import unittest
import numpy as np
from qiskit.quantum_info import Statevector
from aeqga.likelihoods.cosmology import C_LIGHT, distance_integral, distance_moduli
from aeqga.steps.encoding.amplitude_encoding import (
    _normalise, n_qubits_for_population, build_amplitude_circuit,
)
from aeqga.steps.genetic_operators.quantum_gates import verify_u_cross, verify_rx_pi2
from aeqga.steps.decoding.measurement_decoding import decode_random_subset, decode_elite_subset


class QuantumPrimitiveTests(unittest.TestCase):
    def test_distance_units_and_hubble_scaling(self):
        distance = C_LIGHT/70 * 1.5 * distance_integral(.5,.3)
        self.assertTrue(2700 < distance < 2950)
        self.assertTrue(37.5 < distance_moduli(.1,.1,70,.3) < 39)
        delta_mu = distance_moduli(.3,.3,35,.3)-distance_moduli(.3,.3,70,.3)
        self.assertAlmostEqual(float(delta_mu),5*np.log10(2),places=12)
        with self.assertRaises(ValueError): distance_moduli(0,0,70,.3)

    def test_normalization_and_state_preparation(self):
        np.testing.assert_allclose(_normalise(np.zeros(4)),np.ones(4)/2)
        values=np.array([1.,2.,3.,4.])
        np.testing.assert_allclose(Statevector(build_amplitude_circuit(values)).data,
                                   values/np.linalg.norm(values),atol=1e-12)
        with self.assertRaises(AssertionError): n_qubits_for_population(10)

    def test_gate_matrices(self):
        matrix,expected=verify_u_cross()
        np.testing.assert_allclose(matrix,expected,atol=1e-12)
        expected_rx=np.array([[1,-1j],[-1j,1]],dtype=complex)/np.sqrt(2)
        np.testing.assert_allclose(verify_rx_pi2(),expected_rx,atol=1e-12)

    def test_decoding_edge_cases(self):
        np.random.seed(23)
        uniform=decode_random_subset({'00':10,'01':10,'10':10,'11':10},2,60,80,4)
        self.assertTrue(np.all((uniform>=60)&(uniform<=80)))
        zero=decode_elite_subset({},2,.2,.4,4)
        self.assertTrue(np.all((zero>=.2)&(zero<=.4)))


if __name__=='__main__':
    unittest.main()
