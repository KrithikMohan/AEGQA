"""Offline upstream-protocol checks; these do not claim a native HQGA run."""
from types import SimpleNamespace
import unittest
import numpy as np
from aeqga.integrations.hqga import (
    HQGAProblemAdapter, AEQGAProblemForHQGA, parameters_from_hqga, run_qga_aeqga,
)


class Sphere:
    n_dim = 2
    lower_bounds = np.array([-2., 0.])
    upper_bounds = np.array([2., 3.])
    def compute_fitness(self, x):
        return float(np.dot(x, x))


class HQGATests(unittest.TestCase):
    def test_gray_decode_all_codes_and_objective_count(self):
        bridge = AEQGAProblemForHQGA(Sphere(), 3)
        for i in range(8):
            for j in range(8):
                code = f'{i ^ (i >> 1):03b}{j ^ (j >> 1):03b}'
                x = np.array([-2 + i*4/7, j*3/7])
                np.testing.assert_allclose(bridge.convert(code), x, atol=1e-15)
                self.assertAlmostEqual(bridge.evaluate(code), np.dot(x, x))
        self.assertEqual(bridge.evaluations, 64)
        self.assertFalse(bridge.isMaxProblem())
        np.testing.assert_allclose(bridge.resolution, [4/7, 3/7])

    def test_real_objective_roundtrip_and_maximize(self):
        class MaxSphere(Sphere):
            def is_max_problem(self): return True
        hproblem = AEQGAProblemForHQGA(MaxSphere())
        native = HQGAProblemAdapter(hproblem)
        self.assertTrue(native.is_max_problem())
        self.assertEqual(native.compute_fitness([1, 2]), 5)
        self.assertEqual(hproblem.evaluations, 1)

    def test_budget_mapping_does_not_map_hqga_mutation(self):
        params = SimpleNamespace(pop_size=8, max_gen=1, num_shots=0,
                                 progressBar=False, prob_mut=.9, epsilon=.2)
        native = parameters_from_hqga(params)
        self.assertEqual(native.p_mut, .5)
        self.assertFalse(native.progress_bar)
        self.assertEqual(parameters_from_hqga(params, p_mut=.2).p_mut, .2)
        params.pop_size = 3
        with self.assertRaises(ValueError): parameters_from_hqga(params)
        self.assertEqual(parameters_from_hqga(params, pop_size=8).pop_size, 8)

    def test_validation(self):
        with self.assertRaises(AttributeError):
            HQGAProblemAdapter(SimpleNamespace(dim=2))  # BinaryProblem has no bounds
        for bits in [0, True, 1.5, 53]:
            with self.assertRaises(ValueError): AEQGAProblemForHQGA(Sphere(), bits)
        bridge = AEQGAProblemForHQGA(Sphere(), 3)
        for code in ['000', '000002', '000 000', ['0']*6]:
            with self.assertRaises(ValueError): bridge.convert(code)
        for x in [[1], [float('nan'), 0], [3, 0]]:
            with self.assertRaises(ValueError): bridge.computeFitness(x)
        self.assertEqual(bridge.evaluations, 0)

    def test_aeqga_bridge_executes_and_returns_native_results(self):
        np.random.seed(17)
        problem = AEQGAProblemForHQGA(Sphere())
        params = SimpleNamespace(pop_size=8, max_gen=1, num_shots=0,
                                 progressBar=False)
        best, populations, history = run_qga_aeqga(problem, params)
        self.assertEqual(best.evaluations, 16)
        self.assertEqual(problem.evaluations, 16)
        self.assertEqual(len(populations), 2)
        self.assertEqual(len(history), 2)
        self.assertEqual(len(history[0]), 2)
        self.assertEqual(best.x.shape, (2,))


if __name__ == '__main__': unittest.main()
