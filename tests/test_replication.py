"""Regression checks for the paper replication; run with python -m unittest."""
import random
import unittest
from aeqga.paths import PROJECT_ROOT

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from aeqga.steps.evolution.aeqga_algorithm import AEQGAParameters, build_dimension_circuit, run_aeqga_dual, run_aeqga_sv
from aeqga.steps.decoding.measurement_decoding import get_basis_counts
from aeqga.likelihoods.pantheon_problem import load_pantheon_plus, chi2_pantheon_plus, PantheonPlusProblem


class CircuitReadoutTests(unittest.TestCase):
    def test_counts_measure_after_genetic_gates(self):
        values = np.array([1., 2., 3., 7.])
        random.seed(42)
        unitary = build_dimension_circuit(values, 1., 1., 0, "statevector")
        random.seed(42)
        measured = build_dimension_circuit(values, 1., 1., 32768)
        names = [item.operation.name for item in measured.data]
        self.assertLess(names.index("rx"), names.index("measure"))
        self.assertEqual(names.count("measure"), 2)
        expected = Statevector(unitary).probabilities()
        encoded = values**2 / (values @ values)
        self.assertGreater(np.max(np.abs(expected - encoded)), .05)
        backend = AerSimulator()
        counts = backend.run(transpile(measured, backend, seed_transpiler=42),
                             shots=32768, seed_simulator=42).result().get_counts()
        observed = get_basis_counts(counts, 2) / 32768
        np.testing.assert_allclose(observed, expected, atol=.012)


class GenerationTests(unittest.TestCase):
    def test_elite_lower_bound_resampling(self):
        from aeqga.steps.decoding.measurement_decoding import decode_elite_subset
        np.random.seed(11)
        values=decode_elite_subset({'01':100},1,60.,80.,2)
        self.assertTrue(60<values[0]<80)
        self.assertEqual(values[1],80.)
        collapsed=decode_elite_subset({'01':100},1,70.,70.,2)
        np.testing.assert_array_equal(collapsed,[70.,70.])

    def test_initial_population_and_exact_updates(self):
        class Sphere:
            lower_bounds = np.array([60., 0.])
            upper_bounds = np.array([80., .5])
            n_dim = 2
            calls = 0

            def compute_fitness(self, x):
                self.calls += 1
                return float((x[0]-71)**2 + (x[1]-.3)**2)

        for runner, shots in [(run_aeqga_dual, 128), (run_aeqga_sv, 0)]:
            problem = Sphere()
            np.random.seed(123)
            random.seed(123)
            params = AEQGAParameters(pop_size=16, max_gen=3, num_shots=shots,
                                     progress_bar=False, elite_margin=.2)
            best, populations, history = runner(problem, params)
            self.assertEqual(len(populations), 4)
            self.assertEqual(len(history), 4)
            self.assertEqual(problem.calls, 16*4)
            self.assertEqual(best.evaluations, problem.calls)
            for old, new in zip(populations[:-1], populations[1:]):
                order = np.argsort([Sphere.compute_fitness(problem, x) for x in old])
                np.testing.assert_array_equal(new[:4], old[order[:4]])
            for population in populations:
                self.assertTrue(np.all(population >= problem.lower_bounds))
                self.assertTrue(np.all(population <= problem.upper_bounds))
            self.assertTrue(np.all(np.diff([row[1] for row in history]) <= 0))

    def test_zero_updates_returns_initial_best(self):
        class MaxProblem:
            lower_bounds = [0.]
            upper_bounds = [1.]
            n_dim = 1
            def compute_fitness(self, x): return float(x[0])
            def is_max_problem(self): return True
        best, populations, history = run_aeqga_dual(
            MaxProblem(), AEQGAParameters(max_gen=0, progress_bar=False))
        self.assertEqual(len(history), 1)
        self.assertEqual(best.fitness, populations[0].max())

    def test_invalid_settings(self):
        for kwargs in [{"pop_size":4}, {"pop_size":10}, {"max_gen":-1},
                       {"p_cross":1.1}, {"num_shots":-3}, {"elite_margin":-1}]:
            with self.assertRaises(ValueError):
                AEQGAParameters(**kwargs)._validate()


@unittest.skipUnless((PROJECT_ROOT/'sn_data/PantheonPlus/Pantheon+SH0ES.dat').exists(),
                     'Fetch the Pantheon+ data for integration tests')
class PantheonPlusLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = load_pantheon_plus(PROJECT_ROOT/"sn_data/PantheonPlus")
        cls.flow = load_pantheon_plus(PROJECT_ROOT/"sn_data/PantheonPlus", selection="hubble_flow",
                                     redshift="hd_hel")

    def test_full_dataset_and_cholesky(self):
        self.assertEqual(self.full["n_sn"], 1701)
        # CID strings are not globally unique object identifiers across surveys.
        self.assertGreater(len(set(self.full["cid"])), 1500)
        self.assertEqual(self.full["cov_total"].shape, (1701, 1701))
        L = self.full["cholesky"]
        np.testing.assert_allclose(L @ L.T, self.full["cov_total"], atol=1e-12)

    def test_selection_alignment(self):
        rows = self.flow["row_indices"]
        self.assertGreater(len(rows), 1000)
        self.assertLess(len(rows), 1701)
        np.testing.assert_array_equal(self.flow["cov_total"],
                                      self.full["cov_total"][np.ix_(rows, rows)])
        np.testing.assert_array_equal(self.flow["mu_obs"], self.full["mu_obs"][rows])
        self.assertFalse(self.flow["is_calibrator"].any())
        self.assertTrue((self.flow["zHD"] > .01).all())

    def test_calibrated_likelihood_keeps_h0_dependence(self):
        data = self.full
        from aeqga.likelihoods.cosmology import distance_moduli
        residual = distance_moduli(data["zcmb"], data["zhel"], 73., .36)-data["mu_obs"]
        reference = residual @ np.linalg.solve(data["cov_total"], residual)
        self.assertAlmostEqual(chi2_pantheon_plus(73., .36, data), reference, places=7)
        self.assertGreater(chi2_pantheon_plus(65., .36, data)-reference, 100)

    def test_quadrature_against_adaptive_reference(self):
        from aeqga.likelihoods.cosmology import distance_integral
        from scipy.integrate import quad
        for om in [0., .15, .363, .5]:
            for z in [.001, .01, .3, 1., 2.26]:
                reference = quad(lambda zp: 1/np.sqrt(om*(1+zp)**3+1-om), 0, z,
                                 epsabs=1e-12, epsrel=1e-12)[0]
                self.assertAlmostEqual(float(distance_integral(z, om)), reference, places=11)

    def test_paper_grid_matches_direct_on_nodes(self):
        # Reuse validated data to isolate numerical approximation from IO.
        from unittest.mock import patch
        with patch("aeqga.likelihoods.pantheon_problem.load_pantheon_plus", return_value=self.full):
            for size in [100,300]:
                problem = PantheonPlusProblem(grid_size=size)
                for om in problem.omega_grid[[0, size//2, -1]]:
                    for h0 in [60.,72.837,80.]:
                        self.assertAlmostEqual(problem.compute_fitness([h0,om]),
                                               chi2_pantheon_plus(h0,om,self.full),places=7)
                h0s = [69.,73.,75.]
                oms = [.25,.36,.4]
                grid = problem.objective_grid(h0s,oms)
                self.assertEqual(grid.shape,(3,3))
                self.assertEqual(grid[1,2],problem.compute_fitness([75.,.36]))

    def test_distances_do_not_quantize_hubble_constant(self):
        from aeqga.likelihoods.cosmology import distance_moduli
        a=10**((distance_moduli(.3,.3,70.001,.3)-25)/5)
        b=10**((distance_moduli(.3,.3,70.002,.3)-25)/5)
        self.assertAlmostEqual(a/b,70.002/70.001,places=12)


class BAOCMBTests(unittest.TestCase):
    def test_bao_covariance_and_quadratic(self):
        from aeqga.likelihoods.bao_cmb_problem import BAOLikelihood
        bao=BAOLikelihood()
        self.assertEqual(len(bao.z),16)
        np.testing.assert_allclose(bao.factor@bao.factor.T,bao.covariance,atol=1e-12)
        r=bao.prediction(66.76,.323)-bao.observed
        self.assertAlmostEqual(bao.chi2(66.76,.323),r@np.linalg.solve(bao.covariance,r),places=10)
        self.assertTrue(np.isinf(bao.chi2(65,0)))

    def test_missing_probes_raise(self):
        from tempfile import TemporaryDirectory
        from aeqga.likelihoods.bao_cmb_problem import PlanckTTLikelihood
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                PlanckTTLikelihood(data_dir=directory, backend='camb')

    @unittest.skipUnless((PROJECT_ROOT/'data/cmb/COM_PowerSpect_CMB-TT-full_R3.01.txt').exists(),
                         'Run python -m scripts.fetch_cmb_data for TT integration tests')
    def test_tt_units_and_combination(self):
        from aeqga.likelihoods.bao_cmb_problem import BAOCMBProblem
        combined=BAOCMBProblem(backend='camb')
        self.assertEqual(len(combined.cmb.ell),2507)
        x=(66.76,.323)
        dl=combined.cmb.spectrum(*x)
        self.assertTrue(4000<dl[198]<7000)
        self.assertAlmostEqual(combined.compute_fitness(x),
             combined.cmb.chi2(*x)+combined.bao.chi2(*x),places=9)
        self.assertTrue(np.isinf(combined.compute_fitness([70,0])))

    @unittest.skipUnless((PROJECT_ROOT/'data/cmb/pico4_tailmonty_v35_py3.dat').exists(),
                         'Fetch the pinned PICO model for emulator integration tests')
    def test_pico_runtime_reference_and_domain_guard(self):
        import camb
        from aeqga.emulators.pico_runtime import load_pico
        from aeqga.likelihoods.bao_cmb_problem import PlanckTTLikelihood
        # Compare the model's own supported example against matching CAMB.
        p=load_pico(PROJECT_ROOT/'data/cmb/pico4_tailmonty_v35_py3.dat')
        from pypico import CantUsePICO
        inputs=p.example_inputs()
        dl=p.get(outputs=['lensed_TT'],**inputs)['lensed_TT']
        pars=camb.CAMBparams()
        pars.set_cosmology(cosmomc_theta=inputs['theta'],ombh2=inputs['ombh2'],
            omch2=inputs['omch2'],mnu=0,nnu=3.046,YHe=inputs['helium_fraction'],
            tau=inputs['re_optical_depth'])
        pars.InitPower.set_params(As=inputs['scalar_amp(1)'],
            ns=inputs['scalar_spectral_index(1)'],pivot_scalar=inputs['pivot_scalar'])
        pars.set_for_lmax(2508,lens_potential_accuracy=1)
        ref=camb.get_results(pars).get_cmb_power_spectra(CMB_unit='muK')['lensed_scalar'][:,0]
        np.testing.assert_allclose(dl[30:2509],ref[30:2509],rtol=.02,atol=2.)
        strict=PlanckTTLikelihood(backend='pico')
        with self.assertRaises(CantUsePICO): strict.chi2(72,.36)
        hybrid=PlanckTTLikelihood(backend='hybrid')
        self.assertTrue(np.isfinite(hybrid.chi2(72,.36)))
        self.assertEqual(hybrid.fallback_evaluations,1)


if __name__ == "__main__":
    unittest.main()
