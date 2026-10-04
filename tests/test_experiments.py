import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from qiskit.quantum_info import Statevector
from aeqga.steps.evolution import aeqga_algorithm as algorithm
from aeqga.experiments.runner import run_ensemble, validate_history
from aeqga.visualization.scientific_plots import density_contours, joint_levels, outcome_statistics, run_diagnostics


class Bowl:
    n_dim = 2
    lower_bounds = np.array([0.,0.])
    upper_bounds = np.array([1.,1.])
    def compute_fitness(self,x): return float(np.sum((np.asarray(x)-.4)**2))


class ExperimentTests(unittest.TestCase):
    def test_repeatability_each_backend(self):
        params = algorithm.AEQGAParameters(pop_size=8,max_gen=2,num_shots=256,progress_bar=False)
        for engine in ["aer","statevector_shots"]:
            first = algorithm.run_aeqga_dual(Bowl(),params,seed=44,execution_mode=engine,quiet=True)
            second = algorithm.run_aeqga_dual(Bowl(),params,seed=44,execution_mode=engine,quiet=True)
            np.testing.assert_array_equal(first[1],second[1])
            self.assertEqual(first[2],second[2])

    def test_fast_finite_shots_match_post_gate_probabilities(self):
        params = algorithm.AEQGAParameters(pop_size=16,max_gen=1,num_shots=32768,
                                           p_cross=1,p_mut=1,progress_bar=False)
        original_build = algorithm.build_dimension_circuit
        original_decode = algorithm.decode_random_subset
        circuits, counts = [], []
        def build(*args,**kwargs):
            circuit = original_build(*args,**kwargs)
            circuits.append(circuit)
            return circuit
        def decode(c,*args):
            counts.append(c)
            return original_decode(c,*args)
        with patch.object(algorithm,"build_dimension_circuit",side_effect=build), patch.object(algorithm,"decode_random_subset",side_effect=decode):
            algorithm.run_aeqga_dual(Bowl(),params,seed=77,execution_mode="statevector_shots",quiet=True)
        for circuit, c in zip(circuits[::2],counts):
            expected = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False)).probabilities()
            actual = np.array([c[format(i,f"0{circuit.num_qubits}b")] for i in range(len(expected))])
            self.assertEqual(actual.sum(),32768)
            np.testing.assert_allclose(actual/32768,expected,atol=.012,rtol=0)

    def test_checkpoint_resume_and_mismatch(self):
        params = algorithm.AEQGAParameters(pop_size=8,max_gen=2,num_shots=256,progress_bar=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"experiment.json"
            result = run_ensemble(Bowl(),params,[11,12],path,provenance={"dataset":"test"})
            with patch("aeqga.experiments.runner.run_aeqga_dual",side_effect=AssertionError("must reuse checkpoint")):
                resumed = run_ensemble(Bowl(),params,[11,12],path,provenance={"dataset":"test"})
            self.assertEqual(result,resumed)
            for change in [{"seeds":[11,13],"provenance":{"dataset":"test"}},
                           {"seeds":[11,12],"provenance":{"dataset":"other"}}]:
                with self.assertRaises(ValueError): run_ensemble(Bowl(),params,checkpoint=path,**change)
            record = result["runs"][0]
            record["best_history"][0] += 1
            with self.assertRaises(ValueError): validate_history(Bowl(),params,record)

    def test_partial_checkpoint_resumes_only_missing_run(self):
        params = algorithm.AEQGAParameters(pop_size=8,max_gen=1,num_shots=128,progress_bar=False)
        original = algorithm.run_aeqga_dual
        def interrupted(*args,**kwargs):
            if kwargs["seed"] == 12: raise RuntimeError("interruption")
            return original(*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"experiment.json"
            with patch("aeqga.experiments.runner.run_aeqga_dual",side_effect=interrupted):
                with self.assertRaises(RuntimeError): run_ensemble(Bowl(),params,[11,12],path,provenance={})
            self.assertEqual(json.loads(path.read_text())["completed"],[0])
            with patch("aeqga.experiments.runner.run_aeqga_dual",wraps=original) as run:
                result = run_ensemble(Bowl(),params,[11,12],path,provenance={})
                self.assertEqual(run.call_count,1)
            self.assertEqual(len(result["runs"]),2)

    def test_joint_thresholds_and_mass_integrated_kde(self):
        np.testing.assert_allclose(joint_levels((1,2,3)),[2.2957489,6.1800743,11.8291581],atol=1e-6)
        points = np.random.default_rng(91).normal(size=(1000,2))*[.22,.016]+[72.81,.362]
        density = density_contours(points,grid_size=120)
        self.assertGreater(density["captured_mass"],.995)
        for threshold,mass in zip(density["thresholds"],density["masses"]):
            dx,dy=np.diff(density["x"])[0],np.diff(density["y"])[0]
            measured=density["density"][density["density"]>=threshold].sum()*dx*dy
            self.assertAlmostEqual(measured,mass,delta=.006)
        with self.assertRaises(ValueError): outcome_statistics(np.ones((20,2)))
        with self.assertRaises(ValueError): density_contours(points[:10])

    def test_plot_lines_use_actual_chi2_parameter_pairs(self):
        record=dict(best_history=[2000.,1753.],best_parameter_history=[[70.,.2],[72.8,.36]])
        reference=dict(H0=72.83,Omega_m=.363,chi2=1752.9)
        fig=run_diagnostics(record,reference)
        expected=[[2000,1753],[247.1,.1],[70,72.8],[.2,.36]]
        for ax, values in zip(fig.axes,expected):
            np.testing.assert_allclose(ax.lines[0].get_ydata(),values)
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == '__main__': unittest.main()
