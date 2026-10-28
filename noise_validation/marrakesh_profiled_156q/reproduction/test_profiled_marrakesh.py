"""Analytic M, full occupancy, and factorized-stage noise equivalence."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
from qiskit import QuantumCircuit
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits

from aeqga import AEQGA
from aeqga_factorized_noise import simulate_block, stage_circuits, normalized, compile_stable_native
from aeqga_ibm_noise import SnapshotBackend, simulator, in_classical_order, stage_measurement_wires
from aeqga_parallel import connected_placements, marginal_counts
from aeqga_profiled_ibm import ProfiledIBMNoiseAEQGA
from aeqga_profiled_pantheon import ProfiledPantheon
from run_pantheon_aeqga import make_model


class ProfiledMarrakeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread_limit = threadpool_limits(limits=2)
        cls.problem = ProfiledPantheon()

    @classmethod
    def tearDownClass(cls):
        cls.thread_limit.restore_original_limits()

    def test_analytic_m_matches_direct_full_covariance_and_bounds(self):
        p = self.problem
        coordinates = np.vstack(([60,.1],[80,.5],[70,.3],
            np.random.default_rng(4).uniform([60,.1],[80,.5],(17,2))))
        profiled = p.evaluate(coordinates)
        for parameters, (loss, magnitude, raw) in zip(coordinates, profiled):
            mu = make_model(64)(p.x.tolist(), [*parameters, 0])
            white = solve_triangular(p.cholesky, p.y-mu, lower=True)
            expected = (p.ones_white @ white)/p.m_denominator
            self.assertAlmostEqual(raw,expected,places=9)
            self.assertAlmostEqual(magnitude,np.clip(expected,*p.magnitude_bounds),places=9)
            self.assertAlmostEqual(loss,p.direct_check(parameters,magnitude),places=7)
            for delta in (-.001,.001):
                alternative=np.clip(magnitude+delta,*p.magnitude_bounds)
                self.assertGreaterEqual(p.direct_check(parameters,alternative)+1e-7,loss)

    def test_active_m_bound_and_cache_retention(self):
        p=ProfiledPantheon(data=(self.problem.x,self.problem.y,self.problem.covariance),
                           magnitude_bounds=(-19.1,-19.0))
        params=[[73.5,.33],[70,.3]]
        values=p.evaluate(params)
        self.assertEqual(values[0,1],-19.1)
        self.assertLess(values[0,2],-19.1)
        self.assertAlmostEqual(values[0,0],p.direct_check(params[0],-19.1),places=7)
        p.retain([params[0]])
        self.assertEqual(set(p.cache),{tuple(params[0])})

    def test_probability_guard_and_small_trace_drift(self):
        np.testing.assert_allclose(normalized(np.array([.2,.8])*(1-2e-6)),[.2,.8],atol=1e-14)
        with self.assertRaises(AssertionError): normalized([.2,.79])
        with self.assertRaises(AssertionError): normalized([-.01,1.01])

    def test_state_preparation_roundoff_regression(self):
        fixture=json.loads((Path(__file__).parent/'state_preparation_regression.json').read_text())
        amplitude=np.pad(fixture['nonzero_amplitudes'],(0,256-len(fixture['nonzero_amplitudes'])))
        logical=QuantumCircuit(8)
        logical.initialize(amplitude,range(8))
        logical.measure_all()
        backend=SnapshotBackend('marrakesh',8)
        native=compile_stable_native(logical,backend,1252080562)
        repairs=native.metadata.get('state_preparation_roundoff_repairs',[])
        self.assertEqual(len(repairs),1)
        self.assertLess(repairs[0]['max_amplitude_change'],1e-14)
        order=stage_measurement_wires(native)
        circuit=native.remove_final_measurements(inplace=False)
        circuit.save_probabilities()
        result=simulator(backend,'ideal',readout=False,method='statevector').run(circuit,shots=1).result()
        observed=in_classical_order(result.data()['probabilities'],order)
        expected=amplitude**2/np.sum(amplitude**2)
        # Qiskit's isometry synthesis has its own numerical approximation.
        # Bound its distribution error independently of the IBM noise model.
        self.assertLess(float(np.abs(normalized(observed)-expected).sum()/2),1e-6)

    def test_layout_full_budget_and_four_qubit_limit(self):
        backend=SnapshotBackend('marrakesh',156)
        adj={q:set() for q in range(156)}
        for a,b in backend.target['cz']: adj[a].add(b); adj[b].add(a)
        for width,count in ((3,52),(4,38),(5,31),(6,26),(7,22),(8,19)):
            placements=connected_placements([width]*count,adj)
            self.assertEqual(len(set(sum(placements,[]))),width*count)
            for block in placements:
                seen,pending={block[0]},[block[0]]
                while pending:
                    for q in adj[pending.pop()] & set(block)-seen:
                        seen.add(q);pending.append(q)
                self.assertEqual(seen,set(block))
        with self.assertRaisesRegex(ValueError,'No connected layout'):
            connected_placements([4]*39,adj)

    def test_stage_factorization_matches_direct_complete_circuit_sampling(self):
        model=AEQGA(lambda x,p:0,[70,.3],bounds=[(60,80),(.1,.5)],active_qubits=3,
                    parallel_circuits=1,shots=100,seed=4)
        rng=np.random.default_rng(8)
        logical=model._subpopulation_circuit(np.array([[63,.12],[78,.46]]),
            np.array([[61,.13],[66,.22],[74,.37],[79,.48]]),rng,0)
        physical=[0,1,2]
        native,stages,_=simulate_block((logical,physical,'marrakesh',
            Path(__file__).parent/'ibm_backend_snapshots',12,'noisy'))
        backend=SnapshotBackend('marrakesh',physical_qubits=physical)
        shots=8192
        direct=simulator(backend,'noisy',method='matrix_product_state').run(
            native,shots=shots,seed_simulator=415).result()
        self.assertTrue(direct.success)
        counts=direct.get_counts()
        self.assertEqual(len(stage_circuits(native)),4)
        for spec,stage in zip(logical.metadata['stages'],stages):
            observed=AEQGA._counts(marginal_counts(counts,spec['clbits']),8)/shots
            probability=stage['probabilities']
            tolerance=6*np.sqrt(probability*(1-probability)/shots)+.002
            self.assertTrue(np.all(np.abs(observed-probability) < tolerance),
                            (spec,observed,probability))
        # Correlations between stages disappear after the all-wire resets.
        p0=stages[0]['probabilities'][0]; p1=stages[1]['probabilities'][0]
        actual=sum(n for bits,n in counts.items() if int(bits,2)&63==0)/shots
        expected=p0*p1
        self.assertLess(abs(actual-expected),6*np.sqrt(expected*(1-expected)/shots)+.002)

    def test_full_profiled_training_histories_and_packed_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            model=ProfiledIBMNoiseAEQGA(self.problem,record_directory=directory,
                active_qubits=7,parallel_circuits=2,shots=100,seed=7,mode='noisy')
            model.train(generations=2,verbose=False)
            self.assertEqual((model.population_size,model.executed_circuits_,model.quantum_stages_),(16,2,16))
            self.assertEqual(model.n_evaluations_,16+2*12)
            for generation,entry in enumerate(model.history_):
                for sample in entry['population']:
                    self.assertEqual(len(sample['all_parameters']),3)
                    self.assertAlmostEqual(sample['chi2'],self.problem.direct_check(
                        sample['p'],sample['M_profiled']),places=7)
                if generation:
                    previous=model.history_[generation-1]['population']
                    indices=np.argsort([s['loss'] for s in previous],kind='stable')[:4].tolist()
                    self.assertEqual(entry['elite_selection']['global_elite_indices'],indices)
            for entry in model.execution_history_:
                self.assertEqual(entry['used_active_qubits'],6)
                self.assertEqual(len(entry['stage_diagnostics']),8)
                self.assertEqual(sum(entry['executions'][0]['counts'].values()),100)
                self.assertTrue(all(sum(s['counts'].values())==100 for s in entry['stage_measurements']))
            self.assertEqual(np.asarray(model.probability_history_).shape,(2,2,4,2,8))
            model.export_population_history(Path(directory)/'candidates.csv')
            with gzip.open(Path(directory)/'training_history.jsonl.gz','rt') as stream:
                saved=[json.loads(line) for line in stream]
            self.assertEqual(len(saved),3)
            self.assertEqual(saved[-1]['all_parameters'],model.history_[-1]['all_parameters'])


if __name__=='__main__':
    unittest.main()
