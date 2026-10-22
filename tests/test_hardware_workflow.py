"""Offline failure-boundary tests for resumable hardware execution."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from qiskit import QuantumCircuit
from aeqga.execution.ibm_sampler import (
    DryRunExecutor, IBMSamplerExecutor, CheckpointManager, _counts,
    compute_circuit_fingerprint,
)
from aeqga.execution.workflow import run_checkpointed
from aeqga.steps.evolution.aeqga_algorithm import AEQGAParameters
from scripts.recover_ibm_job import recover_job


class ToyProblem:
    """Explicit synthetic test objective; never used by the scientific CLI."""
    lower_bounds = np.array([60., 0.])
    upper_bounds = np.array([80., .5])
    n_dim = 2

    def compute_fitness(self, x):
        return float((x[0]-72)**2 + 100*(x[1]-.3)**2)


class HardwareWorkflowTests(unittest.TestCase):
    def test_scientific_runner_comparison_and_summary_with_explicit_fixture(self):
        # Small generated table exercises the actual loader/likelihood, not a
        # substitute fitness function. It is not an astronomical fit claim.
        from aeqga.likelihoods.cosmology import distance_moduli
        from scripts.run_ibm_generation import run_generation
        from scripts.compare_generation_populations import run_comparison
        from scripts.analyze_hardware_runs import analyze
        with tempfile.TemporaryDirectory() as root:
            data = Path(root)/"data"
            data.mkdir()
            redshifts = np.array([.02, .05, .1, .2, .4, .8])
            mu = distance_moduli(redshifts, redshifts, 72., .3)
            header = "CID zHD zCMB zHEL MU_SH0ES MU_SH0ES_ERR_DIAG CEPH_DIST IS_CALIBRATOR USED_IN_SH0ES_HF\n"
            rows = [f"SN{i} {z} {z} {z} {m} .1 0 0 1\n" for i, (z, m) in enumerate(zip(redshifts, mu))]
            (data/"Pantheon+SH0ES.dat").write_text(header+"".join(rows))
            covariance = np.eye(len(redshifts))*.01
            (data/"Pantheon+SH0ES_STAT+SYS.cov").write_text(
                str(len(redshifts))+"\n"+"\n".join(map(str, covariance.ravel())))
            directory = Path(root)/"pilot"
            result = run_generation(data=data, directory=directory, population=8,
                                      generations=2, shots=64)
            self.assertEqual(result["evaluations"], 24)
            comparison = run_comparison(directory, data=data, samples=10)
            self.assertEqual(len(comparison["hardware_tvds"]), 4)
            summary = analyze(directory, data=data)
            self.assertEqual(summary["recorded_shots"], 2*4*64)
            self.assertTrue((directory/"hardware_convergence.png").is_file())

    def test_missing_scientific_data_fails_before_hardware_access(self):
        from scripts.run_ibm_generation import run_generation
        with tempfile.TemporaryDirectory() as root:
            with patch("scripts.run_ibm_generation.IBMSamplerExecutor") as executor:
                with self.assertRaises(FileNotFoundError):
                    run_generation(data=Path(root)/"missing", submit_hardware=True, confirm=True)
                executor.assert_not_called()

    def test_inflight_runtime_job_recovered_without_new_submission(self):
        from qiskit.providers.fake_provider import GenericBackendV2
        with tempfile.TemporaryDirectory() as root:
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.measure_all()
            job = Mock()
            job.job_id.return_value = "saved-job"
            job.result.side_effect = RuntimeError("lost connection")
            sampler = Mock()
            sampler.run.return_value = job
            service = Mock()
            service.job.return_value = job
            executor = IBMSamplerExecutor(shots=10, checkpoint_dir=root, service=service,
                                           backend=GenericBackendV2(2, seed=4))
            sdk = SimpleNamespace(SamplerV2=Mock(return_value=sampler))
            with patch.dict("sys.modules", {"qiskit_ibm_runtime": sdk}):
                with self.assertRaisesRegex(RuntimeError, "lost connection"):
                    executor.run([qc], generation=1)
                self.assertTrue((Path(root)/"generation_0001.logical.qpy").exists())
                job.result.side_effect = None
                job.result.return_value = [{"00": 5, "01": 5}]
                self.assertEqual(executor.run([qc], generation=1), [{"00": 5, "01": 5}])
            sampler.run.assert_called_once()
            service.job.assert_called_once_with("saved-job")

    def test_uncertain_submission_refuses_automatic_duplicate(self):
        with tempfile.TemporaryDirectory() as root:
            qc = QuantumCircuit(1)
            qc.measure_all()
            CheckpointManager.atomic_write(Path(root)/"generation_0001.json", dict(
                backend="test", shots=10, num_circuits=1, submission_state="submitting",
                circuit_fingerprints=[compute_circuit_fingerprint(qc)]))
            executor = IBMSamplerExecutor(backend_name="test", shots=10, checkpoint_dir=root)
            with self.assertRaisesRegex(RuntimeError, "uncertain"):
                executor.run([qc], generation=1)

    def test_mitigation_policy_and_preparation_equivalence(self):
        from aeqga.analysis.hardware_validation import corrected_probabilities
        from aeqga.steps.encoding.amplitude_encoding import build_amplitude_circuit
        from qiskit.quantum_info import Statevector
        values = np.array([1., 3., 2., 7.])
        np.testing.assert_allclose(
            Statevector(build_amplitude_circuit(values)).probabilities(),
            Statevector(build_amplitude_circuit(values, "state_preparation")).probabilities(), atol=1e-12)
        matrix = np.array([[.9, .2], [.1, .8]])
        probabilities, policy = corrected_probabilities({"0": 550, "1": 450}, matrix)
        np.testing.assert_allclose(probabilities, [.5, .5], atol=1e-12)
        projected, info = corrected_probabilities({"0": 1000}, matrix)
        np.testing.assert_allclose(projected, [1., 0.])
        self.assertGreater(info["clipped_mass"], 0)
        with self.assertRaises(ValueError):
            corrected_probabilities({"0": 10}, np.ones((2, 2))/2)

    def test_restart_after_counts_saved_before_decoding_is_exact(self):
        with tempfile.TemporaryDirectory() as root:
            params = AEQGAParameters(pop_size=8, max_gen=3, num_shots=128, progress_bar=False)
            reference_dir, restart_dir = Path(root)/"reference", Path(root)/"restart"
            reference_executor = DryRunExecutor(shots=128, checkpoint_dir=reference_dir, seed_simulator=4)
            reference = run_checkpointed(ToyProblem(), params, reference_executor,
                                         reference_dir, provenance={"test": True})
            interrupted_executor = DryRunExecutor(shots=128, checkpoint_dir=restart_dir, seed_simulator=4)
            original = interrupted_executor.run
            def interrupt(circuits, generation=0, **kwargs):
                result = original(circuits, generation=generation, **kwargs)
                if generation == 2:
                    raise RuntimeError("simulated power loss after counts persisted")
                return result
            interrupted_executor.run = interrupt
            with self.assertRaisesRegex(RuntimeError, "power loss"):
                run_checkpointed(ToyProblem(), params, interrupted_executor,
                                 restart_dir, provenance={"test": True})
            checkpoint = CheckpointManager.load(restart_dir/"generation_0002.json")
            interrupted_executor.run = original
            restarted = run_checkpointed(ToyProblem(), params, interrupted_executor,
                                         restart_dir, provenance={"test": True})
            self.assertEqual(checkpoint, CheckpointManager.load(restart_dir/"generation_0002.json"))
            for key in ("best_x", "best_chi2", "best_history", "population_history", "evaluations"):
                self.assertEqual(reference[key], restarted[key], key)
            # Fully complete restart must perform no circuit executions.
            interrupted_executor.run = Mock(side_effect=AssertionError("duplicate execution"))
            run_checkpointed(ToyProblem(), params, interrupted_executor,
                             restart_dir, provenance={"test": True})
            changed = copy.copy(params)
            changed.p_mut = .25
            with self.assertRaisesRegex(ValueError, "mismatch"):
                run_checkpointed(ToyProblem(), changed, interrupted_executor,
                                 restart_dir, provenance={"test": True})

    def test_complex_amplitudes_and_small_changes_affect_identity(self):
        def circuit(values):
            qc = QuantumCircuit(1)
            qc.initialize(values)
            qc.measure_all()
            return qc
        first = circuit([1/np.sqrt(2), 1j/np.sqrt(2)])
        other = circuit([1/np.sqrt(2), -1j/np.sqrt(2)])
        self.assertNotEqual(compute_circuit_fingerprint(first), compute_circuit_fingerprint(other))
        self.assertEqual(compute_circuit_fingerprint(first), compute_circuit_fingerprint(first.copy()))

    def test_remote_recovery_extracts_counts_and_updates_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"generation_0001.json"
            saved = dict(job_id="existing", submission_state="submitted", circuit_widths=[2],
                         shots=10, num_circuits=1, register_names=["meas"])
            CheckpointManager.atomic_write(path, saved)
            job = Mock()
            job.status.return_value = "DONE"
            job.result.return_value = [SimpleNamespace(data=SimpleNamespace(
                meas=SimpleNamespace(get_counts=lambda: {"01": 10})))]
            service = Mock()
            service.job.return_value = job
            self.assertEqual(recover_job(checkpoint_path=path, service=service), [{"01": 10}])
            service.job.assert_called_once_with("existing")
            service.run.assert_not_called()
            recover_job(checkpoint_path=path, service=service)
            service.job.assert_called_once()

    def test_reject_invalid_counts_and_cardinality(self):
        for counts in ({"00": -1, "01": 11}, {"00": 10.5}, {"xx": 10}, {"00": True}):
            with self.assertRaises(ValueError):
                _counts(counts, "meas", 10, 2)
        qc = QuantumCircuit(2)
        qc.measure_all()
        with self.assertRaises(ValueError):
            IBMSamplerExecutor(shots=10)._process_results([qc], [qc], [])
        for shots in (True, 1.5, 0):
            with self.assertRaises(ValueError):
                DryRunExecutor(shots=shots)

    def test_reject_mid_circuit_measurement_and_simulator_instructions(self):
        executor = DryRunExecutor(shots=10)
        qc = QuantumCircuit(1)
        qc.measure_all()
        qc.x(0)
        with self.assertRaises(ValueError):
            executor.run([qc])
        from qiskit_aer.library import save_statevector
        qc = QuantumCircuit(1)
        qc.save_statevector()
        qc.measure_all()
        with self.assertRaises(ValueError):
            executor.run([qc])


if __name__ == "__main__":
    unittest.main()
