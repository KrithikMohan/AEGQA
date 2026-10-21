"""
test_ibm_adapter.py
===================
Unit and integration tests for IBM Quantum Hardware execution adapter for AEQGA.

Tests:
1. test_counts: Count extraction and bitstring padding.
2. test_reject_zero_shots: Validation that shots <= 0 raises ValueError.
3. test_shot_total_validation: Detection of shot sum discrepancies.
4. test_correct_measurement_register_width_and_backend_wide_circuits:
   Ensures 156-qubit transpiled device width does not raise invalid register width error.
5. test_correct_bit_ordering: Little-endian bit ordering verification.
6. test_four_circuit_ordering: Validates circuit width order [4, 3, 4, 3].
7. test_offline_generation_execution: Full generation with DryRunExecutor.
8. test_deterministic_circuit_construction: Deterministic fingerprints under seed.
9. test_ideal_probability_equivalence_after_transpilation: Aer MPS equivalence check.
10. test_checkpoint_atomic_write_and_reading: Atomic file creation and loading.
11. test_recovery_without_duplicate_submissions: Job recovery from saved checkpoint.
12. test_rejection_of_mismatched_experiment_fingerprints: Fingerprint collision protection.
13. test_preservation_of_existing_aer_and_statevector_modes: Regression test for simulation.
"""

import sys
import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aeqga.execution.ibm_sampler import (
    BaseExecutor,
    IBMSamplerExecutor,
    DryRunExecutor,
    CheckpointManager,
    CompilationDiagnostics,
    _counts,
    compute_circuit_fingerprint,
    verify_mps_probability_equivalence,
)
from aeqga.analysis import (
    generate_frozen_generation_circuits,
    compute_circuit_ideal_probabilities,
)
from aeqga.likelihoods.pantheon_problem import PantheonPlusProblem
from aeqga.steps.evolution.aeqga_algorithm import AEQGAParameters, run_aeqga_dual


class TestIBMAdapter(unittest.TestCase):
    """Test suite for IBM Quantum execution adapter and offline verification."""

    def setUp(self):
        self.test_dir = REPO_ROOT / "outputs" / "json" / "test_scratch_checkpoints"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_counts(self):
        """Test count extraction, zero-padding, and shot total validation."""
        # 1. Raw dictionary format
        mock_raw = {"0": 512, "1": 512}
        extracted = _counts(mock_raw, register_name="meas", shots=1024, logical_width=2)
        self.assertEqual(extracted, {"00": 512, "01": 512})
        self.assertEqual(sum(extracted.values()), 1024)

        # 2. Mock PubResult with .data attribute
        class MockPubData:
            def __init__(self, counts_dict):
                self._counts = counts_dict

            def get_counts(self):
                return self._counts

        class MockPubResult:
            def __init__(self, data_dict):
                self.data = MagicMock()
                self.data.meas = MockPubData(data_dict)

        pub = MockPubResult({"0011": 400, "1100": 624})
        res = _counts(pub, register_name="meas", shots=1024, logical_width=4)
        self.assertEqual(res, {"0011": 400, "1100": 624})
        self.assertEqual(sum(res.values()), 1024)

    def test_reject_zero_shots(self):
        """Test that shots <= 0 is rejected with ValueError."""
        with self.assertRaises(ValueError):
            DryRunExecutor(shots=0)
        with self.assertRaises(ValueError):
            IBMSamplerExecutor(shots=-10)

    def test_shot_total_validation(self):
        """Test that mismatch in shot sum raises ValueError."""
        bad_counts = {"00": 500, "01": 500}  # total 1000 != 1024
        with self.assertRaises(ValueError):
            _counts(bad_counts, register_name="c0", shots=1024, logical_width=2)

    def test_correct_measurement_register_width_and_backend_wide_circuits(self):
        """Ensure 156-qubit compiled circuit compares classical register against logical width."""
        # 4-qubit circuit with 4-bit classical register
        qc = QuantumCircuit(4, 4)
        qc.h(range(4))
        qc.measure(range(4), range(4))

        fake_156 = GenericBackendV2(num_qubits=156, seed=42)
        compiled = transpile(qc, backend=fake_156, optimization_level=1, seed_transpiler=42)

        # Compiled circuit has 156 physical qubits, but 4 classical bits
        self.assertEqual(compiled.num_qubits, 156)
        self.assertEqual(len(compiled.cregs), 1)
        self.assertEqual(compiled.cregs[0].size, 4)

        # Adapter validation logic: must compare against original.num_qubits (4), NOT compiled.num_qubits (156)
        executor = IBMSamplerExecutor(shots=1024, mode="dry_run")
        # Process results with mock result matching 4-bit register
        mock_result = [{"0000": 1024}]
        processed = executor._process_results([qc], [compiled], mock_result)
        self.assertEqual(processed[0], {"0000": 1024})

        # Failure case: classical register width does not match logical width
        qc_bad = QuantumCircuit(4, 3)
        qc_bad.measure([0, 1, 2], [0, 1, 2])
        compiled_bad = transpile(qc_bad, backend=fake_156, optimization_level=1, seed_transpiler=42)
        with self.assertRaises(ValueError):
            executor._process_results([qc], [compiled_bad], mock_result)

    def test_correct_bit_ordering(self):
        """Test little-endian bit ordering verification."""
        # Basis state |1> on qubit 0 in 4-qubit circuit should be '0001'
        qc = QuantumCircuit(4, 4)
        qc.x(0)
        qc.measure(range(4), range(4))

        sim = DryRunExecutor(shots=100)
        counts = sim.run([qc])[0]
        self.assertEqual(counts, {"0001": 100})

    def test_four_circuit_ordering(self):
        """Validates that 4-circuit generation batch has widths [4, 3, 4, 3]."""
        prob = PantheonPlusProblem()
        _, _, circuits, meta = generate_frozen_generation_circuits(
            prob, pop_size=32, seed=42
        )
        self.assertEqual(len(circuits), 4)
        widths = [c.num_qubits for c in circuits]
        self.assertEqual(widths, [4, 3, 4, 3])
        self.assertEqual(meta["circuit_widths"], [4, 3, 4, 3])

    def test_offline_generation_execution(self):
        """Test a full 1-generation AEQGA run using DryRunExecutor."""
        prob = PantheonPlusProblem()
        params = AEQGAParameters(
            pop_size=32,
            max_gen=1,
            num_shots=1024,
            progress_bar=False,
        )
        executor = DryRunExecutor(shots=1024, seed_simulator=42)
        g_best, pop_evol, bests_log = run_aeqga_dual(
            prob,
            params,
            hardware_executor=executor,
            seed=42,
            quiet=True,
        )
        self.assertEqual(len(pop_evol), 2)  # initial + generation 1
        self.assertEqual(pop_evol[0].shape, (32, 2))
        self.assertEqual(pop_evol[1].shape, (32, 2))
        self.assertIsNotNone(g_best.x)
        self.assertGreater(g_best.fitness, 0.0)

    def test_deterministic_circuit_construction(self):
        """Test deterministic circuit construction produces identical fingerprints."""
        prob = PantheonPlusProblem()
        _, _, c1, _ = generate_frozen_generation_circuits(prob, pop_size=32, seed=42)
        _, _, c2, _ = generate_frozen_generation_circuits(prob, pop_size=32, seed=42)

        fps1 = [compute_circuit_fingerprint(c) for c in c1]
        fps2 = [compute_circuit_fingerprint(c) for c in c2]
        self.assertEqual(fps1, fps2)

    def test_fingerprint_changes_with_encoded_amplitudes(self):
        from aeqga.steps.encoding.amplitude_encoding import build_amplitude_circuit
        first = build_amplitude_circuit(np.arange(1., 9.))
        second = build_amplitude_circuit(np.arange(11., 19.))
        self.assertNotEqual(compute_circuit_fingerprint(first), compute_circuit_fingerprint(second))

    def test_ideal_probability_equivalence_after_transpilation(self):
        """Test Aer MPS simulation verifies ideal output probability equivalence."""
        qc = QuantumCircuit(4)
        qc.h(range(4))
        qc.cx(0, 1)
        qc.cx(2, 3)

        fake_156 = GenericBackendV2(num_qubits=156, seed=42)
        transpiled = transpile(qc, backend=fake_156, optimization_level=1, seed_transpiler=42)

        max_diff = verify_mps_probability_equivalence(qc, transpiled, atol=1e-10)
        self.assertLess(max_diff, 1e-10)

    def test_checkpoint_atomic_write_and_reading(self):
        """Test atomic writing of checkpoint files and reading."""
        ckpt_path = self.test_dir / "generation_0001.json"
        data = {
            "generation": 1,
            "backend": "ibm_kingston",
            "shots": 1024,
            "num_circuits": 4,
            "submission_state": "completed",
        }
        CheckpointManager.atomic_write(ckpt_path, data)
        self.assertTrue(ckpt_path.exists())

        loaded = CheckpointManager.load(ckpt_path)
        self.assertEqual(loaded["generation"], 1)
        self.assertEqual(loaded["backend"], "ibm_kingston")

    def test_recovery_without_duplicate_submissions(self):
        """Test recovery from an already completed checkpoint without resubmission."""
        ckpt_path = self.test_dir / "generation_0001.json"
        fake_counts = [{"00": 1024}]
        saved_data = {
            "generation": 1,
            "backend": "ibm_kingston",
            "shots": 1024,
            "num_circuits": 1,
            "circuit_widths": [2],
            "circuit_fingerprints": ["fake_fp"],
            "job_id": "job_12345",
            "submission_state": "completed",
            "raw_counts": fake_counts,
        }
        CheckpointManager.atomic_write(ckpt_path, saved_data)

        # Build dummy 2-qubit circuit with same fingerprint
        qc = QuantumCircuit(2, 2)
        qc.measure(range(2), range(2))

        executor = IBMSamplerExecutor(
            backend_name="ibm_kingston",
            shots=1024,
            checkpoint_dir=self.test_dir,
            mode="dry_run",
        )
        # Mock compute_circuit_fingerprint to match saved
        import aeqga.execution.ibm_sampler as ibm_mod
        orig_fp = ibm_mod.compute_circuit_fingerprint
        ibm_mod.compute_circuit_fingerprint = lambda c: "fake_fp"
        try:
            counts = executor.run([qc], generation=1)
            self.assertEqual(counts, fake_counts)
        finally:
            ibm_mod.compute_circuit_fingerprint = orig_fp

    def test_rejection_of_mismatched_experiment_fingerprints(self):
        """Test that mismatched configuration raises ValueError rather than reusing bad results."""
        ckpt_path = self.test_dir / "generation_0001.json"
        saved_data = {
            "generation": 1,
            "backend": "ibm_kingston",
            "shots": 512,  # saved with 512 shots
            "num_circuits": 1,
            "submission_state": "completed",
            "raw_counts": [{"00": 512}],
        }
        CheckpointManager.atomic_write(ckpt_path, saved_data)

        qc = QuantumCircuit(2, 2)
        qc.measure(range(2), range(2))

        executor = DryRunExecutor(
            shots=1024,  # requesting 1024 shots
            checkpoint_dir=self.test_dir,
            backend_name="ibm_kingston",
        )
        with self.assertRaises(ValueError) as ctx:
            executor.run([qc], generation=1)
        self.assertIn("mismatch", str(ctx.exception).lower())

    def test_preservation_of_existing_aer_and_statevector_modes(self):
        """Verify standard Aer and statevector execution pathways remain fully intact."""
        prob = PantheonPlusProblem()
        params = AEQGAParameters(pop_size=8, max_gen=2, num_shots=100, progress_bar=False)

        # 1. Aer mode
        g_best_aer, evol_aer, _ = run_aeqga_dual(
            prob, params, execution_mode="aer", seed=42, quiet=True
        )
        self.assertEqual(len(evol_aer), 3)  # gen 0, 1, 2
        self.assertIsNotNone(g_best_aer.x)

        # 2. Statevector mode
        params_sv = AEQGAParameters(pop_size=8, max_gen=2, num_shots=0, progress_bar=False)
        g_best_sv, evol_sv, _ = run_aeqga_dual(
            prob, params_sv, execution_mode="statevector", seed=42, quiet=True
        )
        self.assertEqual(len(evol_sv), 3)
        self.assertIsNotNone(g_best_sv.x)


if __name__ == "__main__":
    unittest.main()
