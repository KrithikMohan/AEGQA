"""Regression checks against committed Kingston records; no QPU access."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from aeqga.analysis.pr10_compatibility import (
    analyze_pilot, analyze_generation, completed_legacy_counts,
)
from aeqga.paths import PROJECT_ROOT
from aeqga.execution.ibm_sampler import IBMSamplerExecutor
from scripts.recover_ibm_job import recover_job


class PR10CompatibilityTests(unittest.TestCase):
    def test_pilot_reproduces_recorded_distribution(self):
        result = analyze_pilot()
        self.assertEqual(result["shots"], 1024)
        self.assertEqual(result["observed_states"], 16)
        self.assertAlmostEqual(result["tvd"], .11056529636640379, places=12)
        self.assertAlmostEqual(result["finite_shot_p99"], .06498679, places=7)
        self.assertEqual(result["trials_at_least_observed"], 0)

    def test_generation_matches_all_four_archived_arrays(self):
        if not (PROJECT_ROOT / "sn_data/PantheonPlus/Pantheon+SH0ES.dat").exists():
            self.skipTest("Real Pantheon+ input is required for historical compatibility")
        result = analyze_generation()
        self.assertEqual(result["evaluations"], 64)
        self.assertAlmostEqual(result["best_chi2"], 1778.7556669812602, places=7)
        self.assertTrue(result["retained_elites_equal"])
        self.assertEqual(set(result["archived_population_max_abs_errors"].values()), {0.0})

    def test_completed_legacy_recovery_is_read_only_and_never_connects(self):
        service = Mock()
        for name, expected in (("ibm_first_pilot", 1), ("ibm_generation_01", 4)):
            path = PROJECT_ROOT / "outputs/json" / name / "generation_0001.json"
            original = path.read_bytes()
            counts = recover_job(checkpoint_path=path, service=service)
            self.assertEqual(len(counts), expected)
            self.assertTrue(all(sum(c.values()) == 1024 for c in counts))
            self.assertEqual(path.read_bytes(), original)
        service.job.assert_not_called()

    def test_legacy_ledger_rejects_corruption_and_unfinished_submission(self):
        path = PROJECT_ROOT / "outputs/json/ibm_generation_01/generation_0001.json"
        record = json.loads(path.read_text())
        bad = copy.deepcopy(record)
        bad["counts"] = bad["counts"][:1]
        with self.assertRaises(ValueError):
            completed_legacy_counts(bad, [4, 3, 4, 3])
        bad = copy.deepcopy(record)
        bad["counts"][0]["0000"] += 1
        with self.assertRaisesRegex(ValueError, "Shot total"):
            completed_legacy_counts(bad, [4, 3, 4, 3])
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "submitted.json"
            record["state"] = "submitted"
            file.write_text(json.dumps(record))
            service = Mock()
            with self.assertRaisesRegex(ValueError, "Legacy recovery"):
                recover_job(checkpoint_path=file, service=service)
            service.job.assert_not_called()

    def test_current_adapter_refuses_unverified_historical_checkpoint(self):
        from qiskit import QuantumCircuit
        path = PROJECT_ROOT / "outputs/json/ibm_first_pilot/generation_0001.json"
        circuit = QuantumCircuit(4)
        circuit.measure_all()
        executor = IBMSamplerExecutor(backend_name="ibm_kingston", shots=1024,
                                       checkpoint_dir=path.parent, service=Mock())
        with self.assertRaisesRegex(ValueError, "mismatch"):
            executor.run([circuit], generation=1)


if __name__ == "__main__":
    unittest.main()
