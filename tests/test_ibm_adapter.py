"""Offline unit checks; no IBM account or QPU required."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from aeqga.execution.ibm_sampler import _counts, IBMSamplerExecutor


class TestIBMReadout(unittest.TestCase):
    def test_counts(self):
        pub = SimpleNamespace(data=SimpleNamespace(meas=SimpleNamespace(
            get_counts=lambda: {'00': 2, '11': 6})))
        self.assertEqual(_counts(pub, 'meas', 8, 2), {'00': 2, '11': 6})
        with self.assertRaises(ValueError): _counts(pub, 'meas', 9, 2)
        with self.assertRaises(ValueError): _counts(pub, 'meas', 8, 3)

    def test_reject_zero_shots(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                IBMSamplerExecutor(SimpleNamespace(), None, directory, shots=0)


if __name__ == '__main__':
    unittest.main()
