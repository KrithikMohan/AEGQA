"""Validation: conda run -n qml python -m unittest -v test_aeqga"""

import csv
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from qiskit.quantum_info import Statevector

from aeqga import AEQGA


def line(x, p):
    return p[0] * x[0] + p[1]


class AEQGATests(unittest.TestCase):
    def test_full_population_history_scores_and_export(self):
        x = np.array([0., 1., 2.])
        y = np.array([1., 2., 3.])
        covariance = np.array([[2., .3, 0.], [.3, 1., .2], [0., .2, 1.5]])
        for weights in ({}, {"sigma": [.5, 1., 2.]}, {"covariance": covariance}):
            with self.subTest(weights=list(weights)):
                fitter = AEQGA(line, [0, 0], bounds=[(-4, 4), (-2, 2)],
                               population_size=8, shots=64, seed=7)
                fitter.train(x, y, generations=3, **weights)
                kind = "chi2" if weights else "sse"
                self.assertEqual(fitter.loss_kind_, kind)
                self.assertEqual([h["generation"] for h in fitter.history_], [0, 1, 2, 3])
                for record in fitter.history_:
                    self.assertEqual(len(record["population"]), 8)
                    self.assertEqual(record["loss_kind"], kind)
                    for candidate in record["population"]:
                        residual = candidate["p"][0] * x + candidate["p"][1] - y
                        if "covariance" in weights:
                            expected = residual @ np.linalg.solve(covariance, residual)
                        elif "sigma" in weights:
                            expected = np.sum((residual / weights["sigma"]) ** 2)
                        else:
                            expected = residual @ residual
                        self.assertAlmostEqual(candidate["loss"], expected)
                        self.assertEqual(candidate["chi2"], candidate["loss"] if weights else None)
                    best = record["population"][record["best_index"]]
                    self.assertEqual(best["p"], record["p"])
                    self.assertEqual(best["loss"], record["loss"])
                    self.assertEqual(record["loss"], min(c["loss"] for c in record["population"]))
                    self.assertIsNot(best["p"], record["p"])
                # Compare the cached elite scores/parameters with the previous population.
                for previous, current in zip(fitter.history_, fitter.history_[1:]):
                    elites = sorted(previous["population"], key=lambda c: c["loss"])[:2]
                    self.assertEqual(current["population"][:2], elites)
                    for old, new in zip(elites, current["population"][:2]):
                        self.assertIsNot(old["p"], new["p"])
                with tempfile.TemporaryDirectory() as directory:
                    path = fitter.export_population_history(
                        Path(directory) / "nested" / "candidates.csv",
                        parameter_names=["slope", "intercept"])
                    with path.open(newline="", encoding="utf-8") as stream:
                        reader = csv.DictReader(stream)
                        self.assertEqual(reader.fieldnames,
                                         ["generation", "individual", "slope", "intercept", kind])
                        rows = list(reader)
                    self.assertEqual(len(rows), 32)
                    for i, row in enumerate(rows):
                        candidate = fitter.history_[i // 8]["population"][i % 8]
                        self.assertEqual((int(row["generation"]), int(row["individual"])), (i // 8, i % 8))
                        self.assertEqual([float(row["slope"]), float(row["intercept"])], candidate["p"])
                        self.assertEqual(float(row[kind]), candidate["loss"])

    def test_population_history_reset_early_stop_and_invalid_candidate(self):
        fitter = AEQGA(lambda x, p: float("nan") if p[0] < 0 else p[0] * x[0],
                       [.2], bounds=[(-1., 1.)], population_size=8, seed=3, shots=64)
        fitter.train([1., 2.], [.6, 1.2], generations=2, sigma=1.)
        initial = fitter.history_[0]["population"]
        self.assertTrue(any(np.isinf(c["chi2"]) for c in initial))
        old_history = fitter.history_
        saved = copy.deepcopy(old_history)
        fitter.train([1., 2.], [.6, 1.2], generations=3, target_loss=1e9)
        self.assertEqual(old_history, saved)
        self.assertEqual(len(fitter.history_), 1)
        self.assertEqual(len(fitter.history_[0]["population"]), 8)
        self.assertTrue(all(c["chi2"] is None for c in fitter.history_[0]["population"]))

    def test_population_export_validation(self):
        fitter = self.make_line()
        with self.assertRaisesRegex(ValueError, "Train first"):
            fitter.export_population_history("unused.csv")
        fitter.train([0, 1], [1, 2], generations=1)
        for names in (["one"], ["a", "a"], ["generation", "b"], ["a", "chi2"], ["", "b"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                fitter.export_population_history("unused.csv", parameter_names=names)

    def test_population_count_defaults_and_explicit_defaults_match(self):
        for size in (8, 16, 32, 64):
            fitter = AEQGA(line, [0, 0], bounds=[(-4, 4), (-2, 2)], population_size=size)
            self.assertEqual((fitter.elite_count, fitter.local_count, fitter.random_count),
                             (size // 4, size // 4, size // 2))
        first = self.make_line()
        second = self.make_line(elite_count=4, local_count=4, random_count=8)
        for fitter in (first, second):
            fitter.train([0, 1, 2], [1, 2, 3], generations=3)
        self.assertEqual(first.history_, second.history_)

    def test_custom_population_counts_with_padding_and_disabled_branches(self):
        for elite, local, random in ((5, 11, 16), (12, 4, 16), (1, 1, 30),
                                     (16, 16, 0), (16, 0, 16), (32, 0, 0)):
            with self.subTest(counts=(elite, local, random)):
                fitter = AEQGA(line, [0, 0], bounds=[(-4, 4), (-2, 2)],
                               elite_count=elite, local_count=local, random_count=random,
                               seed=2, shots=64)
                original = fitter._quantum_offspring
                group_sizes = []

                def checked_groups(elites, random_part, rng):
                    self.assertEqual(elites.shape, (elite, 2))
                    self.assertEqual(random_part.shape, (random, 2))
                    a, b = original(elites, random_part, rng)
                    group_sizes.append((len(a), len(b)))
                    return a, b

                with patch.object(fitter, "_quantum_offspring", side_effect=checked_groups):
                    fitter.train([0, 1, 2], [1, 2, 3], generations=2)
                self.assertEqual(group_sizes, [(local, random)] * 2)
                self.assertEqual(fitter.n_evaluations_, 32 + 2 * (local + random))
                self.assertEqual(fitter.quantum_circuits_, 2 * 2 * ((local > 0) + (random > 0)))
                self.assertTrue(np.all(np.diff([h["loss"] for h in fitter.history_]) <= 0))
                self.assertTrue(np.isfinite(fitter.loss_))
                self.assertTrue(-4 <= fitter.p[0] <= 4 and -2 <= fitter.p[1] <= 2)

    def test_population_count_validation(self):
        for counts in ({"elite_count": 0}, {"elite_count": True}, {"local_count": -1},
                       {"random_count": 1.5}, {"local_count": False},
                       {"elite_count": 4, "local_count": 4, "random_count": 4}):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                AEQGA(line, [0, 0], bounds=[(-4, 4), (-2, 2)], **counts)

    def test_population_eight_padded_encoding_and_training(self):
        fitter = AEQGA(line, [0, 0], bounds=[(-4, 4), (-2, 2)],
                       population_size=8, shots=64, seed=7,
                       crossover_probability=0, mutation_probability=0)
        circuit = fitter._circuit(np.array([3., 4.]), np.random.default_rng(7))
        state = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
        self.assertEqual(circuit.num_qubits, 2)
        np.testing.assert_allclose(state.data, [.6, .8, 0., 0.], atol=1e-12)
        fitter.crossover_probability = fitter.mutation_probability = 1.
        fitter.train([0, 1, 2], [1, 2, 3], generations=3)
        self.assertEqual(fitter.n_evaluations_, 8 + 3 * 6)
        self.assertEqual(fitter.quantum_circuits_, 12)
        self.assertTrue(np.isfinite(fitter.loss_))
        self.assertTrue(np.all(np.diff([h["loss"] for h in fitter.history_]) <= 0))
        residual = fitter.predict([0, 1, 2]) - [1, 2, 3]
        self.assertAlmostEqual(fitter.loss_, float(residual @ residual))

    def make_line(self, **kwargs):
        return AEQGA(line, [0.0, 0.0], bounds=[(-4, 4), (-2, 2)],
                     population_size=16, shots=256, seed=23, **kwargs)

    def test_amplitude_encoding_matches_normalized_vector(self):
        fitter = self.make_line(crossover_probability=0, mutation_probability=0)
        for values in (np.array([1., -2., 3., 4.]), np.array([1e300] * 4), np.zeros(4)):
            circuit = fitter._circuit(values, np.random.default_rng(1))
            state = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
            expected = np.ones(4) / 2 if not np.any(values) else values / np.max(np.abs(values))
            expected = expected / np.linalg.norm(expected)
            np.testing.assert_allclose(state.data, expected, atol=1e-12)

    def test_genetic_gates_are_bidirectional_cry_and_rx(self):
        fitter = self.make_line(crossover_probability=1, mutation_probability=1)
        circuit = fitter._circuit(np.array([1., 2., 3., 4.]), np.random.default_rng(4))
        gates = [item for item in circuit.data if item.operation.name in ("cry", "rx")]
        self.assertEqual([item.operation.name for item in gates], ["cry", "cry", "rx"])
        self.assertEqual(gates[0].qubits, tuple(reversed(gates[1].qubits)))
        for item in gates:
            self.assertAlmostEqual(float(item.operation.params[0]), np.pi / 2)

    def test_real_aer_training_elitism_and_final_generation_evaluation(self):
        fitter = self.make_line()
        with patch.object(fitter._backend, "run", wraps=fitter._backend.run) as run:
            returned = fitter.train([-2, -1, 0, 1, 2], [-3, -1, 1, 3, 5], generations=8)
        self.assertIs(returned, fitter)
        self.assertEqual(run.call_count, 8)
        self.assertEqual(fitter.quantum_circuits_, 8 * 2 * 2)
        self.assertEqual(fitter.n_evaluations_, 16 + 8 * 12)
        self.assertEqual(len(fitter.history_), 9)
        self.assertTrue(np.all(np.diff([h["loss"] for h in fitter.history_]) <= 0))
        residual = fitter.predict([-2, -1, 0, 1, 2]) - [-3, -1, 1, 3, 5]
        self.assertAlmostEqual(fitter.loss_, float(residual @ residual))
        self.assertEqual(fitter.p, fitter.history_[-1]["p"])
        for h in fitter.history_:
            self.assertTrue(-4 <= h["p"][0] <= 4 and -2 <= h["p"][1] <= 2)

    def test_nonlinear_vectorized_fit_against_known_parameters(self):
        def quadratic(x, p):
            self.assertIsInstance(x, list)
            self.assertIsInstance(p, list)
            return p[0] * np.asarray(x) ** 2 + p[1]

        x = np.linspace(-2, 2, 25)
        truth = [1.7, -0.4]
        y = quadratic(x.tolist(), truth)
        fitter = AEQGA(quadratic, [0., 0.], bounds=[(-3, 3), (-2, 2)],
                       vectorized=True, seed=17, shots=512)
        fitter.train(x.tolist(), y.tolist(), generations=120)
        np.testing.assert_allclose(fitter.p, truth, atol=0.025, rtol=0)
        self.assertLess(fitter.loss_, 0.02)

    def test_noisy_line_does_not_prematurely_collapse(self):
        rng = np.random.default_rng(42)
        x = np.linspace(-2, 2, 41)
        y = 2.5 * x - .7 + rng.normal(0, .03, len(x))
        fitter = AEQGA(line, [0., 0.], bounds=[(-5, 5), (-3, 3)], seed=7)
        fitter.train(x, y, generations=150)
        design = np.column_stack((x, np.ones(len(x))))
        reference = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = design @ reference - y
        np.testing.assert_allclose(fitter.p, reference, atol=0.015, rtol=0)
        self.assertLess(fitter.loss_, float(residual @ residual) + 0.005)

    def test_list_input_multiple_features(self):
        calls = []

        def model(x, p):
            self.assertIsInstance(x, list)
            self.assertIsInstance(p, list)
            calls.append(x)
            return p[0] * x[0] + p[1] * x[1]

        fitter = AEQGA(model, [2., -1.], bounds=[(-3, 3), (-3, 3)], seed=1)
        fitter.train([[1, 2], [3, 4]], [0, 2], generations=2, target_loss=0)
        self.assertEqual(fitter.p, [2., -1.])
        self.assertTrue(all(len(row) == 2 for row in calls))
        np.testing.assert_allclose(fitter.predict([[2, 3]]), [1])
        with self.assertRaisesRegex(ValueError, "same number of features"):
            fitter.predict([1, 2])

    def test_same_seed_reproduces_shot_based_training(self):
        first, second = self.make_line(), self.make_line()
        for fitter in (first, second):
            fitter.train([0, 1, 2], [1, 2, 3], generations=4)
        self.assertEqual(first.history_, second.history_)

    def test_getters_do_not_expose_mutable_parameters(self):
        fitter = self.make_line()
        for values in (fitter.p, fitter.get_p()):
            values[0] = 100
        self.assertEqual(fitter.p, [0., 0.])
        with self.assertRaises(AttributeError):
            fitter.p = [1., 2.]

    def test_weighted_losses_and_initial_target_stop(self):
        for weights, expected in (
            ({}, 5.),
            ({"sigma": 2.}, 1.25),
            ({"sigma": [1., 2.]}, 2.),
            ({"covariance": [[2., .5], [.5, 1.]]}, 4.),
        ):
            with self.subTest(weights=weights):
                fitter = AEQGA(lambda x, p: p[0], [0.], bounds=[(-1., 0.)], seed=1)
                fitter.train([0., 1.], [1., 2.], generations=2, target_loss=10., **weights)
                self.assertAlmostEqual(fitter.loss_, expected)
                self.assertEqual(fitter.n_generations_, 0)
                self.assertEqual(fitter.quantum_circuits_, 0)

    def test_missing_counts_and_degenerate_decoding(self):
        counts = AEQGA._counts({"11": 10, "01": 5}, 4)
        np.testing.assert_equal(counts, [0, 5, 0, 10])
        for local in (True, False):
            values = AEQGA._decode(np.ones(4), -2., 3., local, np.random.default_rng(2))
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertTrue(np.all((values >= -2) & (values <= 3)))
        decoded = AEQGA._decode(np.array([1., 2., 3., 4.]), 0., 1., True, np.random.default_rng(2))
        np.testing.assert_allclose(decoded, np.sqrt([.1, .2, .3, .4]))

    def test_nonfinite_candidates_are_rejected(self):
        def partly_defined(x, p):
            return float("nan") if p[0] < 0 else p[0] * x[0]

        fitter = AEQGA(partly_defined, [.2], bounds=[(-1., 1.)], seed=3, shots=64)
        fitter.train([1., 2.], [.6, 1.2], generations=3)
        self.assertTrue(np.isfinite(fitter.loss_))
        self.assertGreaterEqual(fitter.p[0], 0)
        invalid = AEQGA(lambda x, p: np.nan, [0.], bounds=[(-1, 1)])
        with self.assertRaisesRegex(ValueError, "All initial candidates"):
            invalid.train([1], [2])

    def test_validation_rejects_shapes_and_invalid_settings(self):
        for kwargs in (
            {"population_size": 4}, {"population_size": 24}, {"shots": 0},
            {"crossover_probability": 2}, {"mutation_probability": np.nan},
            {"min_local_width": 0}, {"seed": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AEQGA(line, [0, 0], bounds=[(-1, 1), (-1, 1)], **kwargs)
        for bounds in ([(-1, 1)], [(0, 0), (-1, 1)], [(1, 2), (-1, 1)]):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                AEQGA(line, [0, 0], bounds=bounds)
        fitter = self.make_line()
        for kwargs in (
            {"sigma": 0}, {"sigma": [1, 2, 3]},
            {"sigma": 1, "covariance": np.eye(2)},
            {"covariance": [[1, 1], [1, 1]]},
            {"covariance": [[1, 1], [0, 1]]}, {"generations": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fitter.train([0, 1], [1, 2], **kwargs)
        with self.assertRaises(ValueError):
            fitter.train([0, 1], [[1], [2]])
        wrong_output = AEQGA(lambda x, p: [p[0]], [0.], bounds=[(-1, 1)])
        with self.assertRaisesRegex(ValueError, "one scalar per sample"):
            wrong_output.train([0, 1], [1, 2])
        wrong_batch = AEQGA(lambda x, p: p[0], [0.], bounds=[(-1, 1)], vectorized=True)
        with self.assertRaisesRegex(ValueError, "one scalar per sample"):
            wrong_batch.train([0, 1], [1, 2])


if __name__ == "__main__":
    unittest.main()
