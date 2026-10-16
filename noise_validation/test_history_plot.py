"""Plot data alignment and cosmology-model checks without a long training run."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from aeqga import AEQGA
from run_pantheon_aeqga import C_KM_S, make_model


class HistoryPlotTests(unittest.TestCase):
    def make_fitter(self):
        return AEQGA(lambda x, p: p[0] * x[0] + p[1], [0, 0], bounds=[(-2, 2)]*2)

    def test_saved_plot_uses_exact_generations_and_selected_best_parameter(self):
        fitter = self.make_fitter()
        fitter.history_ = [
            {"generation": 0, "p": [1., .2], "loss": 10.},
            {"generation": 1, "p": [.5, .4], "loss": 8.},
            {"generation": 2, "p": [.7, .3], "loss": 4.},
        ]
        captured = []
        original = Figure.savefig

        def capture(figure, *args, **kwargs):
            captured.extend([(axis.lines[0].get_xdata().copy(),
                              axis.lines[0].get_ydata().copy()) for axis in figure.axes])
            return original(figure, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "history.png"
            with patch.object(Figure, "savefig", capture):
                returned = fitter.export_history_plot(path, parameter_indices=[1],
                                                       parameter_names=["slope", "offset"])
            self.assertEqual(returned, path.resolve())
            with Image.open(path) as image:
                image.verify()
            self.assertEqual(len(captured), 2)
            np.testing.assert_array_equal(captured[0][0], [1, 2])
            np.testing.assert_allclose(captured[0][1], [.4, .3])
            np.testing.assert_allclose(captured[1][1], [8, 4])
            fitter.export_history_plot(Path(directory)/"history.svg", include_initial=True)

    def test_empty_history_and_bad_selection(self):
        fitter = self.make_fitter()
        with self.assertRaises(ValueError):
            fitter.export_history_plot("unused.png")
        fitter.history_ = [{"generation": 1, "p": [0, 0], "loss": 0}]
        for options in ({"parameter_indices": [2]}, {"parameter_indices": [0, 0]},
                        {"parameter_names": ["one"]},
                        {"parameter_indices": [], "include_loss": False}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                fitter.export_history_plot("unused.png", **options)

    def test_cosmology_calibrator_and_analytic_distance(self):
        model = make_model()
        x = [[.1, .11, 0, -9], [.001, .002, 1, 30.]]
        # At Omega_m=0, E(z)=1, so the integral is exactly z.
        result = model(x, [70., 0., -19.3])
        expected = 5*np.log10(1.11*C_KM_S/70*.1)+25-19.3
        np.testing.assert_allclose(result, [expected, 10.7], atol=1e-12)
        self.assertAlmostEqual(model(x, [80., .4, -19.3])[1], 10.7)


if __name__ == "__main__":
    unittest.main()
