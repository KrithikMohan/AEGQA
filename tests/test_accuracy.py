"""Analytic accuracy metrics and numerical checks of actual plotted data."""
import unittest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.contour import QuadContourSet
from aeqga.analysis.accuracy import accuracy_report, accuracy_markdown
from aeqga.visualization.scientific_plots import (
    accuracy_comparison, objective_contours, optimizer_scatter, joint_levels,
)


class AccuracyTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_hand_calculated_metrics(self):
        points = [[72., .3], [74., .4], [70., .2]]
        report = accuracy_report(points, [100., 101., 104.], dict(H0=72., Omega_m=.3, chi2=100.))
        h0 = report['parameters']['H0']
        self.assertAlmostEqual(h0['bias'], 0.)
        self.assertAlmostEqual(h0['mae'], 4/3)
        self.assertAlmostEqual(h0['rmse'], np.sqrt(8/3))
        self.assertAlmostEqual(h0['mean_absolute_percent_error'], 100*(4/3)/72)
        self.assertEqual(report['delta_chi2'], [0., 1., 4.])
        self.assertEqual(report['success_fraction'], 2/3)
        low, high = report['success_rate_wilson_95']
        self.assertLess(low, 2/3)
        self.assertGreater(high, 2/3)
        self.assertIn('not confidence', accuracy_markdown(report))

    def test_zero_reference_single_run_and_external_point(self):
        report = accuracy_report([[72., 0.]], [10.], dict(H0=72., Omega_m=0., chi2=10.),
                                  comparison_reference=dict(label='Explicit synthetic truth', H0=71., Omega_m=.1))
        self.assertIsNone(report['parameters']['Omega_m']['absolute_percent_errors'])
        self.assertEqual(report['success_fraction'], 1.)
        self.assertLess(report['success_rate_wilson_95'][0], .5)
        self.assertAlmostEqual(report['external_comparison']['parameters']['H0']['mae'], 1.)

    def test_invalid_or_inconsistent_references_rejected(self):
        ref = dict(H0=72., Omega_m=.3, chi2=100.)
        for points, values in (([], []), ([[72., .3]], [99.]), ([[np.nan, .3]], [100.]),
                               ([[72., .3]], [100., 101.])):
            with self.assertRaises(ValueError):
                accuracy_report(points, values, ref)

    def test_accuracy_plot_uses_report_arrays_without_hiding_outliers(self):
        report = accuracy_report([[72., .3], [74., .4], [70., .2]], [100., 101., 150.],
                                  dict(H0=72., Omega_m=.3, chi2=100.))
        fig = accuracy_comparison(report)
        np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [0., 2., -2.])
        np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), [0., .1, -.1])
        np.testing.assert_array_equal(fig.axes[2].lines[0].get_ydata(), [0., 1., 50.])
        np.testing.assert_allclose(fig.axes[3].lines[0].get_ydata(), [0., 1/3, 2/3, 1.])

    def test_rendered_likelihood_boundaries_match_known_quadratic(self):
        class GaussianObjective:
            grid_size = 0
            @staticmethod
            def objective_grid(h0, omega):
                x, y = np.meshgrid((np.asarray(h0)-72.)/.3, (np.asarray(omega)-.3)/.025)
                return 100.+(x*x-1.2*x*y+y*y)/.64
        ref = dict(H0=72., Omega_m=.3, chi2=100.)
        fig = objective_contours(GaussianObjective(), ref, sigmas=(1, 2, 3))
        contours = next(c for c in fig.axes[0].collections if isinstance(c, QuadContourSet) and not c.filled)
        np.testing.assert_allclose(contours.levels, joint_levels((1, 2, 3)))
        for level, segments in zip(contours.levels, contours.allsegs):
            vertices = np.concatenate(segments)
            x, y = (vertices[:,0]-72.)/.3, (vertices[:,1]-.3)/.025
            np.testing.assert_allclose((x*x-1.2*x*y+y*y)/.64, level, atol=.01, rtol=0)

    def test_covariance_ellipse_uses_sample_covariance_and_joint_radius(self):
        points = np.random.default_rng(10).normal(size=(100, 2))*[.2, .02]+[72., .3]
        fig = optimizer_scatter(points, dict(H0=72., Omega_m=.3))
        ellipse = fig.axes[0].patches[0]
        eigenvalues = np.linalg.eigvalsh(np.cov(points.T, ddof=1))
        np.testing.assert_allclose([ellipse.width, ellipse.height],
                                   2*np.sqrt(joint_levels((1,))[0]*eigenvalues[::-1]))
        np.testing.assert_allclose(ellipse.center, points.mean(axis=0))


if __name__ == '__main__':
    unittest.main()
