"""
test_pantheon_aeqga.py
======================
Integration test: run AEQGA on Pantheon SNe Ia to recover H0 and Omega_m.

Usage
-----
1. Clone the data repo:
       git clone https://github.com/CobayaSampler/sn_data
2. Run:
       python test_pantheon_aeqga.py --data sn_data/Pantheon

The script runs three tests in increasing depth:
  T1  Unit tests on the distance-modulus code (no data needed)
  T2  Data-loading validation (requires --data)
  T3  Full AEQGA optimisation run     (requires --data)
"""

import sys
import os
import math
import argparse
import numpy as np

# ── T1: offline unit tests (no data, no Qiskit) ─────────────────────────────

def test_distance_modulus():
    from pantheon_problem import luminosity_distance_flat_lcdm, distance_modulus

    # At z=0.5, flat ΛCDM, H0=70, Om=0.3:
    # d_L ≈ 2832 Mpc  (standard textbook value)
    d_L = luminosity_distance_flat_lcdm(0.5, 0.5, 70.0, 0.3, n_steps=5000)
    assert 2700 < d_L < 2950, f"d_L out of range: {d_L:.1f} Mpc"
    print(f"  T1a: d_L(z=0.5, H0=70, Om=0.3) = {d_L:.1f} Mpc  [expected ~2832]  PASS")

    # At z→0 the distance modulus should approach 0
    d_L_low = luminosity_distance_flat_lcdm(0.001, 0.001, 70.0, 0.3)
    assert d_L_low < 5.0, f"Low-z d_L too large: {d_L_low}"
    print(f"  T1b: d_L(z=0.001) = {d_L_low*1e3:.3f} kpc  [expected ~4.3 Mpc]  PASS")

    # mu(z=0.1, H0=70, Om=0.3) ~ 38.3
    mu = distance_modulus(0.1, 0.1, 70.0, 0.3)
    assert 37.5 < mu < 39.0, f"mu out of range: {mu:.3f}"
    print(f"  T1c: mu(z=0.1)  = {mu:.3f}  [expected ~38.3]  PASS")

    # Monotonicity: d_L should increase with z
    z_arr = [0.1, 0.3, 0.5, 0.7, 1.0]
    dls   = [luminosity_distance_flat_lcdm(z, z, 70.0, 0.3) for z in z_arr]
    assert all(dls[i] < dls[i+1] for i in range(len(dls)-1)), "d_L not monotone"
    print(f"  T1d: d_L monotone with z: {[f'{d:.0f}' for d in dls]}  PASS")

    # H0 scaling: d_L ∝ 1/H0 at fixed z for H0 >> other scales
    d70 = luminosity_distance_flat_lcdm(0.3, 0.3, 70.0, 0.3)
    d35 = luminosity_distance_flat_lcdm(0.3, 0.3, 35.0, 0.3)
    ratio = d35 / d70
    assert 1.9 < ratio < 2.1, f"H0 scaling wrong: ratio={ratio:.3f}"
    print(f"  T1e: d_L scales ~1/H0: ratio(H0=35/H0=70) = {ratio:.3f}  PASS")

    print("T1: All distance-modulus tests PASSED\n")


def test_chi2_diag():
    from pantheon_problem import _chi2_diag
    # Perfect fit: delta = M*1 (all residuals equal constant) → marginalised chi2 = 0
    delta = np.full(10, 3.0)
    sigma = np.ones(10)
    chi2  = _chi2_diag(delta, sigma)
    assert abs(chi2) < 1e-10, f"Degenerate M not marginalised: chi2={chi2}"
    print("  T1f: chi2_diag marginalises M correctly (chi2=0 for delta=const)  PASS")

    # Non-degenerate case: chi2 > 0
    delta2 = np.array([1.0, -1.0, 0.5, -0.5])
    sigma2 = np.ones(4)
    chi2b  = _chi2_diag(delta2, sigma2)
    assert chi2b > 0, "chi2 should be > 0"
    print(f"  T1g: chi2_diag > 0 for non-degenerate residuals: {chi2b:.4f}  PASS")
    print("T1f-g: Chi2 marginalisation tests PASSED\n")


# ── T2: data loading validation ──────────────────────────────────────────────

def test_data_loading(data_dir: str):
    from pantheon_problem import load_pantheon
    print(f"T2: Loading data from {data_dir}")
    d = load_pantheon(data_dir)

    n = d["n_sn"]
    print(f"  N supernovae         : {n}")
    assert n == 1048, f"Expected 1048 SNe (Pantheon), got {n}"
    print(f"  N = 1048  PASS")

    # Redshift range
    zmin, zmax = d["zcmb"].min(), d["zcmb"].max()
    print(f"  z_cmb range          : [{zmin:.4f}, {zmax:.4f}]")
    assert 0.01 < zmin < 0.05,   f"Unexpected zmin={zmin}"
    assert 1.5  < zmax < 2.5,    f"Unexpected zmax={zmax}"
    print("  z_cmb range plausible  PASS")

    # mb range: standardised magnitudes ~ 22–26 for high-z, ~14–17 for low-z
    mbmin, mbmax = d["mb"].min(), d["mb"].max()
    print(f"  mb range             : [{mbmin:.2f}, {mbmax:.2f}]")
    assert 14 < mbmin < 18, f"Unexpected mbmin={mbmin}"
    assert 24 < mbmax < 28, f"Unexpected mbmax={mbmax}"
    print("  mb range plausible  PASS")

    # dmb should all be small positive
    assert np.all(d["dmb"] > 0),   "Some dmb <= 0"
    assert np.all(d["dmb"] < 0.5), "Some dmb suspiciously large"
    print(f"  dmb in (0, 0.5)      : PASS")

    # Covariance
    if d["cov_sys"] is not None:
        C = d["cov_total"]
        assert C.shape == (n, n), f"Cov shape wrong: {C.shape}"
        # Should be positive semi-definite (diagonal is positive)
        assert np.all(np.diag(C) > 0), "Non-positive diagonal in covariance"
        print(f"  Covariance shape     : {C.shape}  PASS")
        # Symmetry
        assert np.allclose(C, C.T, atol=1e-10), "Covariance not symmetric"
        print("  Covariance symmetric : PASS")
    else:
        print("  sys_full_long.txt not found – using stat-only covariance")

    print("T2: Data loading tests PASSED\n")
    return d


# ── T3: chi2 at known best-fit values ────────────────────────────────────────

def test_chi2_bestfit(data_dir: str):
    from pantheon_problem import PantheonProblem
    print("T3: Chi2 at known best-fit values")
    prob = PantheonProblem(data_dir, use_full_cov=True)

    # Planck 2018 / Scolnic+18 best-fit: H0~67.4, Om~0.298
    x_ref = np.array([67.4, 0.298])
    chi2_ref = prob.compute_fitness(x_ref)
    print(f"  chi2(H0=67.4, Om=0.298) = {chi2_ref:.2f}")

    # chi2 should be near the reduced chi2 ~ 1 → chi2 ~ N-2 ~ 1046
    assert 900 < chi2_ref < 1200, f"chi2 out of expected range: {chi2_ref:.2f}"
    print("  chi2 in plausible range [900, 1200]  PASS")

    # chi2 should decrease as we move toward best-fit vs a bad point
    x_bad = np.array([50.0, 0.5])
    chi2_bad = prob.compute_fitness(x_bad)
    print(f"  chi2(H0=50, Om=0.5)     = {chi2_bad:.2f}")
    assert chi2_bad > chi2_ref, "Bad parameters have lower chi2 than best-fit – suspicious"
    print(f"  chi2 increases away from best-fit  PASS")
    print("T3: Chi2 sanity tests PASSED\n")
    return prob


# ── T4: AEQGA optimisation ────────────────────────────────────────────────────

def test_aeqga_optimisation(prob):
    """Run a short AEQGA to check it recovers H0 and Omega_m."""
    print("T4: AEQGA optimisation on Pantheon data")
    print("    (This requires qiskit + qiskit-aer; skip if not installed)")

    try:
        from aeqga_algorithm import AEQGAParameters, run_aeqga
    except ImportError:
        print("    qiskit/aeqga_algorithm not available – skipping T4")
        return

    params = AEQGAParameters(
        pop_size     = 8,
        max_gen      = 30,        # short run for testing
        p_cross      = 0.7,
        p_mut        = 0.15,
        sigma_mut    = 0.05,      # tight – search space already bounded
        num_shots    = 8192,
        verbose      = False,
        progress_bar = True,
    )

    g_best, _, bests_log = run_aeqga(prob, params)

    H0_best = g_best.x[0]
    Om_best = g_best.x[1]
    chi2_best = g_best.fitness

    print(f"\n  Best fit: H0 = {H0_best:.2f}  Omega_m = {Om_best:.3f}")
    print(f"  chi2_best = {chi2_best:.2f}")

    # Should be in the right ballpark within a short run
    assert 63 < H0_best < 75,   f"H0 far from expectation: {H0_best:.2f}"
    assert 0.20 < Om_best < 0.45, f"Om_m far from expectation: {Om_best:.3f}"
    print("  H0 and Omega_m in plausible range  PASS")

    # Convergence: last chi2 <= first chi2
    assert bests_log[-1][1] <= bests_log[0][1] + 1.0, "AEQGA did not converge"
    print("  AEQGA converged (chi2 did not increase)  PASS")

    print("T4: AEQGA optimisation test PASSED\n")

    # Optional plot
    try:
        from aeqga_algorithm import plot_convergence
        plot_convergence(bests_log, "AEQGA on Pantheon SNe Ia")
    except Exception:
        pass

    return g_best


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AEQGA on Pantheon data")
    parser.add_argument("--data", default=None,
                        help="Path to CobayaSampler/sn_data/Pantheon directory")
    args = parser.parse_args()

    print("=" * 60)
    print("  AEQGA + Pantheon Integration Tests")
    print("=" * 60)

    # T1: pure-Python tests, always run
    test_distance_modulus()
    test_chi2_diag()

    if args.data is None:
        print("No --data path provided. Skipping T2/T3/T4.")
        print("Provide with:  python test_pantheon_aeqga.py --data sn_data/Pantheon")
        sys.exit(0)

    if not os.path.isdir(args.data):
        print(f"ERROR: Directory not found: {args.data}")
        sys.exit(1)

    # T2: data loading
    test_data_loading(args.data)

    # T3: chi2 sanity at known parameters
    prob = test_chi2_bestfit(args.data)

    # T4: full AEQGA (requires qiskit)
    test_aeqga_optimisation(prob)

    print("=" * 60)
    print("  All available tests PASSED")
    print("=" * 60)
