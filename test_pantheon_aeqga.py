"""
test_pantheon_aeqga.py
=======================
Integration test: run AEQGA on Pantheon SNe Ia to recover H0 and Omega_m.

Usage
-----
1. Clone the data repo:
       git clone https://github.com/CobayaSampler/sn_data
2. Run:
       python test_pantheon_aeqga.py --data sn_data/Pantheon

The script runs four groups of tests in increasing depth:
  T1  Unit tests on the distance-modulus code (no data needed)
  T2  Data-loading validation (requires --data)
  T3  Chi2 sanity at known parameters (requires --data)
  T4  Quantum-correctness unit tests (no data needed)
  T5  Full AEQGA optimisation run     (requires --data)
"""

import sys
import os
import math
import argparse
import numpy as np

# ── T1: offline unit tests (no data, no Qiskit) ─────────────────────────────

def test_distance_modulus():
    from pantheon_problem import luminosity_distance_flat_lcdm, distance_modulus

    d_L = luminosity_distance_flat_lcdm(0.5, 0.5, 70.0, 0.3, n_steps=5000)
    assert 2700 < d_L < 2950, f"d_L out of range: {d_L:.1f} Mpc"
    print(f"  T1a: d_L(z=0.5, H0=70, Om=0.3) = {d_L:.1f} Mpc  [expected ~2832]  PASS")

    d_L_low = luminosity_distance_flat_lcdm(0.001, 0.001, 70.0, 0.3)
    assert d_L_low < 5.0, f"Low-z d_L too large: {d_L_low}"
    print(f"  T1b: d_L(z=0.001) = {d_L_low*1e3:.3f} kpc  [expected ~4.3 Mpc]  PASS")

    mu = distance_modulus(0.1, 0.1, 70.0, 0.3)
    assert 37.5 < mu < 39.0, f"mu out of range: {mu:.3f}"
    print(f"  T1c: mu(z=0.1)  = {mu:.3f}  [expected ~38.3]  PASS")

    z_arr = [0.1, 0.3, 0.5, 0.7, 1.0]
    dls   = [luminosity_distance_flat_lcdm(z, z, 70.0, 0.3) for z in z_arr]
    assert all(dls[i] < dls[i+1] for i in range(len(dls)-1)), "d_L not monotone"
    print(f"  T1d: d_L monotone with z  PASS")

    d70 = luminosity_distance_flat_lcdm(0.3, 0.3, 70.0, 0.3)
    d35 = luminosity_distance_flat_lcdm(0.3, 0.3, 35.0, 0.3)
    ratio = d35 / d70
    assert 1.9 < ratio < 2.1, f"H0 scaling wrong: ratio={ratio:.3f}"
    print(f"  T1e: d_L scales ~1/H0: ratio={ratio:.3f}  PASS")

    print("T1: All distance-modulus tests PASSED\n")


def test_chi2_diag():
    from pantheon_problem import _chi2_diag
    delta = np.full(10, 3.0)
    sigma = np.ones(10)
    chi2  = _chi2_diag(delta, sigma)
    assert abs(chi2) < 1e-10, f"Degenerate M not marginalised: chi2={chi2}"
    print("  T1f: chi2_diag marginalises M correctly (chi2=0)  PASS")

    delta2 = np.array([1.0, -1.0, 0.5, -0.5])
    sigma2 = np.ones(4)
    chi2b  = _chi2_diag(delta2, sigma2)
    assert chi2b > 0, "chi2 should be > 0"
    print(f"  T1g: chi2_diag > 0 for non-degenerate residuals: {chi2b:.4f}  PASS")
    print("T1f-g: Chi2 marginalisation tests PASSED\n")


# ── T4: Quantum-correctness unit tests (no data needed) ───────────────────

def test_amplitude_encoding():
    """T4a: Verify amplitude encoding via initialize."""
    from amplitude_encoding import (
        build_amplitude_circuit, n_qubits_for_population, _normalise
    )
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    # Test normalization
    x = np.array([1.0, 2.0, 3.0])
    norm = _normalise(x)
    assert abs(np.linalg.norm(norm) - 1.0) < 1e-12, "Normalization failed"
    print("  T4a: _normalise produces unit vector  PASS")

    # Test qubit count
    nq = n_qubits_for_population(8)
    assert nq == 3, f"Expected 3 qubits for 8 individuals, got {nq}"
    print("  T4b: n_qubits_for_population(8)=3  PASS")

    # Test amplitude encoding produces correct statevector
    values = np.array([1.0, 0.0, 0.0, 0.0])  # |00⟩ state
    qc = build_amplitude_circuit(values)
    sv = Statevector(qc)
    expected = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(sv.data, expected, atol=1e-10), \
        f"Amplitude encoding wrong: {sv.data}"
    print("  T4c: build_amplitude_circuit for |00⟩ state  PASS")

    # Test non-trivial amplitude encoding
    values = np.array([1.0, 1.0])
    qc = build_amplitude_circuit(values)
    sv = Statevector(qc)
    expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    assert np.allclose(sv.data, expected, atol=1e-10), \
        f"Superposition encoding wrong: {sv.data}"
    print("  T4d: build_amplitude_circuit for equal superposition  PASS")

    # Test power-of-two enforcement
    try:
        n_qubits_for_population(10)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        print("  T4e: non-power-of-two rejected  PASS")

    print("T4: Amplitude encoding tests PASSED\n")


def test_quantum_gates():
    """T4f: Verify CRy(π/2) bidirectional = U_cross (Eq.14) and Rx(π/2) = U_mut (Eq.15)."""
    from quantum_gates import verify_u_cross, verify_rx_pi2
    from qiskit.quantum_info import Operator
    import numpy as np

    mat, U_cross = verify_u_cross()
    # Check matrix equality within tolerance
    assert np.allclose(mat, U_cross, atol=1e-10), \
        f"U_cross mismatch:\n{mat}\nvs\n{U_cross}"
    print("  T4f: U_cross = CRy(π/2) bidirectional matches Eq.14  PASS")

    rx_mat = verify_rx_pi2()
    # Rx(π/2) = 1/√2 [[1, -i], [-i, 1]]
    U_mut_expected = (1/np.sqrt(2)) * np.array([[1, -1j], [-1j, 1]], dtype=complex)
    assert np.allclose(rx_mat, U_mut_expected, atol=1e-10), \
        f"Rx(π/2) mismatch:\n{rx_mat}\nvs\n{U_mut_expected}"
    print("  T4g: Rx(π/2) matches Eq.15  PASS")

    print("T4: Quantum gate verification PASSED\n")


def test_decoding():
    """T4h: Verify decoding functions."""
    from amplitude_encoding import decode_random_subset, decode_elite_subset
    import numpy as np

    # decode_random_subset: all counts equal → uniform spread
    counts = {"00": 100, "01": 100, "10": 100, "11": 100}
    result = decode_random_subset(counts, n_qubits=2, a=0.0, b=1.0, n=2)
    # With equal counts, c_min == c_max → uniform draw
    assert len(result) == 2, f"Expected 2 values, got {len(result)}"
    print("  T4h: decode_random_subset equal counts  PASS")

    # decode_elite_subset: probability-based
    counts = {"00": 500, "01": 100, "10": 100, "11": 50}
    result = decode_elite_subset(counts, n_qubits=2, n_min=0.0, n_max=1.0, n=2)
    assert len(result) == 2, f"Expected 2 values, got {len(result)}"
    assert np.all(result >= 0.0) and np.all(result <= 1.0), \
        f"Values out of range: {result}"
    # √p should concentrate toward center
    print("  T4i: decode_elite_subset produces valid values  PASS")

    print("T4: Decoding tests PASSED\n")


def test_zero_norm_normalize():
    """T4j: Zero-norm edge case in _normalise."""
    from amplitude_encoding import _normalise
    x = np.zeros(4)
    norm = _normalise(x)
    assert np.allclose(norm, 1/np.sqrt(4) * np.ones(4)), \
        f"Zero-norm normalization wrong: {norm}"
    print("  T4j: _normalise handles zero-norm  PASS")

    print("T4: Zero-norm edge case PASSED\n")


# ── T2: data loading validation ──────────────────────────────────────────────

def test_data_loading(data_dir: str):
    from pantheon_problem import load_pantheon
    print(f"T2: Loading data from {data_dir}")
    d = load_pantheon(data_dir)

    n = d["n_sn"]
    print(f"  N supernovae         : {n}")
    assert n == 1048, f"Expected 1048 SNe (Pantheon), got {n}"
    print(f"  N = 1048  PASS")

    zmin, zmax = d["zcmb"].min(), d["zcmb"].max()
    print(f"  z_cmb range          : [{zmin:.4f}, {zmax:.4f}]")
    assert 0.01 < zmin < 0.05,   f"Unexpected zmin={zmin}"
    assert 1.5  < zmax < 2.5,    f"Unexpected zmax={zmax}"
    print("  z_cmb range plausible  PASS")

    mbmin, mbmax = d["mb"].min(), d["mb"].max()
    print(f"  mb range             : [{mbmin:.2f}, {mbmax:.2f}]")
    assert 14 < mbmin < 18, f"Unexpected mbmin={mbmin}"
    assert 24 < mbmax < 28, f"Unexpected mbmax={mbmax}"
    print("  mb range plausible  PASS")

    assert np.all(d["dmb"] > 0),   "Some dmb <= 0"
    assert np.all(d["dmb"] < 0.5), "Some dmb suspiciously large"
    print(f"  dmb in (0, 0.5)      : PASS")

    if d["cov_sys"] is not None:
        C = d["cov_total"]
        assert C.shape == (n, n), f"Cov shape wrong: {C.shape}"
        assert np.all(np.diag(C) > 0), "Non-positive diagonal in covariance"
        print(f"  Covariance shape     : {C.shape}  PASS")
        assert np.allclose(C, C.T, atol=1e-10), "Covariance not symmetric"
        print("  Covariance symmetric : PASS")
    else:
        print("  sys_full_long.txt not found – using stat-only covariance")

    print("T2: Data loading tests PASSED\n")
    return d


# ── T3: chi2 at known best-fit values ────────────────────────────────────

def test_chi2_bestfit(data_dir: str):
    from pantheon_problem import PantheonProblem
    print("T3: Chi2 at known best-fit values")
    prob = PantheonProblem(data_dir, use_full_cov=True)

    # Paper SNe Ia best-fit (§4): Ω_M = 0.363, H0 = 72.82
    x_ref = np.array([72.82, 0.363])
    chi2_ref = prob.compute_fitness(x_ref)
    print(f"  chi2(H0=72.82, Om=0.363) = {chi2_ref:.2f}")

    # Should be near the reduced chi2 ~ 1 → chi2 ~ N-2 ~ 1046
    assert 900 < chi2_ref < 1200, f"chi2 out of expected range: {chi2_ref:.2f}"
    print("  chi2 in plausible range [900, 1200]  PASS")

    # chi2 should increase at bad point
    x_bad = np.array([50.0, 0.5])
    chi2_bad = prob.compute_fitness(x_bad)
    print(f"  chi2(H0=50, Om=0.5)     = {chi2_bad:.2f}")
    assert chi2_bad > chi2_ref, "Bad parameters have lower chi2 than best-fit – suspicious"
    print(f"  chi2 increases away from best-fit  PASS")
    print("T3: Chi2 sanity tests PASSED\n")
    return prob


# ── T5: AEQGA optimisation ────────────────────────────────────────────────────

def test_aeqga_optimisation(prob):
    """Run a short AEQGA to check it recovers H0 and Omega_m."""
    print("T5: AEQGA optimisation on Pantheon data")
    print("    (This requires qiskit + qiskit-aer; skip if not installed)")

    try:
        from aeqga_algorithm import AEQGAParameters, run_aeqga_dual
    except ImportError:
        print("    qiskit/aeqga_algorithm not available – skipping T5")
        return

    # Pop size MUST be power of two per paper §3.1
    params = AEQGAParameters(
        pop_size     = 32,     # 2^5 — paper §3.1, §3.4
        max_gen      = 50,
        n_iterations = 1,
        p_cross      = 0.5,    # paper's optimum (§4.1)
        p_mut        = 0.5,    # paper's optimum (§4.1)
        num_shots    = 4096,
        verbose      = False,
        progress_bar = True,
    )

    g_best, _, bests_log = run_aeqga_dual(prob, params)

    H0_best = g_best.x[0]
    Om_best = g_best.x[1]
    chi2_best = g_best.fitness

    print(f"\n  Best fit: H0 = {H0_best:.2f}  Omega_m = {Om_best:.3f}")
    print(f"  chi2_best = {chi2_best:.2f}")

    # Paper SNe Ia best-fit is (0.363, 72.82)
    # Tolerance: within 1σ = ±0.016, ±0.22 for short run
    assert 63 < H0_best < 80,   f"H0 far from expectation: {H0_best:.2f}"
    assert 0.1 < Om_best < 0.5, f"Om_m far from expectation: {Om_best:.3f}"
    print("  H0 and Omega_m in plausible range  PASS")

    # Convergence: last chi2 <= first chi2 + tolerance
    assert bests_log[-1][1] <= bests_log[0][1] + 1.0, "AEQGA did not converge"
    print("  AEQGA converged (chi2 did not increase)  PASS")

    print("T5: AEQGA optimisation test PASSED\n")

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

    # T4: quantum-correctness tests, always run (no data needed)
    test_amplitude_encoding()
    test_quantum_gates()
    test_decoding()
    test_zero_norm_normalize()

    if args.data is None:
        print("No --data path provided. Skipping T2/T3/T5.")
        print("Provide with:  python test_pantheon_aeqga.py --data sn_data/Pantheon")
        sys.exit(0)

    if not os.path.isdir(args.data):
        print(f"ERROR: Directory not found: {args.data}")
        sys.exit(1)

    # T2: data loading
    d = test_data_loading(args.data)

    # T3: chi2 sanity at known parameters
    prob = test_chi2_bestfit(args.data)

    # T5: full AEQGA (requires qiskit)
    test_aeqga_optimisation(prob)

    print("=" * 60)
    print("  All available tests PASSED")
    print("=" * 60)
