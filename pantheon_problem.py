"""
pantheon_problem.py
===================
Pantheon SNe Ia data loader and chi-squared likelihood for use with AEQGA.

Data format (CobayaSampler/sn_data  →  Pantheon/lcparam_full_long_zhel.txt)
----------------------------------------------------------------------------
Columns: #name  zcmb  zhel  dz  mb  dmb  x1  dx1  color  dcolor
         3rdvar  d3rdvar  cov_m_s  cov_m_c  cov_s_c  set  ra  dec  biascor

We need:
  zcmb   – CMB-frame redshift
  zhel   – heliocentric redshift  (used in distance-modulus formula)
  mb     – observed distance modulus (bias-corrected apparent magnitude)
  dmb    – statistical uncertainty on mb

The systematic covariance matrix is in  Pantheon/sys_full_long.txt
(first line = N, then N×N matrix row-by-row).

The cosmological chi-squared is:
    chi2 = delta^T  C^{-1}  delta
where
    delta_i = mb_i  -  mu_theory(z_i, H0, Omega_m)  -  M
    mu_theory = 5*log10(d_L(zcmb_i, zhel_i) / 10 pc)
    d_L computed in flat ΛCDM

M (absolute magnitude offset) is analytically marginalised following
the Pantheon prescription (Conley et al. 2011 Appendix C, Eq C1):
    chi2_marg = chi2  -  (sum_i delta_i/sigma_i)^2 / (sum_i 1/sigma_i)
(diagonal-only approximation used when full covariance is not loaded)

Parameters to fit:
    H0       – Hubble constant  [km/s/Mpc]   range  [60, 80]
    Omega_m  – matter density                range  [0.1, 0.6]
"""

import os
import math
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C_LIGHT = 299792.458   # km/s


# ---------------------------------------------------------------------------
# Flat ΛCDM luminosity distance
# ---------------------------------------------------------------------------

def luminosity_distance_flat_lcdm(z_cmb: float, z_hel: float,
                                   H0: float, Omega_m: float,
                                   n_steps: int = 1000) -> float:
    """Luminosity distance in Mpc for flat ΛCDM.

    The comoving distance integral uses zcmb for the expansion history;
    the (1+z_hel) prefactor uses the heliocentric redshift following the
    Pantheon convention (see Conley+11, Davis+19).
    """
    if z_cmb <= 0.0:
        return 0.0

    Omega_L = 1.0 - Omega_m
    dz = z_cmb / n_steps
    # Gauss quadrature via midpoint rule
    z_arr = (np.arange(n_steps) + 0.5) * dz
    E_inv = 1.0 / np.sqrt(Omega_m * (1.0 + z_arr)**3 + Omega_L)
    comoving = (C_LIGHT / H0) * float(np.sum(E_inv)) * dz
    return (1.0 + z_hel) * comoving   # proper luminosity distance


def distance_modulus(z_cmb: float, z_hel: float,
                     H0: float, Omega_m: float) -> float:
    """mu = 5*log10(d_L / 10 pc) = 5*log10(d_L [Mpc]) + 25"""
    d_L = luminosity_distance_flat_lcdm(z_cmb, z_hel, H0, Omega_m)
    if d_L <= 0.0:
        return -np.inf
    return 5.0 * math.log10(d_L) + 25.0


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_pantheon(data_dir: str) -> dict:
    """Load Pantheon data from a local clone of CobayaSampler/sn_data/Pantheon/.

    Parameters
    ----------
    data_dir : path to the Pantheon sub-folder, e.g. 'sn_data/Pantheon'

    Returns
    -------
    dict with keys:
        zcmb, zhel, mb, dmb  – 1-D numpy arrays (length N_sn)
        cov_stat             – diagonal N×N statistical covariance
        cov_sys              – full N×N systematic covariance (or None)
        cov_total            – stat + sys
        n_sn                 – number of supernovae
    """
    lc_file  = os.path.join(data_dir, "lcparam_full_long_zhel.txt")
    sys_file = os.path.join(data_dir, "sys_full_long.txt")

    if not os.path.exists(lc_file):
        raise FileNotFoundError(
            f"Data file not found: {lc_file}\n"
            "Clone https://github.com/CobayaSampler/sn_data and pass the "
            "path to its Pantheon sub-directory."
        )

    # Parse light-curve parameters
    # Columns: name zcmb zhel dz mb dmb x1 dx1 color dcolor 3rdvar d3rdvar
    #          cov_m_s cov_m_c cov_s_c set ra dec biascor
    data = []
    with open(lc_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            zcmb  = float(parts[1])
            zhel  = float(parts[2])
            mb    = float(parts[4])
            dmb   = float(parts[5])
            data.append((zcmb, zhel, mb, dmb))

    data    = np.array(data)
    zcmb    = data[:, 0]
    zhel    = data[:, 1]
    mb      = data[:, 2]
    dmb     = data[:, 3]
    n_sn    = len(zcmb)

    # Statistical covariance = diagonal dmb^2
    cov_stat = np.diag(dmb ** 2)

    # Systematic covariance (optional)
    cov_sys = None
    if os.path.exists(sys_file):
        with open(sys_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        n_sys = int(lines[0])
        assert n_sys == n_sn, (
            f"Covariance size {n_sys} != data length {n_sn}"
        )
        flat = []
        for line in lines[1:]:
            flat.extend([float(v) for v in line.split()])
        cov_sys = np.array(flat).reshape(n_sn, n_sn)

    cov_total = cov_stat if cov_sys is None else cov_stat + cov_sys

    return dict(zcmb=zcmb, zhel=zhel, mb=mb, dmb=dmb,
                cov_stat=cov_stat, cov_sys=cov_sys,
                cov_total=cov_total, n_sn=n_sn)


# ---------------------------------------------------------------------------
# Chi-squared with analytic M marginalisation
# ---------------------------------------------------------------------------

def chi2_pantheon(H0: float, Omega_m: float, data: dict,
                  use_full_cov: bool = True) -> float:
    """Compute the Pantheon chi^2 for flat ΛCDM (H0, Omega_m).

    M (SN absolute magnitude offset, degenerate with H0) is analytically
    marginalised following Conley+11 Eq. C1.
    """
    zcmb  = data["zcmb"]
    zhel  = data["zhel"]
    mb    = data["mb"]
    n_sn  = data["n_sn"]

    # Theoretical distance moduli
    mu_th = np.array([
        distance_modulus(zcmb[i], zhel[i], H0, Omega_m)
        for i in range(n_sn)
    ])

    # Residuals (M not yet subtracted – marginalised below)
    delta = mb - mu_th    # shape (N,)

    if use_full_cov and data["cov_sys"] is not None:
        C = data["cov_total"]
        try:
            L = np.linalg.cholesky(C)
            # Solve C^{-1} delta via triangular systems
            y = np.linalg.solve(L, delta)
            chi2_full = float(y @ y)
            # Analytic M marginalisation terms
            e = np.ones(n_sn)
            ye = np.linalg.solve(L, e)
            A  = float(ye @ ye)     # e^T C^{-1} e
            B  = float(ye @ y)      # e^T C^{-1} delta
            chi2_marg = chi2_full - B**2 / A
        except np.linalg.LinAlgError:
            chi2_marg = _chi2_diag(delta, data["dmb"])
    else:
        chi2_marg = _chi2_diag(delta, data["dmb"])

    return chi2_marg


def _chi2_diag(delta: np.ndarray, sigma: np.ndarray) -> float:
    """Diagonal (statistical-only) analytically marginalised chi^2."""
    w       = 1.0 / sigma**2
    chi2    = float(np.sum(w * delta**2))
    sum_w   = float(np.sum(w))
    sum_wd  = float(np.sum(w * delta))
    return chi2 - sum_wd**2 / sum_w


# ---------------------------------------------------------------------------
# AEQGA Problem class
# ---------------------------------------------------------------------------

class PantheonProblem:
    """Flat ΛCDM parameter estimation on Pantheon SNe Ia for AEQGA.

    Fits: theta = [H0, Omega_m]
    Minimises chi^2(theta) computed with analytic M marginalisation.
    """

    lower_bounds = np.array([60.0, 0.10])
    upper_bounds = np.array([80.0, 0.60])
    n_dim        = 2

    # Expected best-fit reference values (Scolnic+18 / Planck 2018)
    H0_ref      = 67.4    # km/s/Mpc
    Om_ref      = 0.315

    def __init__(self, data_dir: str, use_full_cov: bool = True,
                 verbose: bool = False):
        print(f"Loading Pantheon data from: {data_dir}")
        self._data      = load_pantheon(data_dir)
        self._use_cov   = use_full_cov
        self._verbose   = verbose
        print(f"  Loaded {self._data['n_sn']} supernovae")
        cov_status = "stat+sys" if self._data["cov_sys"] is not None else "stat only"
        print(f"  Covariance: {cov_status}")

    def compute_fitness(self, x: np.ndarray) -> float:
        H0, Omega_m = float(x[0]), float(x[1])
        # Penalise unphysical values
        if Omega_m <= 0 or Omega_m >= 1 or H0 <= 0:
            return 1e12
        chi2 = chi2_pantheon(H0, Omega_m, self._data, self._use_cov)
        if self._verbose:
            print(f"    H0={H0:.2f}  Om={Omega_m:.3f}  chi2={chi2:.2f}")
        return chi2

    def is_max_problem(self) -> bool:
        return False
