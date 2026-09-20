"""
pantheon_problem.py
===================
Pantheon SNe Ia data loader and chi-squared likelihood for use with AEQGA.

This module provides the SNe Ia component of the AEQGA framework
described in Sarracino et al., arXiv:2602.15459 (2026).

Data format (CobayaSampler/sn_data → Pantheon/lcparam_full_long_zhel.txt)
----------------------------------------------------------------------------
Columns: #name  zcmb  zhel  dz  mb  dmb  x1  dx1  color  dcolor
         3rdvar  d3rdvar  cov_m_s  cov_m_c  cov_s_c  set  ra  dec  biascor

The cosmological chi-squared is:
    chi2 = delta^T  C^{-1}  delta
where
    delta_i = mb_i  -  mu_theory(z_i, H0, Omega_m)  -  M
    mu_theory = 5*log10(d_L(zcmb_i, zhel_i) / 10 pc)
    d_L computed in flat ΛCDM

M (absolute magnitude offset) is analytically marginalised following
the Pantheon prescription (Conley et al. 2011 Appendix C, Eq C1):
    chi2_marg = chi2  -  (sum_i delta_i/sigma_i)^2 / (sum_i 1/sigma_i)

The AEQGA searches over:
    H0       – Hubble constant  [km/s/Mpc]   range  [60, 80]
    Omega_m  – matter density                range  [0.0, 0.5]
    (paper §3 p.7: Ω_M ∈ [0.0, 0.5], H0 ∈ [60, 80])

Note: Paper §2 also uses BAO (16 data points) and CMB (Planck TT)
datasets. This module implements SNe Ia only; BAO/CMB stubs are
provided for future extension. See code-summary.md for details.
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

# Cache for precomputed distance integrals (paper §2.1 p.6)
_D_L_CACHE: dict[tuple, float] = {}


def luminosity_distance_flat_lcdm(z_cmb: float, z_hel: float,
                                   H0: float, Omega_m: float,
                                   n_steps: int = 1000,
                                   use_cache: bool = True) -> float:
    """Luminosity distance in Mpc for flat ΛCDM.

    The comoving distance integral uses zcmb for the expansion history;
    the (1+z_hel) prefactor uses the heliocentric redshift following the
    Pantheon convention (see Conley+11, Davis+19).

    Parameters
    ----------
    use_cache : bool – cache results by (H0, Omega_m) grid lookup.
        Paper §2.1 precomputes integral on 100/300 Ω_M grid and
        uses nearest-neighbor lookup for speed.
    """
    if z_cmb <= 0.0:
        return 0.0

    # Check cache
    if use_cache:
        key = (round(z_cmb, 4), round(z_hel, 4), round(H0, 1), round(Omega_m, 4))
        if key in _D_L_CACHE:
            return _D_L_CACHE[key]

    Omega_L = 1.0 - Omega_m
    dz = z_cmb / n_steps
    z_arr = (np.arange(n_steps) + 0.5) * dz
    E_inv = 1.0 / np.sqrt(Omega_m * (1.0 + z_arr)**3 + Omega_L)
    comoving = (C_LIGHT / H0) * float(np.sum(E_inv)) * dz
    result = (1.0 + z_hel) * comoving

    if use_cache:
        _D_L_CACHE[key] = result

    return result


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
    data_dir : path to the Pantheon sub-folder

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

    cov_stat = np.diag(dmb ** 2)

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

    mu_th = np.array([
        distance_modulus(zcmb[i], zhel[i], H0, Omega_m)
        for i in range(n_sn)
    ])

    delta = mb - mu_th

    if use_full_cov and data["cov_sys"] is not None:
        C = data["cov_total"]
        try:
            L = np.linalg.cholesky(C)
            y = np.linalg.solve(L, delta)
            chi2_full = float(y @ y)
            e = np.ones(n_sn)
            ye = np.linalg.solve(L, e)
            A  = float(ye @ ye)
            B  = float(ye @ y)
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
# BAO stub (paper §2.2, Eq.6-10)
# ---------------------------------------------------------------------------

# Paper §2.2 parameters:
#   Ω_b h² = 0.02237, Ω_ν h² = 0.00064, Σm_ν ≈ 0.06 eV
#   r_d = 55.154·exp(-72.3(Ω_ν h²+0.0006)²) / ((Ω_M h²)^0.25351·(Ω_b h²)^0.12807) Mpc

_Omega_b_h2 = 0.02237
_Omega_nu_h2 = 0.00064
_sum_m_nu = 0.06  # eV


def sound_horizon(Omega_m: float, H0: float) -> float:
    """Compute the sound horizon r_d per Eq.9 (paper §2.2)."""
    Omega_b_h2 = _Omega_b_h2
    Omega_nu_h2 = _Omega_nu_h2
    h = H0 / 100.0
    Omega_M_h2 = Omega_m * h**2
    r_d = (55.154 * np.exp(-72.3 * (Omega_nu_h2 + 0.0006)**2) /
           ((Omega_M_h2)**0.25351 * (Omega_b_h2)**0.12807))
    return r_d  # Mpc


def chi2_bao(H0: float, Omega_m: float, data: dict) -> float:
    """Placeholder BAO chi-squared per paper §2.2 Eq.10.

    Requires BAO dataset (16 measurements) and covariance matrix.
    Not yet implemented; returns 0.0.
    """
    return 0.0


# ---------------------------------------------------------------------------
# CMB stub (paper §2.3, PICO emulator)
# ---------------------------------------------------------------------------

def chi2_cmb(H0: float, Omega_m: float, data: dict) -> float:
    """Placeholder CMB chi-squared per paper §2.3.

    Requires Planck TT spectrum data and PICO emulator (Fendt & Wandelt 2007).
    Not yet implemented; returns 0.0.
    """
    return 0.0


# ---------------------------------------------------------------------------
# AEQGA Problem class
# ---------------------------------------------------------------------------

class PantheonProblem:
    """Flat ΛCDM parameter estimation on Pantheon SNe Ia for AEQGA.

    Fits: theta = [H0, Omega_m]
    Paper §3 p.7 search ranges: Omega_M ∈ [0.0, 0.5], H0 ∈ [60, 80]
    Minimises chi^2(theta) computed with analytic M marginalisation.

    Note: Paper uses Pantheon+ (1701 SNe) with full covariance.
    This class loads Pantheon (1048 SNe). For Pantheon+ compatibility,
    replace lcparam_full_long_zhel.txt with the Pantheon+ file.
    """

    # Paper §3 p.7 ranges (NOT [60, 0.10])
    lower_bounds = np.array([60.0, 0.0])
    upper_bounds = np.array([80.0, 0.5])
    n_dim        = 2

    # Expected best-fit reference values (Scolnic+18)
    # Paper SNe Ia result: (0.363, 72.82)
    H0_ref      = 72.82  # km/s/Mpc (paper SNe Ia)
    Om_ref      = 0.363

    def __init__(self, data_dir: str, use_full_cov: bool = True,
                 verbose: bool = False):
        print(f"Loading Pantheon data from: {data_dir}")
        self._data      = load_pantheon(data_dir)
        self._use_cov   = use_full_cov
        self._verbose   = verbose
        print(f"  Loaded {self._data['n_sn']} supernovae")
        cov_status = "stat+sys" if self._data["cov_sys"] is not None else "stat only"
        print(f"  Covariance: {cov_status}")
        print(f"  Parameter bounds: H0 [{self.lower_bounds[0]}, {self.upper_bounds[0]}], "
              f"Omega_M [{self.lower_bounds[1]}, {self.upper_bounds[1]}]")

    def compute_fitness(self, x: np.ndarray) -> float:
        H0, Omega_m = float(x[0]), float(x[1])
        if Omega_m <= 0 or Omega_m >= 0.5 or H0 <= 0 or H0 > 80:
            return 1e12
        chi2 = chi2_pantheon(H0, Omega_m, self._data, self._use_cov)
        if self._verbose:
            print(f"    H0={H0:.2f}  Om={Omega_m:.3f}  chi2={chi2:.2f}")
        return chi2

    def is_max_problem(self) -> bool:
        return False
