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
datasets. Calibrated Pantheon+ is implemented by PantheonPlusProblem below;
the marginalized legacy Pantheon objective above cannot identify H0.
BAO/CMB implementations live in bao_cmb_problem.py; wrappers require a
configured likelihood rather than silently returning zero.
"""

import os
import math
import numpy as np
from pathlib import Path
from scipy.linalg import cholesky, solve_triangular
from aeqga.likelihoods.cosmology import distance_moduli, distance_integral, in_search_bounds
from aeqga.paths import PROJECT_ROOT

# matplotlib imported lazily in contour helpers to allow headless usage


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
        key = (float(z_cmb), float(z_hel), float(H0), float(Omega_m), int(n_steps))
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


def load_pantheon_plus(data_dir: str, *, selection="all", redshift="hd_hel") -> dict:
    """Load calibrated Pantheon+ moduli and the already-total covariance.

    ``all`` retains 1701 light curves. ``hubble_flow`` selects zHD>0.01
    and excludes Cepheid calibrators, without deduplicating repeated light
    curves. ``cmb`` follows the paper's single-z distance expression;
    ``hd_hel`` uses zHD for the integral and zHEL for its prefactor.
    Every row selection is applied to both covariance axes. No additional
    diagonal error variance is added to STAT+SYS.
    """
    directory = Path(data_dir)
    table_file = directory / "Pantheon+SH0ES.dat"
    covariance_file = directory / "Pantheon+SH0ES_STAT+SYS.cov"
    table = np.genfromtxt(table_file, names=True, dtype=None, encoding="utf-8")
    required = {"CID", "zHD", "zCMB", "zHEL", "MU_SH0ES", "MU_SH0ES_ERR_DIAG",
                "CEPH_DIST", "IS_CALIBRATOR", "USED_IN_SH0ES_HF"}
    missing = required.difference(table.dtype.names or ())
    if missing:
        raise ValueError(f"Missing Pantheon+ columns: {sorted(missing)}")
    n_full = len(table)
    packed = np.fromstring(covariance_file.read_text(), sep=" ")
    if not len(packed) or packed[0] != n_full or len(packed) != 1+n_full**2:
        raise ValueError("Pantheon+ covariance dimension/element count mismatch")
    covariance = packed[1:].reshape(n_full, n_full)
    asymmetry = float(np.max(np.abs(covariance-covariance.T)))
    if not np.all(np.isfinite(covariance)) or asymmetry > 5e-8:
        raise ValueError("Pantheon+ covariance must be finite and symmetric")
    # Published decimal serialization differs by <=3e-8 across the diagonal.
    # Use the symmetric average and retain the measured discrepancy as metadata.
    covariance = (covariance + covariance.T) / 2
    if selection == "all":
        mask = np.ones(n_full, dtype=bool)
    elif selection == "hubble_flow":
        mask = (table["zHD"] > .01) & (table["IS_CALIBRATOR"] == 0)
    else:
        raise ValueError("selection must be 'all' or 'hubble_flow'")
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise ValueError("Pantheon+ selection contains no observations")
    selected = table[mask]
    covariance = covariance[np.ix_(indices, indices)]
    factor = cholesky(covariance, lower=True, check_finite=False)
    if redshift == "cmb":
        z_integral, z_prefactor = selected["zCMB"], selected["zCMB"]
    elif redshift == "hd_hel":
        z_integral, z_prefactor = selected["zHD"], selected["zHEL"]
    else:
        raise ValueError("redshift must be 'cmb' or 'hd_hel'")
    if (np.any(z_integral <= 0) or np.any(z_prefactor <= 0)
            or not np.all(np.isfinite(selected["MU_SH0ES"]))):
        raise ValueError("Invalid Pantheon+ redshift or calibrated modulus")
    return dict(zcmb=np.asarray(z_integral), zhel=np.asarray(z_prefactor),
                zHD=selected["zHD"], zCMB=selected["zCMB"], zHEL=selected["zHEL"],
                mu_obs=selected["MU_SH0ES"], mu_err=selected["MU_SH0ES_ERR_DIAG"],
                cid=selected["CID"], is_calibrator=selected["IS_CALIBRATOR"].astype(bool),
                cepheid_distance=selected["CEPH_DIST"],
                used_in_shoes_hf=selected["USED_IN_SH0ES_HF"].astype(bool),
                cov_total=covariance, cholesky=factor, n_sn=len(indices),
                row_indices=indices, n_full=n_full, selection=selection,
                redshift=redshift, covariance_max_asymmetry=asymmetry,
                table_file=str(table_file), covariance_file=str(covariance_file))


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
        L = data.get("_legacy_cholesky")
        if L is None:
            L = cholesky(data["cov_total"], lower=True)
            data["_legacy_cholesky"] = L
        y = solve_triangular(L, delta, lower=True, check_finite=False)
        ye = data.get("_legacy_ones_white")
        if ye is None:
            ye = solve_triangular(L, np.ones(n_sn), lower=True, check_finite=False)
            data["_legacy_ones_white"] = ye
        chi2_marg = float(y @ y - (ye @ y)**2 / (ye @ ye))
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


def chi2_pantheon_plus(H0: float, Omega_m: float, data: dict) -> float:
    """Eq. 5 evaluated on published, Cepheid-calibrated MU_SH0ES.

    Do not introduce an arbitrary absolute magnitude or marginalize the
    calibrated offset: doing so erases the H0 constraint.
    """
    mu = distance_moduli(data["zcmb"], data["zhel"], H0, Omega_m)
    residual = mu-data["mu_obs"]
    whitened = solve_triangular(data["cholesky"], residual, lower=True,
                               check_finite=False)
    return float(whitened @ whitened)


class PantheonPlusProblem:
    """One calibrated objective for both AEQGA and contour-grid evaluation."""
    lower_bounds = np.array([60., 0.])
    upper_bounds = np.array([80., .5])
    n_dim = 2

    def __init__(self, data_dir=PROJECT_ROOT/"sn_data/PantheonPlus", *, selection="all", redshift="hd_hel",
                 grid_size=300):
        self._data = load_pantheon_plus(data_dir, selection=selection, redshift=redshift)
        if grid_size not in {0, 100, 300}:
            raise ValueError("grid_size must be 0 (direct), 100, or 300")
        self.grid_size = grid_size
        self._ones_white = solve_triangular(self._data["cholesky"],
                                           np.ones(self._data["n_sn"]), lower=True,
                                           check_finite=False)
        self._A = float(self._ones_white @ self._ones_white)
        if grid_size:
            self.omega_grid = np.linspace(0., .5, grid_size)
            # Precompute the distance integral exactly at the specified nodes.
            integrals = np.array([distance_integral(self._data["zcmb"], om)
                                  for om in self.omega_grid])
            mu70 = 5*np.log10(C_LIGHT/70*(1+self._data["zhel"])*integrals)+25
            white = solve_triangular(self._data["cholesky"],
                                     (mu70-self._data["mu_obs"]).T,
                                     lower=True, check_finite=False)
            self._B = self._ones_white @ white
            self._C = np.sum(white**2, axis=0)

    def _coefficients(self, omega_m):
        if self.grid_size:
            # Nearest neighbor: left node wins an exact midpoint tie.
            right = int(np.searchsorted(self.omega_grid, omega_m))
            right = min(right, self.grid_size-1)
            left = max(0, right-1)
            index = left if omega_m-self.omega_grid[left] <= self.omega_grid[right]-omega_m else right
            return float(self._B[index]), float(self._C[index])
        mu70 = distance_moduli(self._data["zcmb"], self._data["zhel"], 70., omega_m)
        white = solve_triangular(self._data["cholesky"], mu70-self._data["mu_obs"],
                                 lower=True, check_finite=False)
        return float(self._ones_white @ white), float(white @ white)

    def compute_fitness(self, x):
        if not in_search_bounds(x):
            return float("inf")
        B, C = self._coefficients(float(x[1]))
        shift = -5*np.log10(float(x[0])/70.)
        # Algebraically exact H0 dependence; only Omega_m is approximated.
        return float(C + 2*shift*B + shift**2*self._A)

    def classical_minimum(self):
        """Profile the exact H0 offset; scan before minimizing in Omega_m."""
        from scipy.optimize import minimize_scalar
        def profiled(omega):
            B, C = self._coefficients(omega)
            shift = np.clip(-B/self._A, -5*np.log10(80/70), -5*np.log10(60/70))
            return float(C+2*shift*B+shift**2*self._A), float(70*10**(-shift/5))
        if self.grid_size:
            candidates = self.omega_grid
        else:
            scan = np.linspace(0., .5, 101)
            best_index = int(np.argmin([profiled(om)[0] for om in scan]))
            result = minimize_scalar(lambda om: profiled(om)[0], method="bounded",
                                     bounds=(scan[max(0,best_index-1)],
                                             scan[min(len(scan)-1,best_index+1)]),
                                     options={"xatol":1e-11})
            candidates = [0., result.x, .5]
        omega = min(candidates, key=lambda om: profiled(om)[0])
        chi2, h0 = profiled(omega)
        return dict(H0=h0, Omega_m=float(omega), chi2=chi2, grid_size=self.grid_size,
                    selection=self._data["selection"], redshift=self._data["redshift"])

    def is_max_problem(self):
        return False

    def objective_grid(self, h0_values, omega_values):
        """Rows are Omega_m, columns H0; evaluate the optimizer's objective."""
        return np.array([[self.compute_fitness([h0, om]) for h0 in h0_values]
                         for om in omega_values])


# ---------------------------------------------------------------------------
# Chi-squared with FIXED M (for contour plots, paper Fig.3)
# ---------------------------------------------------------------------------

def chi2_pantheon_fixed_M(H0: float, Omega_m: float, data: dict,
                          M_ref: float = -19.36,
                          use_full_cov: bool = True) -> float:
    """Compute Pantheon chi^2 with M FIXED (not marginalised).

    This produces elliptical contours in (H0, Omega_m) space because
    the H0-dependence is preserved.  Used for objective-function contour
    plots matching paper Fig.3 (green contours).

    Parameters
    ----------
    H0       : Hubble constant [km/s/Mpc]
    Omega_m  : matter density parameter
    data     : dict from load_pantheon()
    M_ref    : fixed absolute magnitude offset (default -19.36)
    use_full_cov : use full covariance matrix (stat+sys)

    Returns
    -------
    float : chi^2 (no M marginalisation)
    """
    zcmb  = data["zcmb"]
    zhel  = data["zhel"]
    mb    = data["mb"]
    n_sn  = data["n_sn"]

    mu_th = np.array([
        distance_modulus(zcmb[i], zhel[i], H0, Omega_m)
        for i in range(n_sn)
    ])

    delta = mb - mu_th - M_ref

    if use_full_cov and data["cov_sys"] is not None:
        L = data.get("_legacy_cholesky")
        if L is None:
            L = cholesky(data["cov_total"], lower=True)
            data["_legacy_cholesky"] = L
        y = solve_triangular(L, delta, lower=True, check_finite=False)
        chi2 = float(y @ y)
    else:
        chi2 = float(np.sum(delta**2 / data["dmb"]**2))

    return chi2


# ---------------------------------------------------------------------------
# KDE-based confidence contours (for results plots, paper Fig.4)
# ---------------------------------------------------------------------------

def compute_kde_contours(points: np.ndarray,
                         grid_size: int = 100,
                         h0_range: tuple = (60.0, 80.0),
                         om_range: tuple = (0.0, 0.5),
                         ) -> dict:
    """Compute 2D Gaussian KDE from AEQGA best-fit points.

    Implements paper Fig.4 confidence contours from n_iterations runs.

    Parameters
    ----------
    points : shape (n, 2) — array of [H0, Omega_m] best-fits
    grid_size : number of grid points per dimension
    h0_range  : (min, max) for H0 axis
    om_range  : (min, max) for Omega_m axis

    Returns
    -------
    dict with keys:
        H0_arr, Om_arr : 1D grid arrays
        P_grid         : 2D probability density
        mean           : (H0_mean, Om_mean)
        std            : (H0_std, Om_std)
        P_max          : maximum probability density
    """
    from scipy.stats import gaussian_kde

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be (n, 2), got {points.shape}")

    mean = points.mean(axis=0)
    std = points.std(axis=0)

    kde = gaussian_kde(points.T)

    H0_arr = np.linspace(h0_range[0], h0_range[1], grid_size)
    Om_arr = np.linspace(om_range[0], om_range[1], grid_size)
    H0_grid, Om_grid = np.meshgrid(H0_arr, Om_arr)
    coords = np.vstack([H0_grid.ravel(), Om_grid.ravel()])
    P_grid = kde(coords).reshape(grid_size, grid_size)

    P_max = P_grid.max()

    return {
        "H0_arr": H0_arr,
        "Om_arr": Om_arr,
        "H0_grid": H0_grid,
        "Om_grid": Om_grid,
        "P_grid": P_grid,
        "mean": mean,
        "std": std,
        "P_max": P_max,
    }


# ---------------------------------------------------------------------------
# BAO helpers (paper §2.2, Eq.6-10)
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
    """Evaluate an explicitly supplied BAOLikelihood; never silently omit data."""
    if not hasattr(data, 'chi2'):
        raise ValueError('Supply a BAOLikelihood instance')
    return data.chi2(H0, Omega_m)


# ---------------------------------------------------------------------------
# CMB compatibility entry point (paper §2.3, PICO emulator)
# ---------------------------------------------------------------------------

def chi2_cmb(H0: float, Omega_m: float, data: dict) -> float:
    """Evaluate an explicitly supplied PlanckTTLikelihood."""
    if not hasattr(data, 'chi2'):
        raise ValueError('Supply a PlanckTTLikelihood instance')
    return data.chi2(H0, Omega_m)


# ---------------------------------------------------------------------------
# AEQGA Problem class
# ---------------------------------------------------------------------------

class PantheonProblem:
    """Flat ΛCDM parameter estimation on Pantheon SNe Ia for AEQGA.

    Fits: theta = [H0, Omega_m]
    Paper §3 p.7 search ranges: Omega_M ∈ [0.0, 0.5], H0 ∈ [60, 80]
    Minimises chi^2(theta) computed with analytic M marginalisation.

    Note: Paper uses Pantheon+ (1701 SNe) with full covariance.
    This legacy class loads Pantheon (1048 SNe), not Pantheon+.
    Use PantheonPlusProblem for the paper's calibrated H0 fit; its data
    format and calibration cannot be substituted into this loader.
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
        if not in_search_bounds(x):
            return float("inf")
        chi2 = chi2_pantheon(H0, Omega_m, self._data, self._use_cov)
        if self._verbose:
            print(f"    H0={H0:.2f}  Om={Omega_m:.3f}  chi2={chi2:.2f}")
        return chi2

    def is_max_problem(self) -> bool:
        return False
