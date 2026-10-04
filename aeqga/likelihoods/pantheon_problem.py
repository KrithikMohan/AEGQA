"""Calibrated Pantheon+ loading, covariance chi-squared and search problem.

Deprecated legacy magnitude likelihoods, fixed-M contours, midpoint distance
cache and KDE helpers are described in docs/DEPRECATED_CODE.md rather than
kept as executable alternatives. SNe calibration must not be marginalized away.
"""
import numpy as np
from pathlib import Path
from scipy.linalg import cholesky, solve_triangular
from aeqga.likelihoods.cosmology import C_LIGHT, distance_moduli, distance_integral, in_search_bounds
from aeqga.paths import PROJECT_ROOT


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


# Eq. 9 BAO sound horizon retained because BAOLikelihood calls it.
_Omega_b_h2 = 0.02237
_Omega_nu_h2 = 0.00064

def sound_horizon(Omega_m: float, H0: float) -> float:
    """Compute the sound horizon r_d per Eq.9 (paper §2.2)."""
    Omega_b_h2 = _Omega_b_h2
    Omega_nu_h2 = _Omega_nu_h2
    h = H0 / 100.0
    Omega_M_h2 = Omega_m * h**2
    r_d = (55.154 * np.exp(-72.3 * (Omega_nu_h2 + 0.0006)**2) /
           ((Omega_M_h2)**0.25351 * (Omega_b_h2)**0.12807))
    return r_d  # Mpc
