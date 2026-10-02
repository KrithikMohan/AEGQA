"""Flat LCDM low-redshift distances shared by the SNe and BAO likelihoods."""
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss

C_LIGHT = 299792.458


@lru_cache(maxsize=8)
def _quadrature(order):
    return leggauss(order)


def distance_integral(z, omega_m, order=48):
    """Vectorized Gauss–Legendre integral of 1/E(z); no rounded caching."""
    redshifts = np.asarray(z, dtype=float)
    if (not np.isfinite(omega_m) or not 0 <= omega_m <= 1
            or np.any(~np.isfinite(redshifts)) or np.any(redshifts < 0)):
        raise ValueError("Invalid flat LCDM matter density/redshift")
    nodes, weights = _quadrature(order)
    sample = redshifts[..., None] * (nodes+1)/2
    inverse_E = 1/np.sqrt(omega_m*(1+sample)**3 + 1-omega_m)
    return redshifts/2 * np.sum(inverse_E*weights, axis=-1)


def distance_moduli(z_integral, z_prefactor, H0, omega_m):
    if not np.isfinite(H0) or H0 <= 0:
        raise ValueError("H0 must be finite and positive")
    distance = C_LIGHT/H0 * (1+np.asarray(z_prefactor)) * distance_integral(z_integral, omega_m)
    if np.any(distance <= 0):
        raise ValueError("Distance modulus requires positive luminosity distance")
    return 5*np.log10(distance)+25


def in_search_bounds(x):
    values = np.asarray(x, dtype=float)
    return (values.shape == (2,) and np.all(np.isfinite(values))
            and 60 <= values[0] <= 80 and 0 <= values[1] <= .5)
