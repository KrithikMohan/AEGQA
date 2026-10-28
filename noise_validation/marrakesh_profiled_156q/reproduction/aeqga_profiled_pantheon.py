"""Full-covariance Pantheon+ likelihood with the absolute magnitude profiled out.

M_hat = clip((1.T C^-1 (m - mu)) / (1.T C^-1 1), M_lower, M_upper).
The clipping preserves the original three-parameter search bounds exactly.
Only H0 and Omega_m remain quantum-search coordinates.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve_triangular
from run_pantheon_aeqga import load_data, make_model, classical_profile_reference, C_KM_S


class ProfiledPantheon:
    def __init__(self, *, order=32, magnitude_bounds=(-19.6, -19.0), data=None):
        if data is None:
            self.table, self.selected, self.x, self.y, self.covariance, self.covariance_validation = load_data()
        else:
            self.x, self.y, self.covariance = map(np.asarray, data)
            self.table = self.selected = None
            self.covariance_validation = {}
        self.bounds = [(60.,80.),(.1,.5)]
        self.magnitude_bounds = tuple(map(float, magnitude_bounds))
        self.order = order
        self.cholesky = np.linalg.cholesky(self.covariance)
        self.ones_white = solve_triangular(self.cholesky, np.ones(len(self.y)), lower=True)
        self.m_denominator = float(self.ones_white @ self.ones_white)
        self.flow = self.x[:, 2] == 0
        self.flow_white = solve_triangular(self.cholesky, self.flow.astype(float), lower=True)
        nodes, self.weights = leggauss(order)
        z, zhel = self.x[self.flow, 0], self.x[self.flow, 1]
        self.integrand_factor = (1 + z[:, None]*(nodes+1)/2)**3 - 1
        self.distance_factor = (1+zhel)*C_KM_S/70*z/2
        self.cache = {}

    def evaluate(self, parameters, *, chunk_size=256):
        parameters = np.asarray(parameters, dtype=float).reshape(-1, 2)
        result = np.empty((len(parameters), 3))  # chi2, bounded M, unconstrained M
        missing = {}
        for i, p in enumerate(parameters):
            key = tuple(p)
            if key in self.cache:
                result[i] = self.cache[key]
            else:
                missing.setdefault(key, []).append(i)
        keys = list(missing)
        for offset in range(0, len(keys), chunk_size):
            part = keys[offset:offset+chunk_size]
            p = np.asarray(part)
            omegas, inverse = np.unique(p[:, 1], return_inverse=True)
            integration = np.sum(self.weights[None, None, :] / np.sqrt(
                1 + self.integrand_factor[:, None, :]*omegas[None, :, None]), axis=2)
            mu = np.repeat(self.x[:, 3, None], len(omegas), axis=1)
            mu[self.flow] = 5*np.log10(self.distance_factor[:, None]*integration) + 25
            residual = solve_triangular(self.cholesky, self.y[:, None]-mu,
                                        lower=True, check_finite=False)[:, inverse]
            residual += self.flow_white[:, None]*(5*np.log10(p[:, 0]/70))[None, :]
            unconstrained = self.ones_white @ residual / self.m_denominator
            magnitude = np.clip(unconstrained, *self.magnitude_bounds)
            residual -= self.ones_white[:, None]*magnitude[None, :]
            losses = np.einsum('ij,ij->j', residual, residual)
            for j, key in enumerate(part):
                values = (float(losses[j]), float(magnitude[j]), float(unconstrained[j]))
                self.cache[key] = values
                for i in missing[key]:
                    result[i] = values
        return result

    def retain(self, parameters):
        keys = {tuple(p) for p in np.asarray(parameters).reshape(-1,2)}
        self.cache = {key: value for key, value in self.cache.items() if key in keys}

    def predict(self, x, p):
        magnitude = self.evaluate([p])[0, 1]
        return make_model(self.order)(x, [*p, magnitude])

    def reference(self):
        return classical_profile_reference(make_model(64), self.x, self.y,
                                           self.cholesky, self.bounds+[self.magnitude_bounds])

    def direct_check(self, p, magnitude, *, order=64):
        prediction = make_model(order)(self.x.tolist(), [*p, magnitude])
        residual = solve_triangular(self.cholesky, prediction-self.y,
                                    lower=True, check_finite=False)
        return float(residual @ residual)
