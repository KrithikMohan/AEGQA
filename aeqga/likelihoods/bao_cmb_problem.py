"""Paper's 16-point BAO and simplified Planck TT objectives (not Plik).

BAO: Dainotti & Sarracino 2022 Table 1; original DR12/DR16 precision
from CobayaSampler/bao_data commit bb0c1c9009dc76d1391300e169e8df38fd1096db.
WiggleZ inverse covariance: Blake et al. 2011 Table 2. See spec for the
paper's correlated-count discrepancy and TT choices.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from aeqga.likelihoods.cosmology import C_LIGHT, distance_integral, in_search_bounds
from aeqga.paths import PROJECT_ROOT


class BAOLikelihood:
    def __init__(self):
        self.z = np.array([.106,.44,.60,.73,.15,2.33,2.33,
                           .38,.38,.51,.51,.698,.698,.65,1.48,1.48])
        self.kinds = ['DV','A','A','A','DV_fid','DH_rd','DM_rd',
                      'DM_rd','DH_rd','DM_rd','DH_rd','DM_rd','DH_rd',
                      'DV_rd','DM_rd','DH_rd']
        self.observed = np.array([456,.474,.442,.424,664,8.99,37.5,
                                 10.23406,24.98058,13.36595,22.31656,
                                 17.85823691865007,19.32575373059217,
                                 18.33,30.6876,13.2609])
        cov = np.diag(np.array([27,1,1,1,25,.19,1.1,1,1,1,1,1,1,.60,1,1])**2)
        wiggle_inverse = np.array([[1040.3,-807.5,336.8],[-807.5,3720.3,-1551.9],
                                  [336.8,-1551.9,2914.9]])
        cov[1:4,1:4] = np.linalg.solve(wiggle_inverse, np.eye(3))
        cov[7:11,7:11] = [[.02860520,-.04939281,.01489688,-.01387079],
                          [-.04939281,.5307187,-.02423513,.1767087],
                          [.01489688,-.02423513,.04147534,-.04873962],
                          [-.01387079,.1767087,-.04873962,.3268589]]
        cov[11:13,11:13] = [[.1076634008565565,-.05831820341302727],
                            [-.05831820341302727,.2838176386340292]]
        cov[14:16,14:16] = [[.63731604,.1706891],[.1706891,.30468415]]
        self.covariance = cov
        self.factor = cholesky(cov, lower=True)

    def prediction(self, H0, omega_m):
        if not in_search_bounds([H0, omega_m]) or omega_m <= 0:
            raise ValueError('BAO requires positive matter density in the search domain')
        from aeqga.likelihoods.pantheon_problem import sound_horizon
        rd = sound_horizon(omega_m, H0)
        dm = C_LIGHT/H0 * distance_integral(self.z, omega_m)
        dh = C_LIGHT/(H0*np.sqrt(omega_m*(1+self.z)**3+1-omega_m))
        dv = (self.z*dh*dm**2)**(1/3)
        acoustic = dv*np.sqrt(omega_m)*H0/(C_LIGHT*self.z)
        choices = dict(DV=dv,A=acoustic,DV_fid=dv*148.69/rd,
                       DH_rd=dh/rd,DM_rd=dm/rd,DV_rd=dv/rd)
        return np.array([choices[k][i] for i,k in enumerate(self.kinds)])

    def chi2(self, H0, omega_m):
        if not in_search_bounds([H0,omega_m]) or omega_m <= 0:
            return float('inf')
        residual = self.prediction(H0,omega_m)-self.observed
        whitened = solve_triangular(self.factor,residual,lower=True,check_finite=False)
        return float(whitened@whitened)


class PlanckTTLikelihood:
    """Diagonal Eq. 10-style TT approximation, NOT official Planck likelihood.

    Default uses every released ell 2..2508, symmetric average of error bars,
    fixed TTTEEE+lowl+lowE+lensing best-fit parameters and no y_cal adjustment.
    PICO is strict: out-of-training-domain raises, never extrapolates silently.
    Hybrid explicitly falls back to CAMB and records fallback evaluations.
    """
    def __init__(self, data_dir=PROJECT_ROOT/'data/cmb', backend='pico', ell_min=2,
                 ell_max=2508, error_mode='average'):
        if backend not in {'pico','camb','hybrid'}:
            raise ValueError('backend must be pico, camb or hybrid')
        if error_mode not in {'average','asymmetric'}:
            raise ValueError('error_mode must be average or asymmetric')
        table = np.loadtxt(Path(data_dir)/'COM_PowerSpect_CMB-TT-full_R3.01.txt')
        rows = (table[:,0]>=ell_min)&(table[:,0]<=ell_max)
        if not np.any(rows):
            raise ValueError('Empty multipole selection')
        table = table[rows]
        self.ell = table[:,0].astype(int)
        self.observed = table[:,1]
        self.minus, self.plus = table[:,2],table[:,3]
        if np.any(self.minus<=0) or np.any(self.plus<=0):
            raise ValueError('Nonpositive TT errors')
        self.backend, self.error_mode = backend, error_mode
        self.fallback_evaluations = 0
        self.ombh2, self.omnuh2 = .02238280,.00064
        self.tau, self.ns, self.As = .05430842,.9660499,np.exp(3.044784)*1e-10
        self.pico = None
        if backend != 'camb':
            from aeqga.emulators.pico_runtime import load_pico
            self.pico = load_pico(Path(data_dir)/'pico4_tailmonty_v35_py3.dat')

    def camb_parameters(self,H0,omega_m):
        import camb
        cold = omega_m*(H0/100)**2-self.ombh2-self.omnuh2
        if cold <= 0:
            raise ValueError('Matter density below fixed baryon+neutrino densities')
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0,ombh2=self.ombh2,omch2=cold,mnu=.06,nnu=3.046,tau=self.tau)
        pars.omnuh2 = self.omnuh2
        pars.InitPower.set_params(As=self.As,ns=self.ns,pivot_scalar=.05)
        pars.set_for_lmax(int(self.ell[-1]),lens_potential_accuracy=1)
        return pars

    @lru_cache(maxsize=128)
    def spectrum(self,H0,omega_m):
        import camb
        pars = self.camb_parameters(H0,omega_m)
        if self.pico is not None:
            background = camb.get_background(pars)
            inputs = {'ombh2':pars.ombh2,'omch2':pars.omch2,'omnuh2':pars.omnuh2,
                      'theta':background.cosmomc_theta(),'helium_fraction':pars.YHe,
                      'massive_neutrinos':3.046,'re_optical_depth':self.tau,
                      'scalar_amp(1)':self.As,'scalar_spectral_index(1)':self.ns,
                      'scalar_nrun(1)':0.,'pivot_scalar':.05,'Alens':1.}
            try:
                result = self.pico.get(outputs=['lensed_TT'],**inputs)['lensed_TT']
                if len(result)<=self.ell[-1] or not np.all(np.isfinite(result[self.ell])):
                    raise ValueError('Invalid PICO TT output')
                return np.asarray(result)[self.ell]
            except Exception as error:
                from pypico import CantUsePICO
                if not isinstance(error,CantUsePICO) or self.backend != 'hybrid':
                    raise
                self.fallback_evaluations += 1
        results = camb.get_results(pars)
        dl = results.get_cmb_power_spectra(CMB_unit='muK')['lensed_scalar'][:,0]
        return dl[self.ell]

    def chi2(self,H0,omega_m):
        if (not in_search_bounds([H0,omega_m]) or
                omega_m*(H0/100)**2 <= self.ombh2+self.omnuh2):
            return float('inf')
        residual = self.spectrum(float(H0),float(omega_m))-self.observed
        sigma = ((self.minus+self.plus)/2 if self.error_mode=='average'
                 else np.where(residual>=0,self.plus,self.minus))
        return float(np.sum((residual/sigma)**2))


class BAOCMBProblem:
    n_dim = 2
    bounds = np.array([[60.,80.],[0.,.5]])
    lower_bounds = np.array([60.,0.])
    upper_bounds = np.array([80.,.5])
    def __init__(self,**cmb_options):
        self.bao = BAOLikelihood()
        self.cmb = PlanckTTLikelihood(**cmb_options)

    def compute_fitness(self,x):
        if not in_search_bounds(x):
            return float('inf')
        return self.bao.chi2(*x)+self.cmb.chi2(*x)

    def is_max_problem(self):
        return False
