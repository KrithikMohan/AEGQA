# AEQGA replication specification

Source: Sarracino et al., [arXiv:2602.15459v1](https://arxiv.org/html/2602.15459v1).
Implementation order: circuit correctness, generation/bounds correctness,
Pantheon+ loading, calibrated likelihood, numerical acceleration, BAO/CMB.

## Verified requirements

| Component | Requirement | Paper reference |
|---|---|---|
| Parameter order in this repository | `[H0, Omega_m]` | Explicit local convention; paper sometimes reverses the order |
| Domain | H0 in [60,80] km/s/Mpc; Omega_m in [0,0.5] | §3 |
| Population | 25% unchanged elites, 25% copies, 50% fresh uniform draws | Algorithm 1 |
| Encoding | L2-normalized real amplitudes, independent circuit per parameter and subset | Eq. 11, §3.1 |
| Crossover | Two CRY(pi/2) gates on a random ordered qubit pair, swapping control/target | Eqs. 12–14 |
| Mutation | RX(pi/2) on one random qubit | Eq. 15 |
| Readout | Measure only after genetic gates | Algorithm 1 |
| Random decoding | Full-range count min–max transform; resample exact lower-bound outcomes | Eq. 16, Algorithm 2 |
| Elite decoding | Restricted elite interval; square root of basis probabilities; resample exact lower-bound outcomes in both branches | Eqs. 17–18, Algorithm 2 line 11 |
| Main ensemble | 32 individuals, 50 updates, 300 independent iterations, pc=pm=0.5 | Fig. 4 |
| Hyperparameter studies | Usually 16 individuals; settings must follow each figure | §§3.4–4.2 |
| SNe | Pantheon+ calibrated distance moduli and full statistical+systematic covariance | Eq. 5 |
| Distance approximation | 100/300 uniformly spaced Omega_m samples, nearest-neighbor integral lookup | §2.1, Appendix C.1 |
| BAO | 16 measurements; retain the correlations among 10 measurements | §2.2 |
| Sound horizon | Eq. 9, omega_b=0.02237, omega_nu=0.00064 | §2.2 |
| CMB | Planck TT, fixed other parameters; multipole-wise chi-squared; PICO spectra | §2.3 |
| Experiments | SNe separately from CMB+BAO, not an all-probe sum | §2 |
| Uncertainties | Scatter of independent optimizer outcomes, not posterior uncertainty | §4 |

## Details not specified sufficiently for bit-for-bit reproduction

- The exact shot count, random seeds, Qiskit version and simulator configuration.
- The exact numerical construction of the restricted elite decoding interval.
  This code exposes a relative interval margin, defaults to the observed elite
  min/max (zero expansion), and clips the interval to global bounds. A collapsed
  interval remains collapsed. The previous unexplained 5% expansion and absolute
  0.05 fallback are not treated as paper requirements.
- The exact Pantheon+ row mask, redshift convention and handling of calibrators.
  Use published `MU_SH0ES` with the matching supplied covariance, retain a
  recorded selection/redshift mode, and report sensitivity rather than tuning
  calibration to hit the published minimum. Do not add statistical variances
  again to the already-total covariance or marginalize away H0.
- The exact BAO release files/order and exact Planck TT subset, asymmetric-error
  convention, binning and fixed parameter column.
- PICO training-file identity and supported parameter domain. A CAMB reference
  backend is useful for validation but is an explicit deviation from the paper's
  selected emulator. Never silently substitute a different cosmology/backend.

## Validation gates

1. Measured counts match post-gate statevector probabilities within shot error.
2. Initial history plus exactly ng updates; preserved elites and valid bounds.
3. Dataset row masks applied to both axes of covariance; covariance SPD.
4. One likelihood used by optimizer and objective-grid tools; direct classical
   minimum and H0 identifiability checked before quantum convergence claims.
5. Direct integration compared with both paper grid resolutions; performance
   measured after factoring covariance once.
6. BAO and CMB validated independently before combined minimization. Missing
   data or emulator files must raise errors, never contribute zero silently.

The numerical target is statistical consistency with the paper; matching a
hard-coded center or obtaining identical stochastic outputs is not validation.

## Implemented data/backend choices (tasks 4–7)

- Default SNe: all 1701 light-curve rows, `MU_SH0ES`, total covariance,
  from CobayaSampler/sn_data revision `61d96434cafc2770928322c38e5a750e686368ae`.
  Use `zHD` in the distance integral and `zHEL` in its luminosity prefactor.
  Alternative CMB-frame-only and Hubble-flow-only modes are explicit.
  Published covariance decimal asymmetry is at most 3e-8; average both
  triangles before Cholesky factorization and record the measured discrepancy.
  No deduplication by CID: repeated light curves are intentional, and CIDs
  alone are not globally unique.
- BAO uses [Dainotti & Sarracino Table 1](https://academic.oup.com/pasj/article/74/5/1095/6655935),
  [Blake et al. Table 2](https://academic.oup.com/mnras/article/418/3/1707/1061950),
  and [official Cobaya-distributed SDSS tables](https://github.com/CobayaSampler/bao_data/tree/bb0c1c9009dc76d1391300e169e8df38fd1096db).
  There are 16 rows with covariance blocks of 3 (WiggleZ), 4 (DR12), 2
  (DR16 LRG), 2 (DR16 QSO): 11 correlated rows, not the paper's stated 10.
  Preserve all these blocks. Use the reference table's z=0.65 DV/rd row;
  do not silently replace it with a newer ELG release/redshift. The two
  z=2.33 entries have diagonal errors as tabulated by that reference.
- TT defaults: [Planck 2018 full unbinned TT](https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-TT-full_R3.01.txt),
  ell=2..2508, D_ell in microkelvin squared, average lower/upper uncertainties.
  Optional asymmetric errors and ell_min are recorded. This is the paper's
  simplified diagonal objective, NOT the full Planck Plik likelihood.
- Fixed CMB inputs use the release's
  [TTTEEE+lowl+lowE+lensing best-fit column](https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum_R3.01.txt):
  omega_b=0.02238280, tau=0.05430842, ns=0.9660499,
  ln(1e10 As)=3.044784, pivot=0.05/Mpc, Alens=1; omega_nu=0.00064,
  N_eff=3.046 in both backends and BBN-predicted helium from CAMB.
  No fitted foreground/nuisance terms or calibration adjustment. Set
  omega_c=Omega_m h^2-omega_b-omega_nu; physically impossible densities
  return infinity, never a valid zero likelihood.
- [Official PICO tailmonty v35 Python 3 model](https://github.com/marius311/pypico-trainer/releases/tag/tailmony_v35_py3)
  has checksum-pinned loading through a Python-only PICO 4 protocol adapter.
  Model polynomial min/max globals use Python builtins for NumPy 2
  compatibility; fitted coefficients and calculations are otherwise unchanged.
  The original package needs removed numpy.distutils/imp; this adapter avoids
  its obsolete C/Fortran build interface. Executable pickles other than the
  known pinned model are rejected.
- Backend `pico` is strict. The public model rejects the paper target under
  the above fixed parameters and does not cover the full domain. Backend
  `camb` is an explicit reference alternative; `hybrid` records CAMB fallbacks
  on PICO domain rejection. Neither is claimed to exactly reproduce the
  unidentified PICO training file used by the authors.

Tasks 8–12 are now implemented and a full 300-run SNe ensemble has been executed.
See [methodology and measured results](TASKS_8_TO_12_REPORT.md) for provenance,
convergence accuracy, optimizer-distribution contours and remaining deviations.
The CMB+BAO production replication is still limited by the unidentified author
emulator/TT settings; the resumable reference runner does not certify it.
