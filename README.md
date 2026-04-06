# AEQGA — Amplitude-Encoded Quantum Genetic Algorithm
### Applied to Pantheon SNe Ia Cosmological Parameter Estimation

Implementation of the AEQGA described in:
> Sarracino et al., *"A Quantum Genetic Algorithm with application to
> Cosmological Parameters Estimation"*, arXiv:2602.15459 (2026)

Adapted from the Quasar-UniNA HQGA repository:
https://github.com/Quasar-UniNA/HQGA

---

## Files

| File | Purpose |
|------|---------|
| `aeqga_algorithm.py` | Core AEQGA implementation (3 variants) |
| `pantheon_problem.py` | Pantheon data loader + flat ΛCDM chi² likelihood |
| `run_pantheon_aeqga.py` | **Main runner** — clones data, runs AEQGA, plots results |
| `test_pantheon_aeqga.py` | Unit + integration test suite |
| `requirements.txt` | Python dependencies |

---

## Quickstart

```bash
# 1. Install dependencies
pip install qiskit qiskit-aer tqdm numpy matplotlib

# 2. Run — data is cloned automatically from CobayaSampler/sn_data
python run_pantheon_aeqga.py

# 3. Fast test (20 generations, no contour plot)
python run_pantheon_aeqga.py --fast

# 4. Full run with more generations
python run_pantheon_aeqga.py --gen 100 --pop 8 --shots 8192

# 5. If you already have the data
python run_pantheon_aeqga.py --data /path/to/sn_data/Pantheon
```

## Output files produced

- `pantheon_convergence.png` — chi² vs generation
- `pantheon_contours.png` — 1/2/3-sigma contours in (H0, Omega_m) plane
  (skipped with `--fast` or `--no-contour`)

---

## Algorithm summary

The AEQGA encodes each real-valued individual xᵢ ∈ [lower, upper] as a
qubit rotation angle:

    θᵢ = 2 · arcsin(√((xᵢ − lower) / (upper − lower)))

so that P(|1⟩) = sin²(θᵢ/2) encodes the normalised position in the
search space. Each generation:

1. **Encode** — RY(θᵢ) gates initialise each qubit
2. **Crossover** — adjacent pairs entangled via RY–CNOT–RY with prob p_cross
3. **Mutation** — random RY(δ), δ~N(0,σ), applied per qubit with prob p_mut
4. **Measure** — shot-based or statevector readout
5. **Decode** — P(|1⟩) back-mapped to real values
6. **Evaluate** — classical chi² against Pantheon data
7. **Select** — elitism + fitness-ranked re-seeding

Parameters fitted: **H0** ∈ [60, 80] km/s/Mpc, **Ωm** ∈ [0.1, 0.6]

Expected best-fit: H0 ≈ 67–70, Ωm ≈ 0.28–0.32 (consistent with Scolnic+18)

---

## Data

The Pantheon dataset (1048 SNe Ia, Scolnic+18, arXiv:1710.00845) is
fetched automatically from:
https://github.com/CobayaSampler/sn_data

Files used:
- `Pantheon/lcparam_full_long_zhel.txt` — redshifts + standardised magnitudes
- `Pantheon/sys_full_long.txt` — 1048×1048 systematic covariance matrix
