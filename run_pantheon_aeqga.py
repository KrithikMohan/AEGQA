"""
run_pantheon_aeqga.py
=====================
END-TO-END runner: AEQGA on Pantheon SNe Ia data (arXiv:2602.15459).

Steps performed
---------------
1. Clone the data (if not already present)
2. Load Pantheon: 1048 SNe Ia, redshifts + standardised magnitudes + covariance
3. Run the AEQGA to minimise chi^2(H0, Omega_m) in flat LCDM
4. Print the best-fit cosmological parameters
5. Plot convergence curve  →  pantheon_convergence.png
6. Plot 2-D chi^2 contours around the best-fit  →  pantheon_contours.png

Usage
-----
    # Option A: let this script clone the data automatically
    python run_pantheon_aeqga.py

    # Option B: point at an existing local clone
    python run_pantheon_aeqga.py --data /path/to/sn_data/Pantheon

    # Faster test (fewer generations, no contour plot)
    python run_pantheon_aeqga.py --fast

Requirements
------------
    pip install qiskit qiskit-aer tqdm numpy matplotlib
    git  (only if data clone is needed)
"""

import argparse
import os
import subprocess
import sys
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=None,
                    help="Path to Pantheon sub-folder of sn_data clone")
parser.add_argument("--fast", action="store_true",
                    help="Short run: 8 individuals, 20 generations")
parser.add_argument("--pop",  type=int, default=8,
                    help="Population size (default 8)")
parser.add_argument("--gen",  type=int, default=50,
                    help="Number of generations (default 50)")
parser.add_argument("--shots", type=int, default=8192,
                    help="Qiskit shots per generation (default 8192)")
parser.add_argument("--no-contour", action="store_true",
                    help="Skip the 2-D contour plot (faster)")
args = parser.parse_args()

if args.fast:
    args.pop  = 8
    args.gen  = 20
    args.shots = 4096
    args.no_contour = True

# ---------------------------------------------------------------------------
# Step 1 — Data
# ---------------------------------------------------------------------------

DATA_REPO = "https://github.com/CobayaSampler/sn_data"

def get_data_dir():
    if args.data:
        d = args.data
    else:
        d = os.path.join(os.path.dirname(__file__), "sn_data", "Pantheon")

    if not os.path.isdir(d):
        parent = os.path.dirname(d)
        os.makedirs(parent, exist_ok=True)
        print(f"Cloning {DATA_REPO} ...")
        ret = subprocess.run(
            ["git", "clone", "--depth", "1", DATA_REPO,
             os.path.join(parent)],
            check=False
        )
        if ret.returncode != 0:
            sys.exit(
                f"\nERROR: git clone failed.\n"
                f"Clone manually:\n  git clone {DATA_REPO}\n"
                f"Then re-run with:  python run_pantheon_aeqga.py --data sn_data/Pantheon"
            )
    if not os.path.isdir(d):
        sys.exit(f"ERROR: Data directory not found: {d}")
    return d

data_dir = get_data_dir()
print(f"Data directory: {data_dir}")

# ---------------------------------------------------------------------------
# Step 2 — Imports (after verifying data exists)
# ---------------------------------------------------------------------------

try:
    from qiskit_aer import AerSimulator
except ImportError:
    sys.exit(
        "ERROR: qiskit-aer not installed.\n"
        "Install with:  pip install qiskit qiskit-aer tqdm numpy matplotlib"
    )

from pantheon_problem  import PantheonProblem, chi2_pantheon, load_pantheon
from aeqga_algorithm   import AEQGAParameters, run_aeqga, plot_convergence

# ---------------------------------------------------------------------------
# Step 3 — Build problem
# ---------------------------------------------------------------------------

print("\n=== Loading Pantheon data ===")
problem = PantheonProblem(data_dir, use_full_cov=True, verbose=False)

# Quick sanity check: chi2 at Planck best-fit
x_planck = np.array([67.4, 0.298])
chi2_planck = problem.compute_fitness(x_planck)
print(f"chi2 at Planck best-fit (H0=67.4, Om=0.298) = {chi2_planck:.2f}")
print(f"  (reduced chi2 ~ {chi2_planck / (problem._data['n_sn'] - 2):.3f}  "
      f"[expected ~1.0 for a good fit])")

# ---------------------------------------------------------------------------
# Step 4 — Run AEQGA
# ---------------------------------------------------------------------------

print(f"\n=== Running AEQGA  (pop={args.pop}, gen={args.gen}, shots={args.shots}) ===")

params = AEQGAParameters(
    pop_size     = args.pop,
    max_gen      = args.gen,
    p_cross      = 0.7,
    p_mut        = 0.15,
    sigma_mut    = 0.05,   # tight — search bounds already narrow [60,80]x[0.1,0.6]
    num_shots    = args.shots,
    verbose      = False,
    progress_bar = True,
)

g_best, population_evol, bests_log = run_aeqga(problem, params)

H0_best = g_best.x[0]
Om_best = g_best.x[1]
chi2_best = g_best.fitness

print("\n=== Results ===")
print(f"  Best-fit H0      = {H0_best:.2f}  km/s/Mpc")
print(f"  Best-fit Omega_m = {Om_best:.4f}")
print(f"  chi2_min         = {chi2_best:.2f}")
print(f"  Reduced chi2     = {chi2_best / (problem._data['n_sn'] - 2):.4f}")
print(f"  Found at gen     = {g_best.gen}")
print(f"\n  Reference (Scolnic+18): H0 ~ 67.4, Omega_m ~ 0.298")
print(f"  Reference (Planck 2018): H0 ~ 67.4, Omega_m ~ 0.315")

# ---------------------------------------------------------------------------
# Step 5 — Convergence plot
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")   # headless
    import matplotlib.pyplot as plt

    gens   = list(range(len(bests_log)))
    fitvals = [b[1] for b in bests_log]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, fitvals, linewidth=2, color="steelblue")
    ax.axhline(chi2_planck, color="tomato", linestyle="--", linewidth=1.2,
               label=f"Planck ref chi2={chi2_planck:.0f}")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel(r"Best $\chi^2$", fontsize=12)
    ax.set_title("AEQGA convergence — Pantheon SNe Ia", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig("pantheon_convergence.png", dpi=150)
    print("\nConvergence plot saved: pantheon_convergence.png")

except Exception as e:
    print(f"\n(Convergence plot skipped: {e})")

# ---------------------------------------------------------------------------
# Step 6 — 2-D chi^2 contour plot
# ---------------------------------------------------------------------------

if not args.no_contour:
    print("\n=== Computing 2-D chi^2 grid for contour plot ===")
    print("    (this evaluates chi2 on a 30x30 grid — takes ~1 min)")

    data = problem._data
    n_H0 = 30
    n_Om = 30
    H0_arr = np.linspace(62.0, 74.0, n_H0)
    Om_arr = np.linspace(0.20, 0.42, n_Om)
    chi2_grid = np.empty((n_Om, n_H0))

    for i, Om in enumerate(Om_arr):
        for j, H0 in enumerate(H0_arr):
            chi2_grid[i, j] = chi2_pantheon(H0, Om, data, use_full_cov=True)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{n_Om} rows done")

    chi2_min_grid = chi2_grid.min()
    delta_chi2    = chi2_grid - chi2_min_grid   # Delta chi2

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        # 1-sigma, 2-sigma, 3-sigma contours for 2 parameters:
        # Delta chi2 = 2.30, 6.18, 11.83
        levels = [2.30, 6.18, 11.83]
        cf = ax.contourf(H0_arr, Om_arr, delta_chi2,
                         levels=[0] + levels + [30],
                         colors=["#1a6faf", "#3a9fd8", "#8ecde8", "#f0f0f0"])
        cs = ax.contour(H0_arr, Om_arr, delta_chi2,
                        levels=levels,
                        colors=["white"], linewidths=1.2)
        ax.clabel(cs, fmt={2.30: "1σ", 6.18: "2σ", 11.83: "3σ"}, fontsize=9)

        # Mark AEQGA best-fit
        ax.plot(H0_best, Om_best, "r*", markersize=14, label="AEQGA best-fit",
                zorder=5)
        # Mark Planck reference
        ax.plot(67.4, 0.315, "w^", markersize=10, label="Planck 2018",
                zorder=5)

        ax.set_xlabel(r"$H_0$ [km/s/Mpc]", fontsize=12)
        ax.set_ylabel(r"$\Omega_m$", fontsize=12)
        ax.set_title("Pantheon SNe Ia — flat ΛCDM", fontsize=13)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig("pantheon_contours.png", dpi=150)
        print("Contour plot saved: pantheon_contours.png")

    except Exception as e:
        print(f"(Contour plot skipped: {e})")

print("\nDone.")
