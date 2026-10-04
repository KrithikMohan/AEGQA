"""
run_pantheon_aeqga.py
=====================
END-TO-END runner: AEQGA on calibrated Pantheon+ SNe Ia data (arXiv:2602.15459).

Steps performed
---------------
1. Clone the data (if not already present)
2. Load calibrated Pantheon+: moduli, redshifts and full total covariance
3. Run the AEQGA to minimise chi^2(H0, Omega_m) in flat ΛCDM
4. Print the best-fit cosmological parameters
5. Plot convergence curve  →  pantheon_convergence.png
6. Plot 2-D chi^2 contours around the best-fit  →  pantheon_contours.png

Usage
-----
    python -m scripts.run_pantheon_aeqga --data /path/to/sn_data/PantheonPlus
    python -m scripts.run_pantheon_aeqga --fast

Note: pop_size must be a power of two (paper §3.1) for amplitude encoding.

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
from aeqga.paths import PROJECT_ROOT, output_path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=None,
                    help="Path to the selected dataset sub-folder")
parser.add_argument("--selection", choices=["all", "hubble_flow"], default="all")
parser.add_argument("--redshift", choices=["cmb", "hd_hel"], default="hd_hel")
parser.add_argument("--distance-grid", type=int, choices=[0,100,300], default=300,
                    help="0: direct integration; 100/300: paper nearest-neighbor grid")
parser.add_argument("--fast", action="store_true",
                    help="Short run: 8 individuals, 20 generations")
parser.add_argument("--pop",  type=int, default=32,
                    help="Population size (must be power of two, default 32)")
parser.add_argument("--gen",  type=int, default=50,
                    help="Number of generations (default 50)")
parser.add_argument("--shots", type=int, default=4096,
                    help="Qiskit shots per generation (default 4096)")
parser.add_argument("--no-contour", action="store_true",
                    help="Skip the 2-D contour plot (faster)")
args = parser.parse_args()

if args.fast:
    args.pop  = 8
    args.gen  = 20
    args.shots = 4096
    args.no_contour = True

# Validate pop_size is power of two
if args.pop & (args.pop - 1) != 0:
    sys.exit(f"ERROR: pop_size {args.pop} must be a power of two")

# ---------------------------------------------------------------------------
# Step 1 — Data
# ---------------------------------------------------------------------------

DATA_REPO = "https://github.com/CobayaSampler/sn_data"

def get_data_dir():
    if args.data:
        d = args.data
    else:
        d = str(PROJECT_ROOT / "sn_data" / "PantheonPlus")

    if not os.path.isdir(d):
        parent = os.path.dirname(d)
        os.makedirs(parent, exist_ok=True)
        print(f"Cloning {DATA_REPO} ...")
        ret = subprocess.run(
            ["git", "clone", "--depth", 1, DATA_REPO,
             os.path.join(parent)],
            check=False
        )
        if ret.returncode != 0:
            sys.exit(
                f"\nERROR: git clone failed.\n"
                f"Clone manually:\n  git clone {DATA_REPO}\n"
                f"Then re-run with:  python -m scripts.run_pantheon_aeqga --data sn_data/PantheonPlus"
            )
    if not os.path.isdir(d):
        sys.exit(f"ERROR: Data directory not found: {d}")
    return d

data_dir = get_data_dir()
print(f"Data directory: {data_dir}")

# ---------------------------------------------------------------------------
# Step 2 — Imports
# ---------------------------------------------------------------------------

try:
    from qiskit_aer import AerSimulator
except ImportError:
    sys.exit(
        "ERROR: qiskit-aer not installed.\n"
        "Install with:  pip install qiskit qiskit-aer tqdm numpy matplotlib"
    )

from aeqga.likelihoods.pantheon_problem import PantheonPlusProblem
from aeqga.steps.evolution.aeqga_algorithm import AEQGAParameters, run_aeqga_dual

# ---------------------------------------------------------------------------
# Step 3 — Build problem
# ---------------------------------------------------------------------------

print("\n=== Loading calibrated Pantheon+ data ===")
problem = PantheonPlusProblem(data_dir, selection=args.selection,
                             redshift=args.redshift, grid_size=args.distance_grid)
classical_reference = problem.classical_minimum()
print("Classical minimum:", classical_reference)

# Quick sanity check: chi2 at paper's SNe Ia best-fit
x_paper = np.array([72.82, 0.363])
chi2_paper = problem.compute_fitness(x_paper)
print(f"chi2 at paper SNe Ia best-fit (H0=72.82, Om=0.363) = {chi2_paper:.2f}")
print(f"  (reduced chi2 ~ {chi2_paper / (problem._data['n_sn'] - 2):.3f}  "
      f"[expected ~1.0 for a good fit])")

# ---------------------------------------------------------------------------
# Step 4 — Run AEQGA
# ---------------------------------------------------------------------------

print(f"\n=== Running AEQGA  (pop={args.pop}, gen={args.gen}, shots={args.shots}) ===")

params = AEQGAParameters(
    pop_size     = args.pop,
    max_gen      = args.gen,
    p_cross      = 0.5,    # paper's optimum (§4.1)
    p_mut        = 0.5,    # paper's optimum (§4.1)
    num_shots    = args.shots,
    verbose      = False,
    progress_bar = True,
)

g_best, population_evol, bests_log = run_aeqga_dual(problem, params)

H0_best = g_best.x[0]
Om_best = g_best.x[1]
chi2_best = g_best.fitness

print("\n=== Results ===")
print(f"  Best-fit H0      = {H0_best:.2f}  km/s/Mpc")
print(f"  Best-fit Omega_m = {Om_best:.4f}")
print(f"  chi2_min         = {chi2_best:.2f}")
print(f"  Reduced chi2     = {chi2_best / (problem._data['n_sn'] - 2):.4f}")
print(f"  Found at gen     = {g_best.gen}")
print(f"\n  Paper SNe Ia (Sarracino+26): Ω_M = 0.363±0.016, H0 = 72.81±0.22")

# ---------------------------------------------------------------------------
# Step 5 — Convergence plot
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gens   = list(range(len(bests_log)))
    fitvals = [b[1] for b in bests_log]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, fitvals, linewidth=2, color="steelblue")
    ax.axhline(chi2_paper, color="tomato", linestyle="--", linewidth=1.2,
               label=f"Paper best-fit chi2={chi2_paper:.0f}")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel(r"Best $\chi^2$", fontsize=12)
    ax.set_title("AEQGA convergence — calibrated Pantheon+", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path('png', 'pantheon_convergence.png'), dpi=150)
    print("\nConvergence plot saved: pantheon_convergence.png")

except Exception as e:
    print(f"\n(Convergence plot skipped: {e})")

# ---------------------------------------------------------------------------
# Step 6 — 2-D chi^2 contour plot
# ---------------------------------------------------------------------------

if not args.no_contour:
    print("\n=== Computing 2-D chi^2 grid for contour plot ===")
    print("    (using the identical likelihood as the optimizer)")
    print("    (likelihood contours are not optimizer-outcome scatter)")
    H0_arr = np.linspace(max(60., classical_reference['H0']-1.4),
                         min(80., classical_reference['H0']+1.4), 160)
    om_low = max(0., classical_reference['Omega_m']-.085)
    om_high = min(.5, classical_reference['Omega_m']+.085)
    Om_arr = (problem.omega_grid[(problem.omega_grid >= om_low) &
                                (problem.omega_grid <= om_high)]
              if problem.grid_size else np.linspace(om_low, om_high, 100))
    chi2_grid = problem.objective_grid(H0_arr, Om_arr)
    delta_chi2 = chi2_grid - classical_reference['chi2']

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        levels = [2.30, 6.18, 11.83]
        cf = ax.contourf(H0_arr, Om_arr, delta_chi2,
                         levels=[0] + levels + [30],
                         colors=["#2d5a27", "#4a8c3f", "#7db874", "#c8e6c0"])
        cs = ax.contour(H0_arr, Om_arr, delta_chi2,
                        levels=levels,
                        colors=["white"], linewidths=1.2)
        ax.clabel(cs, fmt={2.30: "1σ", 6.18: "2σ", 11.83: "3σ"}, fontsize=9)

        if H0_arr[0] <= H0_best <= H0_arr[-1] and Om_arr[0] <= Om_best <= Om_arr[-1]:
            ax.plot(H0_best, Om_best, "r*", markersize=14, label="AEQGA best-fit", zorder=5)
        else:
            print("AEQGA smoke point is outside the likelihood zoom; no convergence claim.")
        ax.plot(classical_reference['H0'], classical_reference['Omega_m'],
                "k*", markersize=12, label="Classical reference", zorder=5)

        ax.set_xlabel(r"$H_0$ [km/s/Mpc]", fontsize=12)
        ax.set_ylabel(r"$\Omega_m$", fontsize=12)
        ax.set_title("Calibrated likelihood — not optimizer scatter", fontsize=13)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(output_path('png', 'pantheon_contours.png'), dpi=150)
        print("Contour plot saved: pantheon_contours.png")

    except Exception as e:
        print(f"(Contour plot skipped: {e})")

print("\nDone.")
