"""Fit local Pantheon+SH0ES data; export 1000-generation AEQGA trajectories.

Run: conda run --no-capture-output -n qml python run_pantheon_aeqga.py
Follows the selection, calibrator handling and full covariance of the data's
Pantheon+SH0ES_cosmosis_likelihood.py, in flat matter+Lambda cosmology.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import qiskit
import qiskit_aer
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits

from aeqga import AEQGA

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Pantheon+_Data" / "4_DISTANCES_AND_COVAR"
C_KM_S = 299792.458


def load_data():
    table = pd.read_csv(DATA / "Pantheon+SH0ES.dat", sep=r"\s+")
    mask = (table["zHD"] > .01) | (table["IS_CALIBRATOR"] == 1)
    selected = table.loc[mask].copy()
    covariance_path = DATA / "Pantheon+SH0ES_STAT+SYS.cov"
    with covariance_path.open() as stream:
        count = int(stream.readline())
        flat = np.loadtxt(stream)
    if count != len(table) or flat.size != count * count:
        raise ValueError("Covariance dimensions do not match the source table.")
    covariance = flat.reshape(count, count)[np.ix_(mask.to_numpy(), mask.to_numpy())]
    asymmetry = covariance-covariance.T
    max_asymmetry = float(np.max(np.abs(asymmetry)))
    relative_asymmetry = float(np.linalg.norm(asymmetry)/np.linalg.norm(covariance))
    if max_asymmetry > 1e-7 or relative_asymmetry > 1e-6:
        raise ValueError("Source covariance asymmetry is larger than rounding tolerance.")
    # The release has tiny decimal-rounding asymmetries in calibrator entries.
    # Preserve both triangles via their mean, and record the size of this change.
    covariance = (covariance+covariance.T)/2
    # Preserve source row order, including repeated measurements of the same SN.
    x = selected[["zHD", "zHEL", "IS_CALIBRATOR", "CEPH_DIST"]].to_numpy()
    y = selected["m_b_corr"].to_numpy()
    return table, selected, x, y, covariance, {
        "max_source_asymmetry": max_asymmetry,
        "relative_source_asymmetry": relative_asymmetry,
        "treatment": "(C + C.T) / 2 before Cholesky; all correlations retained",
    }


def make_model(order=32):
    nodes, weights = leggauss(order)

    def model(x_list, p_list):
        h0, omega_m, absolute_magnitude = p_list
        x = np.asarray(x_list, dtype=float)
        calibrators = x[:, 2] == 1
        mu = x[:, 3].copy()
        z = x[~calibrators, 0]
        zhel = x[~calibrators, 1]
        integration_z = z[:, None] * (nodes + 1) / 2
        expansion = np.sqrt(omega_m * (1 + integration_z)**3 + 1 - omega_m)
        integral = z / 2 * np.sum(weights / expansion, axis=1)
        distance_mpc = (1 + zhel) * C_KM_S / h0 * integral
        mu[~calibrators] = 5 * np.log10(distance_mpc) + 25
        return mu + absolute_magnitude

    return model


def verify_quadrature(model, x):
    # Check the integrator independently, including edges of the search range.
    noncal = x[x[:, 2] == 0]
    sample = noncal[np.linspace(0, len(noncal) - 1, 20, dtype=int)]
    sample = np.vstack((sample, noncal[np.argmax(noncal[:, 0])]))
    worst = 0.
    for omega_m in (.1, .3, .5):
        predicted = model(sample.tolist(), [70., omega_m, -19.3])
        expected = []
        for z, zhel, _, _ in sample:
            # Independent composite Simpson integration, not Gauss-Legendre.
            grid = np.linspace(0, z, 8193)
            values = 1 / np.sqrt(omega_m*(1+grid)**3 + 1-omega_m)
            integral = (z/8192)/3 * (values[0]+values[-1]
                        + 4*values[1:-1:2].sum() + 2*values[2:-1:2].sum())
            expected.append(5*np.log10((1+zhel)*C_KM_S/70*integral)+25-19.3)
        worst = max(worst, float(np.max(np.abs(predicted - expected))))
    if worst > 1e-8:
        raise AssertionError(f"Quadrature error too large: {worst}")
    return worst


def classical_profile_reference(function, x, y, chol, bounds):
    """Independent post-training benchmark; no SciPy optimize dependency.

    At fixed Omega_m the prediction is base + M + flow*(-5 log10(H0/70)).
    Solve these two linear offsets exactly with full-covariance least squares,
    then search Omega_m with a bounded one-dimensional golden-section search.
    """
    design = np.column_stack((np.ones(len(x)), x[:, 2] == 0))
    whitened_design = solve_triangular(chol, design, lower=True, check_finite=False)
    x_list = x.tolist()

    def profile(omega):
        rhs = solve_triangular(chol, y-function(x_list, [70., omega, 0.]),
                               lower=True, check_finite=False)
        offsets = np.linalg.lstsq(whitened_design, rhs, rcond=None)[0]
        residual = whitened_design @ offsets-rhs
        p = [70*10**(-offsets[1]/5), omega, float(offsets[0])]
        return float(residual @ residual), p

    left, right = bounds[1]
    ratio = (np.sqrt(5)-1)/2
    c, d = right-ratio*(right-left), left+ratio*(right-left)
    fc, fd = profile(c), profile(d)
    while right-left > 1e-10:
        if fc[0] < fd[0]:
            right, d, fd = d, c, fc
            c = right-ratio*(right-left)
            fc = profile(c)
        else:
            left, c, fc = c, d, fd
            d = left+ratio*(right-left)
            fd = profile(d)
    loss, p = min((fc, fd, profile(bounds[1][0]), profile(bounds[1][1])), key=lambda pair: pair[0])
    if not all(lo <= value <= hi for value, (lo, hi) in zip(p, bounds)):
        raise ValueError("The unconstrained linear reference offsets exceeded search bounds.")
    return {"p": p, "chi2": loss, "success": True,
            "method": "GLS profile in M and -5log10(H0/70); golden-section Omega_m search"}


def run(generations=1000, output=ROOT / "output" / "pantheon_aeqga", seed=7):
    started = time.perf_counter()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    table, selected, x, y, covariance, covariance_validation = load_data()
    function = make_model()
    quadrature_error = verify_quadrature(function, x)
    print(f"Selected {len(selected)}/{len(table)} light curves, "
          f"{int(selected.IS_CALIBRATOR.sum())} calibrator rows.", flush=True)
    bounds = [(60., 80.), (.1, .5), (-19.6, -19.)]
    initial_p = [70., .3, -19.3]
    model = AEQGA(function, initial_p, bounds=bounds, vectorized=True,
                  population_size=32, shots=1024, seed=seed)
    with threadpool_limits(limits=1):
        model.train(x, y, covariance=covariance, generations=generations, verbose=True)
        # Check the final chi-square independently with more quadrature nodes.
        chol = np.linalg.cholesky(covariance)
        accurate_model = make_model(64)
        x_list = x.tolist()

        def residual(p):
            return solve_triangular(chol, accurate_model(x_list, p) - y,
                                    lower=True, check_finite=False)

        independently_scored = float(np.sum(residual(model.p)**2))
        reference = classical_profile_reference(accurate_model, x, y, chol, bounds)
        reference["delta_chi2"] = model.loss_-reference["chi2"]
    if not np.isclose(model.loss_, independently_scored, atol=1e-7, rtol=0):
        raise AssertionError("Independent final chi-square check failed.")
    if model.n_generations_ != generations or len(model.history_) != generations + 1:
        raise AssertionError("The requested generation count was not completed.")
    if np.any(np.diff([h["loss"] for h in model.history_]) > 1e-10):
        raise AssertionError("Elitist best loss increased.")
    history = pd.DataFrame([
        {"generation": h["generation"], "H0_km_s_Mpc": h["p"][0],
         "Omega_m": h["p"][1], "M_mag": h["p"][2], "chi2": h["loss"]}
        for h in model.history_ if h["generation"] > 0
    ])
    history.to_csv(output / "history.csv", index=False)
    model.export_population_history(
        output / "population_history.csv", parameter_names=["H0_km_s_Mpc", "Omega_m", "M_mag"])
    labels = [r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]", r"$\Omega_m$", r"$M$ [mag]"]
    plot_title = f"Pantheon+SH0ES | AEQGA | {generations:,} generations"
    for extension in ("png", "svg"):
        model.export_history_plot(output / f"h0_history.{extension}", parameter_indices=[0],
                                  parameter_names=labels, include_loss=False, title=plot_title)
    model.export_history_plot(output / "all_parameters_and_loss.png", parameter_names=labels,
                              title=plot_title, loss_label=r"Best $\chi^2$")
    summary = {
        "dataset": "Pantheon+SH0ES (local release)",
        "model": "flat LambdaCDM, matter + cosmological constant; radiation neglected",
        "selection": "zHD > 0.01 OR IS_CALIBRATOR == 1",
        "source_rows": len(table), "selected_rows": len(selected),
        "unique_supernovae": int(selected.CID.nunique()),
        "calibrator_rows": int(selected.IS_CALIBRATOR.sum()),
        "target": "m_b_corr; calibrator prediction CEPH_DIST + M",
        "covariance": "Pantheon+SH0ES_STAT+SYS.cov (full selected submatrix)",
        "covariance_validation": covariance_validation,
        "parameter_order": ["H0_km_s_Mpc", "Omega_m", "M_mag"],
        "bounds": bounds, "initial_p": initial_p, "seed": seed,
        "population_size": model.population_size, "shots": model.shots,
        "elite_count": model.elite_count, "local_count": model.local_count,
        "random_count": model.random_count,
        "crossover_probability": model.crossover_probability,
        "mutation_probability": model.mutation_probability,
        "local_padding": model.local_padding, "min_local_width": model.min_local_width,
        "generations": model.n_generations_, "plotted_generations": [1, generations],
        "initial_best": model.history_[0], "best_p": model.p, "chi2": model.loss_,
        "independent_chi2_64_point_quadrature": independently_scored,
        "max_quadrature_check_error_mag": quadrature_error,
        "classical_reference_only": reference,
        "fitness_evaluations": model.n_evaluations_, "aer_circuits": model.quantum_circuits_,
        "elapsed_seconds": time.perf_counter()-started,
        "versions": {"numpy": np.__version__, "qiskit": qiskit.__version__,
                     "qiskit_aer": qiskit_aer.__version__},
        "source_sha256": {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest()
                          for name in ("Pantheon+SH0ES.dat", "Pantheon+SH0ES_STAT+SYS.cov")},
        "notes": ["Each plotted point is the minimum-chi-square individual's H0, not minimum H0.",
                  "One 1000-generation run, not 1000 independent runs.",
                  "Classical reference computed after training, never used to seed or correct AEQGA.",
                  "Optimization trajectory is not a posterior distribution or an H0 error bar."]
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return model, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "pantheon_aeqga")
    args = parser.parse_args()
    run(args.generations, args.output, args.seed)
