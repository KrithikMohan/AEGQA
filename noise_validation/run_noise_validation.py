"""Validate AEQGA gate/readout noise across actual 2-5 qubit circuits."""
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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from qiskit.quantum_info import Statevector
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits

from aeqga import AEQGA
from aeqga_noise import (BASIS_GATES, PROFILES, NoiseValidationAEQGA,
                         apply_readout, compile_basis, make_backend)
from run_pantheon_aeqga import (DATA, ROOT, classical_profile_reference,
                                load_data, make_model, verify_quadrature)

BOUNDS = [(60., 80.), (.1, .5), (-19.6, -19.)]
INITIAL = [70., .3, -19.3]
NAMES = ["H0_km_s_Mpc", "Omega_m", "M_mag"]
LABELS = {"ideal": "Ideal", "low": "Low", "medium": "Medium", "high": "High",
          "medium_gate_only": "Medium: gate only", "medium_readout_only": "Medium: readout only"}
COLORS = {"ideal": "#586779", "low": "#1d8a72", "medium": "#e09329", "high": "#c34c55",
          "medium_gate_only": "#5665b1", "medium_readout_only": "#9863a5"}


def probability_vector(values):
    values = np.asarray(values, dtype=float).copy()
    if np.min(values) < -1e-12 or not np.isclose(values.sum(), 1., atol=1e-10):
        raise AssertionError("Invalid probability vector.")
    values[np.abs(values) < 1e-15] = 0.
    values = np.maximum(values, 0.)
    return values / values.sum()


def total_variation(a, b):
    return float(.5 * np.sum(np.abs(np.asarray(a)-np.asarray(b))))


def frozen_catalog(input_repeats):
    """Identical candidates and selected gates are reused across noise profiles."""
    catalog, circuits, ideal_probabilities = [], [], []
    center = np.array([73.53, .332, -19.244])
    widths = np.array([hi-lo for lo, hi in BOUNDS])
    for population in (8, 16, 32, 64):
        model = AEQGA(lambda x, p: 0., INITIAL, bounds=BOUNDS, population_size=population)
        for input_seed in range(input_repeats):
            rng = np.random.default_rng(700+input_seed)
            elites = np.clip(center + rng.uniform(-.02, .02, (population//4, 3))*widths,
                             model._lower, model._upper)
            random_values = rng.uniform(model._lower, model._upper, (population//2, 3))
            for branch, values in (("local", elites), ("global", random_values)):
                for dimension in range(3):
                    lo, hi = (model._local_box(elites[:, dimension], dimension)
                              if branch == "local" else BOUNDS[dimension])
                    for gate_pattern, cross, mutate in (("none", 0., 0.), ("cross", 1., 0.),
                                                        ("mutation", 0., 1.), ("both", 1., 1.)):
                        model.crossover_probability, model.mutation_probability = cross, mutate
                        circuit = model._circuit(values[:, dimension], np.random.default_rng(900+input_seed))
                        ideal = probability_vector(Statevector.from_instruction(
                            circuit.remove_final_measurements(inplace=False)).probabilities())
                        compiled = compile_basis(circuit)
                        catalog.append({"case_id": len(catalog), "population_size": population,
                                        "population_bits": population.bit_length()-1, "branch": branch,
                                        "parameter": NAMES[dimension], "dimension": dimension,
                                        "input_seed": input_seed, "gate_pattern": gate_pattern,
                                        "candidate_count": len(values), "qubits": circuit.num_qubits,
                                        "interval_lower": lo, "interval_upper": hi,
                                        "u_gates": compiled.count_ops().get("u", 0),
                                        "cx_gates": compiled.count_ops().get("cx", 0),
                                        "depth": compiled.depth(), "decoder_seed": 6000+len(catalog)})
                        circuits.append(compiled)
                        ideal_probabilities.append(ideal)
    return catalog, circuits, ideal_probabilities


def fixed_circuit_validation(output, shots, input_repeats=3, sample_repeats=5):
    started = time.perf_counter()
    catalog, circuits, ideal_probabilities = frozen_catalog(input_repeats)
    pd.DataFrame(catalog).to_csv(output / "circuit_catalog.csv", index=False)
    exact_circuits = []
    for circuit in circuits:
        exact = circuit.remove_final_measurements(inplace=False)
        exact.save_probabilities()
        exact_circuits.append(exact)
    exact_rows, sample_rows, archive = [], [], {}
    max_compile_error = 0.
    max_sampling_bound_ratio = 0.
    max_states = max(len(p) for p in ideal_probabilities)
    for profile_index, profile in enumerate(PROFILES):
        print(f"FIXED circuits: {profile}", flush=True)
        backend = make_backend(profile, readout=False)
        result = backend.run(exact_circuits, shots=1).result()
        if not result.success:
            raise AssertionError(f"Probability simulation failed: {result.status}")
        probabilities = []
        for i, case in enumerate(catalog):
            gates_p = probability_vector(result.data(i)["probabilities"])
            p = probability_vector(apply_readout(gates_p, PROFILES[profile]["readout_flip"]))
            p0 = ideal_probabilities[i]
            if profile == "ideal":
                max_compile_error = max(max_compile_error, float(np.max(np.abs(p-p0))))
                np.testing.assert_allclose(p, p0, atol=1e-10, rtol=0)
                # Use one identical compiled ideal vector for every decode baseline.
                # Min/max decoding and exact-lower replacement are discontinuous
                # at tied minima; 1e-12 compiler differences must not count as noise.
                ideal_probabilities[i] = p.copy()
                p0 = ideal_probabilities[i]
            probabilities.append(p)
            lo, hi = case["interval_lower"], case["interval_upper"]
            kwargs = (lo, hi, case["branch"] == "local")
            d0 = AEQGA._decode(p0, *kwargs, np.random.default_rng(case["decoder_seed"]))[:case["candidate_count"]]
            dn = AEQGA._decode(p, *kwargs, np.random.default_rng(case["decoder_seed"]))[:case["candidate_count"]]
            rms = float(np.sqrt(np.mean((dn-d0)**2)))
            row = {**case, "profile": profile, "exact_tvd": total_variation(p, p0),
                   "exact_decode_rmse": rms, "exact_decode_rmse_fraction": rms/(hi-lo),
                   "probability_spread": float(np.ptp(p))}
            exact_rows.append(row)
        packed = np.full((len(catalog), max_states), np.nan)
        for i, p in enumerate(probabilities):
            packed[i, :len(p)] = p
        archive[f"{profile}_probabilities"] = packed
        accumulated_counts = [np.zeros(len(p)) for p in probabilities]
        measured_backend = make_backend(profile)
        for repeat in range(sample_repeats):
            seed = 100000 + 1000*profile_index + repeat
            measured = measured_backend.run(circuits, shots=shots, seed_simulator=seed).result()
            if not measured.success:
                raise AssertionError(f"Sampling failed: {measured.status}")
            packed_counts = np.zeros((len(catalog), max_states), dtype=int)
            for i, case in enumerate(catalog):
                counts = AEQGA._counts(measured.get_counts(i), len(probabilities[i]))
                assert counts.sum() == shots
                accumulated_counts[i] += counts
                packed_counts[i, :len(counts)] = counts
                lo, hi = case["interval_lower"], case["interval_upper"]
                kwargs = (lo, hi, case["branch"] == "local")
                d0 = AEQGA._decode(ideal_probabilities[i], *kwargs,
                                  np.random.default_rng(case["decoder_seed"]))[:case["candidate_count"]]
                decoded = AEQGA._decode(counts, *kwargs,
                                       np.random.default_rng(case["decoder_seed"]))[:case["candidate_count"]]
                assert np.all((decoded >= lo) & (decoded <= hi))
                rms = float(np.sqrt(np.mean((decoded-d0)**2)))
                sample_rows.append({**case, "profile": profile, "repeat": repeat, "sampling_seed": seed,
                                    "shots": shots, "sample_tvd_vs_ideal": total_variation(counts/shots, ideal_probabilities[i]),
                                    "sample_tvd_vs_noisy_exact": total_variation(counts/shots, probabilities[i]),
                                    "sample_decode_rmse": rms, "sample_decode_rmse_fraction": rms/(hi-lo)})
            archive[f"{profile}_counts_repeat_{repeat}"] = packed_counts
        # This verifies classical readout convolution against actual Aer measured output.
        trials = shots*sample_repeats
        for p, counts in zip(probabilities, accumulated_counts):
            bound = 6*np.sqrt(p*(1-p)/trials) + 6/trials
            ratio = float(np.max(np.abs(counts/trials-p)/bound))
            max_sampling_bound_ratio = max(max_sampling_bound_ratio, ratio)
            if ratio > 1:
                raise AssertionError("Measured counts disagree with gate+readout probabilities.")
        pd.DataFrame(exact_rows).to_csv(output / "fixed_circuit_exact.csv", index=False)
        pd.DataFrame(sample_rows).to_csv(output / "fixed_circuit_samples.csv", index=False)
    np.savez_compressed(output / "probabilities_and_counts.npz", **archive)
    exact_table, sample_table = pd.DataFrame(exact_rows), pd.DataFrame(sample_rows)
    exact_summary = exact_table.groupby(["profile", "qubits"])[
        ["exact_tvd", "exact_decode_rmse_fraction"]].agg(["mean", "std", "max"])
    exact_summary.columns = ["_".join(c) for c in exact_summary.columns]
    exact_summary.to_csv(output / "qubit_summary.csv")
    checks = {"fixed_circuits": len(catalog), "noise_profiles": len(PROFILES),
              "input_variants": input_repeats, "gate_patterns": ["none", "cross", "mutation", "both"],
              "shots": shots, "sampling_repeats": sample_repeats,
              "max_compile_probability_error": max_compile_error,
              "max_sampling_bound_ratio": max_sampling_bound_ratio,
              "sampling_check": "Every aggregate frequency within 6*binomial_SE + 6/(shots*repeats).",
              "elapsed_seconds": time.perf_counter()-started}
    (output / "fixed_circuit_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return exact_table, sample_table, pd.DataFrame(catalog), checks


def training_validation(output, generations, shots, seeds):
    table, selected, x, y, covariance, covariance_validation = load_data()
    function = make_model(32)
    quadrature_error = verify_quadrature(function, x)
    accurate = make_model(64)
    rows = []
    with threadpool_limits(limits=1):
        chol = np.linalg.cholesky(covariance)
        for population in (8, 16, 32, 64):
            for seed in seeds:
                initial_record = None
                for profile in ("ideal", "medium", "high"):
                    started = time.perf_counter()
                    print(f"TRAIN population={population}, seed={seed}, profile={profile}", flush=True)
                    model = NoiseValidationAEQGA(function, INITIAL, bounds=BOUNDS,
                                                 population_size=population, vectorized=True,
                                                 shots=shots, seed=seed, noise_profile=profile)
                    model.train(x, y, covariance=covariance, generations=generations)
                    if initial_record is None:
                        initial_record = model.history_[0]
                    else:
                        assert model.history_[0] == initial_record
                    assert model.n_generations_ == generations
                    assert np.all(np.diff([h["loss"] for h in model.history_]) <= 1e-10)
                    residual = solve_triangular(chol, accurate(x.tolist(), model.p)-y,
                                                lower=True, check_finite=False)
                    checked_chi2 = float(residual @ residual)
                    np.testing.assert_allclose(model.loss_, checked_chi2, rtol=0, atol=1e-7)
                    group_qubits = [max(2, (population//divisor-1).bit_length()) for divisor in (4, 2)]
                    row = {"population_size": population, "population_bits": population.bit_length()-1,
                           "local_qubits": group_qubits[0], "global_qubits": group_qubits[1],
                           "profile": profile, "seed": seed, "generations": generations, "shots": shots,
                           "H0_km_s_Mpc": model.p[0], "Omega_m": model.p[1], "M_mag": model.p[2],
                           "chi2": model.loss_, "independent_chi2": checked_chi2,
                           "fitness_evaluations": model.n_evaluations_, "aer_circuits": model.quantum_circuits_,
                           "elapsed_seconds": time.perf_counter()-started}
                    rows.append(row)
                    run_dir = output / "training_runs" / f"N{population}_{profile}_seed{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame([{"generation": h["generation"], **dict(zip(NAMES, h["p"])),
                                   "chi2": h["loss"]} for h in model.history_]).to_csv(run_dir / "best_history.csv", index=False)
                    model.export_population_history(run_dir / "all_candidates.csv", parameter_names=NAMES)
                    pd.DataFrame(rows).to_csv(output / "training_results.csv", index=False)
                    print("DONE " + json.dumps(row), flush=True)
        reference = classical_profile_reference(accurate, x, y, chol, BOUNDS)
    training = pd.DataFrame(rows)
    training["delta_chi2"] = training.chi2 - reference["chi2"]
    training.to_csv(output / "training_results.csv", index=False)
    aggregate = training.groupby(["population_size", "profile"])[NAMES+["chi2", "delta_chi2"]].agg(["mean", "std", "min", "max"])
    aggregate.columns = ["_".join(c) for c in aggregate.columns]
    aggregate.to_csv(output / "training_summary.csv")
    metadata = {"source_rows": len(table), "selected_rows": len(selected),
                "calibrator_rows": int(selected.IS_CALIBRATOR.sum()),
                "covariance": "full STAT+SYS", "covariance_validation": covariance_validation,
                "quadrature_error_mag": quadrature_error, "generations": generations,
                "seeds": seeds, "classical_reference_only": reference}
    return training, metadata


def style_axis(axis):
    axis.grid(alpha=.2)
    axis.spines[["top", "right"]].set_visible(False)


def save_figure(figure, output, name):
    for suffix in ("png", "svg"):
        figure.savefig(output / f"{name}.{suffix}", dpi=180, facecolor="white")
    figure.clear()


def plot_fixed(exact, sampled, catalog, output, shots):
    figure = Figure(figsize=(12, 9.4)); FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    for profile in ("ideal", "low", "medium", "high"):
        exact_group = exact[exact.profile == profile].groupby("qubits").exact_tvd.mean()
        sample_group = sampled[sampled.profile == profile].groupby("qubits").sample_tvd_vs_ideal.mean()
        axes[0, 0].plot(exact_group.index, exact_group.values, "o-", color=COLORS[profile], label=LABELS[profile])
        axes[0, 1].plot(sample_group.index, sample_group.values, "o-", color=COLORS[profile])
        for column, branch in enumerate(("local", "global")):
            subset = sampled[(sampled.profile == profile) & (sampled.branch == branch)]
            grouped = subset.groupby("qubits").sample_decode_rmse_fraction.mean()*100
            axes[1, column].plot(grouped.index, grouped.values, "o-", color=COLORS[profile])
    titles = ["Systematic probability shift (no shot noise)", f"Measured probability error ({shots:,} shots)",
              "Local decode: parameter displacement", "Global decode: parameter displacement"]
    for i, axis in enumerate(axes.ravel()):
        axis.set_title(titles[i], fontsize=12)
        axis.set_xlabel("Actual circuit qubits")
        axis.set_xticks([2, 3, 4, 5] if i < 2 or i == 3 else [2, 3, 4])
        axis.set_ylabel("Mean total variation distance" if i < 2 else "Mean RMSE / decode interval [%]")
        axis.set_ylim(bottom=0)
        style_axis(axis)
    figure.suptitle("AEQGA noise validation across circuit widths", fontsize=18, fontweight="bold", y=.98)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5,.945), ncol=4, frameon=False)
    figure.subplots_adjust(left=.10, right=.97, top=.86, bottom=.15, hspace=.39, wspace=.34)
    figure.text(.08, .035,
                "Synthetic depolarizing + independent readout noise; all-to-all u/cx circuits, including state preparation.\n"
                "Means across fixed inputs, parameters and gate patterns; ideal measured curves include shot noise.\n"
                "Population 8 local branch is padded. Different circuit widths also have different gate counts and encodings.",
                fontsize=9, color="#465565", linespacing=1.6)
    save_figure(figure, output, "noise_vs_actual_qubits")

    figure = Figure(figsize=(11.5, 4.8)); FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for profile in ("medium", "medium_gate_only", "medium_readout_only"):
        subset = exact[exact.profile == profile]
        for i, metric in enumerate(("exact_tvd", "exact_decode_rmse_fraction")):
            grouped = subset.groupby("qubits")[metric].mean()
            axes[i].plot(grouped.index, grouped.values*(100 if i else 1), "o-", color=COLORS[profile], label=LABELS[profile])
    for axis in axes:
        axis.set_xlabel("Actual circuit qubits"); axis.set_xticks([2,3,4,5]); axis.set_ylim(bottom=0); style_axis(axis)
    axes[0].set_ylabel("Mean total variation distance")
    axes[1].set_ylabel("Mean exact decode RMSE / interval [%]")
    figure.suptitle("Separating gate and readout noise | medium profile", fontsize=15, y=.985)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5,.93), ncol=3, frameon=False, fontsize=9)
    figure.subplots_adjust(left=.10, right=.97, top=.77, bottom=.20, wspace=.30)
    figure.text(.09,.035,"Same frozen circuits; exact probability calculations exclude shot noise. Decode includes production lower-bound replacement.",fontsize=8.5)
    save_figure(figure, output, "gate_vs_readout_noise")
    figure = Figure(figsize=(10.8, 4.8)); FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for branch, color in (("local", "#217ca5"), ("global", "#c17b35")):
        subset = catalog[catalog.branch == branch]
        for axis, column in zip(axes, ("cx_gates", "depth")):
            group = subset.groupby("qubits")[column].agg(["mean", "min", "max"])
            axis.errorbar(group.index, group["mean"], yerr=np.vstack((group["mean"]-group["min"],group["max"]-group["mean"])),
                          fmt="o-", color=color, capsize=4, label=branch)
            axis.set_xlabel("Actual circuit qubits"); axis.set_xticks([2,3,4,5]); style_axis(axis)
    axes[0].set_ylabel("CX count"); axes[1].set_ylabel("Compiled circuit depth")
    axes[0].legend(frameon=False)
    figure.suptitle("Circuit resources after u/cx decomposition",fontsize=16,y=.98)
    figure.subplots_adjust(left=.09,right=.97,top=.85,bottom=.20,wspace=.26)
    figure.text(.09,.04,"Markers: mean across circuit cases; bars: observed min-max. No routing or device coupling constraints are imposed.",fontsize=8.8)
    save_figure(figure,output,"compiled_circuit_resources")


def plot_training(training, metadata, output):
    figure = Figure(figsize=(12,9)); FigureCanvasAgg(figure)
    axes = figure.subplots(2,2)
    labels = [r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]",r"$\Omega_m$",r"$M$ [mag]",r"Final $\chi^2$ minus classical reference"]
    for i, (axis, column, label) in enumerate(zip(axes.ravel(), NAMES+["delta_chi2"], labels)):
        for offset, profile in enumerate(("ideal","medium","high")):
            subset = training[training.profile == profile]
            group = subset.groupby("population_bits")[column].agg(["mean","std"])
            xs = group.index.to_numpy()+(offset-1)*.04
            axis.errorbar(xs,group["mean"],yerr=group["std"].fillna(0),fmt="o-",capsize=4,
                          color=COLORS[profile],label=LABELS[profile])
            axis.scatter(subset.population_bits+(offset-1)*.04,subset[column],s=14,alpha=.5,color=COLORS[profile])
        if i < 3:
            axis.axhline(metadata["classical_reference_only"]["p"][i],ls="--",color="#8994a0",lw=1)
        else:
            axis.axhline(0,ls="--",color="#8994a0",lw=1)
        axis.set_xticks([3,4,5,6],["3\nN=8","4\nN=16","5\nN=32","6\nN=64"])
        axis.set_xlabel("Population-equivalent bits: log2(N)");axis.set_ylabel(label);style_axis(axis)
        axis.ticklabel_format(axis="y",style="plain",useOffset=False)
    figure.suptitle(f"Noise effect on {metadata['generations']}-generation fits | {len(metadata['seeds'])} seeds",fontsize=18,fontweight="bold",y=.98)
    figure.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.945),ncol=3,frameon=False)
    figure.subplots_adjust(left=.105,right=.97,top=.86,bottom=.16,hspace=.40,wspace=.34)
    figure.text(.08,.035,"Markers and bars: mean +/- sample SD across seeds; small points: individual runs. Dashed lines: classical reference.\n"
                "Actual local/global qubits: N=8: 2/2; N=16: 2/3; N=32: 3/4; N=64: 4/5. Fixed generations, unequal evaluation budgets.\n"
                "Short-run optimization results are not posterior uncertainties or evidence of hardware readiness.",fontsize=9,color="#465565",linespacing=1.6)
    save_figure(figure,output,"noise_training_final_parameters")


def write_report(output, exact, training, metadata, checks, shots):
    lines = ["AEQGA QUBIT NOISE VALIDATION", "", "METHOD", "Synthetic models; no quantum hardware was used.",
             "Basis gates: u and cx; all-to-all connectivity; optimization_level=0.",
             "State preparation is decomposed and receives the same gate noise as genetic operations.",
             "Noise parameters below are depolarizing channel strengths, not average gate infidelities.",
             "Depolarizing channel: E(rho) = (1-p)*rho + p*I/(2**n).",
             "Readout: independent symmetric bit flip per measured qubit.",
             "Reset is ideal; no thermal relaxation, idle noise, crosstalk, leakage, or device routing is modeled.", ""]
    for name, config in PROFILES.items():
        lines.append(f"{name}: 1Q p={config['one_qubit_depolarizing']}; 2Q p={config['two_qubit_depolarizing']}; readout flip={config['readout_flip']}")
    lines += ["", "FIXED CIRCUITS", json.dumps(checks,indent=2),
              "Candidate values use the same parameter bounds as the Pantheon experiment.",
              "Local inputs lie within 2% of the global parameter widths around [73.53,0.332,-19.244].",
              "Global inputs are uniformly drawn inside the full bounds.",
              "For every case, input amplitudes and selected gate positions are identical across profiles.",
              "TVD = 0.5*sum(abs(p_noisy-p_ideal)); range [0,1].",
              "Decode RMSE is over candidate values, divided by that circuit's decoding interval width.",
              "Decode uses the production random replacement of lower-bound values with the same decoder seed.",
              "That replacement, discrete counts, and min/max normalization can make decode error nonmonotonic.",
              "Exact noisy probabilities: density matrix gate simulation followed by analytic readout convolution.",
              "The compiled ideal probability vector is the common decode reference; ideal exact decode error is zero.",
              f"Measured results: actual Aer gate+readout simulations, {shots} shots, {checks['sampling_repeats']} repetitions.",
              "The ideal measured baseline includes finite-shot noise.", "", "MEAN EXACT TVD BY ACTUAL QUBIT COUNT"]
    means = exact.groupby(["qubits","profile"]).exact_tvd.mean().unstack()
    lines.append(means.to_string(float_format=lambda value:f"{value:.6f}"))
    lines += ["", "CIRCUIT WIDTHS", "N=8: local=2 padded, global=2; N=16: local=2, global=3;",
              "N=32: local=3, global=4; N=64: local=4, global=5.",
              "Width groups also differ in state preparation, gate count, parameter inputs and branch composition.",
              "This experiment does not isolate qubit count as the sole causal variable.", "", "SHORT TRAINING EXPERIMENT",
              f"{metadata['generations']} generations; seeds={metadata['seeds']}; {shots} shots; profiles ideal/medium/high.",
              "Population sizes 8/16/32/64; elite/local/global ratio 1:1:2.",
              "For each population/seed, all profiles have identical generation-zero candidates and chi-square values.",
              f"Data: {metadata['selected_rows']} selected Pantheon+SH0ES rows; full STAT+SYS covariance.",
              "Every final chi-square was independently checked with 64-node quadrature.",
              "All per-generation candidates and chi-square values are exported under training_runs/.",
              f"Classical reference chi2={metadata['classical_reference_only']['chi2']:.12f}; computed after all searches.",
              "The classical reference does not seed or correct any AEQGA run.",
              "Results use short runs; they are not the earlier 1000-generation results.",
              "Standard deviations describe optimization variability, not cosmological confidence intervals.", ""]
    means = training.groupby(["population_size","profile"])[NAMES+["delta_chi2"]].mean()
    lines.append(means.to_string(float_format=lambda value:f"{value:.6f}"))
    lines += ["", "SCOPE", "No pass/fail noise tolerance or acceptable final-chi-square threshold was prespecified.",
              "The tests verify implementation consistency and quantify sensitivity under the stated models.",
              "They do not validate a specific device or demonstrate a speed advantage.", "", "METHOD REFERENCES",
              "https://qiskit.github.io/qiskit-aer/apidocs/aer_noise.html",
              "https://qiskit.github.io/qiskit-aer/tutorials/3_building_noise_models.html", "", "END OF REPORT"]
    (output / "noise_validation_report.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(output, shots=1024, generations=100, seeds=(7,17,27), input_repeats=3, sample_repeats=5):
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    seeds = list(seeds)
    started = time.perf_counter()
    exact, sampled, catalog, checks = fixed_circuit_validation(output,shots,input_repeats,sample_repeats)
    plot_fixed(exact,sampled,catalog,output,shots)
    training, metadata = training_validation(output,generations,shots,seeds)
    plot_training(training,metadata,output)
    write_report(output,exact,training,metadata,checks,shots)
    summary = {"noise_profiles":PROFILES,"basis_gates":BASIS_GATES,"connectivity":"all-to-all synthetic",
               "fixed_circuit_checks":checks,"training":metadata,"elapsed_seconds":time.perf_counter()-started,
               "versions":{"numpy":np.__version__,"qiskit":qiskit.__version__,"qiskit_aer":qiskit_aer.__version__},
               "sha256":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [ROOT/'aeqga.py',ROOT/'aeqga_noise.py',Path(__file__).resolve(),
                                      DATA/'Pantheon+SH0ES.dat',DATA/'Pantheon+SH0ES_STAT+SYS.cov']}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"COMPLETE {output}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/'output'/'qubit_noise_validation')
    parser.add_argument("--shots",type=int,default=1024)
    parser.add_argument("--generations",type=int,default=100)
    parser.add_argument("--seeds",type=int,nargs="+",default=[7,17,27])
    parser.add_argument("--input-repeats",type=int,default=3)
    parser.add_argument("--sample-repeats",type=int,default=5)
    args = parser.parse_args()
    run(args.output,args.shots,args.generations,args.seeds,args.input_repeats,args.sample_repeats)
