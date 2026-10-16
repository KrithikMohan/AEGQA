"""Reproducible Kingston / Fez / Marrakesh snapshot validation and delivery ZIP."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from qiskit import qpy
from qiskit.quantum_info import Statevector
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits

from aeqga import AEQGA
from aeqga_ibm_noise import (DEFAULT_SNAPSHOTS, DEVICES, SNAPSHOT_VERSION,
                             IBMNoiseAEQGA, SnapshotBackend, asymmetric_readout,
                             compile_native, in_classical_order, measurement_order,
                             named_values, simulator)
from run_pantheon_aeqga import (DATA, ROOT, classical_profile_reference,
                                load_data, make_model, verify_quadrature)

BOUNDS = [(60., 80.), (.1, .5), (-19.6, -19.)]
INITIAL = [70., .3, -19.3]
NAMES = ["H0_km_s_Mpc", "Omega_m", "M_mag"]
COLORS = {"kingston": "#3575B5", "fez": "#D27623", "marrakesh": "#198777"}
SOURCES = [
    "https://pypi.org/project/qiskit-ibm-runtime/0.50.0/",
    "https://github.com/Qiskit/qiskit-ibm-runtime",
    "https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/fake-provider-fake-kingston",
    "https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/fake-provider-fake-fez",
    "https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/fake-provider-fake-marrakesh",
    "https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html",
]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probability(values):
    values = np.asarray(values, dtype=float)
    # Aer composes calibrated Kraus channels with finite numerical tolerances.
    # Reject material trace drift; normalize only sub-ppm floating-point drift.
    if np.min(values) < -1e-12 or abs(values.sum()-1) > 1e-6:
        raise AssertionError(f"Invalid probability vector: min={values.min()}, sum={values.sum()}, size={len(values)}")
    values = np.maximum(values, 0.)
    return values / values.sum()


def tvd(a, b):
    return float(np.abs(np.asarray(a)-b).sum()/2)


def snapshots_metadata(output):
    meta = {}
    rows = []
    gate_rows = []
    for name in DEVICES:
        b = SnapshotBackend(name)
        files = {kind: f"{name}/{kind}_{name}.json" for kind in ("conf", "props")}
        meta[name] = {"snapshot_date": b.properties_json["last_update_date"],
                      "device_qubits": b.configuration_json["n_qubits"],
                      "physical_qubits": b.physical_ids, "dt_seconds": b.dt,
                      "basis_gates": b.configuration_json["basis_gates"],
                      "source_distribution": f"qiskit-ibm-runtime=={SNAPSHOT_VERSION}",
                      "sha256": {kind: sha256(DEFAULT_SNAPSHOTS / file) for kind, file in files.items()}}
        for q, p in enumerate(b.calibrations):
            rows.append({"device": name, "physical_qubit": q, "T1_us": p["T1"]*1e6,
                         "T2_us": p["T2"]*1e6, "readout_error": p["readout_error"],
                         "P1_given_0": p["prob_meas1_prep0"], "P0_given_1": p["prob_meas0_prep1"],
                         "readout_duration_ns": p["readout_length"]*1e9})
        for gate in b.properties_json["gates"]:
            if set(gate["qubits"]).issubset(b.physical_ids) and gate["gate"] in b.configuration_json["basis_gates"]:
                values = named_values(gate["parameters"])
                gate_rows.append({"device": name, "gate": gate["gate"],
                                  "physical_qubits": str(gate["qubits"]),
                                  "gate_error": values["gate_error"],
                                  "duration_ns": values["gate_length"]*1e9})
    pd.DataFrame(rows).to_csv(output / "selected_qubit_calibrations.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(output / "selected_gate_calibrations.csv", index=False)
    return meta


def fixed_catalog(input_repeats):
    cases, circuits, probabilities = [], [], []
    center = np.array([73.53, .332, -19.244])
    widths = np.ptp(np.asarray(BOUNDS), axis=1)
    for population in (8, 16, 32, 64):
        model = AEQGA(lambda x, p: 0., INITIAL, bounds=BOUNDS, population_size=population)
        for variant in range(input_repeats):
            rng = np.random.default_rng(700+variant)
            elites = np.clip(center+rng.uniform(-.02, .02, (population//4, 3))*widths,
                             model._lower, model._upper)
            random = rng.uniform(model._lower, model._upper, (population//2, 3))
            for branch, values in (("local", elites), ("global", random)):
                for dim in range(3):
                    lo, hi = model._local_box(elites[:, dim], dim) if branch == "local" else BOUNDS[dim]
                    for pattern, cross, mutate in (("none", 0., 0.), ("cross", 1., 0.),
                                                    ("mutation", 0., 1.), ("both", 1., 1.)):
                        model.crossover_probability, model.mutation_probability = cross, mutate
                        circuit = model._circuit(values[:, dim], np.random.default_rng(900+variant))
                        cases.append({"case_id": len(cases), "population": population,
                                      "population_bits": population.bit_length()-1,
                                      "branch": branch, "parameter": NAMES[dim],
                                      "input_variant": variant, "gate_pattern": pattern,
                                      "qubits": circuit.num_qubits, "candidate_count": len(values),
                                      "lower": lo, "upper": hi})
                        circuits.append(circuit)
                        probabilities.append(probability(Statevector.from_instruction(
                            circuit.remove_final_measurements(inplace=False)).probabilities()))
    return cases, circuits, probabilities


def fixed_validation(output, shots, input_repeats, sample_repeats):
    cases, originals, original_p = fixed_catalog(input_repeats)
    pd.DataFrame(cases).to_csv(output / "fixed_circuit_catalog.csv", index=False)
    with (output / "fixed_logical_circuits.qpy").open("wb") as stream:
        qpy.dump(originals, stream)
    rows, sample_rows, archive = [], [], {}
    worst_compile = worst_sampling = worst_trace = 0.
    for device_index, name in enumerate(DEVICES):
        print(f"FIXED {name}: {len(cases)} circuits", flush=True)
        backends = {n: SnapshotBackend(name, n) for n in range(2, 6)}
        compiled = [compile_native(c, backends[c.num_qubits]) for c in originals]
        with (output / f"fixed_native_{name}.qpy").open("wb") as stream:
            qpy.dump(compiled, stream)
        exact_circuits = []
        for c in compiled:
            exact = c.remove_final_measurements(inplace=False)
            exact.save_probabilities()
            exact_circuits.append(exact)
        ideal = simulator(backends[5], "ideal").run(exact_circuits, shots=1).result()
        noisy = simulator(backends[5], "noisy", readout=False).run(exact_circuits, shots=1).result()
        if not ideal.success or not noisy.success:
            raise AssertionError("Exact simulation failed")
        ideals, noisies = [], []
        for i, (case, c) in enumerate(zip(cases, compiled)):
            order = measurement_order(c)
            worst_trace = max(worst_trace, abs(float(np.sum(noisy.data(i)["probabilities"]))-1))
            p0 = probability(in_classical_order(ideal.data(i)["probabilities"], order))
            pg = probability(in_classical_order(noisy.data(i)["probabilities"], order))
            pn = probability(asymmetric_readout(pg, backends[c.num_qubits], order))
            delta = float(np.max(np.abs(p0-original_p[i])))
            worst_compile = max(worst_compile, delta)
            np.testing.assert_allclose(p0, original_p[i], rtol=0, atol=1e-10)
            ideals.append(p0); noisies.append(pn)
            args = (case["lower"], case["upper"], case["branch"] == "local")
            d0 = AEQGA._decode(p0, *args, np.random.default_rng(6000+i))[:case["candidate_count"]]
            dn = AEQGA._decode(pn, *args, np.random.default_rng(6000+i))[:case["candidate_count"]]
            rows.append({**case, "device": name, "exact_tvd": tvd(pn, p0),
                         "gate_idle_only_tvd": tvd(pg, p0),
                         "decode_rmse_fraction": float(np.sqrt(np.mean((dn-d0)**2)))/(args[1]-args[0]),
                         "cz_count": c.count_ops().get("cz", 0), "depth": c.depth(),
                         "duration_ns": int(c.duration)*backends[5].dt*1e9,
                         "measurement_physical_order": str(order)})
        for label, values in (("ideal", ideals), ("noisy", noisies)):
            packed = np.full((len(cases), 32), np.nan)
            for i, p in enumerate(values):
                packed[i, :len(p)] = p
            archive[f"{name}_{label}_probabilities"] = packed
        for mode, expected in (("ideal", ideals), ("noisy", noisies)):
            backend = simulator(backends[5], mode)
            accumulated = [np.zeros(len(p)) for p in expected]
            for repeat in range(sample_repeats):
                seed = 100000+device_index*1000+repeat
                sampled = backend.run(compiled, shots=shots, seed_simulator=seed).result()
                if not sampled.success:
                    raise AssertionError("Sampling failed")
                packed = np.zeros((len(cases), 32), dtype=np.int64)
                for i, case in enumerate(cases):
                    counts = AEQGA._counts(sampled.get_counts(i), len(expected[i]))
                    assert counts.sum() == shots
                    accumulated[i] += counts
                    packed[i, :len(counts)] = counts
                    sample_rows.append({"device": name, "mode": mode, "case_id": i,
                                        "qubits": case["qubits"], "repeat": repeat,
                                        "sampling_seed": seed, "shots": shots,
                                        "sample_tvd_vs_ideal": tvd(counts/shots, ideals[i]),
                                        "sample_tvd_vs_expected": tvd(counts/shots, expected[i])})
                archive[f"{name}_{mode}_counts_{repeat}"] = packed
            trials = sample_repeats*shots
            for p, counts in zip(expected, accumulated):
                bound = 6*np.sqrt(p*(1-p)/trials)+6/trials
                ratio = float(np.max(np.abs(counts/trials-p)/bound))
                worst_sampling = max(worst_sampling, ratio)
                if ratio > 1:
                    raise AssertionError("Sampled frequencies disagree with exact probabilities")
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "fixed_circuit_results.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(output / "fixed_circuit_samples.csv", index=False)
    np.savez_compressed(output / "fixed_probabilities_and_counts.npz", **archive)
    group = frame.groupby(["device", "qubits"])[["exact_tvd", "gate_idle_only_tvd", "cz_count", "duration_ns"]].agg(["mean", "std", "min", "max"])
    group.columns = ["_".join(c) for c in group.columns]
    group.to_csv(output / "fixed_circuit_summary.csv")
    checks = {"logical_circuits": len(cases), "device_circuits": 3*len(cases),
              "shots": shots, "sample_repeats": sample_repeats,
              "max_ideal_compile_probability_error": worst_compile,
              "max_noisy_probability_normalization_correction": worst_trace,
              "max_sampling_bound_ratio": worst_sampling,
              "sampling_bound": "6*sqrt(p*(1-p)/total_shots) + 6/total_shots"}
    write_json(output / "fixed_checks.json", checks)
    return frame, checks


def train_one(output, population, seed, device, mode, generations, shots, x, y, covariance, chol):
    run_id = f"N{population}_{device}_{mode}_seed{seed}"
    directory = output / "training_runs" / run_id
    config = {"run_id": run_id, "population": population, "population_bits": population.bit_length()-1,
              "local_qubits": max(2, (population//4-1).bit_length()),
              "global_qubits": max(2, (population//2-1).bit_length()),
              "device": device, "mode": mode, "seed": seed,
              "generations": generations, "shots": shots, "bounds": BOUNDS, "initial_p": INITIAL,
              "elite_local_global_counts": [population//4, population//4, population//2],
              "crossover_probability": .5, "mutation_probability": .5,
              "local_padding": .5, "min_local_width": .01,
              "implementation_sha256": {n: sha256(ROOT/n) for n in ("aeqga.py", "aeqga_parallel.py", "aeqga_ibm_noise.py", "run_pantheon_aeqga.py")},
              "snapshot_sha256": {kind: sha256(DEFAULT_SNAPSHOTS/device/f"{kind}_{device}.json") for kind in ("conf", "props")},
              "numerical_versions": {p: importlib.metadata.version(p) for p in ("qiskit", "qiskit-aer", "numpy", "scipy")}}
    if (directory / "result.json").exists():
        prior = json.loads((directory / "result.json").read_text())
        saved = json.loads((directory / "config.json").read_text())
        if saved != json.loads(json.dumps(config)):
            raise ValueError(f"Existing run has different settings: {run_id}")
        print(f"RESUME saved {run_id}", flush=True)
        return prior
    directory.mkdir(parents=True, exist_ok=True)
    print(f"TRAIN {run_id}", flush=True)
    started = time.perf_counter()
    model = IBMNoiseAEQGA(make_model(32), INITIAL, bounds=BOUNDS, population_size=population,
                         shots=shots, vectorized=True, seed=seed, device=device, mode=mode)
    model.train(x, y, covariance=covariance, generations=generations)
    assert model.n_generations_ == generations
    assert np.all(np.diff([h["loss"] for h in model.history_]) <= 1e-10)
    residual = solve_triangular(chol, make_model(64)(x.tolist(), model.p)-y, lower=True, check_finite=False)
    checked = float(residual@residual)
    np.testing.assert_allclose(model.loss_, checked, rtol=0, atol=1e-7)
    model.export_population_history(directory / "all_candidates.csv", parameter_names=NAMES)
    pd.DataFrame([{"generation": h["generation"], "best_index": h["best_index"],
                   **dict(zip(NAMES, h["p"])), "chi2": h["loss"]}
                  for h in model.history_]).to_csv(directory / "best_history.csv", index=False)
    with gzip.open(directory / "population_history.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(model.history_, stream, allow_nan=False)
    with gzip.open(directory / "quantum_measurements.jsonl.gz", "wt", encoding="utf-8") as stream:
        for record in model.circuit_records:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
    write_json(directory / "config.json", config)
    result = {**config, **dict(zip(NAMES, model.p)), "chi2": model.loss_, "independent_chi2": checked,
              "fitness_evaluations": model.n_evaluations_, "quantum_circuits": model.quantum_circuits_,
              "elapsed_seconds": time.perf_counter()-started}
    write_json(directory / "result.json", result)
    print(f"DONE {run_id} chi2={model.loss_:.9f} seconds={result['elapsed_seconds']:.1f}", flush=True)
    return result


def initialize_training_worker():
    global _worker_inputs, _worker_threads
    _worker_threads = threadpool_limits(limits=1)
    _, _, x, y, covariance, _ = load_data()
    _worker_inputs = (x, y, covariance, np.linalg.cholesky(covariance))
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*QuantumCircuit.duration.*")


def train_in_worker(task):
    return train_one(*task, *_worker_inputs)


def training_validation(output, generations, shots, seeds, workers=3):
    table, selected, x, y, covariance, covariance_check = load_data()
    rows = []
    with threadpool_limits(limits=1):
        chol = np.linalg.cholesky(covariance)
        reference = classical_profile_reference(make_model(64), x, y, chol, BOUNDS)
        quadrature = verify_quadrature(make_model(32), x)
        # Independent processes/models/RNGs; each simulator and BLAS use one CPU
        # thread. Parallelism changes wall time, not any experiment setting.
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize_training_worker) as pool:
            futures = [pool.submit(train_in_worker, (output, population, seed, device, mode,
                                   generations, shots))
                       for population in (8,16,32,64) for seed in seeds
                       for device in DEVICES for mode in ("ideal","noisy")]
            for future in as_completed(futures):
                rows.append(future.result())
                pd.DataFrame(rows).sort_values(["population","seed","device","mode"]).to_csv(
                    output / "training_results.csv", index=False)
    rows.sort(key=lambda r: (r["population"],r["seed"],r["device"],r["mode"]))
    frame = pd.DataFrame(rows)
    frame["delta_chi2_reference"] = frame.chi2-reference["chi2"]
    frame.to_csv(output / "training_results.csv", index=False)
    group = frame.groupby(["population", "device", "mode"])[NAMES+["chi2", "delta_chi2_reference"]].agg(["mean", "std", "min", "max"])
    group.columns = ["_".join(c) for c in group.columns]
    group.to_csv(output / "training_summary.csv")
    paired = frame.pivot(index=["population", "population_bits", "seed", "device"], columns="mode", values="chi2").reset_index()
    paired["noisy_minus_ideal_chi2"] = paired.noisy-paired.ideal
    paired.to_csv(output / "paired_noise_effect.csv", index=False)
    checks = verify_exports(output, frame, x, y, chol)
    meta = {"runs": len(frame), "generations": generations, "shots": shots, "seeds": seeds,
            "parallel_workers": workers,
            "source_rows": len(table), "selected_rows": len(selected),
            "calibrator_rows": int(selected.IS_CALIBRATOR.sum()),
            "reference": reference, "quadrature_max_error_mag": quadrature,
            "covariance_validation": covariance_check, "record_validation": checks}
    write_json(output / "training_checks.json", meta)
    return frame, paired, meta


def verify_exports(output, runs, x, y, chol):
    initial = {}
    all_rows = all_circuits = 0
    worst_rescore = 0.
    independent = make_model(64)
    with threadpool_limits(limits=1):
        for row in runs.itertuples():
            directory = output / "training_runs" / row.run_id
            all_candidates = pd.read_csv(directory / "all_candidates.csv")
            best = pd.read_csv(directory / "best_history.csv")
            assert len(all_candidates) == (row.generations+1)*row.population
            assert len(best) == row.generations+1
            np.testing.assert_array_equal(best.generation, np.arange(row.generations+1))
            assert np.all(np.isfinite(all_candidates[NAMES+["chi2"]]))
            for name, (lo, hi) in zip(NAMES, BOUNDS):
                assert all_candidates[name].between(lo, hi).all()
            for generation, candidates in all_candidates.groupby("generation", sort=True):
                np.testing.assert_array_equal(candidates.individual, np.arange(row.population))
                k = int(best.iloc[generation].best_index)
                np.testing.assert_allclose(candidates.iloc[k][NAMES+["chi2"]].to_numpy(dtype=float),
                                           best.iloc[generation][NAMES+["chi2"]].to_numpy(dtype=float), atol=1e-9)
            np.testing.assert_allclose(all_candidates.groupby("generation").chi2.min(), best.chi2, rtol=0, atol=1e-9)
            assert np.all(np.diff(best.chi2) <= 1e-9)
            key = (row.population, row.seed)
            first = all_candidates[all_candidates.generation == 0].to_numpy()
            if key not in initial:
                initial[key] = first
            else:
                np.testing.assert_array_equal(first, initial[key])
            # Independently rescore the final optimum and candidates spread over the entire run.
            indices = np.unique(np.r_[np.linspace(0, len(all_candidates)-1, 12, dtype=int),
                                      row.generations*row.population+int(best.iloc[-1].best_index)])
            for index in indices:
                sample = all_candidates.iloc[index]
                residual = solve_triangular(chol, independent(x.tolist(), sample[NAMES].tolist())-y,
                                            lower=True, check_finite=False)
                error = abs(float(residual@residual)-sample.chi2)
                worst_rescore = max(worst_rescore, error)
                if error > 1e-7:
                    raise AssertionError("Saved candidate chi2 does not match independent integration")
            count = 0
            with gzip.open(directory / "quantum_measurements.jsonl.gz", "rt", encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    assert record["generation"] == count//6+1
                    assert record["circuit_index"] == count%6
                    assert sum(record["counts"].values()) == row.shots
                    assert set(record["measurement_physical_order"]) == set(range(record["qubits"]))
                    count += 1
            assert count == 6*row.generations == row.quantum_circuits
            all_rows += len(all_candidates); all_circuits += count
    return {"verified_runs": len(runs), "candidate_rows": all_rows,
            "quantum_histograms": all_circuits, "matched_initial_population_groups": len(initial),
            "independent_rescore_max_abs_error": worst_rescore,
            "scope": "Every row checked for shape, finiteness and bounds; every generation checked for optimum; 12 spread candidates and final optimum independently rescored per run."}


def summary_plot(output, fixed, training, paired, meta, snapshot_meta):
    fig = Figure(figsize=(16, 10), facecolor="#f5f7fa")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
    for axis in axes.flat:
        axis.set_facecolor("white")
        axis.spines[["right", "top"]].set_visible(False)
        axis.grid(axis="y", alpha=.18)
        axis.tick_params(labelsize=10)
    for device in DEVICES:
        color = COLORS[device]
        subset = fixed[fixed.device == device]
        grp = subset.groupby("qubits").exact_tvd.mean()
        axes[0, 0].plot(grp.index, grp.values*100, "o-", color=color, label=device.capitalize(), lw=2)
        runs = training[(training.device == device) & (training["mode"] == "noisy")]
        for axis, col in [(axes[0, 1], "delta_chi2_reference"), *zip(axes[1], NAMES)]:
            group = runs.groupby("population_bits")[col].agg(["mean", "min", "max"])
            shift = (DEVICES.index(device)-1)*.055
            interval = np.maximum(0., np.vstack((group["mean"]-group["min"], group["max"]-group["mean"])))
            axis.errorbar(group.index+shift, group["mean"], yerr=interval,
                          fmt="o-", lw=1.8, capsize=3, color=color, label=device.capitalize(), ms=5)
        values = paired[paired.device == device].groupby("population_bits").noisy_minus_ideal_chi2.agg(["mean", "min", "max"])
        interval = np.maximum(0., np.vstack((values["mean"]-values["min"], values["max"]-values["mean"])))
        axes[0, 2].errorbar(values.index+(DEVICES.index(device)-1)*.055,
                            values["mean"], yerr=interval, fmt="o-", lw=1.8, capsize=3,
                            color=color, label=device.capitalize(), ms=5)
    axes[0, 0].set(title="A  Measurement-distribution distortion", xlabel="Actual circuit qubits", ylabel="Exact TVD from ideal (%)", xticks=[2,3,4,5])
    axes[0, 0].set_ylim(0, fixed.groupby(["device","qubits"]).exact_tvd.mean().max()*120)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left", ncol=3, columnspacing=1.)
    axes[0, 1].set(title="B  Final fit error above classical reference", ylabel=r"Final $\chi^2 - \chi^2_{ref}$")
    axes[0, 1].set_yscale("symlog", linthresh=.1)
    axes[0, 1].axhline(0, color="#5b6573", ls="--", lw=1)
    axes[0, 2].set(title="C  Paired noise effect on final fit", ylabel=r"$\chi^2_{noisy} - \chi^2_{ideal}$")
    axes[0, 2].set_yscale("symlog", linthresh=1)
    axes[0, 2].axhline(0, color="#5b6573", ls="--", lw=1)
    for axis, label, ref in zip(axes[1], [r"D  Final $H_0$ (km s$^{-1}$ Mpc$^{-1}$)", r"E  Final $\Omega_m$", "F  Final M (mag)"], meta["reference"]["p"]):
        axis.set_title(label)
        axis.axhline(ref, color="#5b6573", ls="--", lw=1.1)
        axis.ticklabel_format(axis="y", useOffset=False)
    for axis in [axes[0, 1], axes[0, 2], *axes[1]]:
        axis.set_xticks([3,4,5,6], ["3\nN=8", "4\nN=16", "5\nN=32", "6\nN=64"])
        axis.set_xlabel(r"Population bits: $\log_2 N$")
    fig.suptitle("IBM public-snapshot noise | AEQGA validation", x=.06, ha="left", fontsize=21, fontweight="bold", y=.98)
    fig.text(.06,.931, f"{meta['runs']} runs  |  {meta['generations']} generations  |  {meta['shots']} shots  |  {len(meta['seeds'])} seeds per condition  |  elite : local : global = 1 : 1 : 2",
             fontsize=11, color="#45546a")
    best_by_width = fixed.groupby(["device", "qubits"]).exact_tvd.mean().unstack(0).idxmin(axis=1)
    if best_by_width.nunique() == 1:
        headline = f"On this physical chain, {best_by_width.iloc[0].capitalize()} has the smallest mean distribution shift at every tested width."
    else:
        headline = "On this physical chain, the smallest mean distribution shift depends on circuit width."
    fig.text(.06,.900, headline, fontsize=11, fontweight="bold", color="#263951")
    fig.text(.06,.056, "A: exact density-matrix probabilities, averaged over fixed circuits.  B-F: final results; points = mean, bars = observed min-max across seeds.\n"
             "C: positive = worse final fit under noise; negative = better final fit for those runs. B/C use symmetric-log scales.\n"
             "Dashed lines in D-F: classical reference. Ranges across seeds are not cosmological confidence intervals.",
             fontsize=9, color="#45546a", linespacing=1.5)
    dates = " | ".join(f"{d.capitalize()} {snapshot_meta[d]['snapshot_date'][:10]}" for d in DEVICES)
    fig.text(.06,.021, f"Snapshots: {dates}.  Fixed physical chain q0-q4 (prefixes for 2-5 qubits); results describe this placement, not whole-device rankings.",
             fontsize=9, color="#45546a")
    fig.subplots_adjust(left=.06, right=.98, top=.845, bottom=.17, hspace=.48, wspace=.30)
    for ext in ("png", "svg"):
        fig.savefig(output / f"ibm_noise_summary.{ext}", dpi=190, facecolor=fig.get_facecolor())


def report(output, fixed, training, paired, meta, snapshot_meta):
    means = fixed.groupby(["device", "qubits"]).exact_tvd.mean().unstack()
    paired_group = paired.groupby(["device", "population"]).noisy_minus_ideal_chi2.agg(["mean", "std"])
    text = f"""IBM PUBLIC-SNAPSHOT NOISE VALIDATION: KINGSTON, FEZ, MARRAKESH
Generated UTC: {datetime.now(timezone.utc).isoformat()}

EXPERIMENT AND SCOPE
These are local Qiskit Aer simulations using the public calibration/configuration
JSON shipped in qiskit-ibm-runtime {SNAPSHOT_VERSION}. No hardware jobs were submitted.
Original JSON is archived unchanged with hashes and its license. The snapshots
have different calibration dates. They do not describe simultaneous measurements
or necessarily present-day device performance.

"""
    for name, info in snapshot_meta.items():
        text += f"{name}: {info['device_qubits']} device qubits; snapshot {info['snapshot_date']}\n"
    text += f"""
The experiment uses the predeclared chain [0,1,2,3,4] on each device, with prefixes
of exactly 2, 3, 4 or 5 qubits. All selected CZ links are calibrated in both
directions. This placement is fixed by physical index, not selected by low error.
The other device qubits are inactive and not simulated. This is not a simulation
of an arbitrary 156-qubit entangled state or a whole-device performance ranking.
Kingston q1 has average readout error 0.168212890625 in this snapshot; this is
retained. Selected-qubit and selected-gate calibration tables are included.

NOISE AND COMPILATION
Calibration units are converted to SI in a Qiskit BackendV2 Target restricted
to the selected subgraph. Gates use the official NoiseModel.from_backend with
gate_error=True, thermal_relaxation=True, temperature=0 mK. Aer constructs
depolarization plus T1/T2 relaxation using each gate's reported error and duration;
reported gate infidelity is not treated as a depolarizing strength directly.
Readout is independent per qubit and preserves both calibrated probabilities
P(measured 1 | prepared 0) and P(measured 0 | prepared 1), instead of replacing
them by their mean. The Target scalar readout channel is disabled before these
asymmetric channels are added. Measurement duration is not separately added as
relaxation to measured qubits; its assignment errors use the calibration matrix.

Initialization, CRy and Rx are decomposed into rz/sx/x/id/cz, with fixed initial
layout, basic routing, optimization level 1 and ALAP scheduling. The official
Aer relaxation pass adds T1/T2 noise to explicit idle delays. dt = 4 ns on all
three snapshots. Reset uses its reported duration and ideal reset followed by
thermal relaxation; no calibrated reset-assignment error is supplied. Virtual Rz has zero
duration/error. No error mitigation or dynamical decoupling is applied.
The model omits leakage, crosstalk, coherent calibration errors, drift, correlated
readout, temperature-dependent excitation and state-preparation/reset errors.
T2 is bounded by 2*T1 inside Aer when needed. Missing frequency is irrelevant
at the chosen zero-temperature setting. The model is an approximation to hardware.

FIXED-CIRCUIT VALIDATION
The same candidates, gate choices and logical circuits are reused across devices.
There are {len(fixed)//3} logical circuits: four populations, two branches, three
parameters, four crossover/mutation patterns and independent input variants.
Each native ideal probability vector is checked against its original circuit,
including the physical-to-classical permutation after routing. Exact noisy
probabilities use density matrices followed by analytic asymmetric readout.
Actual sampled Aer counts (ideal/noisy) are checked against these probabilities.
Numerical probability trace drift up to 1e-6 is normalized; larger drift fails.
The maximum correction actually observed is recorded in fixed_checks.json.
All circuits are saved in QPY; every fixed-circuit probability and histogram is
included in NPZ (only the first 2**qubits columns are used).

Mean exact TVD = 0.5*sum(abs(p_noisy-p_ideal)); columns are actual qubits:
{means.to_string(float_format=lambda v: f'{v:.8f}')}

The width groups also differ in input/state preparation, routing depth and gate
counts. Their differences cannot be attributed to qubit count alone. Decoder
RMSE additionally depends on min/max ties and exact-lower-bound replacement.

TRAINING VALIDATION
{meta['runs']} optimizer executions in a paired design: 3 device targets x 4 populations x
{len(meta['seeds'])} seeds x 2 noise modes. Seeds: {meta['seeds']}.
Each run has {meta['generations']} updates plus generation 0, {meta['shots']} shots per
quantum circuit, initial parameters {INITIAL}, and bounds {BOUNDS}.
Elite/local/global ratio is 1:1:2. Actual local/global widths for populations
8,16,32,64 are 2/2, 2/3, 3/4, 4/5. Population bits log2(N) are 3,4,5,6;
they are not the actual width of either branch. Final results are the best
candidate retained at the fixed training budget, not a convergence guarantee.

Each device has its own equally compiled/scheduled ideal control. All devices
and both modes share identical generation-0 candidates and scores for a given
population/seed. Subsequent trajectories may diverge. The production optimizer,
parameter decoding, candidate selection and chi-square remain unchanged.
Different populations use different numbers of fitness evaluations per update.

Dataset: {meta['selected_rows']} selected Pantheon+SH0ES rows, including
{meta['calibrator_rows']} calibrator rows; selection zHD > 0.01 OR IS_CALIBRATOR=1.
Full STAT+SYS covariance is retained, symmetrized for decimal-rounding asymmetry.
Three fitted parameters are H0, Omega_m and M in flat Lambda-CDM. Training uses
32-point Gauss-Legendre integration; independent final/sample checks use 64 points.
The classical reference is calculated separately and does not alter training:
p_ref = {meta['reference']['p']}
chi2_ref = {meta['reference']['chi2']:.12f}

Paired final chi2(noisy) - chi2(ideal), mean and sample SD across seeds:
{paired_group.to_string(float_format=lambda v: f'{v:.8f}')}

Positive paired differences indicate worse final chi-square in those runs;
negative differences indicate better final chi-square. These short runs and
three seeds do not establish a universal device ranking or a noise benefit.
Reported standard deviations describe optimizer variation, not posterior
uncertainty of cosmological parameters. No hardware-readiness threshold was set.
The overview figure shows mean final values with the observed minimum-to-maximum
range across seeds. Standard deviations are provided separately in the tables.

RECORDS
Each training_runs/N... directory contains:
  all_candidates.csv: all N candidates for every generation 0..G, H0/Omega_m/M/chi2.
  best_history.csv: every generation's optimum and within-generation best index.
  population_history.json.gz: complete population history in original structure.
  quantum_measurements.jsonl.gz: all six histograms per generation, simulator and
    transpiler seeds, native gate counts, depth, duration and measurement mapping.
  config.json, result.json: settings and final checked result.
The individual index is a row within a generation, not an ancestry/lineage ID.
Duplicate elites remain included. Every final-generation offspring is evaluated.

Record verification: {json.dumps(meta['record_validation'], indent=2)}

REPRODUCTION
Source code, pinned numerical dependencies, raw snapshot JSON, selected input
data and covariance are included. Install requirements.txt, then run
python run_ibm_noise_validation.py --output output/reproduced
from the reproduction directory. IBM account credentials are not required.
Existing completed runs are reused only when their settings match; use a new
output directory to repeat the stochastic simulation from scratch.

SOURCE REFERENCES
""" + "\n".join(SOURCES) + "\n"
    (output / "validation_report.txt").write_text(text, encoding="utf-8")
    chinese = f"""IBM 公开校准快照噪声验证

先看 ibm_noise_summary.png：
A：固定量子电路的测量分布偏移（TVD），越小越接近理想电路。
B：有噪声训练的最终卡方减去经典参考最小值，越小越接近参考解。
C：同一设备/种群/种子下，有噪声最终卡方减去无噪声最终卡方。
D-F：有噪声训练最终 H0、Omega_m、M；虚线是经典参考值。
B/C 为对称对数纵轴。点为均值，误差棒是 {len(meta['seeds'])} 个种子的实际最小到最大值，不是参数置信区间。
样本标准差另存于 training_summary.csv。

共 {meta['runs']} 次训练，每次 {meta['generations']} 代、{meta['shots']} shots；第 0 代也完整保留。
population=8/16/32/64；精英/局部/全局=1:1:2；种子={meta['seeds']}。
实际局部/全局电路宽度依次为 2/2、2/3、3/4、4/5 qubit。

每台设备的 training_runs 子目录中同时提供有噪声和无噪声对照。
all_candidates.csv 包含每代全部样本的三个参数及卡方；best_history.csv 保存每代最优值；
quantum_measurements.jsonl.gz 包含每次电路采样的原始计数、种子、门数、时长和比特映射。

模型采用 IBM qiskit-ibm-runtime {SNAPSHOT_VERSION} 发布包中的校准快照：
Kingston：{snapshot_meta['kingston']['snapshot_date']}
Fez：{snapshot_meta['fez']['snapshot_date']}
Marrakesh：{snapshot_meta['marrakesh']['snapshot_date']}
原始 JSON、来源、校验哈希、代码、输入数据均已打包。模拟没有提交到真实量子计算机。

使用固定物理链 q0-q4，各宽度取前缀；没有根据低误差挑选位置。
Kingston 的 q1 在该快照中的平均读出误差约 16.82%，实际用于本次模拟。
包含原生门分解、拓扑路由、门误差、T1/T2、等待退相干和非对称读出误差。
复位使用校准时长及热弛豫，但无复位分配错误；不包含串扰、泄漏、噪声漂移或误差缓解。
这些结果只适用于所用快照、物理位置、算法和训练预算，不能作为整台设备的排名。

详细方法、数值结果与检查见 validation_report.txt、summary.json。
training_results.csv 是全部最终结果，paired_noise_effect.csv 是逐种子配对对比。
reproduction 目录可独立复现；运行说明见其中 README.txt。
"""
    (output / "README_中文.txt").write_text(chinese, encoding="utf-8-sig")


def package_results(output):
    reproduction = output / "reproduction"
    reproduction.mkdir(exist_ok=True)
    for filename in ("aeqga.py", "aeqga_parallel.py", "aeqga_noise.py", "aeqga_ibm_noise.py", "run_pantheon_aeqga.py", "run_ibm_noise_validation.py", "test_aeqga.py", "test_history_plot.py", "test_noise_validation.py", "test_ibm_noise_validation.py", "test_parallel_circuits.py"):
        shutil.copy2(ROOT / filename, reproduction / filename)
    shutil.copytree(DEFAULT_SNAPSHOTS, reproduction / "ibm_backend_snapshots", dirs_exist_ok=True)
    data_target = reproduction / "Pantheon+_Data" / "4_DISTANCES_AND_COVAR"
    data_target.mkdir(parents=True, exist_ok=True)
    for filename in ("Pantheon+SH0ES.dat", "Pantheon+SH0ES_STAT+SYS.cov", "README"):
        shutil.copy2(DATA / filename, data_target / filename)
    dependencies = ["qiskit", "qiskit-aer", "numpy", "scipy", "pandas", "matplotlib", "threadpoolctl"]
    (reproduction / "requirements.txt").write_text("\n".join(f"{p}=={importlib.metadata.version(p)}" for p in dependencies)+"\n")
    (reproduction / "README.txt").write_text(
        "Python 3.11 was used. Create an isolated environment, install requirements.txt,\n"
        "then run: python run_ibm_noise_validation.py --output output/reproduced\n"
        "Full run defaults: 100 generations, 1024 shots, seeds 7 17 27,\n"
        "3 fixed input variants and 5 fixed-circuit sampling repeats.\n"
        "Test with: python -m unittest -v test_ibm_noise_validation\n"
        "qiskit-ibm-runtime is not needed at execution time: its unchanged public\n"
        "snapshot JSON is included. No account, network access or hardware fees.\n"
        "The BackendV2 Target adapter uses documented JSON units and calls the\n"
        "official Aer NoiseModel.from_backend; calibrated asymmetric readout\n"
        "is applied separately. See validation_report.txt for model assumptions.\n",
        encoding="utf-8")
    files = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (output / "SHA256SUMS.txt").write_text("\n".join(f"{sha256(p)}  {p.relative_to(output).as_posix()}" for p in files)+"\n", encoding="utf-8")
    zip_path = output.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in sorted(output.rglob("*")):
            if p.is_file():
                archive.write(p, f"{output.name}/{p.relative_to(output).as_posix()}")
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad:
            raise AssertionError(f"ZIP integrity failed: {bad}")
    return zip_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "ibm_device_noise_validation")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7,17,27])
    parser.add_argument("--input-repeats", type=int, default=3)
    parser.add_argument("--sample-repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--skip-fixed", action="store_true", help="Reuse previously completed fixed-circuit validation")
    args = parser.parse_args()
    if min(args.generations, args.shots, args.input_repeats, args.sample_repeats, args.workers) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Counts must be positive and seeds unique")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*QuantumCircuit.duration.*")
    snapshot_meta = snapshots_metadata(output)
    if args.skip_fixed:
        fixed = pd.read_csv(output / "fixed_circuit_results.csv")
        fixed_checks = json.loads((output / "fixed_checks.json").read_text())
        if fixed_checks["shots"] != args.shots or fixed_checks["sample_repeats"] != args.sample_repeats or fixed_checks["logical_circuits"] != 96*args.input_repeats:
            raise ValueError("Saved fixed-circuit configuration differs")
    else:
        fixed, fixed_checks = fixed_validation(output, args.shots, args.input_repeats, args.sample_repeats)
    training, paired, meta = training_validation(output, args.generations, args.shots, args.seeds, args.workers)
    summary_plot(output, fixed, training, paired, meta, snapshot_meta)
    report(output, fixed, training, paired, meta, snapshot_meta)
    versions = {p: importlib.metadata.version(p) for p in ("qiskit", "qiskit-aer", "numpy", "scipy", "pandas", "matplotlib", "threadpoolctl")}
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "snapshots": snapshot_meta,
               "fixed_circuit_checks": fixed_checks, "training": meta, "versions": versions,
               "python": platform.python_version(), "sources": SOURCES,
               "elapsed_seconds_this_invocation": time.perf_counter()-started,
               "source_hashes": {n: sha256(ROOT/n) for n in ("aeqga.py", "aeqga_parallel.py", "aeqga_ibm_noise.py", "run_ibm_noise_validation.py", "run_pantheon_aeqga.py")},
               "data_hashes": {n: sha256(DATA/n) for n in ("Pantheon+SH0ES.dat", "Pantheon+SH0ES_STAT+SYS.cov")}}
    write_json(output / "summary.json", summary)
    archive = package_results(output)
    print(f"COMPLETE {archive} bytes={archive.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
