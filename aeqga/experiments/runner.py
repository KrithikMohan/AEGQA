"""Seeded independent runs, strict provenance and per-run atomic checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from importlib.metadata import version

import numpy as np
from aeqga.paths import PROJECT_ROOT
from aeqga.steps.evolution.aeqga_algorithm import run_aeqga_dual


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_source_digest():
    digest = hashlib.sha256()
    for directory in ["steps", "likelihoods", "emulators", "experiments"]:
        for path in sorted((PROJECT_ROOT/"aeqga"/directory).rglob("*.py")):
            digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def validate_history(problem, params, record):
    """Recompute every logged pair, check bounds, elitism and final consistency."""
    history = np.asarray(record["best_history"], dtype=float)
    points = np.asarray(record["best_parameter_history"], dtype=float)
    if history.shape != (params.max_gen+1,) or points.shape != (params.max_gen+1, problem.n_dim):
        raise ValueError("Incorrect history shape/generation accounting")
    if (not np.all(np.isfinite(history)) or not np.all(np.isfinite(points))
            or np.any(points < problem.lower_bounds) or np.any(points > problem.upper_bounds)):
        raise ValueError("Non-finite or out-of-bounds history")
    measured = np.asarray([problem.compute_fitness(x) for x in points])
    if not np.allclose(measured, history, rtol=0, atol=1e-7):
        raise ValueError("History chi-squared does not match its parameter pairs")
    maximize = getattr(problem, "is_max_problem", lambda: False)()
    if np.any(np.diff(history) * (-1 if maximize else 1) > 1e-7):
        raise ValueError("Global-best history violates elitism")
    if record["evaluations"] != params.pop_size*(params.max_gen+1):
        raise ValueError("Incorrect objective evaluation budget")
    if (not np.allclose(points[-1], record["best_x"], rtol=0, atol=1e-12)
            or abs(history[-1]-record["best_chi2"]) > 1e-7):
        raise ValueError("Final result disagrees with its history")
    return float(np.max(np.abs(measured-history)))


def run_ensemble(problem, params, seeds, checkpoint, *, provenance,
                 execution_mode="statevector_shots", resume=True, progress=None):
    """Return ALL samples/histories; never reuse mismatched or corrupt checkpoints.

    Each completed run is atomically saved before the manifest is advanced.
    A crash at that boundary reruns only the uncommitted run. Single-writer API.
    """
    params._validate()
    seeds = list(seeds)
    if (not seeds or any(isinstance(s, bool) or not isinstance(s, (int, np.integer))
                         or s < 0 or s >= 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Provide nonempty unique integer seeds in [0, 2**32)")
    seeds = [int(s) for s in seeds]
    checkpoint = Path(checkpoint)
    configuration = dict(schema_version=1, parameters=vars(params), seeds=seeds,
                         execution_mode=execution_mode, provenance=provenance,
                         source_sha256=scientific_source_digest(),
                         versions={name: version(name) for name in
                                   ["numpy", "scipy", "qiskit", "qiskit-aer"]})
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    manifest = dict(configuration=configuration, fingerprint=fingerprint, completed=[])
    if checkpoint.exists():
        if not resume:
            raise FileExistsError("Checkpoint exists; choose a new experiment name")
        manifest = json.loads(checkpoint.read_text())
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Checkpoint configuration/data/source/version mismatch")
    directory = checkpoint.with_suffix(".runs")
    completed = manifest["completed"]
    if completed != list(range(len(completed))) or len(completed) > len(seeds):
        raise ValueError("Invalid checkpoint completed-run sequence")
    records = []
    for index, seed in enumerate(seeds):
        path = directory / f"run_{index:04d}.json"
        if index in completed:
            record = json.loads(path.read_text())
            if record.get("seed") != seed or record.get("fingerprint") != fingerprint:
                raise ValueError("Run checkpoint identity mismatch")
            validate_history(problem, params, record)
        else:
            started = perf_counter()
            best, populations, history = run_aeqga_dual(
                problem, params, seed=seed, execution_mode=execution_mode, quiet=True)
            record = dict(seed=seed, fingerprint=fingerprint, best_x=best.x.tolist(),
                          best_chi2=best.fitness, best_generation=best.gen,
                          evaluations=best.evaluations, elapsed_seconds=perf_counter()-started,
                          best_history=[entry[1] for entry in history],
                          best_parameter_history=[entry[0] for entry in history],
                          population_history=[pop.tolist() for pop in populations])
            record["history_max_abs_error"] = validate_history(problem, params, record)
            atomic_json(path, record)
            completed.append(index)
            atomic_json(checkpoint, manifest)
        records.append(record)
        if progress: progress(index+1, len(seeds))
    points = np.asarray([r["best_x"] for r in records])
    return dict(configuration=configuration, fingerprint=fingerprint, runs=records,
                mean=points.mean(axis=0).tolist(), std_ddof0=points.std(axis=0).tolist(),
                covariance_ddof1=(np.cov(points.T, ddof=1).tolist() if len(seeds)>1 else None),
                interpretation="independent optimizer outcomes, not posterior samples")
