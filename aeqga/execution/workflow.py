"""Single-writer, generation-resumable scientific hardware runs.

No credentials are stored. Callers must explicitly construct a runtime executor
to submit work; the default CLI constructs an offline executor.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter

from aeqga.experiments.runner import file_digest, scientific_source_digest, validate_history
from aeqga.paths import PROJECT_ROOT
from aeqga.steps.evolution.aeqga_algorithm import run_aeqga_dual
from .ibm_sampler import CheckpointManager, get_provenance_metadata


def run_checkpointed(problem, params, executor, directory, *, seed=42, provenance):
    """Resume at a decoded generation, replaying a saved in-flight job if needed.

    Keep one process per directory. A crash after submission but before persisting
    the returned job ID cannot be automatically recovered; inspect IBM jobs first.
    """
    params._validate()
    if params.num_shots != executor.shots:
        raise ValueError("Optimizer and executor shots must match")
    directory = Path(directory)
    if executor.checkpoint_dir != directory:
        raise ValueError("Executor and optimizer must share a checkpoint directory")
    execution_hash = hashlib.sha256()
    for path in sorted((PROJECT_ROOT / "aeqga/execution").glob("*.py")):
        execution_hash.update(path.read_bytes())
    config = dict(schema_version=1, parameters=vars(params), seed=seed,
                  backend=executor.backend_name, executor=type(executor).__name__,
                  shots=executor.shots, provenance=provenance,
                  source=scientific_source_digest(), execution_source=execution_hash.hexdigest(),
                  versions=get_provenance_metadata(),
                  optimization_level=getattr(executor, "optimization_level", None),
                  seed_transpiler=getattr(executor, "seed_transpiler", None),
                  seed_simulator=getattr(executor, "seed_simulator", None),
                  sampler_options=getattr(executor, "sampler_options", {}),
                  initial_layout=getattr(executor, "initial_layout", None),
                  session=getattr(executor, "runtime_mode", None) is not None)
    # Git HEAD alone may change with unrelated commits; source hashes identify code.
    config["versions"].pop("git_sha", None)
    simulation_backend = getattr(executor, "simulation_backend", None)
    if simulation_backend is not None:
        noise = executor.simulator.options.noise_model.to_dict()
        config["noise_model_sha256"] = hashlib.sha256(
            json.dumps(noise, sort_keys=True, default=str).encode()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    path = directory / "optimizer.json"
    saved = CheckpointManager.load(path)
    if saved is not None and saved.get("fingerprint") != fingerprint:
        raise ValueError("Hardware experiment configuration/data/source/version mismatch")
    if saved is None and list(directory.glob("generation_*.json")):
        raise ValueError("Historical directory has no restart identity; use a new experiment directory")
    started = perf_counter()
    elapsed = saved.get("elapsed_seconds", 0) if saved else 0

    def persist(state):
        CheckpointManager.atomic_write(path, dict(
            fingerprint=fingerprint, configuration=config, state=state,
            elapsed_seconds=elapsed+perf_counter()-started))

    best, populations, history = run_aeqga_dual(
        problem, params, hardware_executor=executor, seed=seed, quiet=True,
        resume_state=saved["state"] if saved else None, checkpoint_callback=persist)
    record = dict(best_x=best.x.tolist(), best_chi2=best.fitness,
                  best_generation=best.gen, evaluations=best.evaluations,
                  best_history=[h[1] for h in history],
                  best_parameter_history=[h[0] for h in history],
                  population_history=[p.tolist() for p in populations],
                  fingerprint=fingerprint, configuration=config,
                  elapsed_seconds=elapsed+perf_counter()-started)
    record["history_max_abs_error"] = validate_history(problem, params, record)
    CheckpointManager.atomic_write(directory / "result.json", record)
    return record


def pantheon_provenance(problem):
    data = problem._data
    return dict(objective="Pantheon+ calibrated MU_SH0ES / full STAT+SYS",
                selection=data["selection"], redshift=data["redshift"], grid_size=problem.grid_size,
                bounds=[problem.lower_bounds.tolist(), problem.upper_bounds.tolist()],
                data_hashes={key: file_digest(data[key])
                             for key in ("table_file", "covariance_file")})
