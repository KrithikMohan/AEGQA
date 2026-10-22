"""Resumable real-data SNe hardware pilot; offline unless both flags are set."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aeqga.likelihoods.pantheon_problem import PantheonPlusProblem
from aeqga.steps.evolution.aeqga_algorithm import AEQGAParameters
from aeqga.execution.ibm_sampler import IBMSamplerExecutor, DryRunExecutor
from aeqga.execution.workflow import run_checkpointed, pantheon_provenance


def run_generation(backend_name="ibm_kingston", shots=1024, submit_hardware=False,
                   confirm=False, *, data=None, directory=None, generations=1,
                   population=32, seed=42, session=False, sampler_options=None,
                   preparation="initialize"):
    if submit_hardware and not confirm:
        raise ValueError("Hardware execution requires --confirm-run-qpu")
    if session and not submit_hardware:
        raise ValueError("Sessions require explicit hardware submission")
    # Fail on unavailable real data before contacting the hardware service.
    problem = PantheonPlusProblem(data or REPO_ROOT / "sn_data/PantheonPlus")
    params = AEQGAParameters(pop_size=population, max_gen=generations,
                             num_shots=shots, progress_bar=False, state_preparation=preparation)
    params._validate()
    directory = Path(directory or REPO_ROOT / "outputs/json" /
                     ("sne_hardware_pilot" if submit_hardware else "sne_offline_pilot"))
    executor = (IBMSamplerExecutor(backend_name=backend_name, shots=shots,
                                  checkpoint_dir=directory, seed_transpiler=seed,
                                  sampler_options=sampler_options)
                if submit_hardware else
                DryRunExecutor(shots=shots, checkpoint_dir=directory, seed_simulator=seed))
    context = nullcontext(None)
    if session:
        from qiskit_ibm_runtime import Session
        context = Session(backend=executor._get_backend())
    with context as runtime_session:
        executor.runtime_mode = runtime_session
        record = run_checkpointed(problem, params, executor, directory, seed=seed,
                                  provenance=pantheon_provenance(problem))
    print(f"Best [H0, Omega_m]: {record['best_x']}; chi2={record['best_chi2']:.6f}")
    print(f"Records: {directory}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="ibm_kingston")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-total-shots", type=int, help="Ceiling including all requested runs (not calibration)")
    parser.add_argument("--preparation", choices=["initialize", "state_preparation"], default="initialize")
    parser.add_argument("--session", action="store_true")
    parser.add_argument("--sampler-options", type=Path, help="Runtime options JSON; never credentials")
    parser.add_argument("--submit-hardware", action="store_true")
    parser.add_argument("--confirm-run-qpu", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.generations < 0 or args.shots < 1:
        parser.error("runs/shots must be positive and generations nonnegative")
    planned_shots = args.runs * args.generations * 4 * args.shots
    if args.max_total_shots is not None and planned_shots > args.max_total_shots:
        parser.error(f"Plan requires {planned_shots} shots, exceeding --max-total-shots")
    print(f"Plan: {args.runs} runs, {4*args.generations*args.runs} circuit executions, {planned_shots} shots")
    import json
    options = json.loads(args.sampler_options.read_text()) if args.sampler_options else {}
    import numpy as np
    seeds = ([args.seed] if args.runs == 1 else
             [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(args.seed).spawn(args.runs)])
    base = args.directory or REPO_ROOT/"outputs/json"/(
        "sne_hardware_ensemble" if args.submit_hardware else "sne_offline_ensemble")
    for index, seed in enumerate(seeds):
        directory = args.directory if args.runs == 1 else base/f"run_{index:04d}"
        run_generation(args.backend, args.shots, args.submit_hardware, args.confirm_run_qpu,
                       data=args.data, directory=directory, generations=args.generations,
                       population=args.population, seed=seed, session=args.session,
                       sampler_options=options, preparation=args.preparation)


if __name__ == "__main__":
    main()
