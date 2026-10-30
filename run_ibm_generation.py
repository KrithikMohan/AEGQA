
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService

from aeqga.likelihoods.pantheon_problem import PantheonPlusProblem
from aeqga.steps.evolution.aeqga_algorithm import (
    AEQGAParameters,
    run_aeqga_dual,
)
from aeqga.execution.ibm_sampler import IBMSamplerExecutor

# Hardware experiment configuration
BACKEND = "ibm_kingston"
SHOTS = 1024
SEED = 42

# A separate checkpoint from the earlier single-circuit pilot
CHECKPOINT = Path("outputs/json/ibm_generation_01")

# Prevent accidental reuse of incompatible earlier experiments
CHECKPOINT.mkdir(parents=True, exist_ok=True)

print("Loading Pantheon+ likelihood...", flush=True)

problem = PantheonPlusProblem(
    "sn_data/PantheonPlus",
    selection="all",
    redshift="hd_hel",
    grid_size=300,
)

params = AEQGAParameters(
    pop_size=32,
    max_gen=1,
    p_cross=0.5,
    p_mut=0.5,
    num_shots=SHOTS,
    progress_bar=False,
)

print("Connecting to IBM Quantum...", flush=True)

service = QiskitRuntimeService()
backend = service.backend(BACKEND)

executor = IBMSamplerExecutor(
    backend=backend,
    service=service,
    checkpoint_dir=CHECKPOINT,
    shots=SHOTS,
    optimization_level=1,
    seed_transpiler=SEED,
)

print("\n=== HARDWARE EXPERIMENT ===")
print("Backend:", backend.name)
print("Population:", params.pop_size)
print("Generations:", params.max_gen)
print("Expected circuits: 4")
print("Shots per circuit:", SHOTS)
print("Maximum experiment shots:", 4 * SHOTS)

# Require explicit confirmation before a paid/limited QPU submission
confirmation = input(
    "\nSubmit or recover this checkpointed hardware experiment? "
    "Type YES: "
)

if confirmation != "YES":
    raise SystemExit("Experiment cancelled.")

best, populations, history = run_aeqga_dual(
    problem,
    params,
    seed=SEED,
    hardware_executor=executor,
    quiet=True,
)

assert len(populations) == 2
assert len(history) == 2
assert best.evaluations == 64

print("\n=== IBM AEQGA GENERATION RESULTS ===")
print("Best H0:", best.x[0])
print("Best Omega_m:", best.x[1])
print("Best chi-squared:", best.fitness)
print("Objective evaluations:", best.evaluations)
print("Checkpoint directory:", CHECKPOINT)
print("Hardware generation: COMPLETE")
