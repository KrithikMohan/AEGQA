
import random
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService

from aeqga.steps.evolution.aeqga_algorithm import (
    build_dimension_circuit,
)
from aeqga.execution.ibm_sampler import IBMSamplerExecutor

# Experimental settings
BACKEND = "ibm_kingston"
SHOTS = 1024
SEED = 42

# Reproducible AEQGA input
random.seed(SEED)
rng = np.random.default_rng(SEED)
values = rng.uniform(0.1, 1.0, 16)

circuit = build_dimension_circuit(
    values,
    p_cross=1.0,
    p_mut=1.0,
    num_shots=SHOTS,
    mode="measure",
)

# Ideal noiseless probabilities
ideal_circuit = circuit.remove_final_measurements(
    inplace=False
)
ideal = Statevector.from_instruction(
    ideal_circuit
).probabilities()

# IBM Quantum connection
service = QiskitRuntimeService()
backend = service.backend(BACKEND)

print("Processor:", backend.name)
print("Circuit qubits:", circuit.num_qubits)
print("Shots:", SHOTS)

# Persist IBM job IDs and measured counts for recovery
executor = IBMSamplerExecutor(
    backend=backend,
    service=service,
    checkpoint_dir=Path("outputs/json/ibm_first_pilot"),
    shots=SHOTS,
    optimization_level=1,
    seed_transpiler=SEED,
)

print("Submitting or retrieving IBM quantum job...", flush=True)

counts = executor.execute_generation(
    [circuit],
    generation=1,
)[0]

# Convert measured counts into probabilities
measured = np.array([
    counts.get(format(i, "04b"), 0) / SHOTS
    for i in range(16)
])

# Total variation distance
tvd = 0.5 * np.sum(np.abs(measured - ideal))

print("\n=== IBM QPU RESULTS ===")
print("Measured counts:", counts)
print("Total variation distance:", tvd)
print("Probability normalization:", measured.sum())
print("Checkpoint: outputs/json/ibm_first_pilot")
