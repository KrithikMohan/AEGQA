import numpy as np

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)

from aeqga.steps.evolution.aeqga_algorithm import (
    build_dimension_circuit,
)

service = QiskitRuntimeService()

for backend_name in ["ibm_kingston", "ibm_fez"]:
    backend = service.backend(backend_name)

    print(f"\n=== {backend_name} ===")

    for n_qubits in [3, 4]:
        rng = np.random.default_rng(42)
        values = rng.uniform(0.1, 1.0, 2**n_qubits)

        circuit = build_dimension_circuit(
            values,
            p_cross=1.0,
            p_mut=1.0,
            num_shots=4096,
            mode="measure",
        )

        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=3,
            seed_transpiler=42,
        )

        compiled = pm.run(circuit)

        operations = compiled.count_ops()

        print(f"\n{n_qubits}-qubit AEQGA circuit:")
        print("Original depth:", circuit.depth())
        print("Compiled depth:", compiled.depth())
        print("Compiled operations:", operations)
        print("Two-qubit operations:",
              sum(
                  count
                  for instruction, count in operations.items()
                  if instruction in ("cz", "cx", "ecr")
              ))


from qiskit.quantum_info import Statevector
from qiskit import transpile

backend = service.backend("ibm_kingston")

values = np.random.default_rng(42).uniform(0.1, 1.0, 16)

original = build_dimension_circuit(
    values,
    p_cross=1.0,
    p_mut=1.0,
    num_shots=4096,
    mode="measure",
)

compiled = transpile(
    original,
    backend=backend,
    optimization_level=3,
    seed_transpiler=42,
)

# Remove measurement operations
original_clean = original.remove_final_measurements(inplace=False)
compiled_clean = compiled.remove_final_measurements(inplace=False)

p_original = Statevector.from_instruction(
    original_clean
).probabilities()

from qiskit_aer import AerSimulator

# Find where the original logical qubits ended up
physical_qubits = compiled.layout.final_index_layout(
    filter_ancillas=True
)

# Simulate the hardware-compiled circuit without
# constructing a 156-qubit statevector
compiled_clean.save_probabilities_dict(
    qubits=physical_qubits,
    label="logical_probs"
)

simulator = AerSimulator(method="matrix_product_state")

result = simulator.run(compiled_clean).result()

probability_dict = result.data(0)["logical_probs"]

p_compiled = np.array([
    probability_dict.get(i, probability_dict.get(hex(i), 0.0))
    for i in range(2 ** original.num_qubits)
])

print(
    "\nMaximum probability difference:",
    np.max(np.abs(p_original - p_compiled))
)