from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

for name in ["ibm_fez", "ibm_kingston", "ibm_marrakesh"]:
    backend = service.backend(name)
    props = backend.properties()

    errors = [
        gate.parameters[0].value
        for gate in props.gates
        if gate.gate in ("cz", "ecr", "cx")
        and gate.parameters
        and gate.parameters[0].name == "gate_error"
    ]

    print(f"\nProcessor: {name}")
    print(f"Qubits: {backend.num_qubits}")
    print(f"Pending jobs: {backend.status().pending_jobs}")
    print(f"Median two-qubit error: {sorted(errors)[len(errors)//2]:.5f}" if errors else "No two-qubit error data")