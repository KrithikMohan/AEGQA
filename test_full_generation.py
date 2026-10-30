
import numpy as np
from qiskit.quantum_info import Statevector

from aeqga.likelihoods.pantheon_problem import PantheonPlusProblem
from aeqga.steps.evolution.aeqga_algorithm import (
    AEQGAParameters,
    run_aeqga_dual,
)

class DryRunExecutor:
    def __init__(self, shots=1024):
        self.shots = shots
        self.rng = np.random.default_rng(42)
        self.calls = 0

    def execute_generation(self, circuits, generation):
        self.calls += 1

        assert generation == 1
        assert len(circuits) == 4
        assert [c.num_qubits for c in circuits] == [4, 3, 4, 3]

        results = []

        for circuit in circuits:
            clean = circuit.remove_final_measurements(
                inplace=False
            )
            probabilities = Statevector.from_instruction(
                clean
            ).probabilities()

            samples = self.rng.multinomial(
                self.shots, probabilities
            )

            width = circuit.num_qubits
            counts = {
                format(i, f"0{width}b"): int(count)
                for i, count in enumerate(samples)
                if count > 0
            }

            results.append(counts)

        print("Four circuits processed successfully.")
        return results


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
    num_shots=1024,
    progress_bar=False,
)

executor = DryRunExecutor()

best, populations, history = run_aeqga_dual(
    problem,
    params,
    seed=42,
    hardware_executor=executor,
    quiet=True,
)

assert executor.calls == 1
assert len(populations) == 2
assert len(history) == 2
assert best.evaluations == 64

print("\n=== OFFLINE GENERATION TEST ===")
print("Best parameters:", best.x)
print("Best chi-squared:", best.fitness)
print("Objective evaluations:", best.evaluations)
print("Generation test: PASSED")
