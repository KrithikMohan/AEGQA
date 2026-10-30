
import json
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector

from aeqga.steps.evolution.aeqga_algorithm import (
    build_dimension_circuit,
)

SEED = 42
SHOTS = 1024

# Reconstruct the exact ideal input circuit
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

ideal_circuit = circuit.remove_final_measurements(
    inplace=False
)
ideal = Statevector.from_instruction(
    ideal_circuit
).probabilities()

# Load actual IBM Kingston measurements
path = Path(
    "outputs/json/ibm_first_pilot/recovered_counts.json"
)

counts = json.loads(path.read_text(encoding="utf-8"))

assert sum(counts.values()) == SHOTS

measured = np.array([
    counts.get(format(i, "04b"), 0) / SHOTS
    for i in range(16)
])

# Total variation distance
tvd = 0.5 * np.sum(np.abs(measured - ideal))

print("\n=== AEQGA HARDWARE ANALYSIS ===")
print("Processor: IBM Kingston")
print("Shots:", SHOTS)
print("Total variation distance:", tvd)
print("Maximum probability error:",
      np.max(np.abs(measured - ideal)))

# Plot ideal vs experimental probabilities
labels = [format(i, "04b") for i in range(16)]
x = np.arange(16)
width = 0.38

plt.figure(figsize=(12, 5))
plt.bar(x - width/2, ideal, width, label="Ideal simulator")
plt.bar(x + width/2, measured, width, label="IBM Kingston")

plt.xticks(x, labels, rotation=45)
plt.xlabel("Computational basis state")
plt.ylabel("Measurement probability")
plt.title(f"AEQGA: Ideal vs IBM Kingston (TVD = {tvd:.4f})")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

output = Path("outputs/png/ibm_first_comparison.png")
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=200)

print("Plot saved:", output)
plt.show()

# Monte Carlo finite-shot baseline
rng = np.random.default_rng(12345)

N_TRIALS = 10000

simulated_counts = rng.multinomial(
    SHOTS,
    ideal,
    size=N_TRIALS,
)

simulated_probs = simulated_counts / SHOTS

simulated_tvd = 0.5 * np.sum(
    np.abs(simulated_probs - ideal),
    axis=1,
)

print("\n=== FINITE-SHOT BASELINE ===")
print("Mean ideal-shot TVD:", simulated_tvd.mean())
print("95th percentile:", np.percentile(simulated_tvd, 95))
print("99th percentile:", np.percentile(simulated_tvd, 99))

print(
    "Fraction of ideal-shot experiments with TVD >= hardware:",
    np.mean(simulated_tvd >= tvd),
)

plt.figure(figsize=(9, 5))

plt.hist(
    simulated_tvd,
    bins=50,
    density=True,
    alpha=0.7,
    label="Ideal finite-shot experiments",
)

plt.axvline(
    tvd,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"IBM Kingston: {tvd:.4f}",
)

plt.xlabel("Total variation distance")
plt.ylabel("Probability density")
plt.title("AEQGA: Shot noise vs observed QPU discrepancy")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/png/ibm_shot_noise_comparison.png",
    dpi=200,
)

plt.show()