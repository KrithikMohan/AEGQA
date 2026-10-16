"""Explicit synthetic noise models and compilation for AEQGA validation.

These profiles are experiment assumptions, not calibrations of a real device.
Depolarizing strengths are channel parameters, not average gate infidelities.
"""
from __future__ import annotations

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from aeqga import AEQGA

BASIS_GATES = ["u", "cx"]
PROFILES = {
    "ideal": {"one_qubit_depolarizing": 0., "two_qubit_depolarizing": 0., "readout_flip": 0.},
    "low": {"one_qubit_depolarizing": .0001, "two_qubit_depolarizing": .001, "readout_flip": .005},
    "medium": {"one_qubit_depolarizing": .001, "two_qubit_depolarizing": .01, "readout_flip": .02},
    "high": {"one_qubit_depolarizing": .005, "two_qubit_depolarizing": .05, "readout_flip": .05},
    "medium_gate_only": {"one_qubit_depolarizing": .001, "two_qubit_depolarizing": .01, "readout_flip": 0.},
    "medium_readout_only": {"one_qubit_depolarizing": 0., "two_qubit_depolarizing": 0., "readout_flip": .02},
}


def build_noise_model(profile):
    config = PROFILES[profile] if isinstance(profile, str) else profile
    for value in config.values():
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Noise strengths must be finite and between 0 and 1.")
    noise = NoiseModel(basis_gates=BASIS_GATES)
    if config["one_qubit_depolarizing"]:
        noise.add_all_qubit_quantum_error(
            depolarizing_error(config["one_qubit_depolarizing"], 1), ["u"])
    if config["two_qubit_depolarizing"]:
        noise.add_all_qubit_quantum_error(
            depolarizing_error(config["two_qubit_depolarizing"], 2), ["cx"])
    if config["readout_flip"]:
        flip = config["readout_flip"]
        noise.add_all_qubit_readout_error(ReadoutError([[1-flip, flip], [flip, 1-flip]]))
    return noise


def compile_basis(circuits, seed=12345):
    """Decompose initialization and all logical gates; all-to-all connectivity."""
    compiled = transpile(circuits, basis_gates=BASIS_GATES, optimization_level=0,
                         seed_transpiler=seed, num_processes=1)
    for circuit in compiled if isinstance(compiled, list) else [compiled]:
        unsupported = set(circuit.count_ops()) - {"u", "cx", "reset", "measure", "barrier"}
        if unsupported:
            raise AssertionError(f"Unmodeled instructions remain: {unsupported}")
    return compiled


def make_backend(profile, *, readout=True):
    config = dict(PROFILES[profile])
    if not readout:
        config["readout_flip"] = 0.
    return AerSimulator(method="density_matrix", noise_model=build_noise_model(config),
                        max_parallel_threads=1, max_parallel_experiments=1)


def apply_readout(probabilities, flip):
    """Independent symmetric classical bit flips in Qiskit basis-index order."""
    probabilities = np.asarray(probabilities, dtype=float)
    size = len(probabilities)
    if size < 2 or size & (size-1):
        raise ValueError("The probability vector must have power-of-two length >= 2.")
    if not 0 <= flip <= 1:
        raise ValueError("flip must be in [0, 1].")
    result = probabilities.copy()
    for bit in range(size.bit_length()-1):
        old = result.copy()
        result = (1-flip)*old + flip*old[np.arange(size) ^ (1 << bit)]
    return result


class NoiseValidationAEQGA(AEQGA):
    """Same optimizer and decoder, with explicit gate decomposition and noise."""
    def __init__(self, *args, noise_profile="ideal", **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_profile = noise_profile
        self._backend = make_backend(noise_profile)

    def _compile_circuits(self, circuits, seed):
        return compile_basis(circuits, seed)
