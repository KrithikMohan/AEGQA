"""Local, illustrative trapped-ion noise experiment; no IonQ service calls.

The rates below are assumptions, not device calibration or IonQ's cloud model.
Rotations and XX interactions provide an all-to-all trapped-ion-inspired basis;
they are not a compilation into IonQ's native GPI/GPI2/MS pulse gates.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from .ibm_sampler import DryRunExecutor

BASIS = ["rx", "ry", "rz", "rxx"]


@dataclass(frozen=True)
class IonQDemoNoise:
    """Depolarizing channel strengths (not gate infidelities), and bit-flip rate."""

    one_qubit: float = 0.001
    two_qubit: float = 0.01
    readout: float = 0.005

    def __post_init__(self):
        for value in asdict(self).values():
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Demo noise strengths must be finite numbers in [0, 1]")

    def metadata(self):
        return dict(kind="illustrative_local_aer_approximation", calibrated=False,
                    strengths=asdict(self), basis_gates=BASIS,
                    connectivity="all-to-all", reset_noise=False,
                    limitations="No device calibration, coherent errors, crosstalk, drift or timing model")

    def build(self):
        model = NoiseModel(basis_gates=BASIS)
        if self.one_qubit:
            model.add_all_qubit_quantum_error(
                depolarizing_error(self.one_qubit, 1), ["rx", "ry", "rz"])
        if self.two_qubit:
            model.add_all_qubit_quantum_error(depolarizing_error(self.two_qubit, 2), ["rxx"])
        if self.readout:
            p = self.readout
            model.add_all_qubit_readout_error(ReadoutError([[1-p, p], [p, 1-p]]))
        return model


class IonQDemoExecutor(DryRunExecutor):
    """AEQGA adapter for transient local pilots; persistent restarts are unsupported."""

    def __init__(self, *, shots=100, seed=23, noise=None):
        self.noise = noise if noise is not None else IonQDemoNoise()
        identity = hashlib.sha256(json.dumps(self.noise.metadata(), sort_keys=True).encode()).hexdigest()[:12]
        super().__init__(shots=shots, seed_simulator=seed,
                         backend_name=f"local_ionq_demo_{identity}")
        self.simulator = AerSimulator(method="density_matrix", basis_gates=BASIS,
                                      noise_model=self.noise.build())
        self.batches = []

    def run(self, circuits, generation=0, optimizer_state=None):
        counts = super().run(circuits, generation, optimizer_state)
        if circuits:
            self.batches.append(dict(generation=generation, counts=counts,
                                     compilation=self.diagnostics.summary()))
        return counts
