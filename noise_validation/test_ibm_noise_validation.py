"""Physical-model, routing, relaxation, readout and recording checks."""
import json
import unittest
import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, average_gate_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError

from aeqga import AEQGA
from aeqga_ibm_noise import (DEVICES, IBMNoiseAEQGA, SnapshotBackend,
                             asymmetric_readout, compile_native, device_noise,
                             in_classical_order, measurement_order, simulator)


class IBMNoiseValidationTests(unittest.TestCase):
    def test_selected_target_matches_raw_calibration_units_and_gate_fidelity(self):
        for name in DEVICES:
            b = SnapshotBackend(name, 5)
            self.assertEqual(b.dt, 4e-9)
            for q in range(5):
                raw = {x["name"]: x["value"] for x in b.properties_json["qubits"][q]}
                self.assertAlmostEqual(b.target.qubit_properties[q].t1, raw["T1"]*1e-6, places=14)
                self.assertAlmostEqual(b.target.qubit_properties[q].t2, raw["T2"]*1e-6, places=14)
                np.testing.assert_allclose(b.readout_matrix(q),
                    [[1-raw["prob_meas1_prep0"], raw["prob_meas1_prep0"]],
                     [raw["prob_meas0_prep1"], 1-raw["prob_meas0_prep1"]]], atol=1e-15)
            model = device_noise(b)
            for edge in [(0,1),(1,2),(2,3),(3,4)]:
                raw = next(g for g in b.properties_json["gates"] if g["gate"] == "cz" and g["qubits"] == list(edge))
                raw = {x["name"]: x["value"] for x in raw["parameters"]}
                self.assertAlmostEqual(b.target["cz"][edge].duration, raw["gate_length"]*1e-9, places=14)
                # These selected gates have error above the thermal floor, so Aer
                # must match the reported average gate infidelity exactly.
                error = model._local_quantum_errors["cz"][edge]
                self.assertAlmostEqual(1-average_gate_fidelity(error), raw["gate_error"], places=11)

    def test_native_routing_and_measurement_permutation_preserve_logical_distribution(self):
        model = AEQGA(lambda x,p:p[0], [0.], bounds=[(-1,1)], population_size=64)
        model.crossover_probability = model.mutation_probability = 1.
        for name in DEVICES:
            for width in (2,3,4,5):
                b = SnapshotBackend(name, width)
                original = model._circuit(np.linspace(-.7,1.,1<<width), np.random.default_rng(31))
                c = compile_native(original,b)
                self.assertNotIn("initialize", c.count_ops())
                self.assertNotIn("cry", c.count_ops())
                self.assertNotIn("rx", c.count_ops())
                self.assertGreater(c.count_ops().get("cz", 0), 0)
                ideal = Statevector.from_instruction(c.remove_final_measurements(inplace=False)).probabilities()
                expected = Statevector.from_instruction(original.remove_final_measurements(inplace=False)).probabilities()
                np.testing.assert_allclose(in_classical_order(ideal, measurement_order(c)), expected, atol=1e-10, rtol=0)
        c = QuantumCircuit(3,3)
        c.x(0); c.measure(0,2); c.measure(1,0); c.measure(2,1)
        physical = np.zeros(8); physical[1] = 1
        logical = in_classical_order(physical, measurement_order(c))
        self.assertEqual(np.argmax(logical), 4)

    def test_scheduled_idle_noise_matches_analytic_T1_decay(self):
        for name in DEVICES:
            b = SnapshotBackend(name, 2)
            c = QuantumCircuit(2)
            rho = np.zeros((4,4)); rho[1,1] = 1.
            c.set_density_matrix(rho)
            duration_dt = 10000
            c.delay(duration_dt,0,unit="dt")
            c.save_probabilities()
            result = simulator(b, readout=False).run(c,shots=1).result()
            p = result.data(0)["probabilities"]
            survival = np.exp(-duration_dt*b.dt/b.target.qubit_properties[0].t1)
            np.testing.assert_allclose(p, [1-survival,survival,0,0],atol=1e-10,rtol=0)

    def test_actual_asymmetric_readout_with_nontrivial_classical_order(self):
        for name in DEVICES:
            b = SnapshotBackend(name,2)
            noise = NoiseModel()
            for q in range(2):
                noise.add_readout_error(ReadoutError(b.readout_matrix(q)),[q])
            c = QuantumCircuit(2,2)
            c.x(0); c.measure(0,1); c.measure(1,0)
            original = [0.,0.,1.,0.]
            expected = asymmetric_readout(original,b,[1,0])
            result = AerSimulator(method="density_matrix",noise_model=noise,
                                  max_parallel_threads=1).run(c,shots=60000,seed_simulator=7).result()
            actual = AEQGA._counts(result.get_counts(),4)/60000
            np.testing.assert_allclose(actual,expected,atol=.005,rtol=0)

    def test_full_population_chi2_and_raw_measurement_recording(self):
        x = np.array([0.,1.,2.,3.]); y = np.array([1.,2.,5.,10.])
        function = lambda x,p:p[0]*x[0]**2+p[1]*x[0]+p[2]
        initial = None
        for mode in ("ideal","noisy"):
            model = IBMNoiseAEQGA(function,[1.,0.,0.],bounds=[(-2,2)]*3,
                                  population_size=8,shots=128,seed=7,device="kingston",mode=mode)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",DeprecationWarning)
                model.train(x,y,sigma=.5,generations=2)
            if initial is None:
                initial = model.history_[0]
            else:
                self.assertEqual(initial,model.history_[0])
            self.assertEqual(len(model.circuit_records),12)
            self.assertEqual(len(model.history_),3)
            for h in model.history_:
                for sample in h["population"]:
                    p = sample["p"]
                    residual = (p[0]*x**2+p[1]*x+p[2]-y)/.5
                    self.assertAlmostEqual(sample["chi2"],float(residual@residual))
            for i,r in enumerate(model.circuit_records):
                self.assertEqual(r["generation"],i//6+1)
                self.assertEqual(sum(r["counts"].values()),128)
                json.dumps(r,allow_nan=False)


if __name__ == "__main__":
    unittest.main()
