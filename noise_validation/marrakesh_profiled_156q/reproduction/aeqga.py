"""AEQGA with complete subpopulations on equally sized parallel circuits."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
from scipy.linalg import solve_triangular
from aeqga_legacy import AEQGA as _SinglePopulationAEQGA
from aeqga_parallel import positive_integer, marginal_counts


class AEQGA(_SinglePopulationAEQGA):
    """Complete subpopulations with globally selected, round-robin elites.

    Each of K=parallel_circuits circuits receives q=active_qubits//K qubits
    and N=2**q complete candidates; the qubit remainder is unused. Defaults
    per circuit: N/4 elites, N/4 local offspring, N/2 global offspring.
    Elites are selected from the combined K*N population, then dealt in rank
    order round-robin. Each circuit reuses its q wires for successive
    parameter/branch stages, resetting and measuring into fresh classical bits.
    All K circuits run on disjoint wires. More shots do not add candidates.

    population_size is a derived TOTAL; population_per_circuit and the three
    count attributes refer to ONE group. max_total_candidates guards explicit
    exponential allocation and may be raised after budgeting memory and time.
    The inherited predictor, weighted objective and plotting API are unchanged.
    """

    def __init__(self, function, p, *, bounds, active_qubits=5, parallel_circuits=1,
                 elite_count=None, local_count=None, random_count=None, shots=1024,
                 crossover_probability=.5, mutation_probability=.5, local_padding=.5,
                 min_local_width=.01, vectorized=False, seed=None,
                 max_total_candidates=65536):
        active_qubits = positive_integer(active_qubits, "active_qubits")
        parallel_circuits = positive_integer(parallel_circuits, "parallel_circuits")
        max_total_candidates = positive_integer(max_total_candidates, "max_total_candidates")
        width = active_qubits // parallel_circuits
        if width < 3:
            raise ValueError("active_qubits // parallel_circuits must be at least 3 "
                             "(at least 8 candidates per complete subpopulation).")
        # Reject huge integers before shifting or allocating any arrays.
        if (width >= max_total_candidates.bit_length()
                or parallel_circuits > (max_total_candidates >> width)):
            raise ValueError(f"parallel_circuits * 2**{width} exceeds max_total_candidates="
                             f"{max_total_candidates}; increase parallel_circuits, reduce "
                             "active_qubits, or explicitly budget a larger candidate limit.")
        size = 1 << width
        super().__init__(function, p, bounds=bounds, population_size=size,
                         elite_count=elite_count, local_count=local_count,
                         random_count=random_count, shots=shots,
                         parallel_circuits=parallel_circuits,
                         max_parallel_qubits=width*parallel_circuits,
                         crossover_probability=crossover_probability,
                         mutation_probability=mutation_probability,
                         local_padding=local_padding, min_local_width=min_local_width,
                         vectorized=vectorized, seed=seed)
        self.active_qubits = active_qubits
        self.qubits_per_circuit = width
        self.used_active_qubits = width*parallel_circuits
        self.discarded_qubits = active_qubits-self.used_active_qubits
        self.population_per_circuit = size
        self.population_size = size*parallel_circuits
        self.max_total_candidates = max_total_candidates
        self.quantum_stages_ = 0
        self.selection_history_ = []
        self.subpopulations_ = None

    def _circuit(self, values, rng):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not 0 < len(values) <= self.population_per_circuit:
            raise ValueError("Stage values must fit one subpopulation's amplitude register.")
        return super()._circuit(np.pad(values, (0, self.population_per_circuit-len(values))), rng)

    def _subpopulation_circuit(self, elites, random_part, rng, circuit_index):
        stages, small = [], []
        local_parents = elites[np.arange(self.local_count) % len(elites)]
        for dimension in range(self._p.size):
            for branch, values in (("local", local_parents), ("global", random_part)):
                if not len(values):
                    continue
                start = len(stages)*self.qubits_per_circuit
                stages.append({"stage_index": len(stages), "branch": branch,
                               "parameter_index": dimension, "candidate_count": len(values),
                               "clbits": list(range(start, start+self.qubits_per_circuit))})
                small.append(self._circuit(values[:, dimension], rng))
        if not stages:
            return None
        circuit = QuantumCircuit(self.qubits_per_circuit, len(stages)*self.qubits_per_circuit)
        for stage, operation in zip(stages, small):
            if stage["stage_index"]:
                circuit.barrier()
            circuit.compose(operation, qubits=list(range(self.qubits_per_circuit)),
                            clbits=stage["clbits"], inplace=True)
        circuit.metadata = {"subpopulation_index": circuit_index, "stages": stages,
                            "population_size": self.population_per_circuit,
                            "elite_count": self.elite_count, "local_count": self.local_count,
                            "random_count": self.random_count}
        return circuit

    def _parallel_offspring(self, elites, rngs, execution_rng):
        random_parts = [rng.uniform(self._lower, self._upper,
                                   size=(self.random_count, self._p.size)) for rng in rngs]
        circuits = [self._subpopulation_circuit(e, r, rng, i)
                    for i, (e, r, rng) in enumerate(zip(elites, random_parts, rngs))]
        if all(c is None for c in circuits):
            return np.empty((self.parallel_circuits, 0, self._p.size))
        seed = int(execution_rng.integers(0, 2**31-1))
        histograms = self._measure_circuits(circuits, seed, execution_rng)
        self.quantum_circuits_ += len(circuits)
        self.quantum_stages_ += sum(len(c.metadata["stages"]) for c in circuits)
        offspring, stage_records = [], []
        for i, (circuit, histogram, rng) in enumerate(zip(circuits, histograms, rngs)):
            local = np.empty((self.local_count, self._p.size))
            global_part = np.empty((self.random_count, self._p.size))
            for stage in circuit.metadata["stages"]:
                measured = marginal_counts(histogram, stage["clbits"])
                counts = self._counts(measured, self.population_per_circuit)
                dimension = stage["parameter_index"]
                if stage["branch"] == "local":
                    lo, hi = self._local_box(elites[i, :, dimension], dimension)
                    local[:, dimension] = self._decode(counts, lo, hi, True, rng)[:self.local_count]
                else:
                    global_part[:, dimension] = self._decode(
                        counts, self._lower[dimension], self._upper[dimension], False, rng)[:self.random_count]
                stage_records.append({**stage, "subpopulation_index": i, "counts": measured})
            offspring.append(np.vstack((local, global_part)))
        self.execution_history_[-1].update({
            "active_qubits_requested": self.active_qubits,
            "qubits_per_circuit": self.qubits_per_circuit,
            "used_active_qubits": self.used_active_qubits,
            "discarded_qubits": self.discarded_qubits,
            "population_per_circuit": self.population_per_circuit,
            "total_population": self.population_size, "stage_measurements": stage_records})
        return np.asarray(offspring)

    def _select_elites(self, population, losses, rng, generation):
        """Globally select K*E indices, then deal sorted ranks round-robin."""
        flat, scores = population.reshape(-1, self._p.size), losses.ravel()
        count = self.parallel_circuits*self.elite_count
        indices = np.argsort(scores, kind="stable")[:count].copy()
        valid = indices[np.isfinite(scores[indices])]
        invalid = ~np.isfinite(scores[indices])
        if invalid.any():
            indices[invalid] = rng.choice(valid, size=int(invalid.sum()), replace=True)
        assigned = indices.reshape(self.elite_count, self.parallel_circuits).T
        self.selection_history_.append({
            "generation": generation, "policy": "global_elites_round_robin",
            "global_elite_indices": indices.tolist(),
            "assignments": [{"circuit_index": i, "source_indices": row.tolist(),
                             "source_circuits": (row//self.population_per_circuit).tolist(),
                             "source_individuals": (row % self.population_per_circuit).tolist()}
                            for i, row in enumerate(assigned)]})
        return flat[assigned].copy(), scores[assigned].copy()

    def train(self, x, y, *, generations=100, sigma=None, covariance=None,
              target_loss=None, verbose=False):
        self._positive_integer(generations, "generations")
        inputs, targets = self._inputs(x), self._real_array(y, "y")
        if targets.shape != (len(inputs),):
            raise ValueError("y must be a one-dimensional list with one target per sample.")
        if target_loss is not None and (not np.isfinite(target_loss) or target_loss < 0):
            raise ValueError("target_loss must be finite and nonnegative, or None.")
        scale, cholesky = self._weights(len(targets), sigma, covariance)
        seeds = np.random.SeedSequence(self.seed).spawn(self.parallel_circuits+1)
        rngs = [np.random.default_rng(s) for s in seeds[:-1]]
        execution_rng = np.random.default_rng(seeds[-1])
        population = np.array([rng.uniform(self._lower, self._upper,
                                size=(self.population_per_circuit, self._p.size)) for rng in rngs])
        population[:, 0, :] = self._p
        self.loss_ = None
        self.loss_kind_ = "chi2" if sigma is not None or covariance is not None else "sse"
        self.history_, self.execution_history_, self.execution_circuits_ = [], [], []
        self.selection_history_, self.subpopulations_ = [], None
        self.n_generations_ = self.n_evaluations_ = 0
        self.quantum_circuits_ = self.quantum_stages_ = self.executed_circuits_ = 0
        self._n_features = 1 if inputs.ndim == 1 else inputs.shape[1]

        def evaluate(candidates):
            return self._evaluate_candidates(candidates, inputs, targets, scale, cholesky)

        losses = evaluate(population)
        if not np.any(np.isfinite(losses)):
            raise ValueError("All initial candidates have non-finite loss. Check function and bounds.")
        self._record_subpopulations(population, losses, 0, verbose)
        for generation in range(1, int(generations)+1):
            if target_loss is not None and self.loss_ <= target_loss:
                break
            elites, elite_losses = self._select_elites(population, losses, execution_rng, generation)
            offspring = self._parallel_offspring(elites, rngs, execution_rng)
            population = np.concatenate((elites, offspring), axis=1)
            losses = np.concatenate((elite_losses, evaluate(offspring)), axis=1)
            self._record_subpopulations(population, losses, generation, verbose)
        self.subpopulations_ = population.copy()
        return self

    def _evaluate_candidates(self, candidates, inputs, targets, scale, cholesky):
        """Objective hook; subclasses may batch an equivalent profiled likelihood."""
        shape = candidates.shape[:-1]
        results = np.empty(int(np.prod(shape)))
        for i, parameters in enumerate(candidates.reshape(-1, self._p.size)):
            prediction = self._predict(inputs, parameters)
            self.n_evaluations_ += 1
            if not np.all(np.isfinite(prediction)):
                results[i] = np.inf
                continue
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                residual = prediction-targets
                if scale is not None:
                    residual = residual/scale
                elif cholesky is not None:
                    residual = solve_triangular(cholesky, residual, lower=True, check_finite=False)
                value = float(residual@residual)
            results[i] = value if np.isfinite(value) else np.inf
        return results.reshape(shape)

    def _record_subpopulations(self, population, losses, generation, verbose):
        super()._record(population.reshape(-1, self._p.size), losses.ravel(), generation, verbose)
        record = self.history_[-1]
        record.update({"population_per_circuit": self.population_per_circuit,
                       "parallel_circuits": self.parallel_circuits,
                       "best_circuit": record["best_index"]//self.population_per_circuit,
                       "best_individual": record["best_index"] % self.population_per_circuit,
                       "subpopulations": []})
        for index, candidate in enumerate(record["population"]):
            group, local = divmod(index, self.population_per_circuit)
            origin = ("initial" if generation == 0 else "elite" if local < self.elite_count
                      else "local" if local < self.elite_count+self.local_count else "global")
            candidate.update({"circuit_index": group, "individual": local, "origin": origin})
        for i, (values, scores) in enumerate(zip(population, losses)):
            best = int(np.argmin(scores))
            record["subpopulations"].append({"circuit_index": i, "best_individual": best,
                "p": values[best].tolist(), "loss": float(scores[best]), "population_size": len(values)})
        if generation:
            record["elite_selection"] = self.selection_history_[-1]

    def export_population_history(self, path, *, parameter_names=None):
        if not self.history_:
            raise ValueError("No training history to export. Train first.")
        names = [f"p_{i}" for i in range(self._p.size)] if parameter_names is None else list(parameter_names)
        reserved = {"generation", "circuit_index", "individual", "origin", "chi2", "sse"}
        if (len(names) != self._p.size or not all(isinstance(n, str) and n.strip() for n in names)
                or len(set(names)) != len(names) or set(names) & reserved):
            raise ValueError("parameter_names must be unique nonempty names, one per parameter, "
                             "excluding generation, circuit_index, individual, origin, chi2 and sse.")
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["generation", "circuit_index", "individual", "origin", *names, self.loss_kind_])
            for record in self.history_:
                for candidate in record["population"]:
                    writer.writerow([record["generation"], candidate["circuit_index"],
                                     candidate["individual"], candidate["origin"], *candidate["p"], candidate["loss"]])
        return output
