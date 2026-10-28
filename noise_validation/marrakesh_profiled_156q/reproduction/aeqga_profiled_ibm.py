"""Profiled-M AEQGA with exact factorized IBM noise and streaming records."""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import time
import numpy as np

from aeqga_ibm_noise import IBMNoiseAEQGA
from aeqga_parallel import pack_circuits
from aeqga_factorized_noise import simulate_block


class ProfiledIBMNoiseAEQGA(IBMNoiseAEQGA):
    def __init__(self, problem, *, executor=None, record_directory=None, **kwargs):
        self.problem, self.executor = problem, executor
        self.record_directory = None if record_directory is None else Path(record_directory)
        self._streams = {}
        super().__init__(problem.predict, [70., .3], bounds=problem.bounds,
                         vectorized=True, device='marrakesh', **kwargs)

    def train(self, *, generations=100, verbose=True):
        self.objective_seconds = self.quantum_wall_seconds = 0.
        self.compile_cpu_seconds = self.simulation_cpu_seconds = 0.
        self.scheduled_shot_seconds = 0.
        self.probability_history_ = []
        self._started = time.perf_counter()
        self.problem.cache.clear()
        if self.record_directory is not None:
            self.record_directory.mkdir(parents=True, exist_ok=True)
            self._streams = {name: gzip.open(self.record_directory / f'{name}.jsonl.gz',
                                             'wt', encoding='utf-8')
                             for name in ('training_history', 'execution_history')}
        try:
            return super().train(self.problem.x, self.problem.y, covariance=self.problem.covariance,
                                 generations=generations, verbose=verbose)
        finally:
            for stream in self._streams.values():
                stream.close()
            self._streams = {}

    def _evaluate_candidates(self, candidates, inputs, targets, scale, cholesky):
        start = time.perf_counter()
        flat = candidates.reshape(-1, 2)
        values = self.problem.evaluate(flat)
        self.n_evaluations_ += len(flat)
        self.objective_seconds += time.perf_counter()-start
        return values[:, 0].reshape(candidates.shape[:-1])

    def _measure_circuits(self, circuits, seed, rng):
        started = time.perf_counter()
        tasks = [(circuit, physical, self.device, self.snapshot_root, seed, self.mode)
                 for circuit, physical in zip(circuits, self.placements)]
        outputs = list(map(simulate_block, tasks) if self.executor is None
                       else self.executor.map(simulate_block, tasks))
        compiled = [o[0] for o in outputs]
        bundle = pack_circuits(compiled, qubit_maps=self.placements,
                               register_width=self.full_device.num_qubits)
        sampling_seed = int(rng.integers(0, 2**31-1))
        sampler = np.random.default_rng(sampling_seed)
        raw_samples = [0]*self.shots
        histograms, diagnostics, probabilities = [], [], []
        offset = 0
        for group, (native, stages, timing) in enumerate(outputs):
            group_samples = np.zeros(self.shots, dtype=np.uint64)
            group_probabilities = []
            for spec, stage in zip(circuits[group].metadata['stages'], stages):
                p, ideal = stage['probabilities'], stage['ideal']
                draws = np.searchsorted(np.cumsum(p), sampler.random(self.shots))
                draws = np.minimum(draws, len(p)-1)
                counts = np.bincount(draws, minlength=len(p))
                shift = spec['stage_index']*self.qubits_per_circuit
                group_samples |= draws.astype(np.uint64) << shift
                used = spec['candidate_count']
                diagnostics.append({**spec, 'subpopulation_index': group,
                    'noise_tvd': stage['noise_tvd'],
                    'probability_trace_error': stage['probability_trace_error'],
                    'ideal_probability_trace_error': stage['ideal_probability_trace_error'],
                    'sampling_tvd': float(np.abs(counts/self.shots-p).sum()/2),
                    'sampled_vs_ideal_tvd': float(np.abs(counts/self.shots-ideal).sum()/2),
                    'zero_fraction_used_candidates': float(np.mean(counts[:used] == 0)),
                    'unused_amplitude_probability': float(p[used:].sum()),
                    'mean_ideal_probability_used': float(ideal[:used].mean())})
                group_probabilities.append(np.stack((ideal, p)))
            width = native.num_clbits
            histogram = {format(value, f'0{width}b'): frequency
                         for value, frequency in Counter(map(int, group_samples)).items()}
            histograms.append(histogram)
            for shot, value in enumerate(group_samples):
                raw_samples[shot] |= int(value) << offset
            offset += width
            probabilities.append(group_probabilities)
            bundle.blocks[group].update(timing)
            self.compile_cpu_seconds += timing['compile_cpu_seconds']
            self.simulation_cpu_seconds += timing['simulation_cpu_seconds']
        self.probability_history_.append(np.asarray(probabilities))
        raw_counts = {format(value, f'0{offset}b'): frequency
                      for value, frequency in Counter(raw_samples).items()}
        duration = max(o[2]['duration_seconds'] for o in outputs)
        self.scheduled_shot_seconds += duration*self.shots
        execution = {'execution_index': 0, 'active_qubits': bundle.active_qubits,
                     'register_qubits': bundle.circuit.num_qubits,
                     'depth': bundle.circuit.depth(), 'gate_counts': dict(bundle.circuit.count_ops()),
                     'duration_seconds': duration, 'shots_times_scheduled_duration': self.shots*duration,
                     'blocks': bundle.blocks, 'counts': raw_counts}
        self.execution_circuits_ = [bundle.circuit]
        self.executed_circuits_ += 1  # logical packed executions, not simulator jobs
        self.execution_history_.append({'generation': self.n_generations_+1,
            'parallel_circuits': self.parallel_circuits, 'shots_per_logical_circuit': self.shots,
            'transpiler_seed': seed, 'sampling_seed': sampling_seed,
            'sampling_scheme': 'independent_stage_inverse_cdf',
            'simulation': 'exact_stage_density_matrices_then_finite_shot_sampling',
            'density_probability_evolutions': len(diagnostics) if self.mode == 'noisy' else 0,
            'ideal_probability_evolutions': len(diagnostics),
            'executions': [execution], 'stage_diagnostics': diagnostics})
        self.quantum_wall_seconds += time.perf_counter()-started
        return histograms

    def _record_subpopulations(self, population, losses, generation, verbose):
        super()._record_subpopulations(population, losses, generation, False)
        record = self.history_[-1]
        values = self.problem.evaluate(population.reshape(-1,2))
        if not np.allclose(values[:, 0], losses.ravel(), atol=1e-9, rtol=0):
            raise AssertionError('Recorded losses differ from profiled full-covariance likelihood.')
        for candidate, (_, magnitude, unconstrained) in zip(record['population'], values):
            candidate.update({'M_profiled': float(magnitude), 'M_unconstrained': float(unconstrained),
                              'M_at_bound': bool(magnitude != unconstrained),
                              'all_parameters': [*candidate['p'], float(magnitude)]})
        best = record['population'][record['best_index']]
        record.update({'M_profiled': best['M_profiled'], 'all_parameters': best['all_parameters'],
            'searched_parameter_names': ['H0_km_s_Mpc','Omega_m'],
            'derived_parameter_name': 'M_mag', 'fitness_evaluations': self.n_evaluations_,
            'elapsed_wall_seconds': time.perf_counter()-self._started,
            'objective_seconds_cumulative': self.objective_seconds,
            'quantum_wall_seconds_cumulative': self.quantum_wall_seconds,
            'compile_worker_seconds_cumulative': self.compile_cpu_seconds,
            'simulation_worker_seconds_cumulative': self.simulation_cpu_seconds,
            'shots_times_scheduled_duration_cumulative': self.scheduled_shot_seconds})
        self.problem.retain(population)
        if self._streams:
            stream = self._streams['training_history']
            stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
            if generation:
                stream = self._streams['execution_history']
                stream.write(json.dumps(self.execution_history_[-1], allow_nan=False)+'\n'); stream.flush()
        if verbose and (generation == 0 or generation % 10 == 0):
            print(f'q={self.qubits_per_circuit} {self.mode} generation={generation} '
                  f'chi2={self.loss_:.9f} p={best["all_parameters"]} '
                  f'seconds={record["elapsed_wall_seconds"]:.1f}', flush=True)

    def export_population_history(self, path):
        path = Path(path)
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['generation','circuit_index','individual','origin',
                             'H0_km_s_Mpc','Omega_m','M_mag','chi2','M_unconstrained','M_at_bound'])
            for record in self.history_:
                for sample in record['population']:
                    writer.writerow([record['generation'],sample['circuit_index'],sample['individual'],
                        sample['origin'],*sample['all_parameters'],sample['chi2'],
                        sample['M_unconstrained'],sample['M_at_bound']])
        return path
