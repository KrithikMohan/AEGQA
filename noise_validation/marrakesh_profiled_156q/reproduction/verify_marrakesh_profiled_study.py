"""Audit exported study files without rerunning or changing the experiments."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from qiskit import qpy


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_case(folder, plan):
    start = time.perf_counter()
    summary = read_json(folder/'summary.json')
    config, width = summary['config'], summary['width']
    groups, generations, shots = config['parallel_circuits'], plan['generations'], plan['shots']
    size, population = 2**width, groups*2**width
    elite = size//4
    assert config['generations'] == generations and config['shots'] == shots
    assert summary['validation'] == 'passed'
    assert summary['candidate_rows'] == (generations+1)*population
    assert summary['fitness_evaluations'] == population+generations*groups*3*size//4
    layout = read_json(folder/'physical_layout.json')['physical_groups']
    used = sum(layout, [])
    assert len(layout) == groups and all(len(group) == width for group in layout)
    assert len(used) == len(set(used)) == summary['used_active_qubits']
    assert all(0 <= q < 156 for q in used)

    previous, initialization, rows, count, boundary = None, None, 0, 0, 0
    with gzip.open(folder/'training_history.jsonl.gz', 'rt', encoding='utf-8') as stream, \
         (folder/'population_history.csv').open(newline='', encoding='utf-8') as csv_stream:
        reader = csv.DictReader(csv_stream)
        for expected_generation, line in enumerate(stream):
            entry = json.loads(line)
            assert entry['generation'] == expected_generation
            candidates = entry['population']
            assert len(candidates) == population
            params = np.array([c['all_parameters'] for c in candidates])
            scores = np.array([c['chi2'] for c in candidates])
            assert np.isfinite(params).all() and np.isfinite(scores).all()
            assert ((params >= [60., .1, -19.6]) & (params <= [80., .5, -19.])).all()
            assert entry['best_index'] == int(np.argmin(scores))
            assert entry['loss'] == scores.min()
            assert entry['all_parameters'] == candidates[entry['best_index']]['all_parameters']
            assert entry['fitness_evaluations'] == population+expected_generation*groups*3*size//4
            if previous is None:
                initialization = hashlib.sha256(params.tobytes()+scores.tobytes()).hexdigest()
            else:
                old_params, old_scores = previous
                indices = np.argsort(old_scores, kind='stable')[:groups*elite]
                assignments = indices.reshape(elite, groups).T
                selection = entry['elite_selection']
                assert selection['global_elite_indices'] == indices.tolist()
                for group in range(groups):
                    assert selection['assignments'][group]['source_indices'] == assignments[group].tolist()
                    np.testing.assert_array_equal(params[group*size:group*size+elite], old_params[assignments[group]])
                    np.testing.assert_array_equal(scores[group*size:group*size+elite], old_scores[assignments[group]])
                assert scores.min() <= old_scores.min()+1e-10
            for index, candidate in enumerate(candidates):
                group, individual = divmod(index, size)
                origin = ('initial' if expected_generation == 0 else 'elite' if individual < elite
                          else 'local' if individual < 2*elite else 'global')
                assert candidate['circuit_index'] == group and candidate['individual'] == individual
                assert candidate['origin'] == origin
                assert candidate['p'] == candidate['all_parameters'][:2]
                assert candidate['M_profiled'] == float(np.clip(candidate['M_unconstrained'], -19.6, -19.))
                assert candidate['M_at_bound'] == (candidate['M_profiled'] != candidate['M_unconstrained'])
                boundary += candidate['M_at_bound']
                row = next(reader)
                assert (int(row['generation']), int(row['circuit_index']), int(row['individual'])) == \
                       (expected_generation, group, individual)
                assert row['origin'] == origin
                assert [float(row[k]) for k in ('H0_km_s_Mpc', 'Omega_m', 'M_mag')] == candidate['all_parameters']
                assert float(row['chi2']) == candidate['chi2']
                assert float(row['M_unconstrained']) == candidate['M_unconstrained']
                assert (row['M_at_bound'] == 'True') == candidate['M_at_bound']
                rows += 1
            previous = params, scores
            count += 1
        assert next(reader, None) is None
    assert count == generations+1 and rows == summary['candidate_rows']
    assert boundary == summary['M_boundary_rows']
    assert summary['chi2'] == float(previous[1].min())

    probabilities = np.load(folder/'stage_probabilities.npz')['probabilities']
    assert probabilities.shape == (generations, groups, 4, 2, size)
    assert probabilities.min() >= 0 and np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1., atol=1e-12, rtol=0.)
    if summary['mode'] == 'ideal':
        np.testing.assert_array_equal(probabilities[:, :, :, 0], probabilities[:, :, :, 1])

    executions, checked_stages, repaired_states, max_repair = 0, 0, 0, 0.
    with gzip.open(folder/'execution_history.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for generation, line in enumerate(stream, 1):
            entry = json.loads(line)
            assert entry['generation'] == generation
            assert entry['shots_per_logical_circuit'] == shots
            assert len(entry['executions']) == 1
            execution = entry['executions'][0]
            assert execution['active_qubits'] == groups*width and execution['register_qubits'] == 156
            assert len(execution['blocks']) == groups
            assert sum(execution['counts'].values()) == shots
            assert all(len(bits) == groups*4*width for bits in execution['counts'])
            raw = [(int(bits, 2), n) for bits, n in execution['counts'].items()]
            assert len(entry['stage_measurements']) == len(entry['stage_diagnostics']) == 4*groups
            for measurement, diagnostic in zip(entry['stage_measurements'], entry['stage_diagnostics']):
                group, stage = measurement['subpopulation_index'], measurement['stage_index']
                assert diagnostic['subpopulation_index'] == group and diagnostic['stage_index'] == stage
                shift = group*4*width+stage*width
                marginal = Counter()
                for value, frequency in raw:
                    marginal[format((value >> shift) & (size-1), f'0{width}b')] += frequency
                assert dict(marginal) == measurement['counts']
                assert sum(measurement['counts'].values()) == shots
                exact_ideal, exact_mode = probabilities[generation-1, group, stage]
                sampled = np.zeros(size)
                for bits, frequency in marginal.items():
                    sampled[int(bits, 2)] = frequency/shots
                assert abs(float(np.abs(exact_ideal-exact_mode).sum()/2)-diagnostic['noise_tvd']) < 1e-12
                assert abs(float(np.abs(sampled-exact_mode).sum()/2)-diagnostic['sampling_tvd']) < 1e-12
                assert diagnostic['probability_trace_error'] <= 1e-5
                checked_stages += 1
            for index, block in enumerate(execution['blocks']):
                assert block['qubits'] == layout[index]
                assert block['logical_qubits'] == width
                for repair in block.get('state_preparation_roundoff_repairs', []):
                    assert repair['max_amplitude_change'] <= 1e-14
                    repaired_states += 1
                    max_repair = max(max_repair, repair['max_amplitude_change'])
            executions += 1
    assert executions == generations and checked_stages == groups*4*generations

    with (folder/'last_generation.qpy').open('rb') as stream:
        circuits = qpy.load(stream)
    assert len(circuits) == 1
    circuit = circuits[0]
    assert circuit.num_qubits == 156 and circuit.num_clbits == groups*width*4
    live = {circuit.find_bit(q).index for item in circuit.data for q in item.qubits}
    assert live == set(used)
    block_of = {q:i for i, group in enumerate(layout) for q in group}
    conf = read_json(folder.parent/'reproduction'/'ibm_backend_snapshots'/'marrakesh'/'conf_marrakesh.json')
    edges = {tuple(sorted(pair)) for pair in conf['coupling_map']}
    for item in circuit.data:
        wires = [circuit.find_bit(q).index for q in item.qubits]
        if item.operation.name == 'cz':
            assert tuple(sorted(wires)) in edges
            assert block_of[wires[0]] == block_of[wires[1]]

    for name, expected in config['provenance']['source_sha256'].items():
        assert sha256(folder.parent/'reproduction'/name) == expected
    return {'configuration':folder.name, 'status':'passed', 'candidate_rows':rows,
            'generations_with_initialization':count, 'checked_stage_histograms':checked_stages,
            'state_preparation_roundoff_repairs':repaired_states, 'max_amplitude_repair':max_repair,
            'initial_population_sha256':initialization, 'elapsed_seconds':time.perf_counter()-start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    output = args.folder.resolve()
    plan = read_json(output/'study_plan.json')
    results = []
    for width in plan['widths']:
        paired = []
        for mode in plan['modes']:
            result = audit_case(output/f'q{width}_{mode}', plan)
            print(json.dumps(result), flush=True)
            results.append(result)
            paired.append(result['initial_population_sha256'])
        assert len(set(paired)) == 1, f'q{width}: paired initial populations differ'
    report = {'status':'passed', 'cases':len(results),
              'candidate_rows':sum(r['candidate_rows'] for r in results),
              'stage_histograms':sum(r['checked_stage_histograms'] for r in results),
              'paired_initializations_identical':True, 'configurations':results,
              'audit_script_sha256':sha256(Path(__file__))}
    (output/'export_audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='configurations'}), flush=True)


if __name__ == '__main__':
    main()
