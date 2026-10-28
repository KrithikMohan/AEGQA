"""Compare 3--8 qubit complete populations with profiled M and Marrakesh noise.

Runs locally. No credentials, QPU submissions, sessions, or hardware fees.
"""
from __future__ import annotations

import os
for _variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_variable] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from qiskit import qpy
from threadpoolctl import threadpool_limits

from aeqga_profiled_pantheon import ProfiledPantheon
from aeqga_profiled_ibm import ProfiledIBMNoiseAEQGA
from aeqga_ibm_noise import DEFAULT_SNAPSHOTS

ROOT = Path(__file__).resolve().parent
COUNTS = {3:52, 4:38, 5:31, 6:26, 7:22, 8:19}
SOURCE_FILES = ('aeqga.py','aeqga_legacy.py','aeqga_parallel.py','aeqga_ibm_noise.py',
                'aeqga_ibm_noise_legacy.py','aeqga_profiled_pantheon.py',
                'aeqga_factorized_noise.py','aeqga_profiled_ibm.py',
                'run_pantheon_aeqga.py','run_marrakesh_profiled_study.py',
                'test_profiled_marrakesh.py','state_preparation_regression.json')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def worker_init():
    global _worker_threads
    _worker_threads = threadpool_limits(limits=1)


def run_case(problem, executor, output, width, mode, args, reference, provenance):
    folder = output / f'q{width}_{mode}'
    config = {'qubits_per_circuit':width, 'active_qubits_requested':156,
              'parallel_circuits':COUNTS[width], 'generations':args.generations,
              'shots':args.shots, 'seed':args.seed, 'mode':mode, 'provenance':provenance}
    if args.resume and (folder/'summary.json').exists():
        saved = json.loads((folder/'summary.json').read_text())
        if saved['config'] != config:
            raise ValueError(f'Existing {folder.name} configuration differs; use a new output directory.')
        print(f'SKIP completed {folder.name}', flush=True)
        return saved
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder/'config.json', config)
    optimizer = ProfiledIBMNoiseAEQGA(problem, executor=executor, record_directory=folder,
        active_qubits=156, parallel_circuits=COUNTS[width], mode=mode,
        shots=args.shots, seed=args.seed)
    assert optimizer.qubits_per_circuit == width
    write_json(folder/'physical_layout.json', {'physical_groups':optimizer.placements,
        'requested_active_qubits':156, 'used_active_qubits':optimizer.used_active_qubits,
        'unused_physical_qubits':sorted(set(range(156))-set(sum(optimizer.placements, []))),
        'population_per_circuit':optimizer.population_per_circuit,
        'total_candidates':optimizer.population_size})
    with threadpool_limits(limits=args.objective_threads):
        optimizer.train(generations=args.generations)
        checked = []
        for entry in optimizer.history_:
            sample = entry['population'][entry['best_index']]
            checked.append(abs(problem.direct_check(sample['p'],sample['M_profiled'])-sample['chi2']))
    if max(checked) > 1e-7:
        raise AssertionError(f'Independent high-order chi-square error: {max(checked)}')
    assert all(len(h['population']) == optimizer.population_size for h in optimizer.history_)
    assert len(optimizer.history_) == args.generations+1
    assert optimizer.executed_circuits_ == args.generations
    assert np.all(np.diff([h['loss'] for h in optimizer.history_]) <= 1e-10)
    expected_evaluations = optimizer.population_size + args.generations*COUNTS[width]*(
        optimizer.local_count+optimizer.random_count)
    assert optimizer.n_evaluations_ == expected_evaluations
    optimizer.export_population_history(folder/'population_history.csv')
    probability_values = np.asarray(optimizer.probability_history_)
    np.savez_compressed(folder/'stage_probabilities.npz', probabilities=probability_values,
        axes=np.array(['generation_minus_one','subpopulation','parameter_branch_stage','ideal_or_mode','basis_state']),
        probability_modes=np.array(['ideal','selected_mode']))
    with (folder/'last_generation.qpy').open('wb') as stream:
        qpy.dump(optimizer.execution_circuits_, stream)
    write_json(folder/'last_generation_layout.json', optimizer.execution_history_[-1]['executions'][0]['blocks'])
    histories, metrics = [], []
    for entry in optimizer.history_:
        histories.append({'generation':entry['generation'], 'H0_km_s_Mpc':entry['p'][0],
            'Omega_m':entry['p'][1], 'M_mag':entry['M_profiled'], 'chi2':entry['loss'],
            'delta_chi2':entry['loss']-reference['chi2'], 'fitness_evaluations':entry['fitness_evaluations'],
            'elapsed_wall_seconds':entry['elapsed_wall_seconds'],
            'objective_seconds':entry['objective_seconds_cumulative'],
            'quantum_wall_seconds':entry['quantum_wall_seconds_cumulative'],
            'scheduled_shot_seconds':entry['shots_times_scheduled_duration_cumulative']})
    for entry in optimizer.execution_history_:
        execution = entry['executions'][0]
        stages, blocks = entry['stage_diagnostics'], execution['blocks']
        assert sum(execution['counts'].values()) == args.shots
        assert len(stages) == 4*COUNTS[width]
        assert all(sum(stage['counts'].values()) == args.shots for stage in entry['stage_measurements'])
        metrics.append({'generation':entry['generation'],
            'mean_noise_tvd':np.mean([s['noise_tvd'] for s in stages]),
            'mean_sampling_tvd':np.mean([s['sampling_tvd'] for s in stages]),
            'mean_zero_fraction':np.mean([s['zero_fraction_used_candidates'] for s in stages]),
            'mean_unused_amplitude_probability':np.mean([s['unused_amplitude_probability'] for s in stages]),
            'max_probability_trace_error':max(s['probability_trace_error'] for s in stages),
            'mean_cz_per_group':np.mean([b['gate_counts'].get('cz',0) for b in blocks]),
            'mean_depth_per_group':np.mean([b['depth'] for b in blocks]),
            'maximum_scheduled_circuit_seconds':execution['duration_seconds'],
            'shots_times_scheduled_duration':execution['shots_times_scheduled_duration']})
    for filename, rows in (('best_history.csv',histories), ('generation_metrics.csv',metrics)):
        with (folder/filename).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    thresholds = {}
    for threshold in (.1,.01,.001):
        reached = next((h for h in histories if h['delta_chi2'] <= threshold), None)
        thresholds[str(threshold)] = (None if reached is None else {
            key:reached[key] for key in ('generation','fitness_evaluations','elapsed_wall_seconds','scheduled_shot_seconds')})
    summary = {'config':config, 'width':width, 'mode':mode,
        'used_active_qubits':optimizer.used_active_qubits, 'discarded_qubits':optimizer.discarded_qubits,
        'population_per_circuit':optimizer.population_per_circuit,'total_population':optimizer.population_size,
        'per_group_elite_local_global':[optimizer.elite_count,optimizer.local_count,optimizer.random_count],
        'best_parameters':optimizer.history_[-1]['all_parameters'], 'chi2':optimizer.loss_,
        'delta_chi2':optimizer.loss_-reference['chi2'], 'initial_delta_chi2':histories[0]['delta_chi2'],
        'classical_reference':reference, 'fitness_evaluations':optimizer.n_evaluations_,
        'candidate_rows':len(optimizer.history_)*optimizer.population_size,
        'M_boundary_rows':sum(p['M_at_bound'] for h in optimizer.history_ for p in h['population']),
        'maximum_independent_best_chi2_error':max(checked),
        'maximum_probability_trace_error':max(m['max_probability_trace_error'] for m in metrics),
        'training_wall_seconds':histories[-1]['elapsed_wall_seconds'],
        'objective_wall_seconds':optimizer.objective_seconds,
        'factorized_quantum_wall_seconds':optimizer.quantum_wall_seconds,
        'compile_worker_seconds':optimizer.compile_cpu_seconds,
        'simulation_worker_seconds':optimizer.simulation_cpu_seconds,
        'shots_times_scheduled_duration_total':optimizer.scheduled_shot_seconds,
        'logical_packed_executions':optimizer.executed_circuits_,
        'full_subpopulation_circuits':optimizer.quantum_circuits_,
        'stages':optimizer.quantum_stages_, 'thresholds':thresholds,
        'mean_metrics':{key:float(np.mean([row[key] for row in metrics])) for key in metrics[0] if key!='generation'},
        'validation':'passed'}
    write_json(folder/'summary.json',summary)
    print(f'COMPLETE {folder.name} delta_chi2={summary["delta_chi2"]:.9g} '
          f'seconds={summary["training_wall_seconds"]:.1f}',flush=True)
    return summary


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as stream:
        return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]


def plot_results(output, summaries, args):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    colors = {3:'#2563eb',4:'#0f766e',5:'#b45309',6:'#9333ea',7:'#be123c',8:'#475569'}
    fig = Figure(figsize=(16,12),layout='constrained'); FigureCanvasAgg(fig)
    axes = fig.subplots(2,3)
    by_mode = {mode:sorted([s for s in summaries if s['mode']==mode],key=lambda s:s['width'])
               for mode in ('ideal','noisy')}
    for mode, color in [('ideal','#64748b'),('noisy','#b91c1c')]:
        rows=by_mode[mode]
        if not rows: continue
        axes[0,0].plot([s['width'] for s in rows], [max(s['delta_chi2'],1e-9) for s in rows],
                       'o-',color=color,label=mode)
    axes[0,0].set(xlabel='Qubits per complete circuit',ylabel='Final excess chi-square',
                  title=f'A. Final objective after {args.generations} updates',yscale='log')
    axes[0,0].legend()
    for s in by_mode['noisy']:
        rows=read_csv(output/f'q{s["width"]}_noisy'/'best_history.csv')
        axes[0,1].plot([r['generation'] for r in rows],[max(r['delta_chi2'],1e-9) for r in rows],
                       color=colors[s['width']],label=f'{s["width"]}q')
    axes[0,1].set(xlabel='Generation (0 = initialization)',ylabel='Excess chi-square',
                  title='B. Noisy-search progress',yscale='log')
    axes[0,1].legend(ncol=2)
    noisy=by_mode['noisy']; widths=[s['width'] for s in noisy]
    for key,label,marker in [('mean_noise_tvd','Exact noise distortion','o'),
                             ('mean_sampling_tvd',f'{args.shots}-shot sampling distortion','s')]:
        axes[0,2].plot(widths,[s['mean_metrics'][key] for s in noisy],marker+'-',label=label)
    axes[0,2].set(xlabel='Qubits per complete circuit',ylabel='Mean total variation distance',
                  title='C. Noise and finite-shot effects'); axes[0,2].legend(fontsize=9)
    for mode, color in [('ideal','#64748b'),('noisy','#b91c1c')]:
        rows=by_mode[mode]
        axes[1,0].plot([s['width'] for s in rows], [s['mean_metrics']['mean_zero_fraction'] for s in rows],
                       'o-',label=mode,color=color)
    axes[1,0].set(xlabel='Qubits per complete circuit',ylabel='Fraction of retained bins with zero counts',
                  title='D. Sampling sparsity'); axes[1,0].legend()
    axes[1,1].plot(widths,[s['mean_metrics']['mean_cz_per_group'] for s in noisy],'o-',color='#2563eb')
    axes[1,1].set(xlabel='Qubits per complete circuit',ylabel='Mean native CZ gates / complete circuit',
                  title='E. Circuit cost (four stages)',yscale='log')
    for s in noisy:
        axes[1,2].scatter(s['shots_times_scheduled_duration_total'],max(s['delta_chi2'],1e-9),
                          color=colors[s['width']],s=70)
        axes[1,2].annotate(f'{s["width"]}q', (s['shots_times_scheduled_duration_total'],max(s['delta_chi2'],1e-9)),
                          xytext=(5,5),textcoords='offset points',color=colors[s['width']])
    axes[1,2].set(xlabel='Sum of shots x scheduled packed duration [s]',ylabel='Final excess chi-square',
                  title='F. Objective versus scheduled gate-time cost',yscale='log')
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle(f'Marrakesh snapshot | Profiled M | 156-qubit budget | '
                 f'{args.generations} updates x {args.shots} shots | seed {args.seed}',fontsize=16)
    fig.savefig(output/'tradeoff_overview.png',dpi=180)
    fig.savefig(output/'tradeoff_overview.svg')
    efficiency=Figure(figsize=(13,9),layout='constrained'); FigureCanvasAgg(efficiency)
    ee=efficiency.subplots(2,2)
    comparisons=[('generation','Generation'),('fitness_evaluations','Cumulative candidate evaluations'),
                 ('scheduled_shot_seconds','Cumulative shots x scheduled duration [s]'),
                 ('elapsed_wall_seconds','Local simulator / CPU wall time [s]')]
    for s in noisy:
        rows=read_csv(output/f'q{s["width"]}_noisy'/'best_history.csv')
        for ax,(key,label) in zip(ee.flat,comparisons):
            valid=[r for r in rows if key=='generation' or r[key]>0]
            ax.plot([r[key] for r in valid],[max(r['delta_chi2'],1e-9) for r in valid],
                    color=colors[s['width']],label=f'{s["width"]}q / {s["total_population"]} candidates')
            ax.set(xlabel=label,ylabel='Excess chi-square',yscale='log')
            if key!='generation': ax.set_xscale('log')
            ax.grid(alpha=.2)
    ee[0,0].legend(fontsize=8)
    efficiency.suptitle('Noisy search efficiency | Different denominators answer different cost questions',fontsize=14)
    efficiency.savefig(output/'search_efficiency.png',dpi=180)
    efficiency.savefig(output/'search_efficiency.svg')
    for q in sorted({s['width'] for s in summaries}):
        f=Figure(figsize=(10,10),layout='constrained'); FigureCanvasAgg(f); aa=f.subplots(4,1,sharex=True)
        for mode,color in [('ideal','#64748b'),('noisy','#b91c1c')]:
            path=output/f'q{q}_{mode}'/'best_history.csv'
            if not path.exists(): continue
            rows=read_csv(path)
            for ax,key in zip(aa,['H0_km_s_Mpc','Omega_m','M_mag','delta_chi2']):
                ax.plot([r['generation'] for r in rows],[max(r[key],1e-9) if key=='delta_chi2' else r[key] for r in rows],
                        color=color,label=mode)
                ax.set_ylabel(key); ax.grid(alpha=.2)
        aa[-1].set_yscale('log'); aa[-1].set_xlabel('Generation'); aa[0].legend()
        f.suptitle(f'{q} qubits per circuit | M is derived analytically')
        f.savefig(output/f'q{q}_parameter_histories.png',dpi=160)


def finalize(output, summaries, args, plan):
    summaries=sorted(summaries,key=lambda s:(s['width'],s['mode']))
    write_json(output/'study_summary.json', {'plan':plan,'runs':summaries})
    fields=['width','mode','used_active_qubits','total_population','chi2','delta_chi2',
            'initial_delta_chi2','fitness_evaluations','candidate_rows','training_wall_seconds',
            'shots_times_scheduled_duration_total']
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        writer.writerows({k:s[k] for k in fields} for s in summaries)
    plot_results(output,summaries,args)
    lines=['MARRAKESH CIRCUIT-WIDTH STUDY WITH ANALYTICALLY PROFILED M','',
        'Settings: '+json.dumps({k:plan[k] for k in ('generations','shots','seed','requested_active_qubits')}),
        'Reference: '+json.dumps(plan['classical_reference']), '',
        'Each run searches H0 and Omega_m. M minimizes the same full-covariance chi-square,',
        'subject to the original [-19.6, -19.0] bounds. It is not independently encoded.',
        'Every subpopulation has elite/local/global proportions 1:1:2. Elites are selected',
        'globally each generation and distributed round-robin.', '',
        'The Marrakesh snapshot is dated '+plan['snapshot_date']+'. These are local simulations.',
        '3q and 6q configurations occupy all 156 qubits. 4q uses 38 groups (152 qubits):',
        'enumeration of all 354 connected four-qubit subsets and an integer-programming',
        'optimum prove that 39 disjoint groups are impossible on this calibrated graph.',
        '5q uses 155, 7q uses 154, and 8q uses 152 qubits under integer division.', '',
        'Independent small density matrices give exact stage probabilities for the selected',
        'local Markovian noise model. Exactly the requested shot count is then sampled.',
        'Packed physical circuits and physical layouts are preserved; probability evolution',
        'is factorized across disjoint groups and full-reset stages, not a dense 156-qubit state.',
        'The model includes calibrated native gates, relaxation during scheduled idle periods,',
        'and asymmetric readout. It excludes crosstalk, leakage, drift, and uncalibrated reset error.', '',
        'Long channel compositions have floating-point trace drift. Every stage logs the',
        'pre-normalization error; vectors are normalized only below the 1e-5 rejection limit.',
        'Each run summary reports the maximum observed error. This numerical correction is',
        'separate from the physical noise and the finite-shot sampling statistics.', '',
        'Qiskit state-preparation isometry failures caused by roundoff are retried only',
        'after a uniform amplitude rescaling of less than 1e-14 per component.',
        'This preserves the normalized target state. Repaired instructions are logged',
        'under state_preparation_roundoff_repairs in the affected execution block.', '',
        'Scheduled duration x shots is a gate-time estimate, not measured QPU wall time.',
        'It excludes queueing, network/classical feedback, and controller overhead beyond',
        'the gate durations in the snapshot. Local simulation wall time is reported separately.',
        'Comparisons use one seed. They do not establish a statistically reliable best width.',
        'Population sizes, sampled physical regions, routing, and sampling sparsity change',
        'with width; a noisy/ideal pair controls noise within each configuration.', '',
        'q mode used_qubits candidates initial_delta final_delta simulated_wall_s scheduled_shot_s']
    for s in summaries:
        lines.append(f'{s["width"]} {s["mode"]} {s["used_active_qubits"]} {s["total_population"]} '
                     f'{s["initial_delta_chi2"]:.9g} {s["delta_chi2"]:.9g} '
                     f'{s["training_wall_seconds"]:.3f} {s["shots_times_scheduled_duration_total"]:.6g}')
    lines += ['', 'Files: population_history.csv has every candidate including generation 0.',
              'training_history.jsonl.gz adds global elite assignments and complete parameters.',
              'execution_history.jsonl.gz contains actual sampled counts, stages, seeds, and gate metrics.',
              'stage_probabilities.npz stores ideal and selected-mode exact probability vectors.',
              'last_generation.qpy is the final packed physical circuit, not a hardware submission.',
              'The reference solution is calculated for analysis and never seeds the populations.']
    (output/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    reproduction=output/'reproduction'; reproduction.mkdir(exist_ok=True)
    for name in SOURCE_FILES:
        if (ROOT/name).exists(): shutil.copy2(ROOT/name,reproduction/name)
    shutil.copytree(DEFAULT_SNAPSHOTS/'marrakesh',reproduction/'ibm_backend_snapshots'/'marrakesh',dirs_exist_ok=True)
    data_root=ROOT/'Pantheon+_Data'/'4_DISTANCES_AND_COVAR'
    data_target=reproduction/'Pantheon+_Data'/'4_DISTANCES_AND_COVAR'; data_target.mkdir(parents=True,exist_ok=True)
    for name in ('Pantheon+SH0ES.dat','Pantheon+SH0ES_STAT+SYS.cov','README'):
        shutil.copy2(data_root/name,data_target/name)
    for source in (ROOT/'tmp'/'marrakesh_4q_layout.json',ROOT/'marrakesh_4q_layout.json'):
        if source.exists(): shutil.copy2(source,reproduction/'marrakesh_4q_layout.json'); break
    (reproduction/'requirements.txt').write_text('\n'.join(
        f'{p}=={importlib.metadata.version(p)}' for p in
        ('numpy','scipy','qiskit','qiskit-aer','pandas','matplotlib','threadpoolctl'))+'\n')
    (reproduction/'README.txt').write_text(
        'Python 3.11. Install requirements.txt, then run:\n'
        'python run_marrakesh_profiled_study.py --generations 100 --shots 100 --seed 7 --workers 24\n'
        'Tests: python -m unittest -v test_profiled_marrakesh\n'
        '--resume skips completed configurations with matching settings and source hashes.\n'
        'Worker count changes local runtime; it does not change candidate or shot counts.\n',encoding='utf-8')
    files=sorted(p for p in output.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
    (output/'SHA256SUMS.txt').write_text('\n'.join(f'{sha256(p)}  {p.relative_to(output).as_posix()}' for p in files)+'\n')
    archive_path=output.with_suffix('.zip')
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file(): archive.write(path,f'{output.name}/{path.relative_to(output).as_posix()}')
    with zipfile.ZipFile(archive_path) as archive: assert archive.testzip() is None
    print(f'BUNDLE {archive_path} bytes={archive_path.stat().st_size} sha256={sha256(archive_path)}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--widths',nargs='+',type=int,default=list(range(3,9)))
    parser.add_argument('--modes',nargs='+',choices=['noisy','ideal'],default=['noisy','ideal'])
    parser.add_argument('--generations',type=int,default=100)
    parser.add_argument('--shots',type=int,default=100)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--workers',type=int,default=24)
    parser.add_argument('--objective-threads',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'output'/'marrakesh_profiled_156q')
    args=parser.parse_args()
    if not set(args.widths).issubset(COUNTS) or min(args.generations,args.shots,args.workers,args.objective_threads)<1:
        parser.error('Widths must be 3--8; generations, shots, and worker counts must be positive.')
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=args.objective_threads):
        problem=ProfiledPantheon(); reference=problem.reference()
    snapshot=DEFAULT_SNAPSHOTS/'marrakesh'
    provenance={'source_sha256':{name:sha256(ROOT/name) for name in SOURCE_FILES if (ROOT/name).exists()},
        'snapshot_sha256':{name:sha256(snapshot/name) for name in ('conf_marrakesh.json','props_marrakesh.json')},
        'data_sha256':{name:sha256(ROOT/'Pantheon+_Data'/'4_DISTANCES_AND_COVAR'/name)
                       for name in ('Pantheon+SH0ES.dat','Pantheon+SH0ES_STAT+SYS.cov')}}
    plan={'generations':args.generations,'shots':args.shots,'seed':args.seed,'requested_active_qubits':156,
        'workers':args.workers,'objective_threads':args.objective_threads,'widths':args.widths,'modes':args.modes,
        'snapshot_date':json.loads((snapshot/'props_marrakesh.json').read_text())['last_update_date'],
        'classical_reference':reference,'provenance':provenance,
        'configurations':[{'width':q,'circuits':COUNTS[q],'used_qubits':q*COUNTS[q],
                           'total_candidates':COUNTS[q]*2**q} for q in args.widths]}
    write_json(output/'study_plan.json',plan)
    summaries=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init) as executor:
        for width in args.widths:
            for mode in args.modes:
                summaries.append(run_case(problem,executor,output,width,mode,args,reference,provenance))
    finalize(output,summaries,args,plan)


if __name__=='__main__':
    main()
