"""Add decoding diagnostics and an objective result summary to a completed study."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def decoding_rows(folder, summary):
    result=[]
    with gzip.open(folder/'execution_history.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            entry=json.loads(line)
            replacements={}
            numerator=denominator=any_random=both_random=offspring=0
            for stage in entry['stage_measurements']:
                counts=np.zeros(2**summary['width'],dtype=int)
                for bits,n in stage['counts'].items(): counts[int(bits,2)]=n
                used=stage['candidate_count']
                # Same threshold as AEQGA._decode, evaluated from saved counts.
                replace=(counts==0 if stage['branch']=='local' else
                         np.ones(len(counts),dtype=bool) if np.ptp(counts)==0 else counts==counts.min())[:used]
                numerator+=int(replace.sum()); denominator+=used
                replacements[(stage['subpopulation_index'],stage['branch'],stage['parameter_index'])]=replace
            for group in range(summary['config']['parallel_circuits']):
                for branch in ('local','global'):
                    first,second=(replacements[(group,branch,dimension)] for dimension in (0,1))
                    any_random+=int(np.logical_or(first,second).sum())
                    both_random+=int(np.logical_and(first,second).sum())
                    offspring+=len(first)
            result.append({'generation':entry['generation'],
                'random_replacement_parameter_fraction':numerator/denominator,
                'offspring_any_random_parameter_fraction':any_random/offspring,
                'offspring_both_random_parameters_fraction':both_random/offspring})
    return result


def diversity_rows(folder):
    rows=[]; seen=set(); current=set(); generation=0
    with (folder/'population_history.csv').open(newline='',encoding='utf-8') as stream:
        for candidate in csv.DictReader(stream):
            next_generation=int(candidate['generation'])
            if next_generation!=generation:
                new=len(current-seen); seen.update(current)
                rows.append({'generation':generation,'distinct_candidates':len(current),
                             'new_distinct_candidates':new,'cumulative_distinct_candidates':len(seen)})
                current=set();generation=next_generation
            current.add((float(candidate['H0_km_s_Mpc']),float(candidate['Omega_m'])))
    new=len(current-seen);seen.update(current)
    rows.append({'generation':generation,'distinct_candidates':len(current),
                 'new_distinct_candidates':new,'cumulative_distinct_candidates':len(seen)})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    output=parser.parse_args().folder.resolve()
    study=json.loads((output/'study_summary.json').read_text())
    plan=study['plan']; summaries=study['runs']
    comparisons=[];thresholds=[]
    for summary in summaries:
        folder=output/f'q{summary["width"]}_{summary["mode"]}'
        decoding=decoding_rows(folder,summary); diversity=diversity_rows(folder)
        save_csv(folder/'decoding_diagnostics.csv',decoding)
        save_csv(folder/'candidate_diversity.csv',diversity)
        with (folder/'best_history.csv').open(newline='',encoding='utf-8') as stream:
            history=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
        for threshold in (.01,.001,.0001):
            reached=next((row for row in history if row['delta_chi2']<=threshold),None)
            thresholds.append({'qubits_per_circuit':summary['width'],'mode':summary['mode'],
                'delta_chi2_threshold':threshold,'reached':reached is not None,
                **{key:'' if reached is None else reached[key] for key in
                   ('generation','fitness_evaluations','scheduled_shot_seconds','elapsed_wall_seconds')}})
        metrics=summary['mean_metrics']
        comparisons.append({'qubits_per_circuit':summary['width'],'mode':summary['mode'],
            'parallel_circuits':summary['config']['parallel_circuits'],
            'used_qubits':summary['used_active_qubits'],'candidates_per_circuit':summary['population_per_circuit'],
            'total_candidates':summary['total_population'],
            **dict(zip(('H0_km_s_Mpc','Omega_m','M_mag'),summary['best_parameters'])),
            'chi2':summary['chi2'],'delta_chi2':summary['delta_chi2'],
            'initial_delta_chi2':summary['initial_delta_chi2'],
            'candidate_evaluation_requests':summary['fitness_evaluations'],
            'distinct_candidates_across_run':diversity[-1]['cumulative_distinct_candidates'],
            **{key:metrics[key] for key in ('mean_noise_tvd','mean_sampling_tvd','mean_zero_fraction',
                                           'mean_cz_per_group','mean_depth_per_group')},
            **{key:float(np.mean([row[key] for row in decoding])) for key in decoding[0] if key!='generation'},
            'scheduled_gate_shot_seconds':summary['shots_times_scheduled_duration_total'],
            'local_training_seconds':summary['training_wall_seconds']})
        print(f'Analyzed {folder.name}',flush=True)
    save_csv(output/'tradeoff_metrics.csv',comparisons)
    save_csv(output/'threshold_statistics.csv',thresholds)
    noisy=[r for r in comparisons if r['mode']=='noisy']
    ideal={r['qubits_per_circuit']:r for r in comparisons if r['mode']=='ideal'}
    best=min(noisy,key=lambda row:row['delta_chi2'])
    lines=['OBSERVED RESULTS: MARRAKESH CIRCUIT-WIDTH STUDY','',
        f'All {len(summaries)} configurations completed {plan["generations"]} updates with {plan["shots"]} shots per packed execution.',
        f'The random seed is {plan["seed"]}. Noise and ideal pairs share identical initial populations.',
        f'Archived Marrakesh calibration timestamp: {plan["snapshot_date"]}.', '',
        'M is calculated for each (H0, Omega_m) candidate by full-covariance least squares:',
        'M_raw = [1^T C^-1 (m_b_corr - mu(H0, Omega_m))] / [1^T C^-1 1].',
        'M = clip(M_raw, -19.6, -19.0). The quantum search has two coordinates.',
        'Calibrator distance moduli and all covariance correlations are retained.',
        'The optimum is conditional on each candidate; M is not fixed to one number for the entire search.', '',
        'The independent classical reference is: '+json.dumps(plan['classical_reference']),
        'Delta chi-square below is chi-square minus this reference, not uncertainty in a fitted parameter.', '',
        'Final results (noisy):',
        'q groups used_qubits candidates H0 Omega_m M chi2 excess_chi2 ideal_excess_chi2']
    for row in noisy:
        q=row['qubits_per_circuit']
        lines.append(f'{q} {row["parallel_circuits"]} {row["used_qubits"]} {row["total_candidates"]} '
            f'{row["H0_km_s_Mpc"]:.10f} {row["Omega_m"]:.10f} {row["M_mag"]:.10f} '
            f'{row["chi2"]:.12f} {row["delta_chi2"]:.9g} {ideal[q]["delta_chi2"]:.9g}')
    lines+=['','Noise, sampling and decoding (means across the 100 updates):',
        'q noise_TVD sampling_TVD zero_bin_fraction random_parameter_fraction any_random_offspring_fraction both_random_offspring_fraction mean_CZ_per_group scheduled_gate_shot_seconds']
    for row in noisy:
        lines.append(str(row['qubits_per_circuit'])+' '+' '.join(f'{row[key]:.8g}' for key in
            ('mean_noise_tvd','mean_sampling_tvd','mean_zero_fraction','random_replacement_parameter_fraction',
             'offspring_any_random_parameter_fraction','offspring_both_random_parameters_fraction',
             'mean_cz_per_group','scheduled_gate_shot_seconds')))
    lines+=['',f'The lowest final noisy excess chi-square in this seed is {best["delta_chi2"]:.9g}, at {best["qubits_per_circuit"]} qubits per circuit.',
        'This is a description of this run, not an estimate of the optimal width over repeated trials.', '',
        'Interpretation of the measurements:',
        '- Noise TVD compares exact noisy and ideal probability vectors for the same compiled circuit.',
        '- Sampling TVD compares the 100-shot histogram with that circuit\'s noisy probability vector.',
        '- TVD ranges from 0 to 1. It is a distribution difference, not a gate error probability.',
        '- Zero-bin fraction is averaged over retained amplitude bins in each stage.',
        '- The existing decoder replaces lower-bound outcomes with classical uniform random draws.',
        '  Local draws use the current local box; global draws use the full parameter bounds.',
        '  Consequently, zero counts can increase classical random exploration. They do not remove candidates.',
        '- Random-parameter and offspring fractions are weighted by the number of decoded offspring.',
        '- Circuit width changes candidate population size as well as circuit depth, placement and noise.',
        '  It also changes the number of elites per group and therefore the local search boxes.',
        '  The local decoder is lo + (hi - lo) * sqrt(count / shots). For L equal amplitudes,',
        '  with no intervening mutation/crossover gates, its infinite-shot fraction is 1/sqrt(L).',
        '  Thus the proposal distribution itself depends on width, even before hardware noise.',
        '  See threshold_statistics.csv for generations, evaluations and time to the same objective threshold.',
        '- Candidate evaluation requests count duplicate candidates; cached duplicates do not require a new solve.',
        '  candidate_diversity.csv separately counts exact distinct (H0, Omega_m) pairs.',
        '- Scheduled gate-time times shots excludes queueing, communication and classical-feedback overhead.',
        '  Local training seconds measure this computer and simulator, not a quantum processor.', '',
        'Scope: independent local channels from the archived calibration; no crosstalk, leakage, drift,',
        'or separately calibrated reset infidelity. Independent reset stages are simulated as small',
        'density matrices, then exactly 100 finite-shot outcomes are sampled for each stage.',
        'These are local simulations and contain no hardware executions.', '',
        'Main outputs: tradeoff_overview.png, search_efficiency.png, q*_parameter_histories.png,',
        'tradeoff_metrics.csv, threshold_statistics.csv, report.txt, export_audit.json,',
        'and q*_*/population_history.csv plus full training/execution JSONL and probability arrays.']
    (output/'findings.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
