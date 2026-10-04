"""Separate CMB+BAO experiment; never includes the SNe objective."""
import argparse
import json
from pathlib import Path
from importlib.metadata import version
import numpy as np
from scipy.optimize import minimize
from aeqga.steps.evolution.aeqga_algorithm import AEQGAParameters
from aeqga.likelihoods.bao_cmb_problem import BAOCMBProblem
from aeqga.paths import PROJECT_ROOT, output_path
from aeqga.experiments.runner import run_ensemble, file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['pico','camb','hybrid'],default='pico')
    parser.add_argument('--data',default=str(PROJECT_ROOT/'data/cmb'))
    parser.add_argument('--ell-min',type=int,default=2)
    parser.add_argument('--error-mode',choices=['average','asymmetric'],default='average')
    parser.add_argument('--pop',type=int,default=32)
    parser.add_argument('--gen',type=int,default=50)
    parser.add_argument('--shots',type=int,default=4096)
    parser.add_argument('--runs',type=int,default=1)
    parser.add_argument('--seed',type=int,default=23)
    parser.add_argument('--engine',choices=['aer','statevector_shots'],default='statevector_shots')
    parser.add_argument('--classical-only',action='store_true')
    parser.add_argument('--output',default=str(output_path('json','bao_cmb.json')))
    args = parser.parse_args()
    if args.runs < 1: parser.error('runs must be positive')
    problem = BAOCMBProblem(data_dir=args.data,backend=args.backend,
                            ell_min=args.ell_min,error_mode=args.error_mode)
    # Local independent baseline, not a claim of a proven global minimum.
    baseline = minimize(problem.compute_fitness,[66.76,.323],method='Nelder-Mead',
                        bounds=[(60,80),(0,.5)],
                        options={'xatol':1e-5,'fatol':1e-5,'maxiter':250})
    result = dict(backend=args.backend,ell_min=args.ell_min,error_mode=args.error_mode,
                  baseline=baseline.x.tolist(),chi2=float(baseline.fun),
                  baseline_success=bool(baseline.success))
    if not args.classical_only:
        inputs = [Path(args.data)/'COM_PowerSpect_CMB-TT-full_R3.01.txt']
        if args.backend != 'camb': inputs.append(Path(args.data)/'pico4_tailmonty_v35_py3.dat')
        provenance = dict(objective='CMB+BAO only / diagonal TT reference',
                          backend=args.backend,ell_min=args.ell_min,error_mode=args.error_mode,
                          camb_version=version('camb'),
                          datasets={p.name:file_digest(p) for p in inputs})
        seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(args.seed).spawn(args.runs)]
        experiment = run_ensemble(problem,AEQGAParameters(pop_size=args.pop,max_gen=args.gen,
                                  num_shots=args.shots,progress_bar=False),seeds,
                                  Path(args.output).with_suffix('.checkpoint.json'),
                                  provenance=provenance,execution_mode=args.engine)
        result['ensemble'] = {k:v for k,v in experiment.items() if k != 'runs'}
        result['aeqga_runs'] = [dict(seed=r['seed'],parameters=r['best_x'],fitness=r['best_chi2'],
                                   evaluations=r['evaluations'],best_history=r['best_history'],
                                   best_parameter_history=r['best_parameter_history']) for r in experiment['runs']]
        result['exact_paper_reproduction'] = False
    result['camb_fallback_evaluations'] = problem.cmb.fallback_evaluations
    destination=Path(args.output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
