"""Separate CMB+BAO experiment; never includes the SNe objective."""
import argparse
import json
from pathlib import Path
from scipy.optimize import minimize
from aeqga.steps.evolution.aeqga_algorithm import AEQGAParameters, run_aeqga_dual
from aeqga.likelihoods.bao_cmb_problem import BAOCMBProblem
from aeqga.paths import PROJECT_ROOT, output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['pico','camb','hybrid'],default='pico')
    parser.add_argument('--data',default=str(PROJECT_ROOT/'data/cmb'))
    parser.add_argument('--ell-min',type=int,default=2)
    parser.add_argument('--error-mode',choices=['average','asymmetric'],default='average')
    parser.add_argument('--pop',type=int,default=32)
    parser.add_argument('--gen',type=int,default=50)
    parser.add_argument('--shots',type=int,default=4096)
    parser.add_argument('--classical-only',action='store_true')
    parser.add_argument('--output',default=str(output_path('json','bao_cmb.json')))
    args = parser.parse_args()
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
        best, _, _ = run_aeqga_dual(problem,AEQGAParameters(pop_size=args.pop,
                    max_gen=args.gen,num_shots=args.shots))
        # Core returns a GlobalBest object (history includes generation zero).
        result['aeqga'] = dict(parameters=best.x.tolist(),fitness=float(best.fitness),
                               evaluations=best.evaluations)
    result['camb_fallback_evaluations'] = problem.cmb.fallback_evaluations
    destination=Path(args.output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
