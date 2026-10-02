"""Reproduce task 5/6 accuracy and speed checks; excludes grid construction."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from aeqga.likelihoods.pantheon_problem import PantheonPlusProblem
from aeqga.paths import output_path


def main():
    direct=PantheonPlusProblem(grid_size=0)
    points=np.random.default_rng(23).uniform([70,.3],[75,.42],size=(500,2))
    start=perf_counter()
    reference=np.array([direct.compute_fitness(x) for x in points])
    direct_seconds=perf_counter()-start
    result={'direct':direct.classical_minimum(),'direct_500_seconds':direct_seconds}
    for resolution in [100,300]:
        problem=PantheonPlusProblem(grid_size=resolution)
        start=perf_counter()
        values=np.array([problem.compute_fitness(x) for x in points])
        seconds=perf_counter()-start
        difference=values-reference
        result[str(resolution)]=dict(minimum=problem.classical_minimum(),
            seconds_500=seconds,speedup=direct_seconds/seconds,
            max_abs_delta_chi2=float(np.max(np.abs(difference))),
            rms_delta_chi2=float(np.sqrt(np.mean(difference**2))))
    target=output_path('json','sne_benchmark.json')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
