"""Tasks 8–12: resumable real-data ensemble and shared scientific figures."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from aeqga.experiments.runner import run_ensemble, atomic_json, file_digest
from aeqga.likelihoods.pantheon_problem import PantheonPlusProblem
from aeqga.steps.evolution.aeqga_algorithm import AEQGAParameters
from aeqga.paths import output_path
from aeqga.visualization.scientific_plots import (
    run_diagnostics, objective_contours, optimizer_scatter, export_figure,
)


def sne_provenance(problem):
    data = problem._data
    return dict(objective="calibrated Pantheon+ MU_SH0ES / full STAT+SYS covariance",
                selection=data["selection"], redshift=data["redshift"],
                grid_size=problem.grid_size, n_rows=data["n_sn"],
                bounds=[problem.lower_bounds.tolist(),problem.upper_bounds.tolist()],
                datasets={key: dict(path=data[key],sha256=file_digest(data[key]))
                          for key in ["table_file","covariance_file"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=int,default=300)
    parser.add_argument("--pop",type=int,default=32)
    parser.add_argument("--gen",type=int,default=50)
    parser.add_argument("--shots",type=int,default=4096)
    parser.add_argument("--seed",type=int,default=23)
    parser.add_argument("--grid",type=int,choices=[0,100,300],default=300)
    parser.add_argument("--engine",choices=["aer","statevector_shots"],default="statevector_shots")
    parser.add_argument("--name",default="sne_production")
    args = parser.parse_args()
    if args.runs < 1: parser.error("runs must be positive")
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        parser.error("name must contain only letters, digits, underscore or hyphen")
    problem = PantheonPlusProblem(grid_size=args.grid)
    params = AEQGAParameters(pop_size=args.pop,max_gen=args.gen,num_shots=args.shots,progress_bar=False)
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(args.seed).spawn(args.runs)]
    result = run_ensemble(problem,params,seeds,output_path("json",args.name+"_checkpoint.json"),
                          provenance=sne_provenance(problem),execution_mode=args.engine,
                          progress=lambda n,total: print(f"Completed {n}/{total}",flush=True) if n%10==0 or n==total else None)
    reference = problem.classical_minimum()
    points = np.array([r["best_x"] for r in result["runs"]])
    figures = [(run_diagnostics(result["runs"][0],reference),"convergence"),
               (objective_contours(problem,reference),"likelihood"),
               (objective_contours(problem,reference,points=points),"overlay")]
    if args.runs >= 3: figures.append((optimizer_scatter(points,reference),"scatter"))
    if args.runs >= 20: figures.append((optimizer_scatter(points,reference,kde_regions=True),"density"))
    for fig,kind in figures:
        export_figure(fig,args.name+"_"+kind,provenance={
            "fingerprint":result["fingerprint"],"configuration":result["configuration"]})
        plt.close(fig)
    chi2 = np.array([r["best_chi2"] for r in result["runs"]])
    summary = {k:v for k,v in result.items() if k != "runs"}
    summary.update(classical_reference=reference,
                   best_points=points.tolist(), best_chi2=chi2.tolist(),
                   convergence_gap_quantiles=np.quantile(chi2-reference["chi2"],[0,.5,.9,1]).tolist(),
                   fraction_delta_chi2_below_2_30=float(np.mean(chi2-reference["chi2"]<2.30)),
                   total_optimizer_evaluations=sum(r["evaluations"] for r in result["runs"]),
                   max_history_abs_error=max(r["history_max_abs_error"] for r in result["runs"]),
                   paper_sne_mean=[72.81,.362], paper_sne_std=[.22,.016],
                   exact_paper_reproduction=False)
    atomic_json(output_path("json",args.name+"_summary.json"),summary)
    print("Mean / std [H0, Omega_m]:",summary["mean"],summary["std_ddof0"])
    print("Gap quantiles / coverage:",summary["convergence_gap_quantiles"],summary["fraction_delta_chi2_below_2_30"])


if __name__ == "__main__": main()
