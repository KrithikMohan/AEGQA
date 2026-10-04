"""Validated objective contours and distinctly labeled optimizer diagnostics."""
import numpy as np
from scipy.stats import chi2, norm, gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def joint_levels(sigmas=(1, 2, 3, 4, 5)):
    """Two-parameter likelihood thresholds enclosing 1D Gaussian sigma masses."""
    masses = norm.cdf(sigmas)-norm.cdf(-np.asarray(sigmas))
    return chi2.ppf(masses, df=2)


def outcome_statistics(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.all(np.isfinite(points)):
        raise ValueError("Need at least three finite two-dimensional outcomes")
    covariance = np.cov(points.T, ddof=1)
    if np.linalg.matrix_rank(covariance) != 2:
        raise ValueError("Optimizer outcomes are singular; no covariance ellipse/KDE")
    return points.mean(axis=0), covariance


def density_contours(points, masses=(.682689492, .954499736, .997300204), grid_size=300):
    """KDE highest-density regions by INTEGRATED MASS, never peak fractions.

    KDE bandwidth is Scott's rule. Standardize units before fitting. Reject
    insufficient/singular ensembles rather than manufacturing a contour.
    """
    points = np.asarray(points, dtype=float)
    mean, covariance = outcome_statistics(points)
    masses = np.asarray(masses, dtype=float)
    if len(points) < 20 or grid_size < 40 or np.any((masses <= 0) | (masses >= 1)):
        raise ValueError("KDE needs >=20 outcomes, >=40 grid nodes and masses in (0,1)")
    scale = np.sqrt(np.diag(covariance))
    standardized = (points-mean)/scale
    kde = gaussian_kde(standardized.T)
    axes = [np.linspace(standardized[:,d].min()-5, standardized[:,d].max()+5, grid_size)
            for d in range(2)]
    xx, yy = np.meshgrid(*axes)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    weights = np.ones(density.shape)
    weights[[0,-1], :] *= .5
    weights[:, [0,-1]] *= .5
    area = (axes[0][1]-axes[0][0])*(axes[1][1]-axes[1][0])
    total = float(np.sum(density*weights)*area)
    if not .995 <= total <= 1.005:
        raise ValueError("KDE grid does not resolve/capture its probability mass")
    order = np.argsort(density.ravel())[::-1]
    cumulative = np.cumsum((density*weights).ravel()[order])*area/total
    thresholds = np.array([density.ravel()[order[min(np.searchsorted(cumulative,m), len(order)-1)]]
                           for m in masses]) / np.prod(scale)
    if len(np.unique(thresholds)) != len(thresholds):
        raise ValueError("Density grid cannot resolve distinct requested contour masses")
    return dict(x=axes[0]*scale[0]+mean[0], y=axes[1]*scale[1]+mean[1],
                density=density/np.prod(scale), thresholds=thresholds,
                masses=masses, captured_mass=total,
                interpretation="KDE regions of optimizer outcomes, not posterior confidence")


def export_figure(fig, stem, *, provenance=None):
    from aeqga.paths import output_path
    from aeqga.experiments.runner import atomic_json, file_digest
    from importlib.metadata import version
    paths = []
    for extension in ["png", "svg", "pdf"]:
        path = output_path(extension, stem+"."+extension)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=180)
        paths.append(path)
    atomic_json(output_path("json", stem+"_figure_provenance.json"),
                dict(experiment=provenance, plotting_source_sha256=file_digest(__file__),
                     matplotlib_version=version("matplotlib"),
                     exports=[dict(path=str(path),sha256=file_digest(path)) for path in paths]))
    return paths


def run_diagnostics(record, reference, *, title="AEQGA / calibrated Pantheon+"):
    """Plot the SAME global-best pair and objective value at every update."""
    points = np.asarray(record["best_parameter_history"])
    values = np.asarray(record["best_history"])
    if points.shape != (len(values), 2): raise ValueError("History shape mismatch")
    updates = np.arange(len(values))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    panels = [(values, reference["chi2"], r"Raw $\chi^2$", "Objective value"),
              (values-reference["chi2"], 0, r"$\Delta\chi^2$ above classical minimum", "Convergence gap"),
              (points[:,0], reference["H0"], r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]", "Best-pair Hubble parameter"),
              (points[:,1], reference["Omega_m"], r"$\Omega_m$", "Best-pair matter density")]
    for ax, (data, baseline, label, caption) in zip(axes.ravel(), panels):
        ax.plot(updates, data, color="#245b8a", lw=1.7, label="AEQGA global best")
        ax.axhline(baseline, color="black", ls="--", lw=1,
                   label=f"Classical reference ({baseline:.6f})")
        ax.set(xlabel="Update (0 = initial population)", ylabel=label, title=caption)
        ax.grid(alpha=.18)
    axes[0,1].set_yscale("symlog", linthresh=.01)
    if np.min(values-reference["chi2"]) >= -1e-7:
        axes[0,1].set_ylim(bottom=0)
    axes[0,0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=12)
    return fig


def objective_contours(problem, reference, *, points=None, sigmas=(1,2,3,4,5)):
    h0 = np.linspace(max(60.,reference["H0"]-2), min(80.,reference["H0"]+2), 240)
    low, high = max(0.,reference["Omega_m"]-.12), min(.5,reference["Omega_m"]+.12)
    omega = (problem.omega_grid[(problem.omega_grid >= low)&(problem.omega_grid <= high)]
             if problem.grid_size else np.linspace(low,high,180))
    delta = problem.objective_grid(h0,omega)-reference["chi2"]
    if delta.min() < -1e-6: raise ValueError("Contour reference is not a valid objective minimum")
    levels = joint_levels(sigmas)
    fig, ax = plt.subplots(figsize=(6.4,4.8), constrained_layout=True)
    ax.contourf(h0,omega,delta,levels=[0,*levels],cmap="Blues_r")
    contours = ax.contour(h0,omega,delta,levels=levels,colors="black",linewidths=.8)
    ax.clabel(contours,fmt={v:f"{s}σ" for v,s in zip(levels,sigmas)},fontsize=8)
    ax.plot(reference["H0"],reference["Omega_m"],"*",color="#cc6b2e",ms=12,label="Classical minimum")
    if points is not None:
        points = np.asarray(points)
        ax.scatter(points[:,0],points[:,1],s=9,color="#193d59",alpha=.45,label="Independent AEQGA outcomes")
        # Keep the objective's resolved zoom; report off-view points rather than silently clipping them.
        outside = np.sum((points[:,0]<h0[0])|(points[:,0]>h0[-1])|(points[:,1]<omega[0])|(points[:,1]>omega[-1]))
        ax.set_title(f"Likelihood / optimizer overlay ({outside} outcomes outside zoom)")
    else: ax.set_title("Calibrated likelihood: joint two-parameter contours")
    ax.set(xlim=(h0[0],h0[-1]),ylim=(omega[0],omega[-1]),
           xlabel=r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]",ylabel=r"$\Omega_m$")
    ax.legend(frameon=False,fontsize=8)
    return fig


def optimizer_scatter(points, reference, *, kde_regions=False):
    points = np.asarray(points)
    mean, covariance = outcome_statistics(points)
    fig, ax = plt.subplots(figsize=(6.4,4.8),constrained_layout=True)
    if kde_regions:
        density = density_contours(points)
        levels = np.sort(density["thresholds"])
        ax.contourf(density["x"],density["y"],density["density"],
                    levels=[*levels,density["density"].max()],cmap="Blues",alpha=.6)
        styles = [":", "--", "-"]
        ax.contour(density["x"],density["y"],density["density"],levels=levels,
                   colors="black",linewidths=.9,linestyles=styles)
        # Dense, strongly correlated clouds make inline labels collide.
        # Put mass labels in a compact legend instead of across the data.
        from matplotlib.lines import Line2D
        for style, mass in zip(styles, density["masses"][::-1]):
            ax.add_line(Line2D([],[],color="black",ls=style,lw=.9,
                               label=f"KDE {100*mass:.1f}% enclosed mass"))
    ax.scatter(points[:,0],points[:,1],s=10,alpha=.4,color="#245b8a",label=f"{len(points)} outcomes")
    eig, vec = np.linalg.eigh(covariance)
    angle = np.degrees(np.arctan2(vec[1,-1],vec[0,-1]))
    factor = joint_levels((1,))[0]
    ax.add_patch(Ellipse(mean,2*np.sqrt(factor*eig[-1]),2*np.sqrt(factor*eig[0]),
                         angle=angle,fill=False,ec="#b95f2d",ls="--",lw=1.4,
                         label="68.3% Gaussian covariance ellipse (descriptive)"))
    ax.plot(*mean,"+",color="#b95f2d",ms=12,label="Optimizer mean")
    ax.plot(reference["H0"],reference["Omega_m"],"*",color="black",ms=11,label="Classical reference")
    ax.set(xlabel=r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]",ylabel=r"$\Omega_m$",
           title="Optimizer distribution — not posterior uncertainty")
    ax.legend(frameon=False,fontsize=7,loc="upper center",
              bbox_to_anchor=(.5,-.16),ncol=2)
    return fig
