MARRAKESH: PROFILED-M CIRCUIT-WIDTH STUDY

This directory contains the conclusions and runnable code from the completed
2026-10-26 local experiment. It is distributed on 2026-10-28.

START HERE
  results/findings.txt             Observed results and interpretation of metrics
  results/report.txt               Experimental method and model limitations
  results/tradeoff_overview.png    Final accuracy, convergence, noise and gate cost
  results/search_efficiency.png   Progress versus generations, evaluations and cost
  results/tradeoff_metrics.csv    Detailed comparison of all 12 configurations
  results/threshold_statistics.csv  Cost to reach common objective thresholds
  reproduction/                  Complete Python code, pinned dependencies and inputs

EXPERIMENT
  A 156-qubit budget; 3--8 qubits per complete subpopulation circuit.
  Six noisy runs and six matching ideal controls, seed 7, 100 updates each.
  Each generation has 100 shots of the packed logical execution.
  Each complete subpopulation circuit has four reset/prepare/measure stages.
  Per-group elite/local/global proportions are 1:1:2; elites are selected globally.
  H0 and Omega_m are searched. M is profiled analytically with the full covariance
  and the original [-19.6, -19.0] magnitude bounds.
  The Marrakesh public calibration snapshot is dated 2025-02-26.
  Actual occupied qubits for widths 3--8: 156, 152, 155, 156, 154, 152.
  The four-qubit layout uses 38 connected groups, as documented by the included
  integer-programming certificate and check_marrakesh_layout.py.

RUN
  Use a separate Python 3.11 environment for this standalone experiment.
  Its pinned packages are independent of the parent repository's environment.
  From this directory:

    cd reproduction
    python -m pip install -r requirements.txt
    python -m unittest -v test_profiled_marrakesh
    python run_marrakesh_profiled_study.py --widths 8 3 7 4 6 5 --modes noisy ideal --generations 100 --shots 100 --seed 7 --workers 24

  Results are written to reproduction/output/marrakesh_profiled_156q.
  Reduce --workers to suit local resources; it does not change population or shots.
  --resume skips completed cases only when their configuration/source hashes match.
  No IBM account, hardware submission, or QPU payment is used by this runner.

  After a new complete run, add the saved-record audit and decoding analysis:

    python package_marrakesh_profiled_study.py output/marrakesh_profiled_156q

INCLUDED RESULTS
  Per-run final summaries, generation-best parameters and chi-square,
  physical layouts, noise/sampling/gate metrics, decoding and diversity records,
  six parameter-history plots, and the original completed-export audit.

  The original complete archive contains 2,294,720 candidate rows and 150,400
  stage histograms. This smaller conclusions-and-code distribution does not
  contain those raw per-candidate/measurement histories or probability arrays.
  Re-running the experiment generates the complete records. The audit file is
  evidence about the original complete export, not an audit of this smaller subset.

OBSERVED RESULTS AND LIMITS
  In this single seed, width 4 has the lowest final noisy excess chi-square.
  Width 3 reaches excess chi-square <= 0.001 in 6 updates / 2,288 evaluations;
  width 8 reaches it in 27 updates / 103,360 evaluations.
  Width changes population, elite allocation, local search boxes and decoding,
  in addition to physical placement, gate count and noise.
  Zero-count/lower-bound outcomes trigger classical random replacement in the
  existing decoder; larger circuit width is not an isolated hardware-noise test.
  The model omits crosstalk, leakage, drift and separately calibrated reset error.
  Gate-time estimates are not measured quantum-processor wall time.
  The results do not establish an optimal width across seeds or on real hardware.

PROVENANCE
  results/study_plan.json and each run's summary retain source/data/snapshot hashes.
  Reproduction sources are copied unchanged from the completed experiment.
  SHA256SUMS.txt verifies the files in this smaller distribution.
  results/original_full_archive.txt identifies the original complete archive.
