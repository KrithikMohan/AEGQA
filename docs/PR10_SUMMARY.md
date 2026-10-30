# PR #10: Kingston results and compatibility

[PDF report](../output/pdf/PR10_SUMMARY.pdf) |
[Numeric audit](PR10_COMPATIBILITY.json) |
[Original PR #10](https://github.com/KrithikMohan/AEQGA/pull/10)

Reviewed October 30, 2026. PR #10 merged as `599927f` and preserved nine files:
two Kingston job ledgers, recovered pilot counts, a population-comparison NPZ,
two plots and three analysis/recovery scripts. The ledgers record submissions
on October 11, 2026. All original counts, NPZ arrays and plots are unchanged.

## Recomputed results

- Pilot `db5qnv4vf2bc73d09org`: 4 qubits, 1024 shots, 16 observed basis states,
  TVD 0.110565296366, maximum probability error 0.072804736223.
- The ideal finite-shot baseline uses 10,000 trials: mean TVD 0.040678693805,
  99th percentile 0.064986791024; no trials exceed the observed TVD. This is
  conditional on reconstructed input, not a gate-fidelity measurement.
- Generation `db5r25imb58s738at2v0`: 32 individuals, one update, seed 42,
  four circuits of widths [4,3,4,3], 1024 shots each and gate probabilities 0.5.
- The current algorithm reproduces all four archived initial/final population
  arrays exactly. Best H0=71.95799958, Omega_m=0.46093712,
  chi-squared=1778.755666981; 64 objective evaluations; optimization gap
  25.851620171 above the configured classical minimum.
- All eight classical elites are retained. The best is also retained by the
  simulator comparison; this does not demonstrate improvement from the QPU.

## Compatibility changes

The old replay helper used `execute_generation()`; the current executor exposes
`run(circuits, generation, optimizer_state)`. Historical replay now implements
the current contract with positive shots and count/width checks. Completed old
ledgers use `state`/`counts`; the current format uses `submission_state`/
`raw_counts` and records additional circuit and optimizer identity. Recovery
reads completed historical counts offline without rewriting the original files,
fabricating provenance or requesting remote jobs.

Pilot analysis consistently reconstructs `default_rng(42).uniform(0.1,1.0,16)`,
with forced crossover/mutation, matching the archived experiment. The previous
`scripts/` analyzer assumed a different input and JSON format. Historical root
entry points now delegate to maintained code and do not execute on import.

## Verification and limitations

All 74 tests passed, including five new compatibility regressions. An additional
check exercised installed Runtime 0.50.0 SamplerV2 on a local fake backend and
returned expected counts. No physical QPU jobs were submitted. Existing tests
cover interrupted current jobs and exact optimizer restart.

The old ledgers have no original logical/ISA QPY archives or full optimizer
provenance, so their compiled hashes cannot be verified independently. Historical
count replay is supported for analysis; those files must not be reused as current
resumable checkpoints. Use a new directory with the maintained runner:

```bash
python -m scripts.run_ibm_generation --directory outputs/json/new_offline_pilot
python -m scripts.run_ibm_generation --directory outputs/json/new_qpu_pilot \
  --backend YOUR_BACKEND --submit-hardware --confirm-run-qpu
```

The first command is offline. The second explicitly submits hardware work using
local credentials and consumes the account allocation.

## Reproduce the audit

```bash
python -m aeqga.analysis.pr10_compatibility --output docs/PR10_COMPATIBILITY.json
python compare_generation_populations.py
python analyze_first_qpu_experiment.py
python recover_ibm_job.py --checkpoint outputs/json/ibm_first_pilot/generation_0001.json
MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -q
```

The numeric audit records versions and hashes of all four numeric inputs.
`scripts/build_pr10_summary.py` rebuilds the PDF using ReportLab, the JSON audit
and the unchanged original figure.

## Approved PR #9 resolution

Keep the seven current conflicting implementations. Remove the obsolete overlay
installer, overlay README and five duplicate root helper/test scripts. All nine
experimental artifacts are already on main through PR #10. Clear notebook
execution counters and outputs, retaining exported figures and numeric results.
