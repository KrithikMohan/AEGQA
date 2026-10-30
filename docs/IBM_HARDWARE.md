# IBM Quantum integration and remaining validation

Historical Kingston results from PR #10 are checked in
[the compatibility summary](PR10_SUMMARY.md) and
[PDF report](../output/pdf/PR10_SUMMARY.pdf). Completed legacy `state`/`counts`
ledgers can be read offline using `python -m scripts.recover_ibm_job --checkpoint PATH`.
New resumable jobs require a fresh directory with current circuit and optimizer
provenance; legacy fingerprints are not silently converted.

The maintained hardware runner is `python -m scripts.run_ibm_generation`.
It imports `aeqga.likelihoods.pantheon_problem.PantheonPlusProblem` and the
package evolution loop. Real data are mandatory. Root-level legacy likelihoods
are not part of this workflow.

## Execution and scientific identity

CPU: population generation, fitness, elitism, decoding and checkpointing.
QPU: amplitude preparation, crossover, mutation and final measurement.
For 32 individuals and two parameters, a generation submits [4,3,4,3] logical
qubit circuits, ordered H0-random, H0-elite-copy, Omega_m-random, Omega_m-elite-copy.
The register width is validated against logical width even when ISA circuits
span the entire physical backend.

`IBMSamplerExecutor` uses Runtime `SamplerV2(mode=backend)` by default. Passing
`--session` creates a Runtime Session; account permissions and time limits are
still enforced by IBM. All four circuits are one request, not IBM Batch mode.
Generations and the CLI's independent runs advance sequentially. Parallel run
scheduling is not implemented; use distinct directories for independently
managed processes and account for the combined budget.

The runner fingerprints parameters, seed, data hashes, scientific/execution
source hashes, backend, package versions, preparation method and Runtime
options. Changed configurations are refused. Logical and ISA circuits are
saved as QPY, and circuit fingerprints include complex amplitudes and nested
instructions at full serialization precision. Runtime credentials are loaded
from the local service configuration and never stored by this code.

## Offline use

```bash
python -m scripts.run_ibm_generation --data sn_data/PantheonPlus \
  --population 32 --generations 3 --shots 1024 \
  --directory outputs/json/sne_offline_pilot
python -m scripts.benchmark_hardware --shots 4096
```

The benchmark's default is a reproducible synthetic four-qubit target.
`--backend NAME` instead authenticates to read a real device and derive an Aer
noise model; it still never submits QPU jobs. It compares both preparation
methods and optimization levels 0–3, retains raw counts, reports TVD and shows
both decoder branches' sensitivity. Noise models approximate device behavior;
they do not establish actual hardware fidelity.

For reusable programmatic execution:

```python
from pathlib import Path
from aeqga.likelihoods.pantheon_problem import PantheonPlusProblem
from aeqga.steps.evolution.aeqga_algorithm import AEQGAParameters
from aeqga.execution.ibm_sampler import DryRunExecutor
from aeqga.execution.workflow import run_checkpointed, pantheon_provenance

folder = Path('outputs/json/my_offline_pilot')
problem = PantheonPlusProblem()
params = AEQGAParameters(max_gen=3, num_shots=1024, progress_bar=False)
executor = DryRunExecutor(shots=1024, checkpoint_dir=folder, seed_simulator=42)
result = run_checkpointed(problem, params, executor, folder,
                          seed=42, provenance=pantheon_provenance(problem))
```

`DryRunExecutor(simulation_backend=backend)` uses Aer with backend-derived
noise and MPS simulation. Each generation/circuit gets a distinct deterministic
readout seed. Existing native Aer/statevector paths remain supported by the
optimizer; they have not all been moved into the adapter interface.

## Restart and recovery

Use **one writer per experiment directory**. Files include:

- `optimizer.json`: full population, fitness, best/history, generation and
  Python/NumPy/readout RNG states at the last decoded generation.
- `generation_NNNN.json`: job state/ID, circuit identities, registers, counts,
  optimizer input, compilation, calibration identity and available usage metrics.
- `generation_NNNN.logical.qpy` and `.isa.qpy`: actual circuit definitions.
- `calibration_HASH.json`: immutable backend-properties snapshot when available.
- `result.json`: validated final histories and experiment identity.

Repeat the original command. The optimizer restores the last decoded state,
reconstructs identical next-generation circuits and reuses cached counts or
retrieves a saved in-flight job. Its original decoding RNG state is restored,
so a crash between count persistence and decoding gives the same trajectory.
A completed experiment returns without circuit execution. Histories and
fitnesses are checked against the configured objective before continuing.

The adapter writes `submitting` before calling the service and saves the job
ID before waiting. There is no atomic transaction spanning IBM submission and
a local file. A crash in that gap leaves an uncertain submission: the code
**refuses automatic resubmission**. Inspect IBM's job list, establish the exact
job/circuit identity and attach the confirmed job ID to the checkpoint before
continuing. Do not delete the checkpoint to retry blindly.

```bash
python -m scripts.recover_ibm_job \
  --checkpoint outputs/json/my_hardware_pilot/generation_0001.json \
  --output outputs/json/my_hardware_pilot/recovered_counts.json
```

Recovery handles string/enum statuses, validates register/count cardinality,
widths and shot totals, writes plain JSON counts and updates the existing
checkpoint. Historical completed cached counts remain readable. Incomplete
historical checkpoints without register mapping require the original circuit
mapping; recovery refuses to guess.

## Live submission, budget and suppression

First establish account/instance, accessible backend, session eligibility,
QPU-time/credit budget and pilot scope. Preflight never submits:

```bash
python -m scripts.ibm_preflight --backend ibm_kingston
```

Example for a separately authorized pilot:

```bash
python -m scripts.run_ibm_generation --backend ibm_kingston \
  --generations 1 --shots 1024 --max-total-shots 4096 \
  --directory outputs/json/my_hardware_pilot \
  --submit-hardware --confirm-run-qpu
```

Both flags are required. Add `--session` only with supported access. Runtime
options can be supplied via `--sampler-options options.json`; the installed
Runtime SDK validates supported fields and combinations. Keep untreated,
twirled and dynamical-decoupling experiments in distinct directories, change
one setting at a time, and compare identical archived circuit inputs. Device
compatibility and efficacy must be measured; this repository does not claim
live validation of those suppression settings.

`--runs N` derives independent seeds and keeps each run in its own directory.
The shot ceiling conservatively covers the full requested plan, even on
resume, and excludes calibration/mitigation overhead. It is not a QPU-time or
financial cap. Use IBM's reported job metrics to determine affordable scaling.
50 updates at 4096 shots consume 819,200 circuit shots; 300 such runs consume
245.76 million before extra calibration work. No automatic large QPU run occurs.

Reset-free `--preparation state_preparation` is valid because every circuit
starts in zero. It is a separately fingerprinted experimental variant. The
offline benchmark verifies ideal equivalence with `initialize()` and reports
compiled depth, resets, layout and entangling-gate counts.

## Readout correction policy

`aeqga.analysis.hardware_validation` supplies computational-basis calibration
circuits and a joint assignment matrix. For small 3/4-qubit circuits this needs
8/16 calibration circuits **per measured physical layout**. Calibration must
use the same physical qubits/readout mapping as the experiment; independently
transpiled live calibration is not validated here.

Analysis uses linear inversion followed by clipping negative values and
normalizing. It reports clipped mass and matrix conditioning, rejects singular
matrices, and retains raw counts/calibration data. This projection introduces
bias and may worsen a result. Corrected probabilities are analysis-only: the
scientific optimizer continues to consume raw validated integer counts.
Layout-matched live calibration, its overhead and suppression combinations
remain live-research work.

## Analysis and pilot acceptance

```bash
python -m scripts.compare_generation_populations \
  --directory outputs/json/my_hardware_pilot --samples 1000
python -m scripts.analyze_hardware_runs \
  --directory outputs/json/my_hardware_pilot --max-gap 2.30
```

Comparison loads the actual QPY circuits and verifies their identities against
counts and real-data provenance. It ranks elites using the real objective and
uses the optimizer's actual clipped elite margin. The finite-shot ensemble
reports per-circuit TVD reference quantiles and distribution comparisons.
Re-decoding is explicitly labeled a conditional analysis with an analysis
seed; recorded optimizer histories remain the trajectory evidence.

The summary validates each fit/history and reports optimization gaps, scatter,
success fraction under the supplied gap tolerance, shots, available job usage
metrics and elapsed execution time, plus PNG/SVG/PDF plots. The elapsed field
sums active process invocations; downtime between restarts is excluded. The
2.30 gap is an optimization diagnostic, not posterior coverage. Optional
`--max-tvd VALUE` evaluates an explicitly chosen fidelity threshold; absent
comparisons produce an unknown result, never a pass.

Before escalating a pilot, require:

1. Known-state bit ordering, register integrity and shot totals pass.
2. A deliberate interruption recovers without a second submission and yields
   the same decoded trajectory (automated offline tests cover these boundaries).
3. Representative circuits meet a predeclared TVD/noise tolerance on hardware.
4. Several short independent real-data runs meet declared optimization targets.
5. Measured QPU usage, queue time and calibration drift fit the agreed budget.

There is no automatic promotion to a 50-update run or an ensemble.

## Historical records and remaining work

Local records describe pilot job `db5qnv4vf2bc73d09org` and one-generation job
`db5r25imb58s738at2v0` on `ibm_kingston`, at 1024 shots. Their original serialized
circuits and full restart states are absent. The generation checkpoint marks
circuit provenance unverified. Do not certify exact same-circuit equivalence or
real-data likelihood reproducibility from those records. New comparison tools
refuse historical reconstruction without archives and experiment identity.

Still requiring live access/budget: actual session execution, device-specific
suppression validation, layout-matched readout calibration, known-state QPU
checks, repeatable short scientific runs, measured cost/fidelity thresholds,
and full hardware runs/ensembles. SDK/backend compatibility and scheduling
limits must be established in that environment. The old hardware records and
simulation ensembles do not establish completion of these tasks.
