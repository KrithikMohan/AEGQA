# IBM quantum hardware adaptation plan

This document describes the information and engineering work required to run
the repository's AEQGA implementation on IBM quantum processors.

AEQGA will use a hybrid workflow: the local computer performs population
selection, measurement decoding, cosmological likelihood evaluation and
checkpoint management. IBM hardware performs amplitude preparation,
crossover, mutation and measurement.

The repository currently supports Aer and statevector-based finite-shot
sampling. Reliable QPU execution requires an IBM Sampler adapter,
hardware-aware transpilation and generation-level recovery.

For the paper-sized 32-individual, two-parameter configuration, every
generation constructs four circuits: two four-qubit random-subset circuits and
two three-qubit elite-copy circuits. A 50-generation run therefore requires
200 circuit executions. At 4,096 shots, this is 819,200 shots per run, or
245.76 million shots for 300 runs, before calibration and mitigation overhead.
A staged pilot should establish cost and fidelity before attempting the full
ensemble.

## Information required

| Information | What to provide or establish | Why it matters |
|---|---|---|
| IBM access | Account plan, institution or project instance, region and available backends | Determines hardware access, quotas and scheduling options |
| Authentication | Whether IBM credentials are already configured locally | Enables the adapter; API keys must remain outside the repository and documentation |
| Software environment | Python, Qiskit, `qiskit-aer` and `qiskit-ibm-runtime` versions; local computer or hosted notebook | Determines the supported API and reproducible dependency set |
| Target processor | Preferred backend, or authority to select from accessible devices | Compilation depends on the processor's supported instructions and connectivity |
| Device quality | Calibration snapshot, readout errors, two-qubit errors, gate durations and coherence times | Supports selection of a suitable connected three- or four-qubit region |
| Resource budget | Available QPU time or credits, maximum spend, acceptable queue delay and deadline | Sets feasible shots, generations and independent-run count |
| Scheduling access | Whether sessions are available and their time limits | AEQGA requires classical feedback between generations |
| Initial research scope | SNe first, desired population and generation settings, and hardware comparison goals | Defines the first experiment and its success criteria |
| Noise treatment | Whether to study raw measurements, error suppression and readout mitigation | These treatments can change decoded populations and convergence |
| Research outputs | Required plots, statistical comparisons and reproducibility records | Determines which measurements and metadata must be retained |

IBM documents a newer client-side Sampler implementation introduced in
Runtime 0.50.0. The project should select and pin the interface supported by
the research environment. Sampler is appropriate because AEQGA consumes
measured basis-state counts. See the
[IBM Sampler quickstart](https://quantum.cloud.ibm.com/docs/en/guides/get-started-with-sampler).

Sessions suit adaptive algorithms, but IBM's Open Plan currently cannot
submit session jobs. Batch mode is intended for independently executable jobs,
so it cannot remove dependencies between successive AEQGA generations. See
[IBM execution modes](https://quantum.cloud.ibm.com/docs/en/guides/execution-modes).

## Adaptation tasks

1. **Establish access and freeze the hardware configuration.** Verify
   authentication, list accessible backends and record package versions,
   account limits and budget. Create a configuration file containing backend
   and execution settings while storing credentials separately.

2. **Introduce a common circuit-execution interface.** Extract execution from
   `aeqga_algorithm.py` into adapters for ideal simulation, noisy simulation
   and IBM Sampler. Every adapter should return consistently ordered basis
   counts and execution metadata. Verify that the refactor preserves existing
   simulator results.

3. **Compile complete circuits for the selected processor.** Transpile
   amplitude preparation together with crossover and mutation into the
   backend's supported instructions. Compare layouts, optimization levels,
   circuit depth and two-qubit gate counts. Verify ideal output probabilities
   before and after compilation. IBM requires hardware-compatible ISA
   circuits; see its
   [transpilation guidance](https://quantum.cloud.ibm.com/docs/en/guides/transpile).

4. **Evaluate amplitude-preparation cost.** The repository currently uses
   `initialize()`. Compare its compiled implementation with reset-free state
   preparation where the initial-state assumptions permit it. Preserve the
   encoded amplitudes and the paper's genetic operations. Record approximate
   preparation, if studied, as a separate experimental variant.

5. **Submit all four circuits together per generation.** Construct the four
   circuits before one Sampler request and decode results in a fixed order. Use
   a session where available and ordinary jobs otherwise. Independent runs can
   advance together in generation rounds, subject to resource limits.

6. **Add hardware-safe checkpoints and recovery.** Save the population,
   fitness, random-generator state, circuit identities and IBM job IDs for
   every generation. Persist submitted jobs before waiting for results. On
   restart, retrieve existing jobs and complete decoding without submitting
   duplicate work. Extend the existing run-level checkpointing for this finer
   recovery layer.

7. **Validate result decoding.** Test classical-register selection, bit
   ordering, absent basis states and shot totals. Run known-state circuits to
   verify that hardware results reach the correct decoder. Require positive
   shots for hardware and reject simulator-only statevector instructions.

8. **Benchmark representative circuits under noise.** Compare ideal
   simulation, backend-derived noisy simulation and hardware for identical
   prepared populations. Measure distribution differences and their effect on
   decoded parameters, including the min-max decoder's sensitivity to small
   count changes.

9. **Test noise suppression and mitigation.** Compare untreated measurements
   with supported twirling and dynamical-decoupling settings. Evaluate readout
   mitigation separately while retaining raw counts. Because mitigation can
   produce negative quasi-probabilities, define and validate a probability
   handling policy before passing corrected results to the decoder. IBM's
   supported options and compatibility restrictions depend on the selected
   interface and backend; see
   [Sampler options](https://quantum.cloud.ibm.com/docs/en/guides/sampler-options).

10. **Run a staged hardware pilot.** Progress from known-state checks to
    representative genetic circuits, one complete generation and several
    short independent SNe runs. Advance only after count integrity, recovery,
    fidelity and measured resource use meet documented thresholds.

11. **Scale toward the paper experiment.** Attempt one complete
    32-individual, 50-generation run, followed by a budgeted ensemble.
    Determine the affordable ensemble size from measured usage. Record hardware
    noise, shot count, calibration and preparation changes alongside the paper
    settings.

12. **Update the notebook and research analysis.** Add hardware configuration,
    preflight checks, submission and recovery controls, and comparisons with
    ideal and noisy simulation. Extend the Part 9 analysis with delta
    chi-squared, parameter scatter, success rate, distribution error, QPU usage
    and total elapsed time. Treat each hardware ensemble as a separately
    identified experiment because device noise and calibration drift affect
    reproducibility.

## Initial inputs needed to begin implementation

The first implementation pass requires:

1. IBM account plan and project or service instance.
2. Names of the accessible IBM backends, or permission to choose among them.
3. Installed Python, Qiskit and `qiskit-ibm-runtime` versions.
4. Approximate QPU budget or usage limit.
5. Whether IBM sessions are available.
6. Desired scope of the initial pilot run.

Do not place an IBM API key in configuration files, notebooks, logs,
checkpoints or version control.
