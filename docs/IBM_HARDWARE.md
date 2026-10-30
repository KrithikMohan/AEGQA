# IBM QPU pilot (experimental integration)

This overlay adds an IBM Runtime Sampler V2 execution adapter to AEQGA while
retaining all existing simulator and paper-replication paths. It is **not** a
completed hardware validation or 300-run paper reproduction.

## Install

From the AEGQA repository root, copy this overlay's `aeqga/`, `scripts/`,
`tests/`, and `requirements-ibm.txt` into the matching paths. Copy
`apply_hardware_integration.py` to the root and run it once. Use a new branch.

    pip install -r requirements-ibm.txt
    python -m unittest tests.test_ibm_adapter
    python -m scripts.ibm_preflight --backend YOUR_BACKEND

Credentials must be configured via IBM's secure local account mechanism, NOT
as arguments, notebook cells, tracked configuration, logs or checkpoints.

## Minimal hardware execution

After constructing the existing PantheonPlusProblem and AEQGAParameters:

    from qiskit_ibm_runtime import QiskitRuntimeService
    from aeqga.execution.ibm_sampler import IBMSamplerExecutor
    from aeqga.steps.evolution.aeqga_algorithm import run_aeqga_dual

    service = QiskitRuntimeService()
    backend = service.backend("YOUR_BACKEND")
    executor = IBMSamplerExecutor(backend, service,
                                  "outputs/qpu/pilot_run_000", shots=params.num_shots)
    best, populations, history = run_aeqga_dual(
        problem, params, seed=123, hardware_executor=executor, quiet=True)

Run **one short pilot** first (e.g. 8 individuals, 1 generation, modest shot
count) only after confirming remaining access and budget. Do not commit QPU
output containing sensitive account identifiers.

## Boundaries and caveats

- Four circuits per generation for `n_dim=2`: random/elite for each dimension.
  32 individuals produce 4-, 3-, 4-, and 3-qubit circuits. A 50-generation
  run submits 50 generation jobs containing 200 circuit executions.
- Approximate circuit count is 200 per full run, 819,200 shots at 4,096 shots
  per circuit; 300 runs require 245,760,000 shots before overhead.
- The adapter retrieves saved job IDs after restart and reuses saved counts.
  A *submission_intent* record without a job ID blocks automatic resubmission:
  inspect the IBM jobs dashboard and reconcile manually to avoid duplicates.
- Generation checkpoints contain compiled circuit fingerprints and counts,
  **not** full optimizer populations/RNG state. Restart currently replays
  earlier generations from the original seed and recovered counts. Automatic
  independent-generation resume is NOT yet implemented.
- Circuit fingerprinting includes ISA transpilation; reusing a checkpoint
  requires the same backend, compilation settings and relevant versions.
- Measurement decoding remains the paper's raw-count min/max mapping.
  No readout mitigation, twirling, or dynamical decoupling is enabled by default.
  Such choices must be studied as distinct experimental variants.
- On real devices the number of allowed shots/job and Runtime service limits
  depend on the account and backend; do not assume 4,096 is always valid.
- Before production: confirm raw-count consistency, test state preparation
  against Aer, compare transpilation probability distributions, measure
  calibration/noise and budget, and define statistical acceptance criteria.
- The Pantheon+ selection / likelihood and CMB+BAO PICO ambiguity are separate
  from QPU execution; do not conflate QPU validation with paper equivalence.
