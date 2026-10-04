# HQGA interoperability and fair comparisons

The retained bridge is `aeqga.integrations.hqga`, separate from the paper's
AEQGA loop. It imports no HQGA package and changes neither algorithm's gates.
The old one-way shim in the evolution module is superseded, not reinstated there.

Interfaces were checked against upstream [HQGA](https://github.com/Quasar-UniNA/HQGA)
revision `968f1f908cc3fde3b112d5bf4dd18a8b23003007`: `problems.py`,
`discretization.py`, `hqga_utils.py`, and `hqga_algorithm.py`.

## Existing HQGA real problem → AEQGA

```python
from HQGA.problems import SphereProblem
from aeqga.integrations.hqga import run_qga_aeqga

problem = SphereProblem(num_bit_code=8)
# params is your existing HQGA parameter object; simulator is an AerSimulator.
best, populations, history = run_qga_aeqga(
    problem, params, simulator, pop_size=32, p_cross=0.5, p_mut=0.5)
print(best.x, best.fitness, best.evaluations)
```

`HQGAProblemAdapter` maps `dim` → `n_dim`, `computeFitness(real_vector)` →
`compute_fitness`, and `isMaxProblem` → `is_max_problem`, preserving bounds.
It deliberately does NOT call HQGA `evaluate`, which expects a chromosome.
Binary-only objectives such as OneMax are unsupported.

`parameters_from_hqga` transfers population size, generation budget, shots,
verbosity and `progressBar`. AEQGA population size must be a power of two ≥8;
invalid sizes raise rather than silently changing the experiment. Explicit
overrides are available. HQGA `prob_mut`, `epsilon`, `epsilon_init`, elitism
and repeated-run settings are NOT translated into AEQGA gate probabilities.
The AEQGA defaults remain 0.5 unless explicitly overridden.

The three return values are native AEQGA: `GlobalBest.x`, continuous population
histories, and `[x, fitness]` best-history entries. They are not HQGA `.chr`,
`.phenotype` or three-column chromosome logs; never relabel continuous values
as HQGA chromosomes. AEQGA `.gen` is a generation index; upstream HQGA best
`.gen` stores evaluations to discovery, not the total run budget.

## AEQGA cosmology objective → original HQGA runner

```python
from aeqga.integrations.hqga import AEQGAProblemForHQGA
from HQGA import hqga_utils, hqga_algorithm

shared = AEQGAProblemForHQGA(sne_problem, num_bit_code=8,
                            name="Calibrated Pantheon+ [H0, Omega_m]")
circuit = hqga_utils.setupCircuit(params.pop_size,
                                shared.dim * shared.num_bit_code)
# device_features and params use YOUR HQGA environment's native configuration.
hbest, hpopulations, hhistory = hqga_algorithm.runQGA(
    device_features, circuit, params, shared)
print(hbest.phenotype, hbest.fitness, shared.evaluations)
```

The wrapper supplies `dim`, `num_bit_code`, bounds, `convert`, `convertToReal`,
`computeFitness`, `evaluate` and `isMaxProblem`, the protocol used by runQGA.
Each parameter occupies a consecutive Gray-code chunk; decoding matches the
upstream inclusive grid exactly. Pass chromosomes after HQGA `fromQtoC`, not
raw Qiskit count keys (which upstream reverses/splits first). No encoder is
needed for HQGA's measurement-driven generation loop. Parameter order remains
the objective's order, e.g. `[H0, Omega_m]`. Resolution is available as
`shared.resolution`; HQGA is discretized while AEQGA is continuous.

## Comparison methodology and limits

Use the same dataset selection, calibration, covariance, distances/emulator,
bounds, parameter order and objective direction. Record both parameter sets,
bit depth/resolution, independent seeds, shots, elapsed time and actual objective
evaluations. Reset/create the counting wrapper per HQGA run. Compare fitness
versus evaluation budget, not only generation numbers; retain every independent
run's results. A native HQGA optimizer run must actually be performed before
claiming relative performance. Optimizer best-fit scatter is not a posterior.

The notebook includes a dependency-free real-data objective/Gray-decoding
contract check and an opt-in native HQGA run. Upstream imports old
`qiskit.execute`/`qiskit.Aer`; the current AEQGA environment uses modern Qiskit.
Do not downgrade it automatically. Run native HQGA in a separately compatible
environment (install this package without replacing that environment's pinned
dependencies), or port HQGA's execution layer explicitly. An import failure is
reported, never replaced by AEQGA or mock results. The adapter protocol and
AEQGA side are tested; the complete upstream quantum runner is not certified
against current Qiskit by those tests.

## Verification of this change set

All 29 repository tests passed, including five adapter tests covering Gray-code
grids, objective direction, budget mapping, invalid inputs and an actual AEQGA
bridge run. The default real-data notebook executed with no errors, including
the Pantheon+ adapter contract check. A separate smoke check loaded the pinned
upstream `problems.py` and `discretization.py` without its legacy quantum runner:
all 256 pairs on a two-dimensional four-bit grid matched its actual
`SphereProblem.convert` and `evaluate`; AEQGA also ran directly on that original
SphereProblem (8 individuals, one update, 16 evaluations). This validates the
objective boundary, not native HQGA quantum execution or relative performance.
