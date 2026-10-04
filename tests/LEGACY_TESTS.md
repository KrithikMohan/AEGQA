The retired test_pantheon_aeqga.py script used the 1048-row uncalibrated
Pantheon objective and could not validate H0. It also mixed plotting and
expensive optimization with checks that unittest discovery never executed.

Its useful distance, amplitude normalization, quantum gate and decoding
regressions now live in test_quantum_primitives.py and test_replication.py.
The legacy magnitude fits, fixed-M plots and old runner are not required
for the calibrated paper workflow. Future separate-dataset validation would
need a properly calibrated objective, provenance and meaningful baselines.
See ../docs/DEPRECATED_CODE.md for the full feature audit and future needs.
