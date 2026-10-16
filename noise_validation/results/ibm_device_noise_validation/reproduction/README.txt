Python 3.11 was used. Create an isolated environment, install requirements.txt,
then run: python run_ibm_noise_validation.py --output output/reproduced
Full run defaults: 100 generations, 1024 shots, seeds 7 17 27,
3 fixed input variants and 5 fixed-circuit sampling repeats.
Test with: python -m unittest -v test_ibm_noise_validation
qiskit-ibm-runtime is not needed at execution time: its unchanged public
snapshot JSON is included. No account, network access or hardware fees.
The BackendV2 Target adapter uses documented JSON units and calls the
official Aer NoiseModel.from_backend; calibrated asymmetric readout
is applied separately. See validation_report.txt for model assumptions.
