Historical synthetic-noise reproduction sources. Python 3.11.
The three Python source hashes and two input-data hashes match the
original ../summary.json. The matching AEQGA and synthetic-noise source
files were recovered from the archived IBM reproduction snapshot.
Install: python -m pip install -r requirements.txt
Run: python run_noise_validation.py --output output/reproduced
Defaults: populations 8/16/32/64; 100 generations; 1024 shots;
seeds 7/17/27; 3 fixed-input variants; 5 fixed-circuit sample repeats.
Tests: python -m unittest -v test_aeqga test_history_plot test_noise_validation
The original result files in the parent directory were not modified.
