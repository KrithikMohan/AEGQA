Python 3.11. Install requirements.txt, then run:
python run_marrakesh_profiled_study.py --generations 100 --shots 100 --seed 7 --workers 24
Tests: python -m unittest -v test_profiled_marrakesh
--resume skips completed configurations with matching settings and source hashes.
Worker count changes local runtime; it does not change candidate or shot counts.

Export verification: python verify_marrakesh_profiled_study.py PATH_TO_COMPLETED_STUDY
Decoding/diversity analysis: python summarize_marrakesh_profiled_study.py PATH_TO_COMPLETED_STUDY
Connected 4-qubit packing certificate: python check_marrakesh_layout.py
The audit and summary scripts analyze saved files without rerunning training.
