#!/usr/bin/env python3
"""
setup_env.py — Set up the AEQGA Python virtual environment.

Usage:
    python3 -m scripts.setup_env

This creates a venv at AEGQA-main/venv, installs all dependencies,
and verifies the installation by importing all modules.

After running, activate the environment:
    source venv/bin/activate    # Linux/macOS
    venv\\Scripts\\activate     # Windows
"""

import sys
import os
import subprocess
import platform

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_DIR = os.path.join(PROJECT_DIR, "venv")

def run(cmd, cwd=None):
    """Run a subprocess command and stream output."""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}")
        sys.exit(1)
    return result.returncode


def main():
    print("=" * 60)
    print("  AEQGA Virtual Environment Setup")
    print("=" * 60)

    # Detect Python version
    python_cmd = sys.executable
    print(f"\nPython: {python_cmd} ({sys.version.split()[0]})")

    # Create virtual environment
    if os.path.exists(VENV_DIR):
        print(f"\nvenv already exists at {VENV_DIR}")
        response = input("Remove and recreate? (y/n): ").strip().lower()
        if response == 'y':
            import shutil
            shutil.rmtree(VENV_DIR)
            print("Removed existing venv.")
        else:
            print("Keeping existing venv.")
            return

    print(f"\nCreating virtual environment at {VENV_DIR} ...")
    run([python_cmd, "-m", "venv", VENV_DIR])

    # Determine pip path
    if platform.system() == "Windows":
        pip = os.path.join(VENV_DIR, "Scripts", "pip")
        python = os.path.join(VENV_DIR, "Scripts", "python")
    else:
        pip = os.path.join(VENV_DIR, "bin", "pip")
        python = os.path.join(VENV_DIR, "bin", "python")

    # Upgrade pip
    print("\nUpgrading pip ...")
    run([python, "-m", "pip", "install", "--upgrade", "pip"])

    # Install dependencies
    req_file = os.path.join(PROJECT_DIR, "requirements.txt")
    print(f"\nInstalling dependencies from {req_file} ...")
    run([pip, "install", "-r", req_file])
    run([python, "-m", "pip", "install", "--no-deps", "-e", PROJECT_DIR])

    # Verify installation
    print("\nVerifying installation ...")
    modules = [
        "qiskit",
        "qiskit_aer",
        "numpy",
        "scipy",
        "camb",
        "tqdm",
        "matplotlib",
        "nbformat",
        "jupyter",
        "aeqga.steps.encoding.amplitude_encoding",
        "aeqga.steps.genetic_operators.quantum_gates",
        "aeqga.steps.decoding.measurement_decoding",
        "aeqga.steps.evolution.aeqga_algorithm",
        "aeqga.likelihoods.pantheon_problem",
        "aeqga.likelihoods.bao_cmb_problem",
    ]
    all_ok = True
    for mod in modules:
        result = subprocess.run(
            [python, "-c", f"import {mod}; print(f'  ✓ {mod}')"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ✗ {mod}: FAILED — {result.stderr.strip()}")
            all_ok = False

    print()
    if all_ok:
        print("=" * 60)
        print("  ✓ All dependencies installed and verified!")
        print("=" * 60)
        print(f"""
To activate the virtual environment, run:

    source venv/bin/activate       # Linux/macOS
    venv\\Scripts\\activate          # Windows

Then run the notebook:

    jupyter notebook notebooks/AEQGA.ipynb

Or run the full pipeline:

    python -m scripts.run_pantheon_aeqga --data sn_data/PantheonPlus

To deactivate, simply run:

    deactivate
""")
    else:
        print("Some modules failed to import. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
