"""Read-only IBM hardware preflight. Does not submit jobs or print secrets.

Usage: python -m scripts.ibm_preflight [--backend NAME]
Credentials must be configured securely using IBM's documented local account setup.
"""
import argparse
from importlib.metadata import version, PackageNotFoundError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend')
    args = parser.parse_args()
    for name in ('qiskit', 'qiskit-aer', 'qiskit-ibm-runtime', 'numpy'):
        try:
            print(f'{name}: {version(name)}')
        except PackageNotFoundError:
            print(f'{name}: NOT INSTALLED')
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    backends = [service.backend(args.backend)] if args.backend else service.backends(operational=True, simulator=False)
    for backend in backends:
        print(f'backend={backend.name} qubits={backend.num_qubits} '
              f'pending_jobs={backend.status().pending_jobs}')
    print('No jobs were submitted.')


if __name__ == '__main__':
    main()
