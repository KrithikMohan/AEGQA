"""Stable repository-local data and output paths, independent of working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def output_path(kind, filename):
    if kind not in {'png', 'svg', 'pdf', 'json'}:
        raise ValueError('Unsupported output type: '+kind)
    destination = PROJECT_ROOT/'outputs'/kind/filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination
