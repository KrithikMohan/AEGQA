"""Historical PR #10 replay using the maintained executor interface (offline)."""
import argparse
import json
from pathlib import Path
from aeqga.analysis.pr10_compatibility import analyze_generation
from aeqga.experiments.runner import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze_generation(args.directory, args.data)
    if args.output:
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
