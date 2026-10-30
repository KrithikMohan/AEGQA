"""Historical PR #10 single-circuit analysis; no network access or QPU jobs."""
import json
from aeqga.analysis.pr10_compatibility import analyze_pilot

if __name__ == "__main__":
    print(json.dumps(analyze_pilot(), indent=2))
