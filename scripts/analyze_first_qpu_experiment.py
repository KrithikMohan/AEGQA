"""Analyze the archived PR #10 pilot with its original random input convention."""
import json
from aeqga.analysis.pr10_compatibility import analyze_pilot

if __name__ == "__main__":
    print(json.dumps(analyze_pilot(), indent=2))
