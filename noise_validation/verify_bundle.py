"""Verify the complete bundle and the preserved historical IBM manifest."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent

def check_manifest(manifest):
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = (manifest.parent / name).resolve()
        if not path.is_relative_to(manifest.parent.resolve()):
            raise ValueError(f"Manifest path escapes bundle: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise AssertionError(f"SHA256 mismatch: {path}")
        count += 1
    return count

counts = {"bundle_files": check_manifest(root / "SHA256SUMS.txt"),
          "original_ibm_manifest_files": check_manifest(root / "results/ibm_device_noise_validation/SHA256SUMS.txt")}
summary = json.loads((root / "results/qubit_noise_validation/summary.json").read_text(encoding="utf-8"))
reproduction = root / "results/qubit_noise_validation/reproduction"
for name, expected in summary["sha256"].items():
    path = reproduction / name.replace(chr(92), "/")
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name
counts["original_synthetic_source_and_data_hashes"] = len(summary["sha256"])
print(json.dumps(counts))
