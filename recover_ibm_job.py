
import json
from pathlib import Path
from qiskit_ibm_runtime import QiskitRuntimeService

checkpoint = Path(
    "outputs/json/ibm_first_pilot/generation_0001.json"
)

record = json.loads(checkpoint.read_text(encoding="utf-8"))
job_id = record["job_id"]

service = QiskitRuntimeService()

print("Retrieving existing job:", job_id, flush=True)

job = service.job(job_id)

print("Job status:", job.status(), flush=True)

# Retrieve results without submitting a new job
result = job.result()

pub = result[0]

print("Available measurement registers:", list(pub.data.keys()))

# The original AEQGA circuit uses measure_all(),
# which normally creates a register named 'meas'.
register = pub.data.meas

counts = register.get_counts()

print("\n=== FIRST REAL IBM QUANTUM RESULTS ===")
print("Measured counts:", counts)
print("Total shots:", sum(counts.values()))

# Save the recovered data locally
output = Path("outputs/json/ibm_first_pilot/recovered_counts.json")

output.write_text(
    json.dumps(counts, indent=2),
    encoding="utf-8"
)

print("Results saved to:", output)
