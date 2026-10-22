"""Retrieve an existing IBM job into validated counts; never submits work."""
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aeqga.execution.ibm_sampler import CheckpointManager, _counts


def recover_job(job_id=None, checkpoint_path=None, output_file=None, *, service=None):
    if checkpoint_path is None:
        raise ValueError("A checkpoint is required to establish register names, widths and shots")
    saved = CheckpointManager.load(checkpoint_path)
    if saved is None:
        raise FileNotFoundError(checkpoint_path)
    if job_id is not None and job_id != saved.get("job_id"):
        raise ValueError("Job ID does not match checkpoint")
    job_id = saved.get("job_id")
    if not job_id:
        raise ValueError("Checkpoint has no submitted job ID")
    widths, shots = saved["circuit_widths"], saved["shots"]
    if len(widths) != saved["num_circuits"]:
        raise ValueError("Invalid checkpoint circuit cardinality")
    if saved.get("submission_state") == "completed" and "raw_counts" in saved:
        raw = saved["raw_counts"]
        if len(raw) != len(widths):
            raise ValueError("Cached result cardinality mismatch")
        counts = [_counts(c, "", shots, width) for c, width in zip(raw, widths)]
    else:
        registers = saved.get("register_names")
        if not registers or len(registers) != len(widths):
            raise ValueError("Checkpoint lacks register mapping; recover original circuit archives first")
        if service is None:
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService()
        job = service.job(job_id)
        status = job.status()
        status_name = getattr(status, "name", str(status)).upper()
        if status_name not in {"DONE", "COMPLETED"}:
            raise RuntimeError(f"Job {job_id} is {status_name}; no submission was made")
        results = list(job.result())
        if len(results) != len(widths):
            raise ValueError("Remote result cardinality mismatch")
        counts = [_counts(pub, register, shots, width)
                  for pub, register, width in zip(results, registers, widths)]
        saved.update(submission_state="completed", raw_counts=counts)
        CheckpointManager.atomic_write(checkpoint_path, saved)
    if output_file:
        CheckpointManager.atomic_write(output_file, dict(job_id=job_id, recovered_counts=counts))
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--output")
    args = parser.parse_args()
    print(recover_job(args.job_id, args.checkpoint, args.output))
