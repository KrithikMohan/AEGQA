#!/usr/bin/env python3
"""
recover_ibm_job.py
==================
Job recovery script for IBM Quantum Hardware AEQGA jobs.

Recovers previously submitted jobs from their saved IDs or checkpoint files
rather than submitting duplicate jobs.
"""

import sys
import json
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aeqga.execution.ibm_sampler import CheckpointManager, _counts


def recover_job(job_id: str | None = None, checkpoint_path: str | None = None, output_file: str | None = None):
    print("=" * 60)
    print("  AEQGA IBM Quantum Job Recovery")
    print("=" * 60)

    ckpt_data = None
    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
        ckpt_data = CheckpointManager.load(ckpt_path)
        if ckpt_data is None:
            print(f"[ERROR] Checkpoint file {checkpoint_path} not found.")
            sys.exit(1)
        job_id = ckpt_data.get("job_id", job_id)
        print(f"Loaded checkpoint for generation {ckpt_data.get('generation')}")

    if not job_id:
        print("[ERROR] Neither job_id nor valid checkpoint provided.")
        sys.exit(1)

    print(f"\nTarget Job ID: {job_id}")

    # Check if counts already exist in checkpoint
    if ckpt_data and ckpt_data.get("submission_state") == "completed" and "raw_counts" in ckpt_data:
        print("✓ Results already completed and preserved in checkpoint.")
        counts = ckpt_data["raw_counts"]
    else:
        # Connect to runtime service
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService()
            print("Connecting to IBM Quantum Runtime Service...")
            job = service.job(job_id)
            status = job.status()
            print(f"Job status: {status}")

            if status.name == "DONE":
                result = job.result()
                print("Extracting counts from job result...")
                # Note: exact register mapping depends on original circuits
                counts = result
            else:
                print(f"Job is not yet complete (status: {status}). Please check later.")
                return
        except Exception as e:
            print(f"[INFO] Could not query remote IBM service ({e}).")
            if ckpt_data:
                print("Displaying checkpoint metadata:")
                print(json.dumps(ckpt_data, indent=2))
            return

    if output_file and "counts" in locals():
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump({"job_id": job_id, "recovered_counts": counts}, f, indent=2)
        print(f"Saved recovered counts to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recover IBM Quantum Job")
    parser.add_argument("--job-id", default=None, help="IBM Quantum job ID")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint JSON")
    parser.add_argument("--output", default=None, help="Path to save recovered counts JSON")
    args = parser.parse_args()
    recover_job(args.job_id, args.checkpoint, args.output)
