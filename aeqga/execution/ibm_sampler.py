"""IBM Runtime Sampler V2 generation adapter, with a durable job/result ledger.

No IBM credentials are stored here. All four measured circuits are submitted as
one job per generation. A separate checkpoint is retained for each run.
"""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

from aeqga.experiments.runner import atomic_json


def _digest(circuits):
    from qiskit import qpy
    import io
    stream = io.BytesIO()
    qpy.dump(circuits, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _counts(pub, register_name, shots, width):
    data = getattr(pub.data, register_name)
    counts = data.get_counts()
    output = {}
    for key, value in counts.items():
        bits = str(key).replace(' ', '')
        if len(bits) != width or any(c not in '01' for c in bits):
            raise ValueError(f'Unexpected bitstring {key!r}: expected {width} measured bits')
        if not isinstance(value, int) or value < 0:
            raise ValueError('Invalid Sampler count')
        output[bits] = output.get(bits, 0) + value
    if sum(output.values()) != shots:
        raise ValueError('Sampler result shot total mismatch')
    return output


class IBMSamplerExecutor:
    """Generation executor. Use a unique checkpoint directory per independent run.

    The state of the optimizer itself is not persisted by this class; callers
    should deterministically replay from their seed to restore populations.
    The ledger prevents resubmission of jobs whose IDs were successfully saved.
    """

    def __init__(self, backend, service, checkpoint_dir, *, shots=4096,
                 optimization_level=1, seed_transpiler=11, sampler_factory=None):
        if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
            raise ValueError('Hardware shots must be a positive integer')
        self.backend = backend
        self.service = service
        self.shots = shots
        self.optimization_level = optimization_level
        self.seed_transpiler = seed_transpiler
        self.directory = Path(checkpoint_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sampler_factory = sampler_factory

    def execute_generation(self, circuits, generation):
        """Return counts in the same order as submitted circuits."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        if not circuits or any(c.num_clbits != c.num_qubits or not c.num_clbits for c in circuits):
            raise ValueError('All IBM circuits must measure every qubit exactly once')
        pm = generate_preset_pass_manager(backend=self.backend,
                optimization_level=self.optimization_level, seed_transpiler=self.seed_transpiler)
        compiled = pm.run(circuits)
        if not isinstance(compiled, list):
            compiled = [compiled]
        fingerprint = _digest(compiled)
        path = self.directory / f'generation_{generation:04d}.json'
        if path.exists():
            record = json.loads(path.read_text())
            for field, expected in [('fingerprint', fingerprint), ('shots', self.shots),
                                    ('backend', self.backend.name), ('num_circuits', len(circuits))]:
                if record.get(field) != expected:
                    raise ValueError(f'Checkpoint mismatch on {field}; refusing hardware resubmission')
            if 'counts' in record:
                return record['counts']
            job_id = record.get('job_id')
            if not job_id:
                raise RuntimeError('Submission status uncertain: inspect IBM jobs manually before retrying')
            job = self.service.job(job_id)
        else:
            # An intent file prevents a silent duplicate submission if submission fails
            # or the process stops before the job ID can be persisted.
            record = dict(schema_version=1, generation=generation,
                          fingerprint=fingerprint, shots=self.shots,
                          backend=self.backend.name, num_circuits=len(circuits),
                          submitted_at=datetime.now(timezone.utc).isoformat(),
                          state='submission_intent')
            atomic_json(path, record)
            if self.sampler_factory is None:
                from qiskit_ibm_runtime import SamplerV2
                sampler = SamplerV2(mode=self.backend)
            else:
                sampler = self.sampler_factory(self.backend)
            job = sampler.run(compiled, shots=self.shots)
            record['job_id'] = job.job_id()
            record['state'] = 'submitted'
            atomic_json(path, record)
        result = job.result()
        if len(result) != len(compiled):
            raise ValueError('Sampler returned incorrect number of circuit results')
        counts = []
        for original, circuit, pub in zip(circuits, compiled, result):
            registers = list(circuit.cregs)
            logical_width = original.num_qubits

            if len(registers) != 1 or registers[0].size != logical_width:
                raise ValueError('Exactly one full-width measurement register required')

            counts.append(
                _counts(pub, registers[0].name, self.shots, logical_width)
            )
        record['counts'] = counts
        record['state'] = 'completed'
        atomic_json(path, record)
        return counts
