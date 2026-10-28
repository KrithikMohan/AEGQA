"""Exact independent-block/stage simulation of the archived IBM noise model.

The packed circuit is still constructed on disjoint physical qubits. The local
Markovian noise model has no cross-block channels, and every stage resets all
its qubits. We therefore evolve each stage's density matrix once, then draw the
requested finite-shot outcomes. This is distribution-equivalent to packed-shot
execution under this model, not a model for crosstalk or live IBM execution.
"""
from __future__ import annotations

import time
import warnings
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import Initialize
from qiskit.transpiler.exceptions import TranspilerError
from aeqga_ibm_noise import (SnapshotBackend, compile_native, simulator,
                            stage_measurement_wires, in_classical_order, asymmetric_readout)

_WORKERS = {}


def compile_stable_native(circuit, backend, seed):
    """Repair only floating-point state-preparation synthesis failures.

    Qiskit 2.3 can reject a normalized amplitude vector at machine precision
    during its internal isometry synthesis. A uniform rescaling preserves the
    represented normalized state. The normal compilation path is untouched.
    Every repaired vector must differ by less than 1e-14 per component, and
    the rescaled vector must pass the same isometry decomposition.
    """
    options = dict(routing_method='sabre', optimization_level=3, layout_method='sabre')
    try:
        return compile_native(circuit, backend, seed, **options)
    except TranspilerError as error:
        if 'state_preparation' not in str(error):
            raise
    repaired = circuit.copy()
    changes = []
    for index, item in enumerate(repaired.data):
        if item.operation.name != 'initialize':
            continue
        try:
            item.operation.definition.decompose(reps=6)
            continue
        except ValueError as error:
            if 'not unitary' not in str(error):
                raise
        original = np.asarray(item.operation.params, dtype=complex)
        unit = original/np.linalg.norm(original)
        for ulps in (0, 1, -1, 2, -2, 4, -4):
            adjusted = unit*(1+ulps*np.finfo(float).eps)
            delta = float(np.max(np.abs(adjusted-original)))
            if delta > 1e-14:
                raise AssertionError('State-preparation repair exceeds roundoff tolerance.')
            replacement = Initialize(adjusted)
            try:
                replacement.definition.decompose(reps=6)
            except ValueError as error:
                if 'not unitary' not in str(error):
                    raise
                continue
            repaired.data[index] = item.replace(operation=replacement)
            changes.append({'instruction_index':index, 'max_amplitude_change':delta,
                            'norm_before':float(np.linalg.norm(original)),
                            'norm_after':float(np.linalg.norm(adjusted))})
            break
        else:
            raise RuntimeError('State-preparation roundoff repair did not yield a valid isometry.')
    if not changes:
        raise RuntimeError('State-preparation compilation failed without a recognized roundoff defect.')
    native = compile_native(repaired, backend, seed, **options)
    native.metadata['state_preparation_roundoff_repairs'] = changes
    return native


def stage_circuits(compiled):
    """Remove terminal measurements while retaining all pre-measurement idle time.

    Barriers separate complete reset/prepare/measure stages. Operations after
    a wire's measurement are discarded only until the next all-wire reset stage;
    they cannot alter the already recorded classical outcome.
    """
    specs = compiled.metadata['stages']
    width = compiled.num_qubits
    result, current, measured, stage_index = [], QuantumCircuit(width), set(), 0
    order = stage_measurement_wires(compiled)
    for item in compiled.data:
        name = item.operation.name
        qubits = [compiled.find_bit(q).index for q in item.qubits]
        if name == 'barrier':
            if len(measured) == width:
                current.save_probabilities()
                result.append((current, [order[c] for c in specs[stage_index]['clbits']]))
                stage_index += 1
                current, measured = QuantumCircuit(width), set()
            continue
        if name == 'measure':
            c = compiled.find_bit(item.clbits[0]).index
            if c not in specs[stage_index]['clbits']:
                raise AssertionError('Measurement crossed a sequential-stage boundary.')
            measured.add(qubits[0])
        elif set(qubits) & measured:
            if name != 'delay' or len(qubits) != 1:
                raise AssertionError('A quantum operation after measurement needs explicit handling.')
        else:
            current.append(item.operation, qubits)
    if measured:
        if len(measured) != width:
            raise AssertionError('Incomplete terminal measurement.')
        current.save_probabilities()
        result.append((current, [order[c] for c in specs[stage_index]['clbits']]))
    if len(result) != len(specs):
        raise AssertionError('Stage extraction changed the number of stages.')
    return result


def normalized(values):
    values = np.asarray(values, dtype=float)
    # Thousands of composed calibrated channels can accumulate a few ppm of
    # trace drift. Record it in every stage; reject drift above ten ppm.
    if values.min() < -1e-12 or abs(values.sum()-1) > 1e-5:
        raise AssertionError(f'Invalid probability normalization: {values.min()}, {values.sum()}')
    values = np.maximum(values, 0)
    return values/values.sum()


def simulate_block(task):
    """Process-pool worker: native compilation and exact stage probabilities."""
    circuit, physical, device, root, seed, mode = task
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    key = (tuple(physical), device, str(root))
    if key not in _WORKERS:
        backend = SnapshotBackend(device, snapshot_root=root, physical_qubits=physical)
        _WORKERS[key] = (backend, simulator(backend, 'ideal', readout=False, method='statevector'),
                         simulator(backend, 'noisy', readout=False, method='density_matrix'))
    backend, ideal_simulator, noisy_simulator = _WORKERS[key]
    start = time.perf_counter()
    native = compile_stable_native(circuit, backend, seed)
    compile_seconds = time.perf_counter()-start
    native.metadata.update({'measurement_physical_order':
        [physical[q] for q in stage_measurement_wires(native)], 'duration_dt': int(native.duration)})
    parts = stage_circuits(native)
    start = time.perf_counter()
    ideal_result = ideal_simulator.run([p[0] for p in parts], shots=1, seed_simulator=seed).result()
    noisy_result = (noisy_simulator.run([p[0] for p in parts], shots=1,
                                       seed_simulator=seed).result() if mode == 'noisy' else None)
    if not ideal_result.success or (noisy_result is not None and not noisy_result.success):
        raise RuntimeError('A stage probability simulation failed.')
    stages = []
    for i, (_, order) in enumerate(parts):
        raw_ideal = in_classical_order(ideal_result.data(i)['probabilities'], order)
        ideal = normalized(raw_ideal)
        raw_noisy = (asymmetric_readout(in_classical_order(
            noisy_result.data(i)['probabilities'], order), backend, order)
            if noisy_result is not None else raw_ideal)
        noisy = normalized(raw_noisy)
        stages.append({'ideal': ideal, 'probabilities': noisy,
                       'noise_tvd': float(np.abs(ideal-noisy).sum()/2),
                       'probability_trace_error': float(abs(raw_noisy.sum()-1)),
                       'ideal_probability_trace_error': float(abs(raw_ideal.sum()-1))})
    simulation_seconds = time.perf_counter()-start
    return native, stages, {'compile_cpu_seconds': compile_seconds,
                            'simulation_cpu_seconds': simulation_seconds,
                            'duration_seconds': float(native.duration)*backend.dt}
