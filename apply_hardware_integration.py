"""Apply the narrowly-scoped QPU hook to a checkout of KrithikMohan/AEGQA.

Run at the repository root: python apply_hardware_integration.py
Refuses unknown source layouts; original simulator loop stays untouched.
"""
from pathlib import Path

path = Path('aeqga/steps/evolution/aeqga_algorithm.py')
s = path.read_text(encoding="utf-8")
if 'hardware_executor=None' in s:
    raise SystemExit('Hardware integration already applied')
old = '    *, seed=None, execution_mode="aer", quiet=False,\n'
assert s.count(old) == 1, 'Unknown run_aeqga_dual signature'
s = s.replace(old, '    *, seed=None, execution_mode="aer", quiet=False, hardware_executor=None,\n')
old = '''    if execution_mode not in {"aer", "statevector_shots"}:
        raise ValueError("execution_mode must be aer or statevector_shots")
'''
assert s.count(old) == 1
s = s.replace(old, '''    if execution_mode not in {"aer", "statevector_shots"}:
        raise ValueError("execution_mode must be aer or statevector_shots")
    if hardware_executor is not None and params.num_shots <= 0:
        raise ValueError("IBM hardware requires positive shots")
''')
needle = '''        use_measure = params.num_shots > 0

        for d in range(n_dim):
'''
assert s.count(needle) == 1
replacement = '''        use_measure = params.num_shots > 0

        # IBM path: assemble ALL dimension/subset circuits in the original
        # (random, elite) order before requesting one Sampler job per generation.
        # Classical selection and decoding remain unchanged.
        hardware_counts = None
        if hardware_executor is not None:
            hardware_circuits = []
            for hardware_d in range(n_dim):
                hardware_circuits.append(build_dimension_circuit(
                    P_rand[:, hardware_d], params.p_cross, params.p_mut,
                    params.num_shots, mode="measure"))
                hardware_circuits.append(build_dimension_circuit(
                    P_elite_copy[:, hardware_d], params.p_cross, params.p_mut,
                    params.num_shots, mode="measure"))
            hardware_counts = hardware_executor.execute_generation(hardware_circuits, gen)
            if len(hardware_counts) != 2 * n_dim:
                raise ValueError("Executor count cardinality mismatch")

        for d in range(n_dim):
'''
s=s.replace(needle,replacement)
old = '''            qc_rand = build_dimension_circuit(
                P_rand[:, d],
                p_cross=params.p_cross,
                p_mut=params.p_mut,
                num_shots=params.num_shots,
                mode="measure" if use_measure else "statevector",
            )
'''
assert s.count(old)==1
s=s.replace(old, '''            qc_rand = (None if hardware_counts is not None else build_dimension_circuit(
                P_rand[:, d],
                p_cross=params.p_cross,
                p_mut=params.p_mut,
                num_shots=params.num_shots,
                mode="measure" if use_measure else "statevector",
            ))
''')
old='''            qc_elite = build_dimension_circuit(
                P_elite_copy[:, d],
                p_cross=params.p_cross,
                p_mut=params.p_mut,
                num_shots=params.num_shots,
                mode="measure" if use_measure else "statevector",
            )
'''
assert s.count(old)==1
s=s.replace(old,'''            qc_elite = (None if hardware_counts is not None else build_dimension_circuit(
                P_elite_copy[:, d],
                p_cross=params.p_cross,
                p_mut=params.p_mut,
                num_shots=params.num_shots,
                mode="measure" if use_measure else "statevector",
            ))
''')
# Both branches of both redundant measure/statevector paths get a consistent
# execution source, while remaining simulator behavior stays unmodified.
assert s.count('counts_rand = circuit_counts(qc_rand)')==2
assert s.count('counts_elite = circuit_counts(qc_elite)')==2
s=s.replace('counts_rand = circuit_counts(qc_rand)',
            'counts_rand = (hardware_counts[2*d] if hardware_counts is not None else circuit_counts(qc_rand))')
s=s.replace('counts_elite = circuit_counts(qc_elite)',
            'counts_elite = (hardware_counts[2*d+1] if hardware_counts is not None else circuit_counts(qc_elite))')
path.write_text(s, encoding="utf-8")
print('Integrated IBM generation-batch hook in',path)
