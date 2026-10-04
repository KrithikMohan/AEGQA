"""Offline checks that the reorganized package and notebook remain usable."""
import ast
import importlib
import json
import unittest
from aeqga.paths import PROJECT_ROOT, output_path


class LayoutTests(unittest.TestCase):
    def test_stage_imports(self):
        for name in ['selection.population','encoding.amplitude_encoding',
                     'genetic_operators.quantum_gates','decoding.measurement_decoding',
                     'evolution.aeqga_algorithm']:
            self.assertIsNotNone(importlib.import_module('aeqga.steps.'+name))

    def test_output_paths(self):
        for kind in ['png','svg','pdf','json']:
            self.assertEqual(output_path(kind,'example.'+kind),
                             PROJECT_ROOT/'outputs'/kind/('example.'+kind))
        with self.assertRaises(ValueError): output_path('unknown','example')

    def test_notebook_code_parses(self):
        notebook=json.loads((PROJECT_ROOT/'notebooks/AEQGA.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                source=''.join(cell['source'])
                if not any(line.lstrip().startswith(('%','!')) for line in source.splitlines()):
                    ast.parse(source)

    def test_notebook_has_only_current_scientific_workflow(self):
        notebook=json.loads((PROJECT_ROOT/'notebooks/AEQGA.ipynb').read_text())
        code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        for obsolete in ['PantheonProblem(', 'load_pantheon(', 'chi2_pantheon_fixed_M',
                         'compute_kde_contours', 'run_qga_aeqga', 'MockProblem']:
            self.assertNotIn(obsolete,code)
        self.assertIn('PantheonPlusProblem(',code)
        self.assertIn('BAOCMBProblem(',code)
        self.assertIn('objective_contours(sne_problem,',code)
        config=next(c for c in notebook['cells'] if c['id']=='configuration')
        source=''.join(config['source'])
        for toggle in ['RUN_ENSEMBLE = True','RUN_CMB_BAO = False','RUN_STATEVECTOR = False',
                       'RUN_HQGA_COMPARISON = False']:
            self.assertIn(toggle,source)
        self.assertIn('from aeqga.integrations.hqga import', code)
        self.assertIn('hqga_algorithm.runQGA(', code)
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'],[])

    def test_retired_apis_are_not_executable(self):
        from aeqga.likelihoods import pantheon_problem
        from aeqga.steps.evolution import aeqga_algorithm
        from aeqga.steps.encoding import amplitude_encoding
        for name in ['PantheonProblem','load_pantheon','chi2_pantheon_fixed_M',
                     'compute_kde_contours','chi2_bao','chi2_cmb']:
            self.assertFalse(hasattr(pantheon_problem,name))
        for name in ['run_aeqga','run_qga_aeqga','run_aeqga_iterations','plot_convergence']:
            self.assertFalse(hasattr(aeqga_algorithm,name))
        self.assertFalse(hasattr(amplitude_encoding,'build_amplitude_circuit_with_measure'))


if __name__=='__main__':
    unittest.main()
