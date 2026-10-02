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


if __name__=='__main__':
    unittest.main()
