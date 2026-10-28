"""Zero-physics validation tests for the controlled experiment freeze entry."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'cap-sim/roboicl_reproduction/freeze_controlled.py'
BASE = ROOT / 'experiments/20261015_1047_roboicl_origin_live'


class FreezeContractTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SOURCE.is_file(), 'controlled contract validator has not been implemented')
        spec = importlib.util.spec_from_file_location('freeze_controlled', SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_failed_or_unexecuted_tests_cannot_authorize_launch(self):
        m = self.load_module()
        good = dict(command='python -m pytest -q', exit_code=0, passed=5, failed=0, physics_steps=0)
        m.validate_tests(good)
        for changes in ({'exit_code':1}, {'passed':0}, {'passed':True}, {'failed':1}, {'physics_steps':1}, {'command':''}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                m.validate_tests(dict(good, **changes))
        for key in ('command', 'exit_code', 'passed', 'failed', 'physics_steps'):
            bad = good.copy(); del bad[key]
            with self.subTest(missing=key), self.assertRaises(ValueError): m.validate_tests(bad)

    def test_missing_or_changed_loaded_models_cannot_authorize_launch(self):
        m = self.load_module()
        runtime = json.loads((BASE / 'runtime.json').read_text())
        hashes = json.loads((BASE / 'experiment.json').read_text())['runtime']['model_hashes']
        m.validate_runtime(runtime, hashes)
        for change in ('missing_session', 'gpu_session', 'model_hash', 'version', 'not_loaded'):
            bad = copy.deepcopy(runtime)
            key = next(iter(hashes))
            if change == 'missing_session': del bad['actual_session_providers'][key]
            if change == 'gpu_session': bad['actual_session_providers'][key] = ['CUDAExecutionProvider']
            if change == 'model_hash': bad['model_sha256'][key] = '0' * 64
            if change == 'version': bad['versions']['onnxruntime'] = 'other'
            if change == 'not_loaded': bad['sessions_loaded'] = False
            with self.subTest(change=change), self.assertRaises(ValueError): m.validate_runtime(bad, hashes)

    def test_creation_seconds_are_preserved_and_id_cannot_disagree(self):
        m = self.load_module()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / '20261015_1205_synthetic_fixture'; path.mkdir()
            created = dict(experiment_id=path.name, created_at='2026-10-15T12:05:23+08:00')
            (path / 'created.json').write_text(json.dumps(created))
            self.assertEqual(m.read_creation(path), '2026-10-15T12:05:23+08:00')
            for value in ('2026-10-15T12:05+08:00', '2026-10-15T12:06:23+08:00', '2026-10-15T12:05:23'):
                created['created_at'] = value
                (path / 'created.json').write_text(json.dumps(created))
                with self.subTest(value=value), self.assertRaises(ValueError): m.read_creation(path)

    def test_task_cannot_change_physics_acceptance_or_budget(self):
        m = self.load_module()
        baseline = json.loads((BASE / 'task.json').read_text())
        task = copy.deepcopy(baseline)
        task['budget'].update(max_rows=5, recovery_attempts=2)
        m.validate_task(task, baseline)
        for field, value in (('scene', {'translation':[0, .1, 0]}), ('acceptance', {'kind':'anything'}), ('resources', {})):
            bad = copy.deepcopy(task); bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): m.validate_task(bad, baseline)
        bad = copy.deepcopy(task); bad['budget']['attempts'] = 4
        with self.assertRaises(ValueError): m.validate_task(bad, baseline)

    def test_train_relocation_preserves_measured_values_and_image_bytes(self):
        m = self.load_module()
        with tempfile.TemporaryDirectory() as temp:
            old = Path(temp) / 'old'; out = Path(temp) / 'new'
            old.mkdir(); out.mkdir(); (old/'train_images').mkdir(); (out/'train_images').mkdir()
            (old/'train_images'/'frame.png').write_bytes(b'original image')
            (out/'train_images'/'frame.png').write_bytes(b'original image')
            demo = {'observations':[{'time':0., 'measured_fingers':[1.], 'images':[str(old/'train_images'/'frame.png')]}]}
            source = {'physics_steps':0, 'physics':{'opt.timestep':.005}, 'validation':{'success':True}}
            (old/'train_memory.json').write_text(json.dumps(demo)); (old/'source_evidence.json').write_text(json.dumps(source))
            demo['observations'][0]['images'] = [str(out/'train_images'/'frame.png')]
            (out/'train_memory.json').write_text(json.dumps(demo)); (out/'source_evidence.json').write_text(json.dumps(source))
            self.assertEqual(len(m.validate_train_copy(out, old)), 1)
            demo['observations'][0]['measured_fingers'] = [2.]
            (out/'train_memory.json').write_text(json.dumps(demo))
            with self.assertRaises(ValueError): m.validate_train_copy(out, old)
            demo['observations'][0]['measured_fingers'] = [1.]
            (out/'train_memory.json').write_text(json.dumps(demo)); (out/'train_images'/'frame.png').write_bytes(b'altered')
            with self.assertRaises(ValueError): m.validate_train_copy(out, old)

    def test_contract_requires_template_fields_and_stage_interfaces(self):
        m = self.load_module()
        contract = json.loads((BASE/'experiment.json').read_text())
        template = json.loads((ROOT/'cap-sim/docs/experiment-record.template.json').read_text())
        m.validate_contract(contract, template)
        del contract['control']['stage_interfaces'][0]['handoff_rule']
        with self.assertRaises(ValueError): m.validate_contract(contract, template)


if __name__ == '__main__': unittest.main()
