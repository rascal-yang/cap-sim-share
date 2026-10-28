"""Offline interface checks; these do not start physics or invoke a model."""
import copy
import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name('adapter.py')
    assert path.exists(), 'Independent RoboICL adapter is not implemented'
    spec = importlib.util.spec_from_file_location('roboicl_adapter', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reject_replay_and_mode_bypass():
    mod = module()
    valid = {'analysis': {'evidence': ['observed'], 'hypothesis': 'reach', 'change': 'initial'},
             'actions': [{'stage': 'approach', 'duration_s': .5, 'right_position': [0, 0, .8],
                          'right_quaternion_wxyz': [1, 0, 0, 0], 'replan_after': True}]}
    assert mod.validate_decision(valid) == valid
    for field, value in [('body_keyframes', []), ('tokens', 190), ('mode', 0)]:
        bad = copy.deepcopy(valid)
        bad['actions'][0][field] = value
        with pytest.raises(ValueError):
            mod.validate_decision(bad)
    bad = copy.deepcopy(valid)
    bad['reuse_prefix'] = 'attempt_001'
    with pytest.raises(ValueError):
        mod.validate_decision(bad)
    bad = copy.deepcopy(valid)
    bad['actions'][0]['stage'] = 'native_prefix'
    with pytest.raises(ValueError):
        mod.validate_decision(bad)


def test_nan_or_open_loop_output_rejected():
    mod = module()
    for action in [{'stage': 'approach', 'duration_s': float('nan'), 'replan_after': True},
                   {'stage': 'approach', 'duration_s': .5}]:
        with pytest.raises(ValueError):
            mod.validate_decision({'analysis': {'evidence': ['x'], 'hypothesis': 'y'}, 'actions': [action]})


def test_context_does_not_leak_reference_or_source_paths():
    mod = module()
    request = {'attempt': 1, 'phase': 'solve', 'task': {'instruction': 'pick apple',
               'resources': {'initial_state': '/secret/source.npz'}, 'references': ['original.json'],
               'budget': {'sim_seconds': 30}, 'acceptance': {'kind': 'native'}},
               'interface': {'action': {'body_keyframes': 'forbidden', 'right_position': 'xyz'}},
               'observation': {'time': 0}, 'references': [{'actions': ['secret raw tokens']}],
               'prefixes': [{'actions': ['secret prefix']}], 'successful_trajectory': 'secret',
               'skills': ['hidden skill']}
    payload = mod.build_context(request, {'blocks': []}, [])
    import json
    raw = json.dumps(payload)
    assert 'secret' not in raw and 'original.json' not in raw and 'hidden skill' not in raw
    assert 'body_keyframes' not in payload['interface']['action']
    assert payload['observation']['time'] == 0


def test_memory_keeps_actual_results_not_unexecuted_suffix():
    mod = module()
    memory = mod.InteractionMemory(limit=3)
    memory.start({'time': 0})
    for i in range(5):
        memory.commit([{'stage': 'approach', 'duration_s': .5}], {'time': (i + 1) * .5},
                      {'completed': True, 'end_s': (i + 1) * .5})
    kept = memory.blocks()
    assert len(kept) == 3
    assert kept[0]['observation_before']['time'] == 0
    assert kept[-1]['observation_after']['time'] == 2.5
    assert kept[-1]['receipt']['completed'] is True
    assert kept[1]['gap_before'] is True


def test_launch_gate_rejects_modified_contract_or_code(tmp_path):
    mod = module()
    assert hasattr(mod, 'verify_frozen'), 'Frozen launch gate is not implemented'
    import hashlib
    import json
    source = tmp_path/'source.py'
    source.write_text('original')
    contract = {'record_status': 'frozen', 'preflight': {'passed': True},
                'pipeline': {'source_snapshot_and_hashes': {
                    str(source): hashlib.sha256(source.read_bytes()).hexdigest()}}}
    path = tmp_path/'experiment.json'
    path.write_text(json.dumps(contract))
    (tmp_path/'contract.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    mod.verify_frozen(tmp_path)
    source.write_text('changed')
    with pytest.raises(ValueError):
        mod.verify_frozen(tmp_path)
    source.write_text('original')
    path.write_text(json.dumps({**contract, 'record_status': 'draft'}))
    with pytest.raises(ValueError):
        mod.verify_frozen(tmp_path)
