"""Launch integrity and handoff tests; these never advance a simulator."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def native_module():
    from humanoid_mimicgen import native
    return native


def seal(path, data):
    path.write_text(json.dumps(data))
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())


def contract_fixture(tmp_path):
    template = Path(__file__).resolve().parents[2] / 'docs/experiment-record.template.json'
    data = json.loads(template.read_text())
    def fill(value):
        if value is None:
            return 'explicit fixture evidence'
        if isinstance(value, dict):
            return {k: fill(v) for k, v in value.items()}
        if isinstance(value, list):
            return value or ['explicit fixture evidence']
        return value
    data = fill(data)
    data['record_status'] = 'frozen'
    data['experiment_id'] = '20261015_1200_native_test'
    data['preflight']['passed'] = True
    evidence = tmp_path / 'evidence.json'
    evidence.write_text('{"measured": true}')
    runtime = tmp_path / 'runtime.json'
    runtime.write_text(json.dumps({
        'passed': True, 'sessions_loaded': True,
        'versions': {'tensorrt': '10.13.0.35', 'onnxruntime': '1.23.2',
                     'mujoco': '3.3.6', 'gear_sonic': '0.1.0'},
        'sonic_version': '1.0', 'tensorrt_used_for_sonic_inference': False,
        'actual_session_providers': {'encoder': ['CPUExecutionProvider'],
                                     'decoder': ['CPUExecutionProvider'],
                                     'planner': ['CPUExecutionProvider']},
        'model_sha256': {'encoder': '1'*64, 'decoder': '2'*64, 'planner': '3'*64},
    }))
    data['runtime']['actual_loaded_session_evidence'] = str(runtime)
    data['physics']['asset_and_initial_state_manifest'] = str(evidence)
    data['physics']['physics_parameters_manifest'] = str(evidence)
    data['preflight']['comparison_diff_path'] = str(evidence)
    data['preflight']['evidence_paths'] = [str(evidence), str(runtime)]
    data['preflight']['evidence_sha256'] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (evidence, runtime)
    }
    entry = {k: 'explicit fixture evidence' for k in data['control']['stage_entry_required_fields']}
    data['control']['stage_interfaces'] = [entry]
    data['control']['fallback_allowed'] = False
    path = tmp_path / 'experiment.json'
    seal(path, data)
    return path, data


def test_unfrozen_contract_refuses_before_simulator_import(tmp_path):
    # Removing the launch gate would import/construct HUMANO from a template.
    native = native_module()
    path = tmp_path / 'experiment.json'
    seal(path, {'record_status': 'template_not_executable'})
    with pytest.raises(ValueError, match='frozen'):
        native.run_smoke({}, {}, {}, {}, tmp_path/'absent.parquet', tmp_path/'run', path)
    assert not (tmp_path/'run').exists()


def test_contract_seal_and_evidence_tampering_are_rejected(tmp_path):
    native = native_module()
    path, data = contract_fixture(tmp_path)
    assert native.validate_contract(path)['record_status'] == 'frozen'
    data['purpose']['hypothesis'] = 'changed after freeze'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='seal'):
        native.validate_contract(path)
    seal(path, data)
    Path(data['physics']['physics_parameters_manifest']).write_text('changed physics')
    with pytest.raises(ValueError, match='hash'):
        native.validate_contract(path)


def test_unloaded_runtime_and_incomplete_contract_cannot_launch(tmp_path):
    native = native_module()
    path, data = contract_fixture(tmp_path)
    runtime = Path(data['runtime']['actual_loaded_session_evidence'])
    value = json.loads(runtime.read_text()); value['sessions_loaded'] = False
    runtime.write_text(json.dumps(value))
    data['preflight']['evidence_sha256'][str(runtime)] = hashlib.sha256(runtime.read_bytes()).hexdigest()
    seal(path, data)
    with pytest.raises(ValueError, match='loaded'):
        native.validate_contract(path)
    data['physics']['pd_parameters'] = None
    seal(path, data)
    with pytest.raises(ValueError, match='pd_parameters'):
        native.validate_contract(path)


def test_original_scene_gate_rejects_variant_and_inconsistent_cadence():
    native = native_module()
    native.validate_scene({'scene': {'translation': [0, 0, 0]}})
    for scene in ({'translation': [0, .1, 0]}, {'table_delta': .075}, {'apple_scale': 1.1}):
        with pytest.raises(ValueError, match='original scene'):
            native.validate_scene({'scene': scene})


def test_handoff_reanchors_transit_without_changing_prefix_or_contact():
    # A global shift would corrupt the reusable contact target or retained prefix.
    native = native_module()
    time = np.arange(0, 8.005, .005)
    pose = np.tile([1., 2., 3., 1., 0., 0., 0.], (len(time), 1))
    generated = {'time_s': time, 'left_pose': pose.copy(), 'right_pose': pose.copy(),
                 'metadata': {}}
    annotation = {'stages': [
        {'name': 'approach', 'kind': 'transit', 'start_frame': 0, 'end_frame': 1400},
        {'name': 'contact_grasp', 'kind': 'contact', 'start_frame': 1400, 'end_frame': len(time)},
    ]}
    observation = {'left_position': [0., 2., 3.], 'right_position': [2., 2., 3.],
                   'left_quaternion_wxyz': [1., 0., 0., 0.],
                   'right_quaternion_wxyz': [1., 0., 0., 0.]}
    result, audit = native.reanchor_handoff(generated, annotation, observation, 5.)
    assert np.array_equal(result['right_pose'][:1001], generated['right_pose'][:1001])
    assert np.array_equal(result['right_pose'][1400:], generated['right_pose'][1400:])
    np.testing.assert_allclose(result['right_pose'][1200, :3], [1.5, 2., 3.])
    assert audit['contact_modified'] is False
    assert np.array_equal(generated['right_pose'], pose)


def test_handoff_rejects_direct_entry_into_contact():
    native = native_module()
    generated = {'time_s': np.array([4.995, 5., 5.005, 5.01])}
    annotation = {'stages': [{'name': 'grasp', 'kind': 'contact',
                             'start_frame': 0, 'end_frame': 4}]}
    with pytest.raises(ValueError, match='transit'):
        native.reanchor_handoff(generated, annotation, {}, 5.)


def test_safety_gates_use_continuous_contact_and_table_time():
    native = native_module()
    monitor = native.SafetyMonitor()
    assert monitor.enter('lift_transport', 0., ['lift_transport']) == 'hand_contact_gate'
    assert monitor.enter('contact_grasp', 0.) is None
    for _ in range(23):
        assert monitor.update(hand_force=.2, table_force=0., stable=True, warning=False) is None
    assert monitor.enter('lift_transport', monitor.contact_s, ['lift_transport']) == 'hand_contact_gate'
    monitor.update(hand_force=.2, table_force=0., stable=True, warning=False)
    assert monitor.enter('lift_transport', monitor.contact_s, ['lift_transport']) is None
    for _ in range(5):
        assert monitor.update(hand_force=0., table_force=0., stable=True, warning=False) is None
    assert monitor.update(hand_force=0., table_force=0., stable=True, warning=False) == 'contact_lost'
    monitor.enter('approach', monitor.contact_s)
    for _ in range(9):
        assert monitor.update(hand_force=0., table_force=6., stable=True, warning=False) is None
    assert monitor.update(hand_force=0., table_force=6., stable=True, warning=False) == 'unexpected_table_contact'


def test_contact_gate_uses_explicit_configuration_not_stage_wording():
    native = native_module()
    monitor = native.SafetyMonitor()
    assert monitor.enter('grasp acquisition and initial lift', 0., ['continued lift']) is None
    assert monitor.enter('continued lift', 0., ['continued lift']) == 'hand_contact_gate'


def test_gpt_boundary_does_not_reset_continuous_table_collision():
    monitor = native_module().SafetyMonitor()
    monitor.enter('approach', 0.)
    for _ in range(6):
        assert monitor.update(hand_force=0., table_force=6., stable=True, warning=False) is None
    monitor.enter('contact', 0.)
    for _ in range(3):
        assert monitor.update(hand_force=0., table_force=6., stable=True, warning=False) is None
    assert monitor.update(hand_force=0., table_force=6., stable=True, warning=False) == 'unexpected_table_contact'


def test_input_binding_detects_changed_finger_reference(tmp_path):
    native = native_module()
    parquet = tmp_path/'tokens.parquet'; parquet.write_bytes(b'prefix')
    source = {'time_s': np.array([.005, .01]), 'right_fingers': np.zeros((2, 7)), 'metadata': {}}
    original = native.input_hashes({}, source, source, {}, parquet)
    changed = copy.deepcopy(source); changed['right_fingers'][0, 0] = .1
    actual = native.input_hashes({}, source, changed, {}, parquet)
    assert original['generated'] != actual['generated']
    assert original['demo'] == actual['demo']


def test_success_package_binds_exact_evidence_and_rejects_failed_attempt(tmp_path):
    # Package only accepted physical evidence; never invent sparse action lists.
    native = native_module()
    from cap_sim.io import digest
    attempt = tmp_path/'run/attempts/001'
    attempt.mkdir(parents=True)
    samples = [{'time': .005, 'dt': .005, 'native_success': True, 'stable': True}]
    for filename, payload in {
        'samples.json': samples, 'contacts.json': [{'hand_object_N': 2.}],
        'independent_validation.json': {'success': True, 'physics_steps': 0},
        'commands.json': [{'time_s': .005, 'mode': 1}],
    }.items():
        (attempt/filename).write_text(json.dumps(payload))
    (attempt/'states.npz').write_bytes(b'exact recorded state fixture')
    (attempt/'scene.mjb').write_bytes(b'exact scene fixture')
    (attempt/'executed_reference.npz').write_bytes(b'dense reference fixture')
    accepted = {'success': True, 'failure': None, 'acceptance': {'success': True, 'sim_seconds': .005},
                'observation': {'object_details': {'target': {'root_body': 'apple_001'}}}}
    refused = copy.deepcopy(accepted); refused['success'] = False; refused['failure'] = 'unexpected_table_contact'
    with pytest.raises(ValueError, match='accepted'):
        native.write_success_package(attempt, {'name': 'apple'}, refused)
    assert not (tmp_path/'run/success').exists()
    path = native.write_success_package(attempt, {'name': 'apple'}, accepted)
    trajectory = json.loads(path.read_text())
    assert trajectory['source_attempt'] == str(attempt.resolve())
    assert trajectory['samples_sha256'] == digest(samples)
    assert trajectory['acceptance'] == accepted['acceptance']
    assert 'actions' not in trajectory
    assert trajectory['dense_reference']['sha256'] == hashlib.sha256(b'dense reference fixture').hexdigest()
    for name in ['states.npz', 'scene.mjb', 'samples.json', 'contacts.json', 'independent_validation.json']:
        assert (path.parent/name).read_bytes() == (attempt/name).read_bytes()
