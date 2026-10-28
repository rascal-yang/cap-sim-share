from pathlib import Path
import copy
import json
import numpy as np
import pytest

from humanoid_mimicgen.augmentation.contracts import load_job, validate_job
from humanoid_mimicgen.augmentation.conditions import sample_conditions
from humanoid_mimicgen.augmentation.scene import transform_initial
from humanoid_mimicgen.workflow.contracts import WorkflowError


def job():
    return {'schema_version': 1, 'request_id': 'test', 'baseline_request': '/not-used.json',
            'conditions': {'mode': 'uniform', 'seed': 17, 'count': 3,
                'robot_delta_xyyaw': [[-.2, .2], [0., 0.], [0., 0.]],
                'object_delta_xy': [[0., 0.], [-.1, .1]]},
            'budgets': {'max_cases': 3, 'wall_s': 600}, 'output_root': '/tmp/output',
            'render': False, 'export': True, 'stance_hint_frame': 'object_relative'}


def test_seed_reproduces_conditions():
    a = sample_conditions(job())
    assert a == sample_conditions(job())
    assert len(a) == 3 and len({x['case_id'] for x in a}) == 3
    assert all(-.2 <= x['robot_delta_xyyaw'][0] <= .2 for x in a)
    assert all(x['robot_delta_xyyaw'][1:] == [0., 0.] for x in a)
    other = job(); other['conditions']['seed'] = 18
    assert a != sample_conditions(other)


@pytest.mark.parametrize('patch', [
    {'extra': True}, {'render': 'false'}, {'schema_version': True},
    {'budgets': {'max_cases': 2, 'wall_s': 600}},
    {'conditions': {'mode': 'uniform', 'seed': -1, 'count': 3,
       'robot_delta_xyyaw': [[0, 0]]*3, 'object_delta_xy': [[0, 0]]*2}},
])
def test_invalid_inputs_fail_before_physics(patch):
    value = job(); value.update(patch)
    with pytest.raises(WorkflowError):
        sample_conditions(value)


def test_nonfinite_and_reversed_ranges_rejected():
    for bounds in ([0, float('nan')], [1, -1]):
        value = job(); value['conditions']['object_delta_xy'][0] = bounds
        with pytest.raises(WorkflowError): sample_conditions(value)


def test_relative_input_paths_are_resolved(tmp_path):
    value = job(); value.update(baseline_request='base.json', output_root='output')
    p = tmp_path/'job.json'; p.write_text(json.dumps(value))
    loaded = load_job(p)
    assert loaded['baseline_request'] == str(tmp_path/'base.json')
    assert loaded['output_root'] == str(tmp_path/'output')


def model_and_initial():
    import mujoco
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="pelvis" pos="0 0 1"><freejoint/><geom size=".1" mass="1"/></body>
    <body name="apple" pos="1 0 1"><freejoint/><geom size=".03" mass=".1"/></body>
    </worldbody></mujoco>''')
    return m, {'qpos': m.qpos0.copy(), 'qvel': np.zeros(m.nv), 'ctrl': np.zeros(m.nu)}


def test_object_shift_does_not_shift_robot(monkeypatch):
    import mujoco
    m, initial = model_and_initial()
    monkeypatch.setattr(mujoco, 'mj_step', lambda *a: pytest.fail('No physics allowed'))
    actual = transform_initial(m, initial, 'pelvis', 'apple', [0, 0, 0], [0, .1], 1e-5)
    np.testing.assert_array_equal(actual['qpos'][:7], initial['qpos'][:7])
    np.testing.assert_allclose(actual['qpos'][7:10], [1, .1, 1])
    np.testing.assert_array_equal(initial['qpos'][7:10], [1, 0, 1])


def test_robot_yaw_and_translation_leave_object_fixed():
    m, initial = model_and_initial()
    actual = transform_initial(m, initial, 'pelvis', 'apple', [.2, -.3, np.pi/2], [0, 0], 1e-5)
    np.testing.assert_allclose(actual['qpos'][:3], [.2, -.3, 1])
    np.testing.assert_allclose(actual['qpos'][3:7], [2**-.5, 0, 0, 2**-.5], atol=1e-12)
    np.testing.assert_array_equal(actual['qpos'][7:], initial['qpos'][7:])


def test_initial_penetration_rejected():
    m, initial = model_and_initial()
    with pytest.raises(WorkflowError, match='initial_collision'):
        transform_initial(m, initial, 'pelvis', 'apple', [0, 0, 0], [-1, 0], 1e-5)


def test_changed_input_rejected(tmp_path):
    from humanoid_mimicgen.augmentation.contracts import verify_inputs
    from humanoid_mimicgen.workflow.contracts import file_hash
    p = tmp_path/'evidence'; p.write_text('original')
    hashes = {str(p): file_hash(p)}
    verify_inputs(hashes)
    p.write_text('changed')
    with pytest.raises(WorkflowError, match='integrity_error'): verify_inputs(hashes)


def test_runtime_gpt_trigger_rejected(tmp_path):
    from humanoid_mimicgen.augmentation.contracts import baseline_request
    p = tmp_path/'baseline.json'
    p.write_text(json.dumps({'decision': {'mode': 'deterministic'},
        'strategy': {'adapter': 'g1_sonic_native_v1', 'policy': {'grasp_trigger': {}}}}))
    with pytest.raises(WorkflowError, match='trigger disabled'):
        baseline_request({'baseline_request': str(p)})


def test_explicit_conditions_reject_duplicate_and_boolean_values():
    value = job(); row = {'case_id': 'a', 'robot_delta_xyyaw': [0, 0, 0], 'object_delta_xy': [0, 0]}
    value['conditions'] = {'mode': 'explicit', 'cases': [row, copy.deepcopy(row)]}
    with pytest.raises(WorkflowError): sample_conditions(value)
    value['conditions']['cases'] = [row]; row['object_delta_xy'][0] = True
    with pytest.raises(WorkflowError): sample_conditions(value)


def test_materialized_contract_names_new_case_and_binds_job_skill(tmp_path, monkeypatch):
    import mujoco
    from humanoid_mimicgen.augmentation import scene
    from humanoid_mimicgen.workflow.contracts import file_hash
    m, initial = model_and_initial()
    model = tmp_path/'model.mjb'; mujoco.mj_saveModel(m, str(model), None)
    np.savez(tmp_path/'initial.npz', **initial)
    (tmp_path/'task.json').write_text(json.dumps({'resources':{'initial_state':'old'}}))
    (tmp_path/'record.json').write_text(json.dumps({'cases':[{'id':'historical'}], 'physics':{},
          'pipeline':{'skill_versions_and_hashes':[]}, 'preflight':{'evidence_sha256':{}}}))
    request={'files':{k:str(tmp_path/v) for k,v in [('model','model.mjb'),('initial','initial.npz'),('task','task.json'),('record','record.json')]},
        'strategy':{'bindings':{'root_body':'pelvis','object':'apple'}, 'policy':{'planning':{'penetration_tolerance_m':1e-5}},'provenance':{}}}
    monkeypatch.setattr(scene,'baseline_request',lambda j:copy.deepcopy(request))
    value=job(); value['output_root']=str(tmp_path/'output')
    condition={'case_id':'new_case','robot_delta_xyyaw':[0,0,0],'object_delta_xy':[0,.1]}
    path=scene.materialize_case(value,condition,tmp_path/'case')
    actual=json.loads(path.read_text()); record=json.loads(Path(actual['files']['record']).read_text())
    assert record['cases'][0]['id']=='new_case'
    assert record['cases'][0]['initial_state_sha256']==file_hash(actual['files']['initial'])
    assert json.loads(Path(actual['files']['augmentation_job']).read_text())['conditions']['seed']==17
    assert record['pipeline']['skill_versions_and_hashes'][-1]['sha256']==file_hash(actual['files']['augmentation_skill'])


def test_baseline_keeps_symlink_alias_for_hash_validation(tmp_path, monkeypatch):
    from humanoid_mimicgen.augmentation import contracts
    old=tmp_path/'old'; old.write_text('old'); new=tmp_path/'new'; new.write_text('new')
    alias=tmp_path/'targets'; alias.symlink_to(old)
    request={'decision':{'mode':'deterministic'},'strategy':{'adapter':'g1_sonic_native_v1','policy':{}},
             'files':{k:str(alias) for k in ('model','initial','targets','reference_states','task','record','alignment')}}
    path=tmp_path/'baseline.json'; path.write_text(json.dumps(request))
    monkeypatch.setattr(contracts,'validate_strategy',lambda s:s)
    monkeypatch.setattr(contracts,'validate_policy',lambda p:p)
    normalized=contracts.baseline_request({'baseline_request':str(path)})
    from humanoid_mimicgen.workflow.contracts import file_hash
    hashes={p:file_hash(p) for p in normalized['files'].values()}
    alias.unlink(); alias.symlink_to(new)
    with pytest.raises(WorkflowError,match='integrity_error'): contracts.verify_inputs(hashes)
