import json
from pathlib import Path

import pytest

from humanoid_mimicgen.augmentation.lateral_agent import run_session
from humanoid_mimicgen.augmentation.privileged_lateral import PrivilegedSession, validate_cached_plan
from humanoid_mimicgen.workflow.contracts import WorkflowError, file_hash


class Driver:
    def __init__(self, root):
        self.root, self.calls = root, 0

    def cycle(self, plan, value, proof):
        self.calls += 1
        case = self.root / 'case'
        case.mkdir()
        (case / 'experiment.json').write_text('{}')
        return case, dict(case=str(case), native_success=True, clean_success=False, physics_steps=100,
                          failure=None, max_obstacle_N=13, max_hand_object_N=40)


def make_session(tmp_path, monkeypatch):
    import humanoid_mimicgen.augmentation.privileged_lateral as module
    monkeypatch.setattr(module, 'validate_cached_plan', lambda config: dict(selected={'id': 'selected'},
                         offset_m=.05, search_incomplete=True, scene_change={}))
    cfg = dict(cached_plan=str(tmp_path/'plan.json'), input_sha256={}, max_attempts=1,
               max_rounds=2, wall_s=20, offset_m=.05, navigation_mode='direct_stance')
    (tmp_path/'plan.json').write_text('{}')
    driver = Driver(tmp_path)
    return PrivilegedSession(tmp_path, cfg, driver), cfg, driver


def test_numeric_loop_finishes_in_two_calls_without_images_or_final_summary_rpc(tmp_path, monkeypatch):
    session, cfg, driver = make_session(tmp_path, monkeypatch)
    class Provider:
        config = {'model': 'fixture'}
        calls = 0
        def respond(self, items, definitions, instructions):
            assert 'input_image' not in json.dumps(items)
            assert {d['name'] for d in definitions} == {'run_grasp', 'reflect_grasp'}
            self.calls += 1
            if self.calls == 1:
                name, args = 'run_grasp', dict(candidate_id='selected', wrist_raise_m=.026, rationale='Measured prior')
            else:
                ident = next(iter(session.state['prepared']))
                name, args = 'reflect_grasp', dict(prepared_id=ident, action='retry', rationale='Table contact', next_change='Test next batch')
            return dict(status='completed', output=[dict(type='function_call', name=name,
                        call_id=str(self.calls), arguments=json.dumps(args))])
    provider = Provider()
    result = run_session(tmp_path, cfg, session, provider)
    assert provider.calls == 2 and driver.calls == 1
    assert result['status'] == 'completed'
    assert not result['cases'][0]['clean_success']
    assert result['cases'][0]['reflection']['action'] == 'retry'


def test_interrupted_run_is_never_repeated_and_accept_requires_clean_result(tmp_path, monkeypatch):
    session, cfg, driver = make_session(tmp_path, monkeypatch)
    args = dict(candidate_id='selected', wrist_raise_m=.026, rationale='Prior')
    proof = dict(call_id='once')
    first = session.dispatch('run_grasp', args, proof)
    assert session.dispatch('run_grasp', args, proof) == first
    assert driver.calls == 1
    ident = first['prepared_id']
    reflect = dict(prepared_id=ident, action='accept', rationale='Lift', next_change='None')
    with pytest.raises(WorkflowError):
        session.dispatch('reflect_grasp', reflect, {'observed_result_ids': [ident]})
    session.state['prepared'][ident].pop('result')
    session.persist()
    restarted = PrivilegedSession(tmp_path, cfg, driver)
    with pytest.raises(WorkflowError):
        restarted.dispatch('run_grasp', args, proof)
    assert driver.calls == 1


def test_reflection_requires_result_from_earlier_api_request(tmp_path, monkeypatch):
    session, _, _ = make_session(tmp_path, monkeypatch)
    first = session.dispatch('run_grasp', dict(candidate_id='selected', wrist_raise_m=.026, rationale='Prior'), {'call_id': 'a'})
    with pytest.raises(WorkflowError):
        session.dispatch('reflect_grasp', dict(prepared_id=first['prepared_id'], action='stop', rationale='Done', next_change='None'), {})


def test_cached_plan_rejects_a_different_offset_before_launch(tmp_path):
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(status='completed', selected={'feasible': True}, offset_m=.1)))
    with pytest.raises(WorkflowError):
        validate_cached_plan(dict(cached_plan=str(plan), offset_m=.05))


def test_cached_plan_binds_source_and_shifted_initial_separately(tmp_path, monkeypatch):
    import humanoid_mimicgen.workflow.navigation.reference_bundle as module
    monkeypatch.setattr(module, 'load_reference_bundle', lambda value: {})
    paths = {k: tmp_path/k for k in ('manifest', 'initial', 'shifted', 'options')}
    for key, path in paths.items():
        path.write_text('{}' if key == 'options' else key)
    values = dict(status='completed', selected={'feasible': True}, offset_m=.05,
                  source_manifest=str(paths['manifest']), lateral_options={},
                  scene_files={'initial': str(paths['shifted'])},
                  scene_input_sha256={'initial': file_hash(paths['shifted'])},
                  evidence_sha256={str(p): file_hash(p) for p in paths.values()}, generated_reference={})
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps(values))
    cfg = {k: str(v) for k, v in paths.items() if k != 'shifted'}
    cfg.update(cached_plan=str(plan), offset_m=.05)
    assert validate_cached_plan(cfg)['status'] == 'completed'
    paths['initial'].write_text('different source initial')
    with pytest.raises(WorkflowError):
        validate_cached_plan(cfg)


def test_numeric_audit_does_not_initialize_a_policy_and_preserves_old_output(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from humanoid_mimicgen.augmentation import lateral_delivery, direct_grasp
    execution = tmp_path/'execution'
    execution.mkdir()
    states = np.zeros((4, 14));states[:, 0] = np.arange(4)*.005
    nexts = states.copy();nexts[:, 0] += .005
    np.savez(execution/'trajectory.npz', state=states, next_state=nexts, commands=np.zeros((4, 30)))
    pose = [0., 0., .1, 1., 0., 0., 0.]
    obs = [dict(object_pose=pose, right_pose=pose, clearance_m=.1, contacts=dict(hand_object_N=1., support_N=0.),
                feet={'left': pose, 'right': pose}) for _ in range(4)]
    values = {'execution/controller.json': [dict(target=[0.]*29, mode=1)],
              'execution/observations.json': obs, 'execution/commands.json': [{}]*4,
              'result.json': dict(physics_steps=4, native_success=True), 'config.json': {}, 'experiment.json': {}}
    for name, value in values.items():
        (tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'experiment.sha256').write_text(file_hash(tmp_path/'experiment.json'))
    original = tmp_path/'independent_audit.json'
    original.write_text('preserve original audit')
    closed = []
    fake = SimpleNamespace(m=SimpleNamespace(nq=7), d=SimpleNamespace(time=0., qpos=np.zeros(7), qvel=np.zeros(6)),
                           native_success=lambda: True, close=lambda: closed.append(True))
    monkeypatch.setattr(lateral_delivery, 'native_scene', lambda case: fake)
    monkeypatch.setattr(direct_grasp, 'initialize', lambda case: pytest.fail('Verification loaded a policy'))
    monkeypatch.setattr(direct_grasp.mujoco, 'mj_forward', lambda *args: None)
    out = lateral_delivery.audit(tmp_path, scene_only=True, output_path=tmp_path/'new_audit.json')
    assert out['independent_native_success'] and out['physics_steps_in_verifier'] == 0
    assert closed == [True] and original.read_text() == 'preserve original audit'


def test_navigation_schema_and_context_default_to_far_numeric_execution(tmp_path, monkeypatch):
    _, cfg, driver = make_session(tmp_path, monkeypatch)
    cfg['navigation_mode'] = 'precision'
    session = PrivilegedSession(tmp_path, cfg, driver)
    assert session.model_context['navigation']['initialization'] == 'original_far_state'
    params = session.tool_definitions()[0]['parameters']['properties']
    assert params['execution']['enum'] == ['restart', 'navigation_start', 'parked']
    assert params['feedback_gain']['maximum'] == 3 and 'parent_prepared_id' in params
    assert 'force_stop_N' not in session.model_context['thresholds']


def test_navigation_branch_requires_observed_parent_and_unchanged_early_parameters(tmp_path, monkeypatch):
    from humanoid_mimicgen.augmentation.privileged_lateral import validate_branch
    parent = dict(case=str(tmp_path), result={'available_checkpoints': ['parked', 'navigation_start']},
                  decision={'wrist_raise_m': 0.0}, reflection={'action': 'retry'})
    args = dict(execution='parked', parent_prepared_id='parent', wrist_raise_m=0.0, feedback_gain=1.5)
    with pytest.raises(WorkflowError):
        validate_branch(args, {'parent': parent}, {})
    assert validate_branch(args, {'parent': parent}, {'observed_result_ids': ['parent']}) == {'case': str(tmp_path), 'name': 'parked'}
    with pytest.raises(WorkflowError):
        validate_branch(args | {'wrist_raise_m': .02}, {'parent': parent}, {'observed_result_ids': ['parent']})
    assert validate_branch(args | {'execution': 'restart', 'parent_prepared_id': ''}, {}, {}) is None


def test_navigation_api_retry_uses_observed_parent_and_is_idempotent(tmp_path, monkeypatch):
    _, cfg, _ = make_session(tmp_path, monkeypatch)
    cfg.update(navigation_mode='precision', max_attempts=2, max_rounds=4)
    class NavigationDriver:
        calls = []
        def navigation_cycle(self, plan, value, proof, gain, resume):
            self.calls.append((proof, gain, resume))
            case = tmp_path / str(len(self.calls));case.mkdir()
            (case/'experiment.json').write_text('{}')
            return case, dict(case=str(case), native_success=True, clean_success=False,
                physics_steps=100, navigation_executed=True, available_checkpoints=['navigation_start', 'parked'])
    driver = NavigationDriver()
    session = PrivilegedSession(tmp_path, cfg, driver)
    class Provider:
        config = {'model': 'fixture'}
        calls = 0
        def respond(self, items, definitions, instructions):
            assert 'input_image' not in json.dumps(items)
            self.calls += 1
            if self.calls in (1, 3):
                parent = next(iter(session.state['prepared'])) if self.calls == 3 else ''
                name, args = 'run_grasp', dict(candidate_id='selected', wrist_raise_m=0., rationale='Numeric candidate',
                    execution='parked' if parent else 'restart', parent_prepared_id=parent, feedback_gain=1.5 if parent else 2.)
            else:
                ident = list(session.state['prepared'])[-1]
                name, args = 'reflect_grasp', dict(prepared_id=ident, action='retry' if self.calls == 2 else 'stop', rationale='Measured contact', next_change='Evaluate feedback')
            return dict(status='completed', output=[dict(type='function_call', name=name,
                call_id=str(self.calls), arguments=json.dumps(args))])
    result = run_session(tmp_path, cfg, session, Provider())
    assert result['status'] == 'completed' and len(driver.calls) == 2
    assert result['navigation_executed'] is True
    assert driver.calls[1][2] == dict(case=str(tmp_path/'1'), name='parked')
    ident, row = list(session.state['prepared'].items())[-1]
    assert driver.calls[1][0]['call_id'] == '3'
    session.dispatch('run_grasp', row['decision'], driver.calls[1][0])
    assert len(driver.calls) == 2
