import json
from pathlib import Path

import pytest

from humanoid_mimicgen.workflow.contracts import WorkflowError, file_hash
from humanoid_mimicgen.augmentation.lateral_tools import ToolSession, validate_raise, review_images, experience_summary


@pytest.mark.parametrize('value', [True, -0.001, 0.031, float('nan'), float('inf')])
def test_raise_limits_reject_nonfinite_and_unavailable_controls(value):
    with pytest.raises(WorkflowError):
        validate_raise(value)


def test_raise_accepts_boundary():
    assert validate_raise(0.03) == 0.03


def test_old_experience_missing_raise_field_is_explicitly_unknown(tmp_path):
    value = dict(experiment_id='legacy', offset_m=.05, native_success=True, success=False,
                 max_clearance_m=.13, max_obstacle_N=15., obstacle_contact_gt_0p1_s=1.16)
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(value))
    (tmp_path / 'config.json').write_text('{}')
    assert experience_summary(path)['wrist_raise_m'] is None


def test_no_contact_trial_still_returns_existing_review_evidence(tmp_path):
    folder = tmp_path / 'inspection'
    folder.mkdir()
    sheet = folder / 'contact_sheet.jpg'
    sheet.write_bytes(b'sheet')
    assert review_images(tmp_path) == [str(sheet)]


class LocalDriver:
    def __init__(self, root):
        self.root = root
        self.calls = 0

    def prepare(self, plan, raise_m, provenance):
        case = self.root / 'case'
        case.mkdir()
        (case / 'experiment.json').write_text(json.dumps({'raise_m': raise_m, 'provenance': provenance}))
        (case / 'initial.png').write_bytes(b'fixture preview')
        return case

    def execute(self, case):
        self.calls += 1
        return {'case': str(case), 'native_success': True, 'clean_success': False,
                'physics_steps': 10, 'max_obstacle_N': 17.0}


def session(tmp_path):
    driver = LocalDriver(tmp_path)
    cfg = {'offset_m': 0.05, 'max_attempts': 2}
    s = ToolSession(tmp_path, cfg, driver=driver)
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'status': 'completed', 'selected': {'id': 'support_0'}}))
    s.state.update(skill_read=True, inspected=True, plan_file=str(plan),
                   plan_sha256=file_hash(plan), selected_id='support_0')
    s.persist()
    return s, driver


def prepare(s):
    return s.dispatch('prepare_grasp', {'candidate_id': 'support_0', 'wrist_raise_m': 0.02,
                                     'rationale': 'Test middle finger table clearance'}, {'transport': 'injected'})


def test_initial_image_must_be_exposed_before_physics(tmp_path):
    s, driver = session(tmp_path)
    p = prepare(s)
    with pytest.raises(WorkflowError):
        s.dispatch('execute_grasp', {'prepared_id': p['prepared_id'], 'initial_review': 'Open hands clear'})
    assert driver.calls == 0


def test_execute_once_even_across_session_reconstruction(tmp_path):
    s, driver = session(tmp_path)
    p = prepare(s)
    s.expose_image(p['prepared_id'])
    args = {'prepared_id': p['prepared_id'], 'initial_review': 'Open hands clear; proceed'}
    first = s.dispatch('execute_grasp', args, {'observed_prepared_ids': [p['prepared_id']]})
    second = ToolSession(tmp_path, s.config, driver=driver).dispatch('execute_grasp', args)
    assert first == second
    assert driver.calls == 1


def test_changed_prepared_contract_blocks_execution(tmp_path):
    s, driver = session(tmp_path)
    p = prepare(s)
    s.expose_image(p['prepared_id'])
    Path(p['case']).joinpath('experiment.json').write_text('{}')
    with pytest.raises(WorkflowError):
        s.dispatch('execute_grasp', {'prepared_id': p['prepared_id'], 'initial_review': 'Proceed'})
    assert driver.calls == 0


def test_reflection_cannot_accept_native_only_success(tmp_path):
    s, _ = session(tmp_path)
    p = prepare(s)
    s.expose_image(p['prepared_id'])
    s.dispatch('execute_grasp', {'prepared_id': p['prepared_id'], 'initial_review': 'Proceed'},
               {'observed_prepared_ids': [p['prepared_id']]})
    with pytest.raises(WorkflowError):
        s.dispatch('reflect_grasp', {'prepared_id': p['prepared_id'], 'action': 'accept',
                                   'rationale': 'Native passed', 'next_change': 'none'},
                   {'observed_result_ids': [p['prepared_id']]})
    assert not s.complete()
    with pytest.raises(WorkflowError):
        prepare(s)
    s.dispatch('reflect_grasp', {'prepared_id': p['prepared_id'], 'action': 'stop',
                               'rationale': 'Native passed but 17 N table impact', 'next_change': 'Finger preshape needs new tool'},
               {'observed_result_ids': [p['prepared_id']]})
    assert s.complete()


def test_started_without_result_never_retries_automatically(tmp_path):
    s, driver = session(tmp_path)
    p = prepare(s)
    s.expose_image(p['prepared_id'])
    s.state['prepared'][p['prepared_id']]['started'] = True
    s.persist()
    with pytest.raises(WorkflowError):
        s.dispatch('execute_grasp', {'prepared_id': p['prepared_id'], 'initial_review': 'Proceed'})
    assert driver.calls == 0


def test_same_response_cannot_reflect_on_unobserved_physics(tmp_path):
    from humanoid_mimicgen.augmentation.lateral_agent import apply_response
    s, driver = session(tmp_path)
    p = prepare(s)
    s.expose_image(p['prepared_id'])
    items = [{'type': 'function_call_output', 'call_id': 'prepare', 'output': json.dumps(p)}]
    raw = {'status': 'completed', 'output': [
        {'type': 'function_call', 'name': 'execute_grasp', 'call_id': 'execute',
         'arguments': json.dumps({'prepared_id': p['prepared_id'], 'initial_review': 'Clear'})},
        {'type': 'function_call', 'name': 'reflect_grasp', 'call_id': 'premature_reflect',
         'arguments': json.dumps({'prepared_id': p['prepared_id'], 'action': 'stop',
                                  'rationale': 'Prewritten guess', 'next_change': 'none'})}]}
    apply_response(raw, s, tmp_path, {}, items)
    assert driver.calls == 1
    assert 'reflection' not in s.state['prepared'][p['prepared_id']]
    assert not s.complete()
