import json
import numpy as np
import pytest
from repro.offline import create_policy, observation, response


def receipts(payload):
    return [json.loads(x['output']) for x in payload['input'] if x.get('type') == 'function_call_output']


def test_invalid_action_corrected_at_same_observation(tmp_path):
    bad = [[True] * 14]
    p = create_policy(tmp_path, [response('bad', bad), response('good')])
    p.update_obs(observation(0))
    actions = p.get_action()
    assert len(actions) == 2 and p.step == 0 and not p.ledger
    rejected = receipts(p.requests[1])[-1]
    assert rejected['executed_steps'] == 0 and rejected['observation_step'] == 0
    assert p.pending['call_id'] == 'good'


def test_partial_execution_and_terminal_receipt(tmp_path):
    p = create_policy(tmp_path, [response('a'), response('b')])
    p.update_obs(observation(0))
    a = p.get_action()
    p.update_obs(observation(1, a[0], event={'type': 'chunk_interrupted_tracking_error', 'discarded_commands': 1}))
    p.get_action()
    r = receipts(p.requests[1])[-1]
    assert r['executed_steps'] == 1 and r['unexecuted_steps'] == 1
    assert r['executed_step_interval'] == [0, 1]
    p.finalize()
    r = json.loads(p.messages[-1]['output'])
    assert r['terminal'] is True and r['executed_steps'] == 0


def test_ik_zero_step_rejection_is_next_request_context(tmp_path):
    p = create_policy(tmp_path, [response('a'), response('b')])
    p.update_obs(observation(0)); p.get_action()
    p.notify_action_rejection({'step': 0, 'event': {'type': 'action_rejected_ik_continuity',
        'rejected_action_executed': False, 'action_index': 0, 'discarded_commands': 2}})
    p.get_action()
    r = receipts(p.requests[1])[-1]
    assert r['executed_steps'] == 0 and len(r['controller_events']) == 1
    assert len(p.archive) == 1 and p.live_memory.stats['captured_chunks'] == 0


def test_no_inference_without_execution_and_no_privileged_state(tmp_path):
    p = create_policy(tmp_path, [response('a')])
    ob = observation(0); ob['state']['object_pose'] = 'SECRET_OBJECT'; ob['reward'] = 'SECRET_REWARD'
    p.update_obs(ob); p.get_action()
    assert 'SECRET_' not in json.dumps(p.requests)
    with pytest.raises(ValueError, match='no executed observation'):
        p.get_action()
    assert len(p.requests) == 1


def test_reset_removes_live_history(tmp_path):
    p = create_policy(tmp_path, [response('a')]); p.update_obs(observation(0)); p.get_action()
    p.reset()
    assert p.pending is None and not p.messages and not p.archive and not p.seen_call_ids


def test_invalid_budget_uses_explicit_original_joint_hold(tmp_path):
    p = create_policy(tmp_path, [response(str(i), [[False]*14]) for i in range(3)])
    p.update_obs(observation(0)); a = p.get_action()
    assert len(p.requests) == 3 and len(a) == 1
    assert 'left_arm_joint_state' in a[0] and p.pending['local_safety_hold']
