import json
import pytest
from repro.session import Session, StepResult, CorrectionExhausted


def ob():
    return {'state': {'x': 1}, 'images': {'head': 'synthetic'}}


def answer(cid, actions):
    return {'type': 'function_call', 'name': 'act', 'call_id': cid,
            'arguments': json.dumps({'actions': actions, 'execution_note': 'test'})}


def convert(rows, state):
    if any(type(x) is not int or x < 0 for x in rows):
        raise ValueError('invalid numeric command')
    return rows


def make():
    return Session(ob(), convert=convert, state_keys=('x',), cameras=('head',), horizon=2, maximum=20)


def test_rejection_then_repair_enters_live_and_never_executes_train():
    s = make(); called = []
    s.submit(answer('bad', [-1]), lambda c: called.append(c))
    assert not called and s.step == 0
    assert 'invalid_action' in json.dumps(s.context())
    s.submit(answer('good', [1]), lambda c: StepResult(ob(), executed=True, elapsed_s=.1))
    assert s.step == 1 and 'bad' in json.dumps(s.context())


def test_partial_interrupt_drops_suffix_and_next_context_has_receipt():
    s = make(); called = []
    def execute(c):
        called.append(c)
        return StepResult(ob(), executed=True, elapsed_s=.05, interrupt='tracking_error')
    s.submit(answer('a', [1, 2]), execute)
    assert called == [1] and s.step == 1
    r = s.receipts[-1]
    assert r['executed_steps'] == 1 and r['unexecuted_steps'] == 1
    assert r['actual_elapsed_s'] == .05 and not r['terminal']


def test_collision_terminal_is_visible_and_blocks_further_execution():
    s = make()
    s.submit(answer('a', [1, 2]), lambda c: StepResult(ob(), True, .04, terminal=True, interrupt='collision'))
    assert s.receipts[-1]['terminal'] is True and 'collision' in json.dumps(s.context())
    with pytest.raises(RuntimeError, match='terminal'):
        s.submit(answer('b', [1]), lambda c: None)


def test_zero_step_rejection_and_correction_budget_no_hold_fallback():
    s = make()
    for i in range(2):
        s.submit(answer(str(i), [1]), lambda c: StepResult(ob(), False, 0, interrupt='controller_rejected'))
    with pytest.raises(CorrectionExhausted):
        s.submit(answer('third', [1]), lambda c: StepResult(ob(), False, 0, interrupt='controller_rejected'))
    assert s.step == 0 and s.terminal and not s.ledger


def test_duplicate_call_and_privileged_observation_blocked():
    s = make(); s.submit(answer('a', [-1]), lambda c: None)
    with pytest.raises(ValueError, match='call_id'):
        s.submit(answer('a', [1]), lambda c: None)
    dirty = ob(); dirty['state']['target_pose'] = 'SECRET'
    with pytest.raises(ValueError, match='whitelist'):
        Session(dirty, convert=convert, state_keys=('x',), cameras=('head',))


def test_execution_exception_poison_session_without_fabricating_receipt():
    s = make()
    def execute(c):
        raise RuntimeError('unknown partial execution')
    with pytest.raises(RuntimeError): s.submit(answer('a', [1]), execute)
    assert s.terminal and not s.receipts


def test_unknown_execution_after_prefix_preserves_readable_terminal_history():
    s = make(); n = 0
    s.submit(answer("previous", [0]), lambda c: StepResult(ob(), True, .1))
    def execute(c):
        nonlocal n
        n += 1
        if n == 2:
            raise RuntimeError('lost execution evidence')
        return StepResult(ob(), True, .1)
    with pytest.raises(RuntimeError):
        s.submit(answer('a', [1,2]), execute)
    text = json.dumps(s.context())
    assert 'execution_outcome_unknown' in text and 'confirmed_executed_steps' in text
    assert s.step == 2 and s.terminal and len(s.ledger) == 2
