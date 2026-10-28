"""Confirmed-prefix feedback must survive the normal LIVE context path."""
import importlib
import importlib.util
import json

import pytest

from repro.session import StepResult


def observation(step=0):
    return {'state':{'x':step},'images':{'head':'synthetic_no_physics'}}


def make():
    assert importlib.util.find_spec('repro.feedback_session') is not None, 'feedback session is missing'
    cls=importlib.import_module('repro.feedback_session').FeedbackSession
    return cls(observation(),convert=lambda rows,state:rows,state_keys=['x'],cameras=['head'],horizon=5,maximum=30)


def call(cid,rows):
    return {'type':'function_call','name':'act','call_id':cid,
            'arguments':json.dumps({'actions':rows,'execution_note':'Synthetic unit fixture'})}


def result(step,peak=0.,*,interrupt=None,terminal=False):
    return StepResult(observation(step),True,.1,terminal=terminal,interrupt=interrupt,
        feedback={'command_contacts':{'peak_robot_table_N':peak,
            'contact_pairs':[{'bodies':['table','earlier_right_middle_link'],'peak_force_N':peak}] if peak else []}})


def test_early_transient_and_named_contact_reach_next_live_context():
    s=make()
    s.submit(call('first',[1,2]),lambda row:result(row,2. if row==1 else 0.))
    feedback=s.receipts[-1]['latest_feedback']
    assert feedback['command_contacts']['peak_robot_table_N']==0.
    prefix=feedback['executed_prefix_feedback']
    assert [row['action_index'] for row in prefix]==[0,1]
    assert prefix[0]['feedback']['command_contacts']['peak_robot_table_N']==2.
    assert prefix[0]['elapsed_s']==.1
    assert 'earlier_right_middle_link' in json.dumps(s.context())
    assert all('executed_prefix_feedback' not in row['feedback'] for row in prefix)


def test_partial_interrupt_retains_only_confirmed_prefix_and_discards_suffix():
    s=make();executed=[]
    def execute(row):
        executed.append(row)
        return result(row,3.,interrupt='tracking_error')
    s.submit(call('partial',[1,2,3]),execute)
    assert executed==[1]
    receipt=s.receipts[-1]
    assert receipt['executed_steps']==1 and receipt['unexecuted_steps']==2
    prefix=receipt['latest_feedback']['executed_prefix_feedback']
    assert len(prefix)==1 and prefix[0]['interrupt']=='tracking_error'
    assert not s.terminal


def test_zero_step_rejection_retains_earlier_feedback_without_inventing_a_row():
    s=make()
    def execute(row):
        if row==1:return result(1,4.)
        return StepResult(observation(1),False,0.,interrupt='standing_stability_gate',
                          feedback={'admission':{'allowed':False}})
    s.submit(call('gate',[1,2,3]),execute)
    receipt=s.receipts[-1];feedback=receipt['latest_feedback']
    assert feedback['admission']['allowed'] is False
    assert len(feedback['executed_prefix_feedback'])==1
    assert feedback['executed_prefix_feedback'][0]['feedback']['command_contacts']['peak_robot_table_N']==4.
    assert receipt['executed_steps']==1 and receipt['unexecuted_steps']==2
    assert len(s.ledger)==1


def test_invalid_call_and_next_chunk_cannot_leak_previous_chunk_accumulator():
    s=make()
    s.submit(call('first',[1,2]),lambda row:result(row,7.))
    s.submit({'bad':'envelope'},lambda row:pytest.fail('invalid envelope executed'))
    s.submit(call('next',[3]),lambda row:result(row))
    prefix=s.receipts[-1]['latest_feedback']['executed_prefix_feedback']
    assert len(prefix)==1 and prefix[0]['action_index']==0
    assert prefix[0]['feedback']['command_contacts']['peak_robot_table_N']==0.
    assert len(s.receipts[0]['latest_feedback']['executed_prefix_feedback'])==2


def test_exception_after_prefix_remains_unknown_and_terminal_without_false_receipt():
    s=make()
    def execute(row):
        if row==1:return result(1,5.)
        raise RuntimeError('execution evidence lost')
    with pytest.raises(RuntimeError,match='execution evidence lost'):
        s.submit(call('unknown',[1,2,3]),execute)
    assert s.terminal and s.step==1 and not s.receipts
    assert len(s.ledger)==1
    assert s.ledger[0]['feedback']['executed_prefix_feedback'][0]['feedback']['command_contacts']['peak_robot_table_N']==5.
    unknown=[json.loads(m['output']) for m in s.context()
             if m.get('type')=='function_call_output'][-1]
    assert unknown['error']=='execution_outcome_unknown'
    assert unknown['confirmed_executed_steps']==1 and unknown['executed_steps'] is None
    assert unknown['uncertain_action_index']==1


def test_executor_feedback_is_not_mutated_or_recursively_copied():
    s=make();raw=result(1,6.)
    s.submit(call('immutable',[1]),lambda row:raw)
    assert 'executed_prefix_feedback' not in raw.feedback
    prefix=s.receipts[-1]['latest_feedback']['executed_prefix_feedback']
    assert len(prefix)==1 and 'executed_prefix_feedback' not in prefix[0]['feedback']
