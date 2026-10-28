"""Actual Session bookkeeping around a no-physics wait backend."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from repro.event_session import submit_wait
from repro.feedback_session import FeedbackSession
from repro.session import StepResult
from test_event_wait import Fixture, command, pair


class Backend(Fixture):
    def __init__(self):
        super().__init__(ready_at=3)
        self.d=SimpleNamespace(time=0.)

    def control_status(self):return {'stability':self.status()['stability']}
    def contacts(self):return deepcopy(self.status()['contacts'])
    def step(self,cmd,remaining):
        result=self.execute(cmd);self.d.time=self.time
        return StepResult({'state':{'time':self.time},'images':{'head':'synthetic'}},
                          result.executed,result.elapsed_s,feedback=result.feedback)


def setup():
    b=Backend()
    s=FeedbackSession({'state':{'time':0.},'images':{'head':'synthetic'}},
        convert=lambda rows,state:rows,state_keys=['time'],cameras=['head'],horizon=5,maximum=100)
    return s,b


def act(s,b):
    s.submit({'type':'function_call','name':'act','call_id':'start',
              'arguments':json.dumps({'actions':[command()],'execution_note':'Explicit fixture action'})},lambda c:b.step(c,30))


def wait_call():
    return {'type':'function_call','name':'wait_until_stable','call_id':'wait',
            'arguments':json.dumps({'max_wait_s':1.,'execution_note':'Fixture wait'})}


def test_wait_commits_each_compiled_command_and_serializable_outer_receipt():
    s,b=setup();act(s,b)
    report=submit_wait(s,wait_call(),b,30)
    assert report['reason']=='stable' and report['command_count']==2
    assert report['executed_step_interval']==[1,3]
    assert len(s.ledger)==3 and len(s.receipts)==3
    assert s.ledger[1]['command']==s.ledger[2]['command']
    assert 'executor_compiled_wait' in json.dumps(s.context())
    assert json.loads(json.dumps(report))['elapsed_s']==pytest.approx(.2)


def test_wait_without_prior_command_or_with_invalid_target_does_not_mutate_history():
    s,b=setup()
    with pytest.raises(ValueError):submit_wait(s,wait_call(),b,30)
    assert not s.seen and not s.tail and not b.calls
    act(s,b);s.ledger[-1]['command'].pop('right_position')
    seen=set(s.seen);tail=deepcopy(s.tail)
    with pytest.raises(ValueError):submit_wait(s,wait_call(),b,30)
    assert s.seen==seen and s.tail==tail and len(b.calls)==1


def test_initial_named_pair_identity_is_available_to_detect_transient_new_pair():
    s,b=setup();b.hand=.8;b.pairs=[pair()];act(s,b)
    original=b.execute
    def execute(cmd):
        result=original(cmd)
        result.feedback['command_contacts']['contact_pairs'].append(pair('right_hand_thumb_link',3))
        return result
    b.execute=execute
    report=submit_wait(s,wait_call(),b,30)
    assert report['reason']=='hand_contact_changed' and report['command_count']==1
    assert any(e['type']=='hand_contact_pair_changed' for e in report['events'])


def test_postexecution_feedback_error_is_terminal_not_a_zero_step_rejection():
    s,b=setup();act(s,b)
    original=b.execute
    def execute(cmd):
        result=original(cmd);result.feedback['command_contacts']['peak_robot_table_N']='invalid'
        return result
    b.execute=execute
    with pytest.raises(ValueError):submit_wait(s,wait_call(),b,30)
    assert s.step==2 and len(b.calls)==2 and s.terminal
    assert len(s.ledger)==2
