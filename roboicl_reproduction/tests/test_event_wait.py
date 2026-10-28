"""Bounded waits execute recorded targets and return on measured events."""
from copy import deepcopy
import importlib
import importlib.util
import json

import pytest

from repro.session import StepResult


def api():
    assert importlib.util.find_spec('repro.event_wait') is not None, 'event wait helper missing'
    return importlib.import_module('repro.event_wait')


def command():
    return {'stage':'approach','duration_s':.1,'velocity_xy':[.2,0],
            'pelvis_height_m':.78,'facing_yaw':.2,'torso_quaternion_wxyz':[1,0,0,0],
            'left_position':[.1,.2,.8],'right_position':[.1,-.2,.8],
            'left_quaternion_wxyz':[1,0,0,0],'right_quaternion_wxyz':[1,0,0,0],
            'left_fingers':[0]*7,'right_fingers':[0,0,-.1,.1,.2,.1,.2]}


def pair(name='right_hand_middle_link',count=20):
    return {'bodies':['apple_001',name],'geoms':['apple_geom',name+'_geom'],
            'peak_force_N':.8,'sample_count':count,'first_sample_s':.005,'last_sample_s':.1}


class Fixture:
    def __init__(self,ready_at=99,hand=0.):
        self.calls=[];self.time=0.;self.ready_at=ready_at;self.hand=hand
        self.pairs=[pair()] if hand>.1 else []
        self.terminal=False;self.remaining=30.

    def status(self):
        return {'stability':{'ready':len(self.calls)>=self.ready_at},
                'contacts':{'hand_object_N':self.hand,'robot_table_N':0.,
                            'events':[{'bodies':p['bodies'],'geoms':p['geoms'],'force_N':p['peak_force_N']} for p in self.pairs]},
                'terminal':self.terminal,'remaining_sim_seconds':self.remaining-self.time}

    def execute(self,cmd):
        self.calls.append(deepcopy(cmd));self.time+=.1
        intervals=[{'first_sample_s':self.time-.095,'last_sample_s':self.time,
                    'sample_count':20,'observed_duration_s':.1,'peak_force_N':self.hand}] if self.hand>.1 else []
        feedback={'contacts':{'hand_object_N':self.hand,'robot_table_N':0.},
                  'command_contacts':{'sample_count':20,'sample_dt_s':.005,
                    'peak_robot_table_N':0.,'peak_hand_object_N':self.hand,
                    'hand_object_intervals':intervals,'robot_table_intervals':[],
                    'contact_pairs':deepcopy(self.pairs)}}
        return StepResult({'state':{'time':self.time},'images':{}},True,.1,feedback=feedback)


def test_stable_early_preserves_absolute_targets_and_never_reanchors():
    f=Fixture(ready_at=2);original=command()
    out=api().execute_wait(original,5.,f.status,f.execute)
    assert out['reason']=='stable' and out['command_count']==2
    assert out['executed_commands']==2 and out['elapsed_s']==pytest.approx(.2)
    assert original==command()
    assert f.calls[0]==f.calls[1]
    assert f.calls[0]['stage']=='hold' and f.calls[0]['velocity_xy']==[0,0]
    for key in original.keys()-{'stage','velocity_xy','duration_s'}:
        assert f.calls[0][key]==original[key]


def test_already_ready_is_valid_zero_step_completion():
    f=Fixture(ready_at=0)
    out=api().execute_wait(command(),1.,f.status,lambda cmd:pytest.fail('already ready executed'))
    assert out['reason']=='stable' and out['command_count']==0 and out['elapsed_s']==0
    assert not out['receipts']


def test_timeout_is_bounded_by_actual_commands_and_measured_elapsed():
    f=Fixture()
    out=api().execute_wait(command(),.3,f.status,f.execute)
    assert out['reason']=='timeout' and len(f.calls)==3
    assert out['elapsed_s']==pytest.approx(.3)
    assert len(out['receipts'])==3


@pytest.mark.parametrize('terminal,remaining,reason',[(True,30.,'terminal'),(False,.09,'simulation_budget')])
def test_terminal_and_insufficient_budget_never_call_executor(terminal,remaining,reason):
    f=Fixture();f.terminal=terminal;f.remaining=remaining
    out=api().execute_wait(command(),1.,f.status,lambda cmd:pytest.fail('unavailable simulation executed'))
    assert out['reason']==reason and out['command_count']==0


def test_same_sustained_hand_contact_does_not_make_wait_return_forever():
    f=Fixture(ready_at=2,hand=.8)
    out=api().execute_wait(command(),1.,f.status,f.execute)
    assert out['reason']=='stable' and len(f.calls)==2


@pytest.mark.parametrize('initial_hand,intervals,end_hand,event',[
    (0.,[(.02,.04,5)],0.,'new_hand_contact'),
    (.8,[],0.,'lost_hand_contact'),
    (.8,[(.005,.02,4),(.04,.1,13)],.8,'lost_hand_contact'),
])
def test_first_command_transient_contact_and_loss_return_even_if_endpoint_recovers(initial_hand,intervals,end_hand,event):
    f=Fixture(hand=initial_hand)
    def execute(cmd):
        result=f.execute(cmd);f.hand=end_hand
        result.feedback['contacts']['hand_object_N']=end_hand
        cc=result.feedback['command_contacts'];cc['peak_hand_object_N']=.8 if intervals else 0.
        cc['hand_object_intervals']=[{'first_sample_s':a,'last_sample_s':b,'sample_count':n,
            'observed_duration_s':n*.005,'peak_force_N':.8} for a,b,n in intervals]
        return result
    out=api().execute_wait(command(),1.,f.status,execute)
    assert out['reason']=='hand_contact_changed' and len(f.calls)==1
    assert event in [e['type'] for e in out['events']]


def test_new_named_contact_pair_returns_while_aggregate_force_stays_constant():
    f=Fixture(hand=.8)
    def execute(cmd):
        result=f.execute(cmd)
        result.feedback['command_contacts']['contact_pairs'].append(pair('right_hand_thumb_link',3))
        return result
    out=api().execute_wait(command(),1.,f.status,execute)
    assert out['reason']=='hand_contact_changed' and len(f.calls)==1
    assert 'hand_contact_pair_changed' in [e['type'] for e in out['events']]


def test_transient_pair_loss_returns_even_when_aggregate_contact_and_endpoint_recover():
    f=Fixture(hand=.8)
    def execute(cmd):
        result=f.execute(cmd)
        result.feedback['command_contacts']['contact_pairs'][0]['sample_count']=18
        return result
    out=api().execute_wait(command(),1.,f.status,execute)
    assert out['reason']=='hand_contact_changed' and len(f.calls)==1
    assert 'hand_contact_pair_changed' in [e['type'] for e in out['events']]


def test_reported_pair_change_is_not_hidden_by_individual_force_below_aggregate_hand_threshold():
    f=Fixture(hand=.8)
    def execute(cmd):
        result=f.execute(cmd);new=pair('right_hand_thumb_link',3);new['peak_force_N']=.08
        result.feedback['command_contacts']['contact_pairs'].append(new)
        return result
    out=api().execute_wait(command(),1.,f.status,execute)
    assert out['reason']=='hand_contact_changed' and len(f.calls)==1


def test_transient_table_contact_returns_without_waiting_for_endpoint_force():
    f=Fixture()
    def execute(cmd):
        result=f.execute(cmd);result.feedback['command_contacts']['peak_robot_table_N']=.01
        return result
    out=api().execute_wait(command(),1.,f.status,execute)
    assert out['reason']=='robot_table_contact' and len(f.calls)==1


@pytest.mark.parametrize('executed,elapsed,terminal,interrupt,reason',[
    (True,.04,True,'unexpected_table_contact','terminal'),
    (True,.1,False,'chunk_interrupted_tracking_error','controller_interrupt'),
    (False,0.,False,'standing_stability_gate','zero_step'),
])
def test_terminal_tracking_and_zero_step_stop_without_synthetic_elapsed(executed,elapsed,terminal,interrupt,reason):
    f=Fixture()
    def execute(cmd):
        f.calls.append(cmd)
        return StepResult({'state':{'time':elapsed},'images':{}},executed,elapsed,
                          terminal=terminal,interrupt=interrupt,feedback={'admission':{'allowed':False}})
    out=api().execute_wait(command(),1.,f.status,execute)
    assert out['reason']==reason and out['command_count']==1
    assert out['elapsed_s']==elapsed and out['executed_commands']==int(executed)


@pytest.mark.parametrize('value',[True,0.,.15,5.1,float('nan'),float('inf'),'1'])
def test_wait_duration_rejects_unbounded_or_non_tick_aligned_values(value):
    with pytest.raises(ValueError):api().execute_wait(command(),value,Fixture().status,lambda cmd:None)


def test_wait_requires_prior_absolute_command_and_rejects_extra_motion_channels():
    for cmd in (None,{},dict(command(),body_keyframes=[])):
        with pytest.raises(ValueError):api().execute_wait(cmd,1.,Fixture().status,lambda cmd:None)


def test_wait_call_validates_envelope_and_exact_arguments():
    call={'type':'function_call','name':'wait_until_stable','call_id':'wait1',
          'arguments':json.dumps({'max_wait_s':2.,'execution_note':'Observe settling'})}
    assert api().validate_wait_call(call)=={'max_wait_s':2.,'execution_note':'Observe settling'}
    with pytest.raises(ValueError):api().validate_wait_call(call,seen_call_ids=['wait1'])
    for bad in (dict(call,name='act'),dict(call,arguments=json.dumps({'max_wait_s':2.,'execution_note':'x','right_fingers':[1]*7})),dict(call,arguments={})):
        with pytest.raises(ValueError):api().validate_wait_call(bad)


def test_incomplete_tick_feedback_cannot_silently_continue_waiting():
    f=Fixture()
    def execute(cmd):
        result=f.execute(cmd);result.feedback['command_contacts']['sample_count']=19
        return result
    out=api().execute_wait(command(),1.,f.status,execute)
    assert out['reason']=='feedback_incomplete' and len(f.calls)==1


def test_contact_coordinates_never_escape_wait_report():
    f=Fixture(hand=.8)
    def execute(cmd):
        result=f.execute(cmd)
        result.feedback['contacts']['events']=[dict(pair(),position=[12345,23456,34567])]
        result.feedback['command_contacts']['contact_pairs'][0]['position']=[12345,23456,34567]
        result.feedback['command_contacts']['contact_pairs'].append(pair('right_hand_thumb_link',3))
        return result
    out=api().execute_wait(command(),1.,f.status,execute)
    assert out['reason']=='hand_contact_changed'
    assert '12345' not in json.dumps(out)
