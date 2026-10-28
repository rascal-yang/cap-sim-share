"""No-physics checks of admission and measured-prefix feedback."""
from copy import deepcopy
import importlib
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest

from repro.bridge import NativeBridge
from repro.idle_bridge import StandingIdlePlanner
from repro.session import StepResult


def api():
    assert importlib.util.find_spec('repro.controlled_bridge') is not None, 'control-aware bridge is missing'
    return importlib.import_module('repro.controlled_bridge')


def spec():
    return SimpleNamespace(
        init_qpos=(0.,)*7,
        close_qpos_left=(.35, -.10, .28, -.51, -1.20, -.54, -.98),
        close_qpos_right=(.02, -.02, -.22, .26, 1.34, .31, .98),
        left_actuated_joints=tuple('left_'+str(i) for i in range(7)),
        right_actuated_joints=tuple('right_'+str(i) for i in range(7)))


def names():
    return {side:list(getattr(spec(), side+'_actuated_joints')) for side in ('left','right')}


def test_continuous_stability_uses_every_measured_tick_not_latest_low_speed():
    gate=api().MeasuredStabilityGate(initial_time=0.)
    for i in range(1,100):
        gate.update(i*.005, [0,0,0,0,0,0])
    assert not gate.feedback()['ready']
    gate.update(.5,[.03,0,0,0,0,-.1])
    assert gate.feedback()['ready']
    gate.update(.505,[0,0,0,0,0,.1001])
    assert gate.feedback()['continuous_s']==0
    gate.update(.510,[0,0,0,0,0,0])
    assert not gate.feedback()['ready']
    assert gate.feedback()['continuous_s']==pytest.approx(.005)


def test_stability_gap_duplicate_and_nonfinite_do_not_credit_unobserved_time():
    gate=api().MeasuredStabilityGate(initial_time=0.)
    gate.update(.005,[0]*6); gate.update(.005,[0]*6)
    assert gate.feedback()['continuous_s']==pytest.approx(.005)
    gate.update(.020,[0]*6)
    assert gate.feedback()['continuous_s']==0
    gate.update(.025,[0,0,0,0,0,float('nan')])
    assert not gate.feedback()['ready']
    assert gate.feedback()['continuous_s']==0
    gate.update(.030,[.025,.025,0,0,0,0])
    assert gate.feedback()['continuous_s']==0
    with pytest.raises(ValueError,match='time'):
        gate.update(.029,[0]*6)


def test_closure_profile_uses_joint_order_and_sign_not_any_finger_change():
    profile=api().dex3_closure_profile(spec(), names())
    current=[[0]*7,[0]*7]
    target=deepcopy(current); target[1][2]=-.2
    assert api().new_closure(target,current,profile)['new_closure']
    current=deepcopy(target); target[1][2]=-.1
    assert not api().new_closure(target,current,profile)['new_closure']
    target[0][4]=-.1
    assert api().new_closure(target,current,profile)['new_closure']
    bad=names();bad['right'][1]=bad['right'][2]
    with pytest.raises(ValueError,match='order'):
        api().dex3_closure_profile(spec(),bad)


def test_closure_profile_maps_named_source_joints_into_compiled_model_order():
    actual=names();actual['right'][3:]=[actual['right'][5],actual['right'][6],actual['right'][3],actual['right'][4]]
    profile=api().dex3_closure_profile(spec(),actual)
    assert profile['joint_names']['right']==['right_0','right_1','right_2','right_5','right_6','right_3','right_4']
    assert profile['close_qpos']['right']==[.02,-.02,-.22,.31,.98,.26,1.34]


def test_unknown_direction_is_explicit_and_fails_closed_for_changed_joint():
    hand=spec();hand.close_qpos_right=(0.,)+hand.close_qpos_right[1:]
    profile=api().dex3_closure_profile(hand,names())
    current=[[0]*7,[0]*7];target=deepcopy(current);target[1][0]=.2
    status=api().new_closure(target,current,profile)
    assert status['requires_stability'] and status['unknown_direction_changes']
    assert not status['new_closure']


def feed(mode=0,transformed=True,time=0.):
    return {'time':time,'transformed':transformed,'after':{'mode':[mode], 'target_vel':[-1],
        'movement_direction':[[0,0,0]],'height':[-1]}}


def command(stage='hold',right=None,velocity=None):
    return {'stage':stage,'duration_s':.1,'left_fingers':[0]*7,
            'right_fingers':right or [0]*7,'velocity_xy':velocity or [0,0],
            'pelvis_height_m':.78}


@pytest.fixture
def bridge(tmp_path):
    b=api().ControlAwareBridge(tmp_path,{},standing_idle=True)
    b.closure_profile=api().dex3_closure_profile(spec(),names())
    b.m=SimpleNamespace(nq=7,nv=6)
    b.d=SimpleNamespace(time=0.)
    b.states=[];b.contact_records=[];b.fingers=[np.zeros(7),np.zeros(7)]
    b.cached={'state':{'time':0.,'fingers':[[0]*7,[0]*7]},'images':{'head':'fixture','external':'fixture'}}
    b.idle_adapter=StandingIdlePlanner(SimpleNamespace(),enabled=True)
    b.idle_adapter.records=[feed()]
    b.stability=api().MeasuredStabilityGate(initial_time=0.)
    b.sonic=SimpleNamespace(planner=b.idle_adapter)
    return b


def make_stable(b):
    for i in range(1,101):
        b.stability.update(i*.005,[0]*6)
    b.d.time=.5


def test_unstable_contact_is_zero_step_and_preserves_exact_observation(bridge,monkeypatch):
    def forbidden(*args):raise AssertionError('rejected action reached physics')
    monkeypatch.setattr(NativeBridge,'step',forbidden)
    before=deepcopy(bridge.cached)
    result=bridge.step(command('contact'),30)
    assert not result.executed and result.elapsed_s==0 and not result.terminal
    assert result.observation==before and bridge.cached==before
    assert result.interrupt=='standing_stability_gate'
    assert not result.feedback['admission']['allowed']
    assert not result.feedback['stability']['ready']


@pytest.mark.parametrize('stage',['approach','hold','recover','walk'])
def test_stage_label_cannot_bypass_new_closure_gate(bridge,monkeypatch,stage):
    monkeypatch.setattr(NativeBridge,'step',lambda *a:pytest.fail('closure executed while moving'))
    result=bridge.step(command(stage,right=[0,0,-.1,0,0,0,0]),30)
    assert not result.executed
    assert result.feedback['admission']['closure']['new_closure']


@pytest.mark.parametrize('case',['disabled','old_feed','requested_motion','squat','missing_feed'])
def test_stable_state_alone_cannot_bypass_actual_idle_mapping(bridge,monkeypatch,case):
    make_stable(bridge);action=command('lift')
    if case=='disabled':bridge.idle_adapter.enabled=False
    elif case=='old_feed':bridge.idle_adapter.records=[feed(4,False)]
    elif case=='requested_motion':action['velocity_xy']=[.1,0]
    elif case=='squat':action['pelvis_height_m']=.6
    else:bridge.idle_adapter.records=[]
    monkeypatch.setattr(NativeBridge,'step',lambda *a:pytest.fail('inadmissible action executed'))
    assert not bridge.step(action,30).executed


def fake_physics(monkeypatch, *, terminal=False, velocities=None, forces=None):
    """Replace only the existing physics bridge, preserving real new admission/summary code."""
    def execute(b,action,remaining):
        start=b.d.time
        for i in range(20):
            b.d.time=start+(i+1)*.005
            velocity=[0]*6 if velocities is None else velocities[i]
            b.states.append(np.r_[b.d.time,np.zeros(7),velocity])
            table,hand=(0.,0.) if forces is None else forces[i]
            b.contact_records.append({'robot_table_N':table,'hand_object_N':hand,'events':[]})
        b.fingers=[np.array(action[side+'_fingers']) for side in ('left','right')]
        b.cached=deepcopy(b.cached);b.cached['state']['time']=b.d.time
        return StepResult(deepcopy(b.cached),True,.1,terminal=terminal,feedback={'tracking':{'fixture':'existing'}})
    monkeypatch.setattr(NativeBridge,'step',execute)


def test_opening_recovery_executes_without_stability_but_new_closure_does_not(bridge,monkeypatch):
    bridge.fingers[1][2]=-.2
    fake_physics(monkeypatch)
    result=bridge.step(command('recover',right=[0,0,-.1,0,0,0,0]),30)
    assert result.executed
    assert not result.feedback['admission']['closure']['new_closure']
    assert result.feedback['tracking']=={'fixture':'existing'}
    assert result.feedback['stability']['continuous_s']==pytest.approx(.1)


def test_stable_idle_admits_contact_and_reports_transients_not_only_endpoint(bridge,monkeypatch):
    make_stable(bridge)
    forces=[(0.,0.)]*20;forces[2:4]=[(2.,.5),(3.,.8)];forces[8]=(0.,2.)
    velocities=[[0]*6 for _ in range(20)];velocities[10]=[0,0,0,0,0,.2]
    fake_physics(monkeypatch,forces=forces,velocities=velocities)
    result=bridge.step(command('contact'),30)
    assert result.executed and not result.feedback['stability']['ready']
    assert result.feedback['stability']['continuous_s']==pytest.approx(.045)
    contacts=result.feedback['command_contacts']
    assert contacts['peak_robot_table_N']==3.
    assert contacts['peak_hand_object_N']==2.
    assert contacts['robot_table_intervals'][0]['observed_duration_s']==pytest.approx(.01)
    assert contacts['robot_table_intervals'][0]['first_sample_s']==pytest.approx(.515)
    assert len(contacts['hand_object_intervals'])==2
    assert contacts['hand_object_intervals'][-1]['last_sample_s']==pytest.approx(.545)


def test_requested_movement_resets_dwell_even_before_lower_planner_refresh(bridge,monkeypatch):
    make_stable(bridge)
    fake_physics(monkeypatch)
    result=bridge.step(command('walk',velocity=[.1,0]),30)
    assert result.executed
    assert result.feedback['stability']['continuous_s']==0
    assert not result.feedback['stability']['ready']


def test_stability_credits_only_ticks_covered_by_actual_idle_feed(bridge,monkeypatch):
    bridge.idle_adapter.records=[feed(4,False,time=0.),feed(time=.05)]
    fake_physics(monkeypatch)
    result=bridge.step(command(),30)
    assert result.feedback['stability']['continuous_s']==pytest.approx(.05)
    assert not result.feedback['stability']['ready']


def test_disabled_idle_adapter_never_accumulates_eligible_dwell(bridge,monkeypatch):
    bridge.idle_adapter.enabled=False
    fake_physics(monkeypatch)
    result=bridge.step(command(),30)
    assert result.feedback['stability']['continuous_s']==0
    assert not result.feedback['stability']['standing_idle_active_at_last_sample']


def test_terminal_result_latches_and_prevents_even_safe_future_physics(bridge,monkeypatch):
    fake_physics(monkeypatch,terminal=True)
    assert bridge.step(command(),30).terminal
    before=len(bridge.states)
    with pytest.raises(RuntimeError,match='terminal'):
        bridge.step(command('recover'),30)
    assert len(bridge.states)==before


def test_missing_saved_prefix_evidence_poison_bridge(bridge,monkeypatch):
    def broken(b,*args):
        b.d.time=.1
        return StepResult(deepcopy(b.cached),True,.1,feedback={})
    monkeypatch.setattr(NativeBridge,'step',broken)
    with pytest.raises(ValueError,match='saved'):
        bridge.step(command(),30)
    with pytest.raises(RuntimeError,match='terminal'):
        bridge.step(command(),30)


def test_export_includes_actual_lower_feeds_and_grounded_closure_profile(bridge,monkeypatch,tmp_path):
    monkeypatch.setattr(NativeBridge,'export',lambda *a:None)
    bridge.export(tmp_path)
    assert json.loads((tmp_path/'planner_inputs.json').read_text())[0]['after']['mode']==[0]
    assert json.loads((tmp_path/'closure_profile.json').read_text())['joint_names']==names()


def test_contact_summary_retains_disappearing_named_pairs_without_positions(bridge,monkeypatch):
    fake_physics(monkeypatch)
    inherited=NativeBridge.step
    def execute(b,*args):
        result=inherited(b,*args)
        b.contact_records[2].update(robot_table_N=2.,events=[{'bodies':['table','right_hip_yaw_link'],
            'geoms':['table_geom','right_hip_geom'],'force_N':2.,'position':[7,8,9]}])
        b.contact_records[3].update(robot_table_N=3.,events=[{'bodies':['right_hip_yaw_link','table'],
            'geoms':['right_hip_geom','table_geom'],'force_N':3.,'world_position':[7,8,9]}])
        return result
    monkeypatch.setattr(NativeBridge,'step',execute)
    result=bridge.step(command(),30)
    pairs=result.feedback['command_contacts']['contact_pairs']
    assert len(pairs)==1
    assert set(pairs[0]['bodies'])=={'table','right_hip_yaw_link'}
    assert pairs[0]['peak_force_N']==3.
    assert pairs[0]['last_sample_s']==pytest.approx(.020)
    assert set(pairs[0])=={'bodies','geoms','peak_force_N','first_sample_s','last_sample_s','sample_count'}


def test_reset_counts_inherited_steps_and_verifies_restored_state(bridge,monkeypatch,tmp_path):
    import mujoco
    from humano.robots import hands
    monkeypatch.setattr(hands,'DEX3_HAND',spec())
    def physical_boundary(model,data,nstep=1):data.time+=model.opt.timestep*nstep
    monkeypatch.setattr(mujoco,'mj_step',physical_boundary)
    def reset_boundary(b,task,out):
        allnames=names()['left']+names()['right']
        b.m=SimpleNamespace(opt=SimpleNamespace(timestep=.005),joint=lambda i:SimpleNamespace(name=allnames[i]))
        b.robot=SimpleNamespace(left_hand_actuated_index=np.arange(7),right_hand_actuated_index=np.arange(7,14))
        b.initial={'qpos':np.zeros(7),'qvel':np.zeros(6),'ctrl':np.zeros(2),'qacc_warmstart':np.zeros(6)}
        b.initial_q=b.initial['qpos'];b.initial_v=b.initial['qvel']
        b.d=SimpleNamespace(time=0.,**{k:v.copy() for k,v in b.initial.items()})
        b.sonic=None
        mujoco.mj_step(b.m,b.d,nstep=1)
        b.d.time=0.
        b.sonic=SimpleNamespace(planner=SimpleNamespace())
        return deepcopy(b.cached)
    monkeypatch.setattr(NativeBridge,'reset',reset_boundary)
    result=bridge.reset({},tmp_path)
    proof=json.loads((tmp_path/'reset_physics.json').read_text())
    assert result==bridge.cached and mujoco.mj_step is physical_boundary
    assert proof['actual_bootstrap_physics_steps']==1
    assert proof['calls'][0]['before_time_s']==0.
    assert proof['calls'][0]['after_time_s']==.005
    assert not proof['calls'][0]['sonic_constructed']
    assert proof['restored_initial_state']['all_hashes_match']
    assert proof['restored_initial_state']['time_s']==0
    assert bridge.sonic.planner is bridge.idle_adapter
    assert not bridge.control_status()['stability']['ready']


def test_reset_restores_step_function_even_when_inherited_reset_raises(bridge,monkeypatch,tmp_path):
    import mujoco
    original=mujoco.mj_step
    def failure(*args):raise RuntimeError('fixture reset failure')
    monkeypatch.setattr(NativeBridge,'reset',failure)
    with pytest.raises(RuntimeError,match='fixture reset failure'):
        bridge.reset({},tmp_path)
    assert mujoco.mj_step is original
    proof=json.loads((tmp_path/'reset_physics.json').read_text())
    assert proof['actual_bootstrap_physics_steps']==0
    assert proof['restored_initial_state'] is None
